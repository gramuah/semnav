#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List
import wandb
import numpy as np
import torch
import tqdm
import re
from gym import spaces
from habitat import Config, logger
from habitat.utils import profiling_wrapper
from habitat.utils.render_wrapper import overlay_frame
from habitat.utils.visualizations.utils import observations_to_image
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter, get_writer
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    init_distrib_slurm,
    is_slurm_batch_job,
    load_resume_state,
    rank0_only,
    requeue_job,
    save_resume_state,
)
from habitat_baselines.rl.ppo.ppo_trainer import PPOTrainer
from habitat_baselines.utils.common import (
    ObservationBatchingCache,
    action_array_to_dict,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
    linear_decay,
)

from torch import nn as nn
from torch.optim.lr_scheduler import CyclicLR
from torch.optim.lr_scheduler import LambdaLR


from pirlnav.algos.agent import DDPILAgent
from pirlnav.algos.agent import Semantic_DDPILAgent
from pirlnav.common.rollout_storage import RolloutStorage
import cv2
import sys
from types import SimpleNamespace
from PIL import Image
sys.path.append("/home/rafa/dgx_repositorios/semnav/ESANet")
from ESANet.src.prepare_data import prepare_data
from src.build_model import build_model
from src.args import ArgumentParserRGBDSegmentation
from src.prepare_data import prepare_data
import torch.nn.functional as F
from contextlib import contextmanager
import matplotlib.pyplot as plt

import pynvml

@contextmanager
def measure_resources():
    """Context manager para medir tiempo, memoria GPU, uso de GPU y potencia"""
    # Inicializar NVML para potencia
    power_samples = []
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        power_available = True
    except:
        power_available = False

    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()  # Reset peak tracking
        mem_before = torch.cuda.memory_allocated()
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        start_event.record()

    wall_start = time.time()

    if power_available:
        try:
            power_samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        except:
            pass

    yield

    if power_available:
        try:
            power_samples.append(pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0)
        except:
            pass

    wall_time = time.time() - wall_start

    if torch.cuda.is_available():
        end_event.record()
        torch.cuda.synchronize()
        gpu_time = start_event.elapsed_time(end_event) / 1000.0
        mem_after = torch.cuda.memory_allocated()
        mem_delta = mem_after - mem_before
        mem_peak = torch.cuda.max_memory_allocated() - mem_before  # Memoria adicional usada desde el reset

        power_info = ""
        if power_samples:
            avg_power = sum(power_samples) / len(power_samples)
            power_info = f", power={avg_power:.2f}W"

        print(f"Resources: wall={wall_time:.4f}s, "
              f"gpu={gpu_time:.4f}s, "
              f"mem_delta={mem_delta / (1024**2):.2f}MB, "
              f"mem_peak_used={mem_peak / (1024**2):.2f}MB{power_info}")

    if power_available:
        pynvml.nvmlShutdown()


def classes_to_semantic_rgb(semantic_classes, sem_constant):
    """
    Convierte clases semánticas a tensor RGB semántico.

    Args:
        semantic_classes: Tensor (B, H, W, 1) con clases semánticas
        constant: Constante para la conversión (e.g., 414534)

    Returns:
        Tensor (B, H, W, 3) RGB semántico
    """
    device = semantic_classes.device

    # Multiplicar las clases por la constante
    semantic_int = (semantic_classes.squeeze(-1).long() * sem_constant)

    # Extraer componentes RGB usando operaciones de bits
    rgb_tensor = torch.stack([
        (semantic_int >> 16) & 0xFF,  # R
        (semantic_int >> 8) & 0xFF,  # G
        semantic_int & 0xFF  # B
    ], dim=-1).float()

    return rgb_tensor

def semantic_rgb_to_classes(semantic_rgb_tensor, sem_constant):
    """
    Convierte tensor RGB semántico a clases originales.

    Args:
        semantic_rgb_tensor: Tensor (B, H, W, 3) RGB semántico

    Returns:
        Tensor (B, H, W, 1) con clases semánticas
    """
    device = semantic_rgb_tensor.device

    # Reconstruir el valor entero desde RGB
    semantic_int = (
            (semantic_rgb_tensor[..., 0].long() << 16) |
            (semantic_rgb_tensor[..., 1].long() << 8) |
            semantic_rgb_tensor[..., 2].long()
    )

    # Dividir por la constante para obtener las clases
    semantic_classes = (semantic_int // sem_constant).unsqueeze(-1)  # (B, H, W, 1)

    return semantic_classes


def add_semantic_class_confusion(semantic_tensor, prob=0.1, num_classes=40, sem_constant=414534):
    """
    Confunde categorías semánticas completas con probabilidad dada (optimizado para GPU).

    Args:
        semantic_tensor: Tensor (B, H, W, 1) con clases semánticas
        prob: Probabilidad de confusión por clase (0.0 a 1.0)
        num_classes: Número total de clases (default 40)

    Returns:
        Tensor RGB (B, H, W, 3)
    """
    device = semantic_tensor.device
    B, H, W, _ = semantic_tensor.shape
    semantic_float = semantic_tensor.float().squeeze(-1).clone()  # (B, H, W)

    # Vectorizado: obtener clases únicas por batch
    for b in range(B):
        unique_classes = torch.unique(semantic_float[b])

        # Generar decisiones de confusión para todas las clases a la vez
        confusion_decisions = torch.rand(len(unique_classes), device=device) < prob

        # Filtrar clases a confundir (excluir clase 0)
        classes_to_confuse = unique_classes[(unique_classes != 0) & confusion_decisions]

        if len(classes_to_confuse) == 0:
            continue

        # Generar nuevas clases aleatorias (excluir 0 y la clase original)
        new_classes = torch.randint(1, num_classes + 1, (len(classes_to_confuse),), device=device).float()

        # Crear máscara y reemplazar en una sola operación
        for old_class, new_class in zip(classes_to_confuse, new_classes):
            # Asegurar que new_class != old_class
            while new_class == old_class:
                new_class = torch.randint(1, num_classes + 1, (1,), device=device).float().item()

            mask = (semantic_float[b] == old_class)
            semantic_float[b][mask] = float(new_class)  # Conversión explícita a float

    # Conversión a RGB (operaciones vectorizadas en GPU)
    semantic_result_int = semantic_float.unsqueeze(-1).to(torch.int32)
    observations_mult = semantic_result_int * sem_constant

    # Operación de bits optimizada
    rgb_matrix = torch.stack([
        (observations_mult[:, :, :, 0] >> 16) & 0xFF,
        (observations_mult[:, :, :, 0] >> 8) & 0xFF,
        observations_mult[:, :, :, 0] & 0xFF
    ], dim=-1).float()

    return rgb_matrix



def add_semantic_class_blobs_perlin(semantic_tensor, num_blobs_range=(1, 10), blob_scale=50, sem_constant=414534):
    """
    Añade manchas grandes y localizadas en pocos puntos de la imagen.
    """
    device = semantic_tensor.device
    B, H, W, _ = semantic_tensor.shape
    semantic_float = semantic_tensor.float().clone()

    # Generar número de manchas para cada batch
    num_blobs = torch.randint(num_blobs_range[0], num_blobs_range[1] + 1, (B,), device=device)
    max_blobs = num_blobs.max().item()

    # Pre-generar todas las clases aleatorias
    random_classes = torch.randint(1, 41, (B, max_blobs), device=device)

    # Generar centros aleatorios para cada mancha
    center_y = torch.randint(0, H, (B, max_blobs), device=device)
    center_x = torch.randint(0, W, (B, max_blobs), device=device)

    # Crear grids de coordenadas
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )

    # Aplicar manchas localizadas
    for blob_idx in range(max_blobs):
        valid_mask = (num_blobs > blob_idx).float().view(B, 1, 1)

        for b in range(B):
            if valid_mask[b, 0, 0] > 0:
                # Centro de esta mancha
                cy = center_y[b, blob_idx]
                cx = center_x[b, blob_idx]

                # Generar ruido localizado solo alrededor del centro
                local_size = blob_scale * 2
                y_start = max(0, cy - local_size)
                y_end = min(H, cy + local_size)
                x_start = max(0, cx - local_size)
                x_end = min(W, cx + local_size)

                local_h = y_end - y_start
                local_w = x_end - x_start

                # Generar ruido solo en la región local
                noise = torch.randn((1, 1, local_h // 4, local_w // 4), device=device)
                noise = F.interpolate(noise, size=(local_h, local_w), mode='bilinear', align_corners=False)

                # Suavizar
                kernel_size = max(blob_scale // 3, 3)
                if kernel_size % 2 == 0:
                    kernel_size += 1
                noise = F.avg_pool2d(noise, kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

                # Normalizar
                noise = noise.squeeze()
                noise = (noise - noise.min()) / (noise.max() - noise.min() + 1e-8)

                # Crear máscara con umbral alto para manchas más definidas
                threshold = 0.6  # Umbral más alto = manchas más pequeñas y definidas
                blob_mask = (noise > threshold).float()

                # Aplicar la mancha solo en la región local
                current_class = random_classes[b, blob_idx].float()
                semantic_float[b, y_start:y_end, x_start:x_end, 0] = (
                    semantic_float[b, y_start:y_end, x_start:x_end, 0] * (1 - blob_mask) +
                    current_class * blob_mask
                )

    # Conversión a RGB
    semantic_result_int = semantic_float.to(torch.int32)
    observations_mult = semantic_result_int * sem_constant

    rgb_matrix = torch.stack([
        (observations_mult[:, :, :, 0] >> 16) & 0xFF,
        (observations_mult[:, :, :, 0] >> 8) & 0xFF,
        observations_mult[:, :, :, 0] & 0xFF
    ], dim=-1).float()

    return rgb_matrix



def add_semantic_noise_with_true_adjacent(semantic_tensor, sigma=1.5, warp_strength=5.0, neighborhood=3, sem_constant=414534):
    """
    Mezcla bordes solo con clases adyacentes reales.
    """
    device = semantic_tensor.device
    B, H, W, _ = semantic_tensor.shape
    semantic_float = semantic_tensor.float().permute(0,3,1,2)  # (B,1,H,W)

    # --- (1) Deformación espacial como antes ---
    noise = torch.randn((B,2,H//16,W//16),device=device)*warp_strength
    noise = F.interpolate(noise,size=(H,W),mode='bilinear',align_corners=False)
    kernel_size = int(2*round(2*sigma)+1)
    coords = torch.arange(kernel_size, device=device).float() - kernel_size//2
    gauss_1d = torch.exp(-0.5*(coords/sigma)**2)
    gauss_1d /= gauss_1d.sum()
    gauss_x = gauss_1d.view(1,1,1,-1)
    gauss_y = gauss_1d.view(1,1,-1,1)
    for c in range(2):
        n=noise[:,c:c+1,:,:]
        n=F.conv2d(n,gauss_x,padding=(0,kernel_size//2))
        n=F.conv2d(n,gauss_y,padding=(kernel_size//2,0))
        noise[:,c:c+1,:,:]=n

    yy,xx=torch.meshgrid(torch.linspace(-1,1,H,device=device),
                         torch.linspace(-1,1,W,device=device),
                         indexing='ij')
    base_grid = torch.stack((xx,yy),dim=-1).unsqueeze(0).repeat(B,1,1,1)
    norm_noise = noise / torch.tensor([W/2,H/2],device=device).view(1,2,1,1)
    norm_noise = norm_noise.permute(0,2,3,1)
    flow_grid = base_grid + norm_noise
    warped = F.grid_sample(semantic_float, flow_grid, mode='nearest', padding_mode='reflection', align_corners=False)

    # --- (2) Detectar bordes ---
    pad = neighborhood//2
    max_pool = F.max_pool2d(warped, kernel_size=neighborhood, stride=1, padding=pad)
    min_pool = -F.max_pool2d(-warped, kernel_size=neighborhood, stride=1, padding=pad)
    border_mask = (max_pool != min_pool).float()

    # --- (3) Para cada píxel de borde, elegir aleatoriamente una clase vecina ---
    # Extraer ventanas usando unfold
    unfolded = F.unfold(warped, kernel_size=neighborhood, padding=pad)  # (B, neighborhood^2, H*W)
    # Convertimos a int para manejar clases
    unfolded_int = unfolded.to(torch.int64)
    # Creamos tensor para clases únicas (por ventana)
    B_, K, N = unfolded_int.shape
    new_classes = torch.zeros_like(warped)
    for b in range(B_):
        vals = unfolded_int[b]  # (K, H*W)
        # Selección aleatoria de clase vecina
        idx = torch.randint(0, K, (N,), device=device)
        new_classes[b,0].view(-1).copy_(vals[idx, torch.arange(N, device=device)])
    # Mezclamos solo donde hay borde
    mixed = warped*(1-border_mask) + new_classes*border_mask

    # --- (4) Convertir a int ---
    semantic_result_int = mixed.to(torch.int32).permute(0,2,3,1)

    # --- (5) RGB ---
    observations_mult = semantic_result_int * sem_constant
    rgb_matrix = torch.zeros((B,H,W,3),dtype=torch.float32,device=device)
    rgb_matrix[:,:,:,0] = (observations_mult[:,:,:,0] >> 16) & 0xFF
    rgb_matrix[:,:,:,1] = (observations_mult[:,:,:,0] >> 8) & 0xFF
    rgb_matrix[:,:,:,2] = observations_mult[:,:,:,0] & 0xFF

    return rgb_matrix

@baseline_registry.register_trainer(name="pirlnav-il")
class ILEnvDDPTrainer(PPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)

        # LOAD ESANET
        self.segmentator_path = 'ESANet/trained_models/nyuv2_r34_NBt1D_scenenet/nyuv2/r34_NBt1D_scenenet.pth'
        self.segmentator_args = SimpleNamespace(activation='relu',aug_scale_max=1.4,aug_scale_min=1.0,batch_size=8,batch_size_valid=None,c_for_logarithmic_weighting=1.02, channels_decoder=128, ckpt_path=self.segmentator_path, class_weighting='median_frequency', context_module='ppm', dataset='nyuv2', dataset_dir=None, debug=False, decoder_channels_mode='decreasing', depth_scale=0.1, encoder='resnet34', encoder_block='NonBottleneck1D', encoder_decoder_fusion='add', encoder_depth=None, epochs=500, finetune=None, freeze=0, fuse_depth_in_rgb_encoder='SE-add', he_init=False, height=480, last_ckpt="", lr=0.01, modality='rgbd', momentum=0.9, nr_decoder_blocks=[3], optimizer='SGD', pretrained_dir='./trained_models/imagenet', pretrained_on_imagenet=False, pretrained_scenenet="", raw_depth=True, upsampling='learned-3x3-zeropad', valid_full_res=False, weight_decay=0.0001, width=640, workers=8)
        self.segmentator_args.pretrained_on_imagenet = False
        self.ESANET_dataset, self.ESANET_preprocessor = prepare_data(self.segmentator_args,with_input_orig=False)
        self.ESANET_n_classes = self.ESANET_dataset.n_classes_without_void
        self.ESANET_color_constant = np.floor((255*255*255)/40)
        self.seg_model, self.segmentator_device = build_model(self.segmentator_args, n_classes=self.ESANET_n_classes)
        # Prefer GPU for the segmentator if available to avoid CPU<->GPU copies

        self.ESANET_checkpoint = torch.load(self.segmentator_args.ckpt_path,
                                map_location=lambda storage, loc: storage)
        self.seg_model.load_state_dict(self.ESANET_checkpoint['state_dict'])
        print('Loaded ESANet model from {}'.format(self.segmentator_args.ckpt_path))
        self.seg_model.eval()
        self.seg_model.to(self.segmentator_device)

        # Constant for semantic RGB to class conversion
        self.constant = int(np.floor((255*255*255)/self.config.NUM_CATEGORIES))

        # Add noise to semantic segmentation
        self.add_noise = self.config.ADD_NOISE

        self.use_esanet = self.config.USE_ESANET

    def _setup_actor_critic_agent(self, il_cfg: Config) -> None:
        r"""Sets up actor critic and agent for IL.

        Args:
            il_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        observation_space = self.envs.observation_spaces[0]
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )
        self.obs_space = observation_space

        policy = baseline_registry.get_policy(self.config.IL.POLICY.name)
        self.actor_critic = policy.from_config(
            self.config, observation_space, self.envs.action_spaces[0]
        )
        self.actor_critic.to(self.device)
        if 'semantic' in observation_space.spaces:
            self.agent = Semantic_DDPILAgent(
                actor_critic=self.actor_critic,
                num_envs=self.envs.num_envs,
                num_mini_batch=il_cfg.num_mini_batch,
                lr=il_cfg.lr,
                encoder_lr=il_cfg.encoder_lr,
                eps=il_cfg.eps,
                max_grad_norm=il_cfg.max_grad_norm,
                wd=il_cfg.wd,
                entropy_coef=il_cfg.entropy_coef,
                semantic_constant = self.constant
            )
        else:
            self.agent = DDPILAgent(
                actor_critic=self.actor_critic,
                num_envs=self.envs.num_envs,
                num_mini_batch=il_cfg.num_mini_batch,
                lr=il_cfg.lr,
                encoder_lr=il_cfg.encoder_lr,
                eps=il_cfg.eps,
                max_grad_norm=il_cfg.max_grad_norm,
                wd=il_cfg.wd,
                entropy_coef=il_cfg.entropy_coef,
            )

    def _init_train(self):
        #If there is some checkpoint we will want to take this checkpoint in order to start working again

        resume_state = load_resume_state(self.config)

        #Unless is there is no checkpoint, in that case we will have nothing to unfreeze
        #This line changes the configuration to the one the ckpt was done, not interesting for us in this moment
        if resume_state is not None:
            #None
            # resume_state["config"]['NUM_UPDATES'] = 60000000
            # resume_state["config"]['NUM_CHECKPOINTS'] = 1000
            self.config: Config = resume_state["config"]
        #Distributed is in order to parallel the work, it seems to be necessary
        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        # Add replay sensors
        self.config.defrost()

        #This are the sensors that will be repeated in behavior cloning
        self.config.TASK_CONFIG.TASK.SENSORS.extend(
            ["DEMONSTRATION_SENSOR", "INFLECTION_WEIGHT_SENSOR"]
        )

        self.config.freeze()

        if self._is_distributed:
            #All of this seems to be configurations in order to parallel the work
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )


            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            #Save the gpus to be used in the task
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            #Generate seeds for every GPU and for every environment
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)

            #A rollout will be a sequence of steps without interruption
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        #Environments are initialized
        self._init_envs()

        #What actions are going to be used 4 or 6
        action_space = self.envs.action_spaces[0]
        self.policy_action_space = action_space

        #We check if we are taking discrete point nav or continuous pointnav
        if is_continuous_action_space(action_space):
            # Assume ALL actions are NOT discrete
            action_shape = (get_num_actions(action_space),)
            discrete_actions = False
        else:
            # For discrete pointnav
            action_shape = None
            discrete_actions = True
        #Save IL configurations
        il_cfg = self.config.IL.BehaviorCloning
        #Save Policy configurations
        policy_cfg = self.config.POLICY
        #POLICY INCLUDES: RGB ENCODER: hidden_size  512 image size 256
        # State encoder: hidden size 2048 recurrent layers 2 (GRU)
        # use previous action: True
        #critic: in il no_critic True
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        #Set the critic
        self._setup_actor_critic_agent(il_cfg)
        #Distributed training start, find_unused_params is a way to construct the training and communicate different
        #proccesses in a distributed training
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)  # type: ignore

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )
        #starts getting the space information
        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if il_cfg.use_double_buffered_sampler else 1
        #sets rollouts initialization
        self.rollouts = RolloutStorage(
            il_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            policy_cfg.STATE_ENCODER.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=il_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()

        ###########
        if 'semantic' in obs_space.spaces:
            for i in range(self.envs.num_envs):
                observations[i]["semantic_rgb"] = np.zeros([480,640,3])
        ###########
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=il_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )
        t_sample_action = time.time()
        # fetch actions from replay buffer
        step_batch = self.rollouts.buffers[
            self.rollouts.current_rollout_step_idxs[buffer_index],
            env_slice,
        ]

        next_actions = step_batch["observations"]["next_actions"]
        actions = next_actions.long().unsqueeze(-1)

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if act.shape[0] > 1:
                step_action = action_array_to_dict(
                    self.policy_action_space, act
                )
            else:
                step_action = act.item()
            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            actions=actions,
            buffer_index=buffer_index,
        )

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        t_update_model = time.time()

        self.agent.train()

        (
            action_loss,
            rnn_hidden_states,
            dist_entropy,
            _,
        ) = self.agent.update(self.rollouts)

        self.rollouts.after_update(rnn_hidden_states)
        self.pth_time += time.time() - t_update_model

        return (
            action_loss,
            dist_entropy,
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0
        il_cfg = self.config.IL.BehaviorCloning
        lr_scheduler = LambdaLR(
            lr_lambda=lambda x: 1 - self.percent_done(),
        )
        # lr_scheduler = CyclicLR(
        #     optimizer=self.agent.optimizer,
        #     mode='exp_range',
        #     base_lr=il_cfg.lr,
        #     max_lr=il_cfg.lr*il_cfg.CYCLIC_LR.multiplication_factor,
        #     gamma=il_cfg.CYCLIC_LR.gamma,
        #     cycle_momentum=False,
        #     step_size_up=il_cfg.CYCLIC_LR.step_size_up
        # )
        resume_state = load_resume_state(self.config)

        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        #PPO config and il_config are saved
        ppo_cfg = self.config.RL.PPO
        il_cfg = self.config.IL.BehaviorCloning

        with (
            #Tensorboard
            get_writer(self.config, flush_secs=self.flush_secs)
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if il_cfg.use_linear_clip_decay:
                    self.agent.clip_param = il_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
                    #Resume state is saved
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                self.agent.eval() #When is training, not done
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(il_cfg.num_steps):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == il_cfg.num_steps
                    )

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                profiling_wrapper.range_push(
                                    "_collect_rollout_step"
                                )

                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                (
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                if il_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        action_loss=action_loss,
                        entropy=dist_entropy,
                    ),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()
            cv2.destroyAllWindows()

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"losses/{k}", v, self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)
        writer.add_scalar("metrics/fps", fps, self.num_steps_done)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                    "  ".join(
                        "{}: {:.3f}".format(k, v) for k, v in losses.items()
                    ),
                )
            )

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.EVAL.SHOULD_LOAD_CKPT:
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
        else:
            ckpt_dict = {}

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.DATASET.TYPE = "ObjectNav-v1"
        config.freeze()

        if (
            len(self.config.VIDEO_OPTION) > 0
            and self.config.VIDEO_RENDER_TOP_DOWN
        ):
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

        action_space = self.envs.action_spaces[0]
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = action_space
            if is_continuous_action_space(action_space):
                # Assume NONE of the actions are discrete
                action_shape = (get_num_actions(action_space),)
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = (1,)
                discrete_actions = True

        il_cfg = config.IL.BehaviorCloning
        policy_cfg = config.POLICY
        self._setup_actor_critic_agent(il_cfg)

        if self.agent.actor_critic.should_load_agent_state:
            self.agent.load_state_dict({
                k.replace("model.", "actor_critic."): v
                for k, v in ckpt_dict["state_dict"].items()
            })
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        if self.add_noise:
            # Aplicar ruido con bordes difuminados
            batch["semantic_rgb"] = add_semantic_class_confusion(batch["semantic"],prob=0.1, num_classes=40, sem_constant=self.constant)
            batch["semantic"] = semantic_rgb_to_classes(batch["semantic_rgb"], self.constant)
            batch["semantic_rgb"]=add_semantic_noise_with_true_adjacent(batch["semantic"], sigma=1.5, warp_strength=5.0, neighborhood=3)
        # En tu código de evaluación:

            batch["semantic"] = semantic_rgb_to_classes(batch["semantic_rgb"], self.constant)
            batch["semantic_rgb"] = add_semantic_class_blobs_perlin(
                batch["semantic"], num_blobs_range=(1, 10), blob_scale=50
            )

        elif self.use_esanet:

            rgb_batch = batch['rgb'].cpu().numpy()  # -> (B, 3, H, W)
            depth_batch = batch['depth'].cpu().numpy()  # -> (B, 1, H, W)
            depth_batch = (depth_batch * 1000).astype(np.uint16)

            rgb_list = [np.transpose(rgb, (0, 1, 2)).copy() for rgb in rgb_batch]  # (H, W, 3)
            depth_list = [np.transpose(depth, (0, 1, 2)).copy() for depth in depth_batch]  # (H, W)

            sample_to_semantic = [self.ESANET_preprocessor({'image': rgb, 'depth': depth}) for rgb, depth in
                                  zip(rgb_list, depth_list)]
            image_sem = torch.stack([s['image'] for s in sample_to_semantic]).to(
                self.segmentator_device)  # (B, 3, H, W)
            depth_sem = torch.stack([s['depth'] for s in sample_to_semantic]).to(
                self.segmentator_device)  # (B, 1, H, W)

            pred = self.seg_model(image_sem, depth_sem)  # (B, num_classes, H, W)

            pred = F.interpolate(pred, size=(480, 640), mode='bilinear',
                                 align_corners=False)  # (B, num_classes, 480, 640)

            pred = torch.argmax(pred, dim=1)  # (B, 480, 640)

            pred = pred.cpu().numpy().astype(np.uint8) + 1
            pred = pred * self.ESANET_color_constant
            B, H, W = pred.shape
            pred_uint64 = pred.astype(np.uint64)

            rgb_matrix = np.zeros((B, H, W, 3), dtype=np.uint8)

            rgb_matrix[..., 0] = (pred_uint64 >> 16) & 0xFF  # R
            rgb_matrix[..., 1] = (pred_uint64 >> 8) & 0xFF  # G
            rgb_matrix[..., 2] = pred_uint64 & 0xFF  # B
            batch["semantic_rgb"] = rgb_matrix
            rgb_matrix_uint8 = rgb_matrix.astype(np.uint8)

            # Convertimos a tensor
            semantic_rgb_tensor = torch.from_numpy(rgb_matrix_uint8)  # dtype torch.uint8, shape (B, H, W, 3)

            # Guardamos en el batch
            batch["semantic_rgb"] = semantic_rgb_tensor.to(self.device)

        else:
            batch["semantic_rgb"] = classes_to_semantic_rgb(batch["semantic"], self.constant)

        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.net.num_recurrent_layers,
            policy_cfg.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        logger.info("Sampling actions deterministically...")
        self.actor_critic.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()
            with torch.no_grad():
                actions, test_recurrent_hidden_states = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    deterministic=True,
                )
                # batch["rgb"] = torch.zeros(2,480,640,3).to("cuda:0")
                # (
                #     actions,
                #     test_recurrent_hidden_states,
                # ) = self.actor_critic.act(
                #     batch,
                #     prueba_test_recurrent_hidden_states,
                #     prev_actions,
                #     not_done_masks,
                #     deterministic=True,
                # )
                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            if actions[0].shape[0] > 1:
                step_data = [
                    action_array_to_dict(self.policy_action_space, a)
                    for a in actions.to(device="cpu")
                ]
            else:
                step_data = [a.item() for a in actions.to(device="cpu")]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            if self.add_noise:
                # Aplicar ruido con bordes difuminados
                batch["semantic_rgb"] = add_semantic_class_confusion(batch["semantic"], prob=0.1, num_classes=40, sem_constant=self.constant)
                batch["semantic"] = semantic_rgb_to_classes(batch["semantic_rgb"], self.constant)
                batch["semantic_rgb"] = add_semantic_noise_with_true_adjacent(batch["semantic"], sigma=1.5,
                                                                              warp_strength=5.0, neighborhood=3)
                # En tu código de evaluación:

                batch["semantic"] = semantic_rgb_to_classes(batch["semantic_rgb"], self.constant)
                batch["semantic_rgb"] = add_semantic_class_blobs_perlin(
                    batch["semantic"], num_blobs_range=(1, 10), blob_scale=50
                )
            elif self.use_esanet:

                rgb_batch = batch['rgb'].cpu().numpy()  # -> (B, 3, H, W)
                depth_batch = batch['depth'].cpu().numpy()  # -> (B, 1, H, W)
                depth_batch = (depth_batch * 1000).astype(np.uint16)

                rgb_list = [np.transpose(rgb, (0, 1, 2)).copy() for rgb in rgb_batch]  # (H, W, 3)
                depth_list = [np.transpose(depth, (0, 1, 2)).copy() for depth in depth_batch]  # (H, W)

                sample_to_semantic = [self.ESANET_preprocessor({'image': rgb, 'depth': depth}) for rgb, depth in
                                      zip(rgb_list, depth_list)]
                image_sem = torch.stack([s['image'] for s in sample_to_semantic]).to(
                    self.segmentator_device)  # (B, 3, H, W)
                depth_sem = torch.stack([s['depth'] for s in sample_to_semantic]).to(
                    self.segmentator_device)  # (B, 1, H, W)

                pred = self.seg_model(image_sem, depth_sem)  # (B, num_classes, H, W)

                pred = F.interpolate(pred, size=(480, 640), mode='bilinear',
                                     align_corners=False)  # (B, num_classes, 480, 640)

                pred = torch.argmax(pred, dim=1)  # (B, 480, 640)

                pred = pred.cpu().numpy().astype(np.uint8) + 1
                pred = pred * self.ESANET_color_constant
                B, H, W = pred.shape
                pred_uint64 = pred.astype(np.uint64)

                rgb_matrix = np.zeros((B, H, W, 3), dtype=np.uint8)

                rgb_matrix[..., 0] = (pred_uint64 >> 16) & 0xFF  # R
                rgb_matrix[..., 1] = (pred_uint64 >> 8) & 0xFF  # G
                rgb_matrix[..., 2] = pred_uint64 & 0xFF  # B
                batch["semantic_rgb"] = rgb_matrix
                rgb_matrix_uint8 = rgb_matrix.astype(np.uint8)

                # Convertimos a tensor
                semantic_rgb_tensor = torch.from_numpy(rgb_matrix_uint8)  # dtype torch.uint8, shape (B, H, W, 3)

                # Guardamos en el batch
                batch["semantic_rgb"] = semantic_rgb_tensor.to(self.device)
            else:
                batch["semantic_rgb"] = classes_to_semantic_rgb(batch["semantic"], self.constant)

            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.VIDEO_FPS,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.EVAL_KEYS_TO_INCLUDE_IN_NAME,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    frame = observations_to_image(
                        {k: v[i] for k, v in batch.items()}, infos[i]
                    )
                    if self.config.VIDEO_RENDER_ALL_INFO:
                        frame = overlay_frame(frame, infos[i])

                    rgb_frames[i].append(frame)

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()


import contextlib
import os
import random
import time
from collections import defaultdict, deque
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import tqdm
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
from semnav.common.rollout_storage import RolloutStorage
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
    action_array_to_dict,
    batch_obs,
    generate_video,
    get_num_actions,
    is_continuous_action_space,
)

from semnav.algos.ppo import DDPPO, PPO
from semnav.utils.lr_scheduler import PIRLNavLRScheduler


@baseline_registry.register_trainer(name="semnav-ddppo")
@baseline_registry.register_trainer(name="semnav-ppo")
class PIRLNavPPOTrainer(PPOTrainer):
    def __init__(self, config=None):
        super().__init__(config)
    def _update_agent(self):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]
            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        self.agent.train()

        value_loss, action_loss, dist_entropy = self.agent.update(
            self.rollouts
        )

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )
    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]
        constant = 414534
        # constant = 9994
        semantic_array = torch.tensor(
            np.array([obs["semantic"] for obs in observations]),
            dtype=torch.int32,
            device=self.device
        ) * constant

        # Crear el tensor `rgb_matrix` en el formato esperado (batch, 480, 640, 3)
        rgb_matrix = torch.zeros((semantic_array.size(0), 480, 640, 3), dtype=torch.uint8, device=semantic_array.device)
        rgb_matrix[:, :, :, 0] = (semantic_array[:, :, :, 0] >> 16) & 0xFF  # R
        rgb_matrix[:, :, :, 1] = (semantic_array[:, :, :, 0] >> 8) & 0xFF  # G
        rgb_matrix[:, :, :, 2] = semantic_array[:, :, :, 0] & 0xFF  # B

        # Añadir `semantic_rgb` a cada observación sin bucle
        for i, obs in enumerate(observations):
            obs["semantic_rgb"] = rgb_matrix[i].cpu().numpy()
        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        constant = 414534
        # constant = 9994

        # print(batch["observations"]["rgb"])
        # print(torch.any(batch["observations"]["semantic"]))
        observations_mult = self.rollouts.buffers["observations"]["semantic"] * constant

        rgb_matrix = torch.zeros((observations_mult.size(0), observations_mult.size(1), 480, 640, 3), dtype=torch.uint8,
                                 device=observations_mult.device)
        rgb_matrix[:, :, :, :, 0] = (observations_mult[:, :, :, :, 0] >> 16) & 0xFF  # R
        rgb_matrix[:, :, :, :, 1] = (observations_mult[:, :, :, :, 0] >> 8) & 0xFF  # G
        rgb_matrix[:, :, :, :, 2] = observations_mult[:, :, :, :, 0] & 0xFF  # B
        self.rollouts.buffers["observations"]["semantic_rgb"] = rgb_matrix
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

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
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )
    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.
        Args:
            ppo_cfg: config node with relevant params
        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        logger.info("Setting up policy in PIRLNav trainer..........")

        self.actor_critic = policy.from_config(
            self.config, observation_space, self.policy_action_space
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):
            try:
                pretrained_state = torch.load(
                    self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
                )
                logger.info("Weights Fine.")
            except Exception as e:
                logger.error("Weights not fine: %s", e)

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {k[len("actor_critic."):]: v for k, v in pretrained_state["state_dict"].items()}, strict=False
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = (DDPPO if self._is_distributed else PPO)(
            actor_critic=self.actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
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

        lr_scheduler = PIRLNavLRScheduler(
            optimizer=self.agent.optimizer,
            agent=self.agent,
            num_updates=self.config.NUM_UPDATES,
            base_lr=self.config.RL.PPO.lr,
            finetuning_lr=self.config.RL.Finetune.lr,
            ppo_eps=self.config.RL.PPO.eps,
            start_actor_update_at=self.config.RL.Finetune.start_actor_update_at,
            start_actor_warmup_at=self.config.RL.Finetune.start_actor_warmup_at,
            start_critic_update_at=self.config.RL.Finetune.start_critic_update_at,
            start_critic_warmup_at=self.config.RL.Finetune.start_critic_warmup_at,
        )
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

        ppo_cfg = self.config.RL.PPO

        with (
            get_writer(self.config, flush_secs=self.flush_secs)
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
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

                self.agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")
                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(ppo_cfg.num_steps):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == ppo_cfg.num_steps
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
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        value_loss=value_loss,
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

    def _init_train(self):

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config: Config = resume_state["config"]
            self.using_velocity_ctrl = (
                self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
            ) == ["VELOCITY_CONTROL"]

        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
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
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
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

        self._init_envs()

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
                # Assume ALL actions are NOT discrete
                action_shape = (get_num_actions(action_space),)
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = None
                discrete_actions = True

        ppo_cfg = self.config.RL.PPO
        policy_cfg = self.config.POLICY
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)  # type: ignore

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

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

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            policy_cfg.STATE_ENCODER.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        if 'semantic' in obs_space.spaces:
            for i in range(self.envs.num_envs):
                observations[i]["semantic_rgb"] = np.zeros([480,640,3])
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
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

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

        lrs = {}
        for i, param_group in enumerate(self.agent.optimizer.param_groups):
            lrs["pg_{}".format(i)] = param_group["lr"]
        writer.add_scalars("learning_rate", lrs, self.num_steps_done)

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
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
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
        for seed in random.sample(range(10000), 14):
            logger.info('Setting seed to %d', seed)
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

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

            ppo_cfg = config.RL.PPO
            policy_cfg = config.POLICY

            config.defrost()
            config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
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

            self._setup_actor_critic_agent(ppo_cfg)

            if self.agent.actor_critic.should_load_agent_state:
                self.agent.load_state_dict(ckpt_dict["state_dict"])
            self.actor_critic = self.agent.actor_critic

            observations = self.envs.reset()
            batch = batch_obs(
                observations, device=self.device, cache=self._obs_batching_cache
            )
            constant = 414534
            #constant = 9994

            observations_mult = batch["semantic"] * constant

            rgb_matrix = torch.zeros((observations_mult.size(0), 480, 640, 3), dtype=torch.uint8,
                                     device=observations_mult.device)
            rgb_matrix[:, :, :, 0] = (observations_mult[:, :, :, 0] >> 16) & 0xFF  # R
            rgb_matrix[:, :, :, 1] = (observations_mult[:, :, :, 0] >> 8) & 0xFF  # G
            rgb_matrix[:, :, :, 2] = observations_mult[:, :, :, 0] & 0xFF  # B

            batch["semantic_rgb"] = rgb_matrix
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            current_episode_reward = torch.zeros(
                self.envs.num_envs, 1, device="cpu"
            )

            test_recurrent_hidden_states = torch.zeros(
                self.config.NUM_ENVIRONMENTS,
                self.actor_critic.num_recurrent_layers,
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
            self.actor_critic.eval()
            while (
                len(stats_episodes) < number_of_eval_episodes
                and self.envs.num_envs > 0
            ):
                current_episodes = self.envs.current_episodes()

                with torch.no_grad():
                    (
                        _,
                        actions,
                        _,
                        test_recurrent_hidden_states,
                    ) = self.actor_critic.act(
                        batch,
                        test_recurrent_hidden_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=False,
                    )

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
                constant = 414534
                # constant = 9994

                observations_mult = batch["semantic"] * constant

                rgb_matrix = torch.zeros((observations_mult.size(0), 480, 640, 3), dtype=torch.uint8, device=observations_mult.device)
                rgb_matrix[:, :, :, 0] = (observations_mult[:, :, :, 0] >> 16) & 0xFF  # R
                rgb_matrix[:, :, :, 1] = (observations_mult[:, :, :, 0] >> 8) & 0xFF  # G
                rgb_matrix[:, :, :, 2] = observations_mult[:, :, :, 0] & 0xFF  # B
                batch["semantic_rgb"] = rgb_matrix
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
                        # TODO move normalization / channel changing out of the policy and undo it here
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
            logger.info(f"Seed: {seed}")

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

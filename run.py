#!/usr/bin/env python3
import argparse
import os
import random
from datetime import datetime

import numpy as np
import numba
import quaternion
import torch
import habitat
import wandb
from habitat import logger
from habitat.config import Config
from habitat_baselines.common.baseline_registry import baseline_registry

from semnav.config import get_config
from habitat_baselines.rl.ddppo.ddp_utils import load_resume_state

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )
    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def execute_exp(config: Config, run_type: str) -> None:
    r"""This function runs the specified config with the specified runtype
    Args:
    config: Habitat.config
    runtype: str {train or eval}
    """
    # set a random seed (from detectron2)
    seed = (
            os.getpid()
            + int(datetime.now().strftime("%S%f"))
            + int.from_bytes(os.urandom(2), "big")
    )
    logger.info("Using a generated random seed {}".format(seed))
    config.defrost()
    config.RUN_TYPE = run_type
    config.TASK_CONFIG.SEED = seed
    config.freeze()
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    if config.FORCE_TORCH_SINGLE_THREADED and torch.cuda.is_available():
        torch.set_num_threads(1)

    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """

    config = get_config(exp_config, opts)
    resume_state = load_resume_state(config)
    if resume_state is not None:
        wandb_id = resume_state["config"]["WANDB_UNIQUE_ID"]
        if wandb_id is not None:
            config.defrost()
            config.WANDB_UNIQUE_ID = wandb_id
            config.freeze()
        del resume_state
    if getattr(config, "WANDB_UNIQUE_ID", None) is None:
        config.defrost()
        # if we're going to restart the experiment, this will be saved to a json file
        config.WANDB_UNIQUE_ID = f'{run_type}-{config.TENSORBOARD_DIR.split("/")[-1]}_{datetime.now().strftime("%Y%m%d_%H%M%S_%f")}'
        config.freeze()

    if config.WANDB_ENABLED:
        if os.environ.get("LOCAL_RANK", None) is not None:
            world_rank = int(os.environ["RANK"])
        # Else parse from SLURM is using SLURM
        elif os.environ.get("SLURM_JOBID", None) is not None:
            world_rank = int(os.environ["SLURM_PROCID"])
        # Else parse from TSUBAME if using TSUBAME
        elif os.environ.get("JOB_ID", None) is not None:
            world_rank = int(os.environ["OMPI_COMM_WORLD_RANK"])
        else:
            world_rank = 0
        # print(f"local_rank: {local_rank}, global_rank: {global_rank}")
        if int(world_rank) == 0:  # multinode job
            wandb.init(project="semnav",
                             name=f'{run_type}-{config.TENSORBOARD_DIR.split("/")[-1]}',
                             sync_tensorboard=True,
                             config=config,
                             entity='gram-uah',
                             tags=[f'{run_type}', 'dgx'],
                             id=config.WANDB_UNIQUE_ID,
                             resume="allow")
    execute_exp(config, run_type)


if __name__ == "__main__":
    main()

#!/bin/bash

export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet

config="configs/experiments/il_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d/objectnav_hm3d_v1"
TENSORBOARD_DIR="tb/pirlnavwDINO31"
CHECKPOINT_DIR="data/checkpoints/pirlnavwDINO31"
EVAL_CKPT_PATH_DIR="data/checkpoints/pirlnavwDINO31"

echo "In ObjectNav IL DDP"
python -u -m run \
    --exp-config $config \
    --run-type eval \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 50000 \
    NUM_ENVIRONMENTS 20 \
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \

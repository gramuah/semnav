#!/bin/bash

# ******************* Setup dirs *************************************************
config="configs/experiments/il_objectnav.yaml"
DATA_PATH="data/datasets/objectnav/objectnav_hm3d_hd"
TENSORBOARD_DIR="tb/semantic_rgb_lrcycliccor0.00001dgx_NOpretrainedencoder40categories2"
CHECKPOINT_DIR="data/checkpoints/semantic_rgb_lrcyliccor0.0000100001dgx_NOpretrainedencoder40categories2"
INFLECTION_COEF=3.234951275740812

# ******************* Set nvidia-smi to log GPU usage ******************************
mkdir -p "$TENSORBOARD_DIR"
mkdir -p "$CHECKPOINT_DIR"

nvidia-smi --query-gpu=timestamp,name,gpu_bus_id,utilization.gpu,utilization.memory,memory.used,memory.free \
    --format=csv -l 1 > "${TENSORBOARD_DIR}"/gpu-usage.log &
NVIDIA_SMI_PID=$!

# ******************* Setup number of cpus and gpus *******************************
NUM_CPUS=$(nproc)
NGPU_PER_NODE=$(nvidia-smi -L | wc -l)
NHOSTS=1

# ******************* Export variables ********************************************
export CONFIG
export DATA_PATH
export TENSORBOARD_DIR
export CHECKPOINT_DIR
export GLOG_minloglevel=2
export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export LOG_DIR
export NGPU_PER_NODE
export OMP_NUM_THREADS=$((NUM_CPUS/NGPU_PER_NODE))

# ******************* Run the training script *******************************
echo "Getting number of cpus and cpus per node..."
echo "NUM_CPUS: $NUM_CPUS", "NGPU_PER_NODE: $NGPU_PER_NODE", "CPUS PER GPU: $OMP_NUM_THREADS"
echo "Running imitation learning..."

torchrun --nnodes="${NHOSTS}" \
  --nproc_per_node="${NGPU_PER_NODE}" \
  --max_restarts 3 \
  run.py \
    TENSORBOARD_DIR $TENSORBOARD_DIR \
    CHECKPOINT_FOLDER $CHECKPOINT_DIR \
    NUM_UPDATES 320000 \
    NUM_ENVIRONMENTS 16 \
    IL.BehaviorCloning.num_mini_batch 2\
    EVAL.USE_CKPT_CONFIG True\
    RL.DDPPO.force_distributed True \
    TASK_CONFIG.DATASET.DATA_PATH "$DATA_PATH/{split}/{split}.json.gz" \
    TASK_CONFIG.TASK.INFLECTION_WEIGHT_SENSOR.INFLECTION_COEF $INFLECTION_COEF

kill $NVIDIA_SMI_PID
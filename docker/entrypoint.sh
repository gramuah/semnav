#!/bin/bash

echo "Starting training script: $(date)"

. activate habitat
cd ~/code/ && pip install -e .
bash scripts/launch_training.sh
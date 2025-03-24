#!/bin/bash

echo "Iniciando script de entrenamiento: $(date)"

. activate habitat
cd ~/code/ && pip install -e .
bash scripts/launch_training.sh
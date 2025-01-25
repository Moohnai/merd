#!/bin/bash

# Script: run_distributed.sh
# Usage: bash run_distributed.sh

# Ensure you have the correct number of GPUs
NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
if [ "$NUM_GPUS" -le 0 ]; then
  echo "No GPUs available. Exiting."
  exit 1
fi

# Set the master address and port for distributed training
export MASTER_ADDR=localhost
export MASTER_PORT=12345  # Change this if the port is already in use

# Run the distributed training script
python -m torch.distributed.run \
    --nproc_per_node=$NUM_GPUS \
    test.py

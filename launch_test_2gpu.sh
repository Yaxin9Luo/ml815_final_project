#!/bin/bash

# Launch script for testing with 2 GPUs (GPU 0 and 1) using 1% of dataset
# This script uses CUDA_VISIBLE_DEVICES to restrict to GPUs 0 and 1

echo "Launching training with GPUs 0 and 1..."
echo "Using 1% of dataset and reduced iterations for testing"

# Set CUDA_VISIBLE_DEVICES to use only GPUs 0 and 1
export CUDA_VISIBLE_DEVICES=0,1

# Launch with DeepSpeed using 2 GPUs
deepspeed --num_gpus=2 train_deepspeed.py config/train_gpt2_deepspeed_zero1.py --disable_flash_attention=True

echo "Training completed!"


#!/bin/bash

# Set environment variables
set -a
source .env
set +a
export WANDB_NAME="TASK_2_1_4"
export VLLM_USE_V1=0

# Run lighteval command
lighteval vllm \
  "model_name=/root/autodl-tmp/output/metamath/checkpoint-375,dtype=bfloat16,max_model_length=4096,max_num_batched_tokens=4096,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:4096,temperature:0.15,top_p:0.95}" \
  "lighteval|gsm8k|0" \
  --wandb --save-details
  
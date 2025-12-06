#!/bin/bash

# Set environment variables
source .env
export WANDB_NAME="TASK_2_1_1"

# Run lighteval command
lighteval vllm \
  "model_name=Qwen/Qwen2.5-0.5B-Instruct,dtype=bfloat16,max_model_length=4096,gpu_memory_utilization=0.9,generation_parameters={max_new_tokens:4096,temperature:0.6,top_p:0.95}" \
  "lighteval|gsm8k|5" \
  --wandb --save-details
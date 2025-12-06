#!/bin/bash

source .env
export WANDB_NAME="TASK_2_1_3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd /root/LLaMA-Factory

FORCE_TORCHRUN=1 llamafactory-cli train \
  /root/LLaMA-Factory/examples/train_full/llama3_full_sft.yaml \
  model_name_or_path=Qwen/Qwen2.5-0.5B-Instruct \
  report_to=wandb \
  dataset=MetaMathQA \
  per_device_train_batch_size=4 \
  deepspeed=null\
  finetuning_type=lora \ 
  output_dir=/root/autodl-tmp/nlphw2/part2/output
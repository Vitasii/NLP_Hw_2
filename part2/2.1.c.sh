#!/bin/bash

set -a
source .env
set +a
export WANDB_NAME="TASK_2_1_3"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

rm -rf /root/autodl-tmp/output/metamath

cd /root/autodl-tmp/LLaMA-Factory

FORCE_TORCHRUN=1 llamafactory-cli train \
  /root/autodl-tmp/LLaMA-Factory/examples/train_full/llama3_full_sft.yaml \
  model_name_or_path=Qwen/Qwen2.5-0.5B-Instruct \
  report_to=wandb \
  dataset=MetaMathQA \
  template=qwen \
  per_device_train_batch_size=4 \
  deepspeed=null \
  output_dir=/root/autodl-tmp/output/metamath
#!/bin/bash

source .env
export WANDB_NAME=TASK_2_1_2
cd /root/LLaMA-Factory
# 运行训练
FORCE_TORCHRUN=1 llamafactory-cli train \
  /root/LLaMA-Factory/examples/train_full/llama3_full_sft.yaml \
  model_name_or_path=Qwen/Qwen2.5-0.5B-Instruct \
  report_to=wandb \
  dataset=alpaca_en_demo \
  per_device_train_batch_size=4\
  deepspeed=null
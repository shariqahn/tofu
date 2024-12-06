#!/bin/bash

# Set up correct environment
source /etc/profile
module load anaconda/Python-ML-2024b

export HF_HOME=/state/partition1/user/$USER/hug
export HYDRA_FULL_ERROR=1
export TRITON_CACHE_DIR=/state/partition1/user/$USER/triton

master_port=18765
split=forget01
model=llama2-7b
lr=1e-5

CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 --master_port=$master_port forget.py --config-name=forget.yaml split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} lr=${lr}
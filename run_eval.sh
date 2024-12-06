#!/bin/bash

# Set up correct environment
source /etc/profile
module load anaconda/Python-ML-2024b

# master_port=18765
# split=forget10_perturbed
# model_family=llama2-7b
# lr=1e-5
# # todo check all below
# # model_path=/home/gridsan/shossain/tofu/model_outputs/tofu_baseline/grad_ascent_${lr}_${split}_5
# model_path=/home/gridsan/shossain/tofu/scr/models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port evaluate_util.py\
#     model_family=$model_family split=$split\
#     model_path=$model_path

path_to_eval_result=model_outputs/tofu_baseline/eval_results/ds_size300/eval_log_aggregated.json

python aggregate_eval_stat.py retain_result=$path_to_eval_result ckpt_result=$path_to_eval_result method_name=Page8Right save_file=./model_outputs/tofu_baseline/eval_results/ds_size300/Page8Right.csv

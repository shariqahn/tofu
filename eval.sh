#!/bin/bash

# Set up correct environment
module load anaconda/Python-ML-2024b
source /state/partition1/llgrid/pkg/anaconda/python-ML-2024b/etc/profile.d/conda.sh 

master_port=18765
split=forget01
model_family=llama2-7b
lr=1e-5
# todo check all below
data_path: locuslab/TOFU
model_path=/home/gridsan/shossain/tofu/model_outputs/tofu_baseline/grad_ascent_${lr}_${split}_5
data_path: [locuslab/TOFU, locuslab/TOFU, locuslab/TOFU, locuslab/TOFU]
split: forget10_perturbed
split_list:
  - retain_perturbed
  - real_authors_perturbed
  - world_facts_perturbed
  - ${split}

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port evaluate_util.py\
 model_family=$model_family split=$split\
 model_path=$model_path



# path_to_aggregated_retain_result=data/ft_epoch5_lr1e-05_llama2-7b_full_wd0/eval_results/ds_size300/eval_log_aggregated.json

# python aggregate_eval_stat.py retain_result=$path_to_aggregated_retain_result ckpt_result=$path_to_aggregated_retain_result method_name=Testing save_file=full_wd0.csv

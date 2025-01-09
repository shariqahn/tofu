#!/bin/bash

# Set up correct environment
source /etc/profile
module load anaconda/Python-ML-2024b
# which python
# module list
# conda list

master_port=18765
# master_port=18775

split=forget10_perturbed
model_family=llama2-7b
# lr=1e-5
experiment_name=WISE_dummy
eval_name=${experiment_name}_${split}
save_root=/home/gridsan/shossain/tofu/model_outputs/$eval_name
# model_path=/home/gridsan/shossain/tofu/model_outputs/tofu_baseline/grad_ascent_${lr}_${split}_5
# model_path=/home/gridsan/shossain/tofu/scr/models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd
model_path=/home/gridsan/shossain/EasyEdit/outputs/$experiment_name/model

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port evaluate_util.py\
    model_family=$model_family split=$split\
    model_path=$model_path\
    save_root=$save_root

# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port evaluate_util.py model_family=$model_family split=$split model_path=$model_path save_root=$save_root

path_to_eval_result=model_outputs/$eval_name/eval_results/ds_size300/eval_log_aggregated.json

python aggregate_eval_stat.py retain_result=$path_to_eval_result ckpt_result=$path_to_eval_result method_name=$eval_name save_file=./model_outputs/$eval_name/eval_results/ds_size300/aggr_result.csv

# LLsub run_eval.sh -s 3 -g volta:1
# WISE needs > 20
# LLsub run_eval.sh -s 20 -g volta:1
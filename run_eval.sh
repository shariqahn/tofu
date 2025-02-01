#!/bin/bash

# Set up correct environment
source /etc/profile
module load anaconda/Python-ML-2024b
# which python
# module list
# conda info --envs

master_port=18765
# master_port=18775

split=forget10_perturbed
model_family=llama2-7b
# lr=1e-5
# experiment_name=KL_baseline
# experiment_name=gradient_ascent_baseline
experiment_name=tofu_baseline_single_old
# experiment_name=preference_optimization_baseline
# experiment_name=gradient_difference_baseline
eval_name=${experiment_name}_${split}
save_root=~/tofu/model_outputs/$eval_name
model_path=~/tofu/scr/models--locuslab--tofu_ft_llama2-7b/snapshots/8fa500e8f345f1dd9cfe95bb4689878c944c9cbd
# model_path="/home/gridsan/$USER/EasyEdit/outputs/$experiment_name/model"
# model_path=~/tofu/scr/models--locuslab--llama2-7b_KL_1e-05_forget10/snapshots/697ae16fe82fcfaccd7d6764b44f5de948093fdd/
# model_path=~/tofu/scr/models--locuslab--llama2-7b_grad_ascent_1e-05_forget10/snapshots/b409e0ce7906d973ece7753ddd16bca363039cf3/
# model_path=~/tofu/scr/models--locuslab--llama2-7b_idk_1e-05_forget10/snapshots/d03eb1249eceb6cae05ee9593fdc54554a7d2e25/
# model_path=~/tofu/scr/models--locuslab--llama2-7b_grad_diff_1e-05_forget10/snapshots/7d569764437680ad37ff8c22d17f24a034f06914/


# CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port evaluate_util.py\
#     model_family=$model_family split=$split\
#     model_path=$model_path\
#     save_root=$save_root
#     # for idk, KL
#     batch_size=16

CUDA_VISIBLE_DEVICES=0 torchrun --nproc_per_node=1 --master_port=$master_port evaluate_util_old.py\
    model_family=$model_family split=$split\
    model_path=$model_path\
    save_root=$save_root
    # for idk, KL
    # batch_size=16

path_to_eval_result=model_outputs/$eval_name/eval_results/ds_size300/eval_log_aggregated.json
path_to_retain_result=./data/ft_epoch5_lr1e-05_llama2-7b_retain90_wd0.01/eval_results/ds_size300/eval_log_aggregated.json

# python aggregate_eval_stat.py retain_result=$path_to_retain_result ckpt_result=$path_to_eval_result method_name=$eval_name save_file=./model_outputs/$eval_name/eval_results/ds_size300/aggr_result.csv

python aggregate_eval_stat.py retain_result=$path_to_eval_result ckpt_result=$path_to_eval_result method_name=$eval_name save_file=./model_outputs/$eval_name/eval_results/ds_size300/aggr_result.csv

# python aggregate_eval_stat.py retain_result=$path_to_retain_result ckpt_result=$path_to_retain_result method_name=retain90_wd0.01 save_file=./model_outputs/$eval_name/eval_results/ds_size300/aggr_result.csv

# LLsub run_eval.sh -s 3 -g volta:1
# WISE needs > 20
# LLsub run_eval.sh -s 20 -g volta:1
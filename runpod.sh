#!/bin/bash

# rsync -rvz -e "ssh -p 22065" ~/EasyEdit/outputs/WISE_avoidant/model/ root@69.30.85.10:/workspace/model/

# git clone https://github.com/shariqahn/tofu.git
# cd tofu

# python -c "import torch; print(torch.__version__)"
# nvcc --version
pip install --no-cache-dir -r EasyEdit_requirements.txt
pip install --no-cache-dir -r requirements.txt
pip install --no-cache-dir flash-attn --no-build-isolation

# export HF_TOKEN
export PYTHONUNBUFFERED=1  # Enable unbuffered output for Python scripts

echo "Starting the first Python script (evaluate_util)..."
master_port=18765
split=forget10_perturbed
model_family=llama2-7b
experiment_name=WISE_avoidant
eval_name=${experiment_name}_${split}
save_root=./model_outputs/$eval_name
model_path=/workspace/model

CUDA_VISIBLE_DEVICES=0 stdbuf -oL torchrun --nproc_per_node=1 --master_port=$master_port evaluate_util.py \
    model_family=$model_family split=$split \
    model_path=$model_path \
    save_root=$save_root \
    > evaluate_util.log 2>&1 &

EVAL_PID=$!  # Capture the PID of the first background process

echo "First script is running with PID: $EVAL_PID"

wait $EVAL_PID
echo "First script finished!"

echo "Starting the second Python script (aggregate_eval_stat)..."
path_to_eval_result=model_outputs/$eval_name/eval_results/ds_size300/eval_log_aggregated.json

nohup python3 -u aggregate_eval_stat.py \
    retain_result=$path_to_eval_result \
    ckpt_result=$path_to_eval_result \
    method_name=$eval_name \
    save_file=./model_outputs/$eval_name/eval_results/ds_size300/aggr_result.csv \
    > aggregate_eval_stat.log 2>&1 &

AGGR_PID=$!  # Capture the PID of the second background process

echo "Second script is running with PID: $AGGR_PID"

wait $AGGR_PID
echo "Second script finished! Commit changes now."

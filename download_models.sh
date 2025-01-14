#!/bin/bash

# Set up correct environment
source /etc/profile
module load anaconda/Python-ML-2024b
# source /state/partition1/llgrid/pkg/anaconda/python-ML-2024b/etc/profile.d/conda.sh 
# conda deactivate 
# conda activate tofu

# export HYDRA_FULL_ERROR=1
# This is where model is downloaded
export HF_HOME=/state/partition1/user/$USER/hug
mkdir -p $HF_HOME
HF_LOCAL_DIR=$HOME/tofu/scr
mkdir -p $HF_LOCAL_DIR

# Remove existing models so that they will be replaced with fresh ones
# rm -r $HF_LOCAL_DIR/*
# rm -r $HF_LOCAL_DIR/models--locuslab--tofu_ft_llama2-7b/
# echo "Existing models removed. Here's what local looks like:"
# ls $HF_LOCAL_DIR

echo "Dirs created:"
ls /state/partition1/user/$USER
echo "downloading"
python -u download.py

# Copy the model from HF_HOME into HF_LOCAL_DIR
echo "Model collected. Here is what home looks like:"
ls $HF_HOME
# cp -rf $HF_HOME/* $HF_LOCAL_DIR
rsync -a --ignore-existing ${HF_HOME}/ $HF_LOCAL_DIR
echo "Model copied. Here is what local looks like:"
ls $HF_LOCAL_DIR
rm -rf $HF_HOME
echo "Home cleared. Here is what home looks like:"
ls /state/partition1/user/$USER
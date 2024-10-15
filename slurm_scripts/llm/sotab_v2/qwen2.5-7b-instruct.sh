#!/bin/bash
#SBATCH --job-name=qwen2.5-7b-instruct
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate huggingface

python ./openforge/llm_ensemble/msft_phi3.py \
    --config_path=./openforge/llm_ensemble/exp_configs/sotab-v2_0-shot/qwen2.5-7b-instruct.ini  \
    --mode=inference

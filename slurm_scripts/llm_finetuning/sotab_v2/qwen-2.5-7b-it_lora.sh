#!/bin/bash
#SBATCH --job-name=qwen-2.5-7b_lora
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=2:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate huggingface

python ./openforge/llm_finetuning/lora_finetuning_sotab.py \
    --config_path=./openforge/llm_finetuning/exp_configs/sotab_v2/qwen-2.5-7b_lora.ini 

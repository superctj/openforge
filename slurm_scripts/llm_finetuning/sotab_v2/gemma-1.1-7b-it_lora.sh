#!/bin/bash
#SBATCH --job-name=gemma-1.1-7b-it_lora
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=6:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate huggingface

python ./openforge/llm_finetuning/evaluate_sotab.py \
    --config_path=./openforge/llm_finetuning/exp_configs/sotab_v2/gemma-1.1-7b-it_lora_mrf-inputs.ini

#!/bin/bash
#SBATCH --job-name=prior-inference_gemma-2-9b-it_lora
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

python ./openforge/llm_finetuning/evaluate_em_walmart_amazon.py \
    --config_path=./openforge/llm_finetuning/exp_configs/em_walmart-amazon/prior_gemma-2-9b-it_lora.ini
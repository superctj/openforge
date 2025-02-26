#!/bin/bash
#SBATCH --job-name=gemma-2-9b-it
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate huggingface

python ./openforge/llm_ensemble/llm_icpsr_hyper_hypo.py \
    --config_path=./openforge/llm_ensemble/exp_configs/icpsr-hyper-hypo_0-shot/gemma-2-9b-it.ini  \
    --mode=inference

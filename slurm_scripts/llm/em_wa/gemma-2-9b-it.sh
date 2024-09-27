#!/bin/bash
#SBATCH --job-name=gemma-2-9b-it_predictions
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32g
#SBATCH --time=04:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate openforge-gpu

python ./openforge/llm_ensemble/msft_phi3.py \
    --config_path=./openforge/llm_ensemble/exp_configs/em-wa_5-shots/gemma-2-9b-it.ini \
    --mode=inference

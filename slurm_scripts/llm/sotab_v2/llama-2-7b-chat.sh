#!/bin/bash
#SBATCH --job-name=llama-2-7b-chat_predictions
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=03:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate openforge-gpu

python ./openforge/llm_ensemble/msft_phi3.py \
    --config_path=./openforge/llm_ensemble/exp_configs/sotab-v2_10-shots/llama-2-7b-chat.ini  \
    --mode=inference

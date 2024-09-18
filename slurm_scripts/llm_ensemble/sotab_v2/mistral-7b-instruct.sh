#!/bin/bash
#SBATCH --job-name=mistral-7b-instruct-v0.2_predictions
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64g
#SBATCH --time=02:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate pgmax-gpu

python ./openforge/llm_ensemble/msft_phi3.py \
    --config_path=./openforge/llm_ensemble/exp_configs/sotab-v2_5-shots/mistral-7b-instruct-v0.2.ini  \
    --mode=inference

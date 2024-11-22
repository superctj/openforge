#!/bin/bash
#SBATCH --job-name=populate-mrfs_gemma-2-9b-it
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate huggingface

python ./openforge/prepare_mrf_inputs/em-wa_populate-mrfs_llm.py \
    --config_path=./openforge/prepare_mrf_inputs/exp_configs/em_walmart-amazon/populate-mrfs_nv-embed-v2_gemma-2-9b-it.ini
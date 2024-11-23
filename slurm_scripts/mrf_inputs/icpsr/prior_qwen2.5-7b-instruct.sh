#!/bin/bash
#SBATCH --job-name=icpsr_prior-inference_qwen2.5-7b-instruct
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

python ./openforge/prepare_mrf_inputs/icpsr_prior_llm.py \
    --config_path=./openforge/prepare_mrf_inputs/exp_configs/icpsr_10-shots/prior_qwen2.5-7b-instruct.ini
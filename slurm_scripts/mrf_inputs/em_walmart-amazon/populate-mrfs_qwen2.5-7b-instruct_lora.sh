#!/bin/bash
#SBATCH --job-name=populate-mrfs_qwen2.5-7b-instruct_lora
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
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib

python ./openforge/llm_finetuning/populate_mrf_inputs_llm.py \
    --config_path=./openforge/llm_finetuning/exp_configs/em_walmart-amazon/populate-mrfs_nv-embed-v2_qwen2.5-7b-instruct_lora.ini
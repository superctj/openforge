#!/bin/bash
#SBATCH --job-name=prepare-mrf-inputs_nv-embed-v2_qwen2.5-7b-instruct
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=10:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate llm2vec
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib

python ./openforge/prepare_mrf_inputs/em-wa_create-mrfs_nv-embed-v2.py \
    --config_path=./openforge/prepare_mrf_inputs/exp_configs/em_walmart-amazon/create-mrfs_nv-embed-v2_qwen2.5-7b-instruct.ini

conda deactivate
conda activate huggingface
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib

python ./openforge/prepare_mrf_inputs/em-wa_populate-mrfs_llm.py \
    --config_path=./openforge/prepare_mrf_inputs/exp_configs/em_walmart-amazon/populate-mrfs_nv-embed-v2_qwen2.5-7b-instruct.ini

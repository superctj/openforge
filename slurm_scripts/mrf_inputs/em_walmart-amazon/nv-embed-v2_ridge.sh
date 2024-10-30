#!/bin/bash
#SBATCH --job-name=prepare-mrf-inputs_nv-embed-v2_ridge
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=12:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate llm2vec
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib

python ./openforge/prepare_mrf_inputs/prepare_mrf_inputs_em_wa.py \
    --config_path=./openforge/prepare_mrf_inputs/exp_configs/em_walmart-amazon/nv-embed-v2_ridge.ini

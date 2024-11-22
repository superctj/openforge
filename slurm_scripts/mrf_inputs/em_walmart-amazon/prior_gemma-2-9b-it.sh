#!/bin/bash
#SBATCH --job-name=prior-inference_gemma-2-9b-it
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=8:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate huggingface
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib/python3.11/site-packages/nvidia/cuda_nvrtc/lib

python ./openforge/prepare_mrf_inputs/em-wa_prior_llm.py \
    --config_path=./openforge/prepare_mrf_inputs/exp_configs/em_walmart-amazon/prior_gemma-2-9b-it.ini
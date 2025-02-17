#!/bin/bash
#SBATCH --job-name=mrf-inference_qwen2.5-7b-instruct-lora
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=2:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate pgmax-gpu
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

python ./openforge/mrf_inference/pgmax_lbp_icpsr_hyper_hypo.py \
    --config_path=./openforge/mrf_inference/tuning_exp_configs/icpsr_hyper_hypo/qwen2.5-7b-instruct-lora.ini \
    --mode=hp_tuning

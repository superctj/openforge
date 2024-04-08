#!/bin/bash
#SBATCH --job-name=pgmax-gpu-lbp-icpsr-hp-tuning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=1-00:00:00
#SBATCH --account=jag0
#SBATCH --output=/home/congtj/openforge/exps/openforge_icpsr/ridge_pgmax_gpu_lbp/outputs.log

source ~/.bash_profile
conda activate pgmax-gpu

python ./openforge/mrf_inference/pgmax_lbp_icpsr.py \
    --config_path=./openforge/mrf_inference/tuning_exp_configs/icpsr_ridge_pgmax_lbp.ini \
    --mode=hp_tuning

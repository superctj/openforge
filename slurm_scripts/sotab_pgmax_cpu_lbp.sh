#!/bin/bash
#SBATCH --job-name=sotab_pgmax-cpu-lbp_inference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=01:00:00
#SBATCH --account=jag98
#SBATCH --output=/home/congtj/openforge/openforge/mrf_inference/sotab_ridge_pgmax_cpu_lbp_inference.log

source ~/.bashrc
conda activate pgmax-cpu

python ./openforge/mrf_inference/pgmax_lbp_sotab.py \
    --config_path=./openforge/mrf_inference/tuning_exp_configs/sotab_xlarge_ridge_pgmax_lbp.ini \
    --mode=inference

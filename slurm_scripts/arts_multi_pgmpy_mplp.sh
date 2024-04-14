#!/bin/bash
#SBATCH --job-name=arts-multi_pgmpy-mplp_inference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=1-00:00:00
#SBATCH --account=jag98
#SBATCH --output=/home/congtj/openforge/openforge/mrf_inference/arts_multi_rf_pgmpy_mplp_inference.log

source ~/.bashrc
conda activate pgmpy

python ./openforge/mrf_inference/pgmpy_mplp_arts_multi.py \
    --config_path=./openforge/mrf_inference/tuning_exp_configs/arts_multi_rf_pgmpy_mplp.ini \
    --mode=inference
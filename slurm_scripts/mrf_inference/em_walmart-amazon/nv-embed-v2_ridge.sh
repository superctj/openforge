#!/bin/bash
#SBATCH --job-name=mrf-inference_nv-embed-v2_ridge
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=10:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate pgmax-gpu
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

python ./openforge/mrf_inference/pgmax_lbp_em_wa.py \
    --config_path=./openforge/mrf_inference/tuning_exp_configs/em_walmart-amazon/nv-embed-v2_ridge.ini \
    --mode=hp_tuning

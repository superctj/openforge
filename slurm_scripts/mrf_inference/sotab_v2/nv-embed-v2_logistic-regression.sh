#!/bin/bash
#SBATCH --job-name=mrf-inference_nv-embed-v2_logistic-regression
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=3:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate pgmax-gpu
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

python ./openforge/mrf_inference/pgmax_lbp_sotab.py \
    --config_path=./openforge/mrf_inference/tuning_exp_configs/sotab-v2_0-shot/nv-embed-v2_logistic-regression.ini \
    --mode=hp_tuning

#!/bin/bash
#SBATCH --job-name=prior-finetuning_nv-embed-v2_gbdt
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

python ./openforge/prior_models/gbdt.py \
    --config_path=./openforge/prior_models/exp_configs/em_walmart-amazon/nv-embed-v2_gbdt.ini \
    --mode=train_w_hp_tuning
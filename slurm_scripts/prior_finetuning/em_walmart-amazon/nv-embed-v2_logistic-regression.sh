#!/bin/bash
#SBATCH --job-name=prior-finetuning_nv-embed-v2_logistic-regression
#SBATCH --partition=standard
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=10:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate pgmax-cpu
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

python ./openforge/prior_models/logistic_regression.py \
    --config_path=./openforge/prior_models/exp_configs/em_walmart-amazon/nv-embed-v2_logistic-regression.ini \
    --mode=train_w_hp_tuning
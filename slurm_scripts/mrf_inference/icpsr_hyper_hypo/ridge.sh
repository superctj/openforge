#!/bin/bash
#SBATCH --job-name=icpsr_mrf-inference_ridge
#SBATCH --partition=largemem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=256g
#SBATCH --time=8:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate pgmax-gpu
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

python ./openforge/mrf_inference/pgmax_lbp_icpsr_hyper_hypo.py \
    --config_path=./openforge/mrf_inference/tuning_exp_configs/icpsr_hyper_hypo/ridge.ini \
    --mode=hp_tuning

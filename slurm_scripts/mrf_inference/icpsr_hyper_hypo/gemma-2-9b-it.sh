#!/bin/bash
#SBATCH --job-name=icpsr_mrf-inference_gemma-2-9b-it
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

python ./openforge/mrf_inference/pgmax_lbp_icpsr_hyper.py \
    --config_path=./openforge/mrf_inference/tuning_exp_configs/icpsr/0_shot/gemma-2-9b-it.ini \
    --mode=hp_tuning

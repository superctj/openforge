#!/bin/bash
#SBATCH --job-name=arts_pgmax-gpu-lbp_hp-tuning
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=02:00:00
#SBATCH --account=jag98
#SBATCH --output=/home/congtj/openforge/openforge/mrf_inference/arts_pgmax-gpu-lbp_hp_tuning.log

module load cuda/12.1.1
source ~/.bashrc
conda activate pgmax-gpu

python ./openforge/mrf_inference/pgmax_lbp_arts.py \
    --config_path=./openforge/mrf_inference/tuning_exp_configs/arts_rf_pgmax_lbp.ini \
    --mode=hp_tuning

#!/bin/bash
#SBATCH --job-name=icpsr-hyper_pgmax-gpu-lbp_inference
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=01:00:00
#SBATCH --account=jag98
#SBATCH --output=/home/congtj/openforge/openforge/mrf_inference/icpsr_hyper_gbdt_pgmax_gpu_lbp_inference.log

module load cuda/12.1.1
source ~/.bashrc
conda activate pgmax-gpu

python ./openforge/mrf_inference/pgmax_lbp_icpsr_hyper.py \
    --config_path=./openforge/mrf_inference/tuning_exp_configs/icpsr_hyper_gbdt_pgmax_lbp.ini \
    --mode=inference

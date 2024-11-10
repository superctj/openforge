#!/bin/bash
#SBATCH --job-name=mrf-inference_nv-embed-v2_unicorn_cpu
#SBATCH --partition=largemem
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=16g
#SBATCH --time=6:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate pgmax-cpu

python ./openforge/mrf_inference/pgmax_lbp_em_wa_cpu.py \
    --config_path=./openforge/mrf_inference/tuning_exp_configs/em_walmart-amazon/nv-embed-v2_unicorn_cpu.ini \
    --mode=hp_tuning

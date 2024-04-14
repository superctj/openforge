#!/bin/bash
#SBATCH --job-name=pgmax-cpu-lbp_scalability
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=3-00:00:00
#SBATCH --account=jag98
#SBATCH --output=/home/congtj/openforge/openforge/pgmax_cpu_lbp_scalability.log

source ~/.bashrc
conda activate pgmax-cpu

for n in $(seq 200 100 400)
do
    python openforge/evaluate_lbp_efficiency.py \
        --num_concepts=${n} \
        --log_dir=/home/congtj/openforge/logs/synthesized_mrf/pgmax_cpu_lbp
done
#!/bin/bash
#SBATCH --job-name=pgmax-gpu-lbp-scalability
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=1-00:00:00
#SBATCH --account=jag98
#SBATCH --output=/home/congtj/openforge/logs/synthesized_mrf/pgmax_gpu_lbp_scalability.log

# # Stop on errors
# set -Eeuo pipefail

module load cuda/12.1.1
source ~/.bashrc
conda activate pgmax-gpu

for n in $(seq 300 100 500)
do
    python openforge/evaluate_lbp_efficiency.py --num_concepts=${n}
done
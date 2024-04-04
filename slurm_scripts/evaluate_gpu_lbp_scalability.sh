#!/bin/bash
#SBATCH --job-name=gpu-lbp-scalability
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=64gb
#SBATCH --time=3-00:00
#SBATCH --account=jag0
#SBATCH --output=/home/congtj/openforge/logs/synthesized_mrf

# Stop on errors
set -Eeuo pipefail

source ~/.conda/envs/openforge-pgmax-gpu/bin/activate
conda activate openforge-pgmax-gpu

for n in $(seq 200 200 1000)
do
    python openforge/evaluate_lbp_efficiency.py --num_concepts=${n}
done
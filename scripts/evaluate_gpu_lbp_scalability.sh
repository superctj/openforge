#!/bin/bash
#SBATCH --job-name=gpu-lbp-scalability
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16gb
#SBATCH --time=1-00:00
#SBATCH --account=jag0

# Stop on errors
set -Eeuo pipefail

for n in $(seq 20 20 100)
do
    python openforge/evaluate_lbp_efficiency.py --num_concepts=${n}
done
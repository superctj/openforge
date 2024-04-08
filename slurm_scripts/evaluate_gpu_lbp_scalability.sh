#!/bin/bash
#SBATCH --job-name=gpu-lbp-scalability
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=1-00:00
#SBATCH --account=jag98
#SBATCH --output=/home/congtj/openforge/logs/synthesized_mrf/evaluate_gpu_lbp_scalability.outputs

# Stop on errors
set -Eeuo pipefail

source /home/congtj/.bash_profile
conda activate pgmax-gpu

for n in $(seq 20 20 100)
do
    python openforge/evaluate_lbp_efficiency.py --num_concepts=${n}
done
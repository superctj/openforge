#!/bin/bash
#SBATCH --job-name=pgmax-gpu-lbp-scalability
#SBATCH --partition=spgpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=256g
#SBATCH --time=1-00:00:00
#SBATCH --account=jag0

source ~/.bash_profile
conda activate pgmax-gpu
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib

for n in $(seq 3000 1000 5000); do
    for k in $(seq 4 4 16); do 
        python openforge/evaluate_lbp_efficiency.py --num_concepts=$n --k=$k
    done
done
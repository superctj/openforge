#!/bin/bash
#SBATCH --job-name=install-pgmax-gpu-env
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128g
#SBATCH --time=02:00:00
#SBATCH --account=jag0
#SBATCH --output=/home/congtj/openforge/install_pgmax_gpu_env.outputs

# Stop on errors
set -Eeuo pipefail

source /home/congtj/.bash_profile
conda create -n pgmax-gpu python=3.10
conda activate pgmax-gpu
conda install conda-forge::configspace
conda install anaconda::pandas
conda install anaconda::scikit-learn
conda install jaxlib=*=*cuda* jax cuda-nvcc -c conda-forge -c nvidia
conda install conda-build

pip install pgmax
pip install smac

conda develop /home/congtj/openforge

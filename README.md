# OpenForge

## Environment Setup
Assume using [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) for Python package management on Linux machines. 

1. Clone this repo in your working directory:

    ```
    git clone <OpenForge repo url>
    ```
    
    ```
    cd openforge
    ```

2. Create and activate the Python environment:
    ```
    conda env create -f pgmax_cpu_env.yml
    ```

    ```
    conda activate pgmax-cpu
    ```

    If you have access to GPUs, you can create and activate the GPU environment:
    ```
    conda env create -f pgmax_gpu_env.yml
    ```

    ```
    conda activate pgmax-gpu
    ```

    Heads-up: The pgmax-gpu environment depends on the [GPU version of Jax](https://jax.readthedocs.io/en/latest/installation.html), which requires Nvidia driver version to be >= 525.60.13 for CUDA 12 on Linux.

3. Import OpenForge as an editable package to the Python environment

    ```
    conda develop <path to OpenForge repo>
    ```

    e.g.,
    
    ```
    conda develop /home/congtj/openforge
    ```

## Quick Start
1. Generate prior beliefs (e.g., Gradient Boosting Decision Tree on OpenForge-ICPSR benchmark):
    ```
    cd openforge/prior_models

    python icpsr_hyper_gbdt.py \
        --config_path=./tuning_exp_configs/icpsr_hyper_gbdt.ini \
        --mode=train_w_default_hp
    ```

2. Run Markov Random Field inference to obtain posterior beliefs:
    ```
    cd openforge/mrf_inference

    python pgmax_lbp_icpsr_hyper.py \
        --config_path=./tuning_exp_configs/icpsr_hyper_gbdt_pgmax_lbp.ini \
        --mode=inference
    ```

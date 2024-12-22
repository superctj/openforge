# OpenForge: Probabilistic Metadata Integration

## Environment Setup
We use [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) for Python package management on Linux machines. Due to the complexity of the dependencies, we separate the environments for training prior models and running inference over Markov Random Fields (MRFs).

1. Clone this repo in your working directory:

    ```
    git clone <OpenForge repo url>
    ```
    
    ```
    cd openforge
    ```

2. Create two independent environments, one for training prior models and the other for running MRF inference:

    ```
    conda env create -f huggingface_env.yml
    ```

    ```
    conda env create -f pgmax_gpu_env.yml
    ```

    Heads-up: The pgmax-gpu environment depends on the [GPU version of Jax](https://jax.readthedocs.io/en/latest/installation.html), which requires Nvidia driver version to be >= 525.60.13 for CUDA 12 on Linux. You can also choose to create the CPU version of the pgmax environment for running MRF inference:
    
    ```
    conda env create -f pgmax_cpu_env.yml
    ```


3. Import OpenForge as an editable package to both environments:

    ```
    conda activate huggingface
    conda develop <path to the OpenForge repository, e.g., /home/congtj/openforge>
    ```

    ```
    conda activate pgmax-gpu
    conda develop <path to the OpenForge repository, e.g., /home/congtj/openforge>
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

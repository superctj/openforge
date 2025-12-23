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

## Datasets
We provide the following datasets for training prior models and running MRF inference. Each dataset is stored in a separate Google Drive folder, which contains raw data and preprocessed data that are ready for running MRF inference.

- [SOTAB-v2](https://drive.google.com/drive/folders/1AxxW_-rueMo58tLGsZx2CsRz8ibjpFeD?usp=sharing)
- [Walmart-Amazon](https://drive.google.com/drive/folders/1zUIHjL8fneBMYk2J54QLoWMJpQK2hGSr?usp=sharing)
- [ICPSR](https://drive.google.com/drive/folders/1f0Dm3vscFF4aPnlJwsb7dHvfbAySKQiy?usp=sharing)

## Quick Start
- Run hyperparameter tuning and MRF inference to obtain posterior beliefs:

    ```
    conda activate pgmax-gpu

    cd openforge/mrf_inference
    
    python pgmax_lbp_icpsr_hyper.py \
        --config_path=./tuning_exp_configs/icpsr/qwen2.5-7b-instruct-lora.ini \
        --mode=hp_tuning
    ```

- Run MRF inference to obtain posterior beliefs (with the best found hyperparameters hard-coded in the program):

    ```
    conda activate pgmax-gpu

    cd openforge/mrf_inference

    python pgmax_lbp_icpsr_hyper.py \
        --config_path=./tuning_exp_configs/icpsr/qwen2.5-7b-instruct-lora.ini \
        --mode=inference
    ```

- Fine-tune a LLM with LoRA:

    ```
    conda activate huggingface

    cd openforge/llm_finetuning

    python google_gemma_lora_icpsr.py \
        --config_path=./exp_configs/icpsr/qwen2.5-7b-instruct_lora.ini
    ```

## Citing This Repository
If you find this repository useful for your work, please cite the following BibTeX:

```bibtex
@article{DBLP:journals/pvldb/CongNXJ25,
  author       = {Tianji Cong and
                  Fatemeh Nargesian and
                  Junjie Xing and
                  H. V. Jagadish},
  title        = {OpenForge: Probabilistic Metadata Integration},
  journal      = {Proc. {VLDB} Endow.},
  volume       = {18},
  number       = {9},
  pages        = {2914--2927},
  year         = {2025},
  url          = {https://www.vldb.org/pvldb/vol18/p2914-cong.pdf},
  doi          = {10.14778/3746405.3746417},
  timestamp    = {Wed, 17 Dec 2025 16:44:24 +0100},
  biburl       = {https://dblp.org/rec/journals/pvldb/CongNXJ25.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

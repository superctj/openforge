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
    conda env create -f environment.yml
    ```

    ```
    conda activate openforge
    ```

3. Import OpenForge as an editable package to the Python environment

    ```
    conda develop <path to OpenForge>
    ```

    e.g.,
    
    ```
    conda develop /home/congtj/openforge
    ```

import os
import random
import shutil

from configparser import ConfigParser

import numpy as np
import torch


def get_proj_dir(filepath: str, file_level: int = 3) -> str:
    """Get the project directory.

    Args:
        filepath (str): The path of the file.
        file_level (int): The level of the file relative to the project
                          directory starting at 1.

    Returns:
        (str): The project directory.
    """

    abs_path = os.path.abspath(filepath)
    proj_dir = "/".join(abs_path.split("/")[:-file_level])
    return proj_dir


def create_dir(dir_path: str, force: bool = False):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    else:
        if force:
            shutil.rmtree(dir_path)
            os.makedirs(dir_path)
        else:
            raise ValueError(
                f"Directory {dir_path} already exists and not forced to create the directory."
            )


def fix_global_random_state(seed: int):
    """
    Control sources of randomness.

    Args:
        seed (int): The seed to use.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_config(config_path: str) -> ConfigParser:
    """
    Parse an experiment configuration file.

    Args:
        config_path (str): Path to the configuration file.

    Returns:
        parser (dict): The parsed configuration.
    """

    parser = ConfigParser()
    parser.read(config_path)

    return parser

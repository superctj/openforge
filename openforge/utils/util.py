import os


def get_proj_dir(filepath: str, file_level: int=3) -> str:
    """Get the project directory.

    Args:
        filepath (str): The path of the file.
        file_level (int): The level of the file relative to the project directory starting at 1.
    
    Returns:
        (str): The project directory.
    """

    abs_path = os.path.abspath(filepath)
    proj_dir = "/".join(abs_path.split("/")[:-file_level])
    return proj_dir


def create_dir(dir_path: str):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

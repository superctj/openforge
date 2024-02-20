import logging
import os

from datetime import datetime


def get_custom_logger(log_dir: str, log_level=logging.INFO):
    if not os.path.isdir(log_dir):
        log_filepath = log_dir
    else:
        cur_datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        log_filepath = os.path.join(log_dir, f"{cur_datetime}.log")

    logger = logging.getLogger("openforge")
    logger.setLevel(log_level)

    # create the logging file handler
    f_handler = logging.FileHandler(log_filepath, mode="w")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(f_handler)

    return logger

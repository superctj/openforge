import logging


def get_logger(log_filepath):
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # create the logging file handler
    f_handler = logging.FileHandler(log_filepath)

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(f_handler)
    return logger
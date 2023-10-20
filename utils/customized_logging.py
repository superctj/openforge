import logging


logging_level_map = {"CRITICAL": 50, "ERROR": 40, "WARNING": 30, "INFO": 20, "DEBUG": 10, "NOTSET": 0}


def get_logger(log_filepath, level=logging.INFO):
    logger = logging.getLogger(log_filepath)
    logger.setLevel(level)

    # create the logging file handler
    f_handler = logging.FileHandler(log_filepath, mode="w")
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    f_handler.setFormatter(formatter)

    # add handler to logger object
    logger.addHandler(f_handler)
    return logger

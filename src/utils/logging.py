
""" Logging program execution """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import logging

def init_logging(name: str, log_file: str, loglevel: str):
    """
    Init a logger with given name.

    :param str name: The name of the logger.
    :param log_file: The file used for logging.
    :param log_level: The level used for logging.
    """
    logger = logging.getLogger(name)

    # Global config
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    logger.setLevel(numeric_level)

    # Logging to file
    fileFormatter = logging.Formatter("%(asctime)s"\
                                      " [%(levelname)s]"\
                                      " %(message)s")
    fileHandler = logging.FileHandler(log_file)
    fileHandler.setFormatter(fileFormatter)
    logger.addHandler(fileHandler)

    # Logging to console
    consoleFormatter = logging.Formatter("[%(levelname)s] %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(consoleFormatter)
    logger.addHandler(consoleHandler)

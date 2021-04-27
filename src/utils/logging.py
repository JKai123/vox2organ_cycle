
""" Logging program execution """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import time
import logging
import functools

import wandb
import numpy as np
import nibabel as nib

from utils.modes import ExecModes

debug = False
log_time = False

def get_log_dir(experiment_dir: str):
    return os.path.join(experiment_dir, "logs")

def log_losses(losses, iteration):
    """ Logging with wandb and std logging """
    losses = {k: v.detach() for k, v in losses.items()}
    trainLogger = logging.getLogger(ExecModes.TRAIN.name)
    trainLogger.info("Iteration: %d", iteration)
    for k, v in losses.items():
        trainLogger.info("%s: %.5f", k, v)

    if not debug:
        wandb.log(losses, step=iteration)

def log_val_results(val_results, iteration):
    """ Logging with wandb and std logging """
    val_results = {"Val_" + k: v for k, v in val_results.items()}
    trainLogger = logging.getLogger(ExecModes.TRAIN.name)
    for k, v in val_results.items():
        trainLogger.info("%s: %.5f", k, v)

    if not debug:
        wandb.log(val_results, step=iteration)

def init_wandb_logging(exp_name, log_dir, wandb_proj_name,
                       wandb_group_name, wandb_job_type, params):
    """ Initialization for logging with wandb
    """
    wandb.init(name=exp_name, dir=log_dir, config=params, project=wandb_proj_name,
               group=wandb_group_name, job_type=wandb_job_type)

def init_std_logging(name, log_dir, loglevel, mode):
    """ The standard logger with levels 'INFO', 'DEBUG', etc.
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
    log_file = os.path.join(log_dir, mode.name.lower() + ".log")
    fileHandler = logging.FileHandler(log_file, mode='a')
    fileHandler.setFormatter(fileFormatter)
    logger.addHandler(fileHandler)

    # Logging to console
    consoleFormatter = logging.Formatter("[%(levelname)s] %(message)s")
    consoleHandler = logging.StreamHandler()
    consoleHandler.setFormatter(consoleFormatter)
    logger.addHandler(consoleHandler)

def init_logging(logger_name: str, exp_name: str, log_dir: str, loglevel: str, mode: ExecModes,
                 proj_name: str, group_name: str, params: dict, time_logging: bool):
    """
    Init a logger with given name.

    :param str logger_name: The name of the logger.
    :param str exp_name: The name of the experiment.
    :param log_file: The file used for logging.
    :param log_level: The level used for logging.
    :param exec_modes: TRAIN or TEST, see utils.modes.ExecModes
    :param str proj_name: The project name of the wandb logger.
    :param str group_name: The group name of the experiment.
    :param dict params: The experiment configuration.
    :param bool measure_time: Enable time measurement for some functions.
    """
    if exp_name == 'debug':
        global debug
        debug = True
        loglevel='DEBUG'

    init_std_logging(name=logger_name, log_dir=log_dir, loglevel=loglevel, mode=mode)
    if not debug and not mode == ExecModes.TEST: # no wanb when debugging or just testing
        init_wandb_logging(exp_name=exp_name,
                           log_dir=log_dir,
                           wandb_proj_name=proj_name,
                           wandb_group_name=group_name,
                           wandb_job_type=mode.name.lower(),
                           params=params)

    global log_time
    log_time = time_logging
    if time_logging:
        # Enable time logging
        timeLogger = logging.getLogger("TIME")
        timeLogger.setLevel('DEBUG')
        time_file = os.path.join(log_dir, "times.txt")
        fileHandler = logging.FileHandler(time_file, mode='a')
        timeLogger.addHandler(fileHandler)

def write_array_if_debug(data_1, data_2):
    """ Write data if debug mode is on.
    """
    file_1 = "../misc/array_1.npy"
    file_2 = "../misc/array_2.npy"
    if debug:
        np.save(file_1, data_1)
        np.save(file_2, data_2)

def write_img_if_debug(img_1: np.ndarray, img_2: np.ndarray):
    """ Write data if debug mode is on.
    """
    file_1 = "../misc/img_1.nii.gz"
    file_2 = "../misc/img_2.nii.gz"
    if debug:
        img_1 = nib.Nifti1Image(img_1, np.eye(4))
        nib.save(img_1, file_1)
        img_2 = nib.Nifti2Image(img_2, np.eye(4))
        nib.save(img_2, file_2)

def measure_time(func):
    """ Decorator for time measurement """
    @functools.wraps(func)
    def time_wrapper(*args, **kwargs):
        if log_time:
            tic = time.perf_counter()
            return_value = func(*args, **kwargs)
            toc = time.perf_counter()
            time_elapsed = toc - tic
            logging.getLogger("TIME").debug("Function %s takes %.5f s",
                                            func.__name__, time_elapsed)

        else:
            return_value = func(*args, **kwargs)

        return return_value

    return time_wrapper


""" Logging program execution """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import logging

import wandb

from utils.modes import ExecModes

debug = False

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
                 proj_name: str, group_name: str, params: dict):
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
    """
    init_std_logging(name=logger_name, log_dir=log_dir, loglevel=loglevel, mode=mode)
    if exp_name == 'debug':
        global debug
        debug = True
    if not debug: # no wanb when debugging
        init_wandb_logging(exp_name=exp_name,
                           log_dir=log_dir,
                           wandb_proj_name=proj_name,
                           wandb_group_name=group_name,
                           wandb_job_type=mode.name.lower(),
                           params=params)

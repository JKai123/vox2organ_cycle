
""" Logging program execution """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import time
import logging
import functools

import torch
import wandb
import numpy as np
import nibabel as nib

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from utils.modes import ExecModes

# global variables
use_wandb = True
debug = False
log_time = False

def get_log_dir(experiment_dir: str):
    return os.path.join(experiment_dir, "logs")

def log_losses(losses, iteration):
    """ Logging with wandb and std logging """
    losses = {k: v.detach() for k, v in losses.items()}
    trainLogger = logging.getLogger(ExecModes.TRAIN.name)
    for k, v in losses.items():
        trainLogger.info("%s: %.5f", k, v)

    if use_wandb:
        wandb.log(losses, step=iteration)

def log_deltaV(coords, iteration):
    """ Logging with wandb and std logging """
    deltaV_avg = coords.norm(dim=1).mean().detach().cpu()

    trainLogger = logging.getLogger(ExecModes.TRAIN.name)
    trainLogger.info("Average displacement: %.5f", deltaV_avg)

    if use_wandb:
        wandb.log({"Avg_displacement": deltaV_avg}, step=iteration)

def log_epoch(epoch: int, iteration: int):
    """ Logging with wandb and std logging """

    trainLogger = logging.getLogger(ExecModes.TRAIN.name)
    trainLogger.info("Epoch: %d", epoch)

    if use_wandb:
        wandb.log({"epoch": epoch}, step=iteration)

def log_coords(coords, iteration):
    """ Logging with wandb and std logging """
    avg_coords = coords.mean(dim=(0,1)).detach().cpu()

    trainLogger = logging.getLogger(ExecModes.TEST.name)
    trainLogger.info("Average coordinates: {}", avg_coords)

    if use_wandb:
        wandb.log({"Coord_x": avg_coords[0],
                   "Coord_y": avg_coords[1],
                   "Coord_z": avg_coords[2]}, step=iteration)

def log_val_results(val_results, iteration):
    """ Logging with wandb and std logging """
    val_results = {"Val_" + k: v for k, v in val_results.items()}
    trainLogger = logging.getLogger(ExecModes.TRAIN.name)
    for k, v in val_results.items():
        trainLogger.info("%s: %.5f", k, v)

    if use_wandb:
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
    # no wanb when debugging or not in training mode
    global use_wandb
    global debug
    if exp_name == 'debug' or loglevel == 'DEBUG':
        use_wandb = False
        debug = True
        loglevel='DEBUG'

    if mode != ExecModes.TRAIN:
        use_wandb = False

    init_std_logging(name=logger_name, log_dir=log_dir, loglevel=loglevel, mode=mode)
    if use_wandb:
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

def write_img_if_debug(img: np.ndarray, path: str):
    """ Write data if debug mode is on.
    """
    if debug:
        img = nib.Nifti1Image(img, np.eye(4))
        nib.save(img, path)

def write_scatter_plot_if_debug(points, path: str):
    """ Write a screenshot of a 3d scatter plot """
    if isinstance(points, torch.Tensor):
        points = points.cpu()
    if debug:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(points[:,0],
                   points[:,1],
                   points[:,2])
        plt.savefig(path)
        plt.close()

def measure_time(func):
    """ Decorator for time measurement """
    @functools.wraps(func)
    def time_wrapper(*args, **kwargs):
        if log_time:
            tic = time.perf_counter()
            return_value = func(*args, **kwargs)
            toc = time.perf_counter()
            time_elapsed = toc - tic
            logging.getLogger("TIME").info("Function %s takes %.5f s",
                                            func.__name__, time_elapsed)

        else:
            return_value = func(*args, **kwargs)

        return return_value

    return time_wrapper

def log_model_tensorboard_if_debug(model, img):
    if debug:
        from torch.utils.tensorboard import SummaryWriter
        writer = SummaryWriter("../misc/tensorboard/")
        writer.add_graph(model, img, verbose=True)
        writer.close()


""" Training procedure """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import json
import numpy as np

from utils.utils import serializable_dict
from data.dataset import dataset_split_handler

class Solver():
    """
    Solver class for neural network training.

    :param torch.optim optimizer: The optimizer to use, e.g. Adam.
    :param dict optim_params: The parameters for the optimizer. If empty,
    default values are used.
    :param list loss_func: A list of loss functions to apply.
    :param list loss_func_weights: A list of the same length of 'loss_func'
    with weights for the losses.
    :param str save_path: The path where results and stats are saved.

    """

    def __init__(self,
                 optimizer,
                 optim_params,
                 loss_func,
                 loss_func_weights,
                 save_path):
        pass


def training_routine(hps: dict, experiment_name=None, verbose=False):
    """
    A full training routine including setup of experiments etc.

    :param dict hps: Hyperparameters to use.
    :param str experiment_name (optional): The name of the experiment
    directory. If None, a name is created automatically.
    """
    ###### Prepare folder ######

    experiment_base_dir = hps['EXPERIMENT_BASE_DIR']
    experiment_name = hps.get('EXPERIMENT_NAME', None)

    if experiment_name is not None:
        experiment_dir = os.path.join(experiment_base_dir, experiment_name)
    else:
        # Automatically enumerate experiments exp_i
        ids_exist = []
        for n in os.listdir(experiment_base_dir):
            try:
                ids_exist.append(int(n.split("_")[-1]))
            except ValueError:
                pass
        if len(ids_exist) > 0:
            new_id = np.max(ids_exist) + 1
        else:
            new_id = 1

        experiment_name = "exp_" + str(new_id)
        hps['EXPERIMENT_NAME'] = experiment_name

        experiment_dir = os.path.join(experiment_base_dir, experiment_name)

    if experiment_name=="debug":
        # Overwrite
        os.makedirs(experiment_dir, exist_ok=True)
    else:
        # Throw error if directory exists already
        os.makedirs(experiment_dir)

    if verbose:
        print(f"Starting training '{experiment_name}'...")

    # Store hyperparameters
    param_file = os.path.join(experiment_dir, "params.json")
    hps_to_write = serializable_dict(hps)
    with open(param_file, 'w') as f:
        json.dump(hps_to_write, f)

    ###### Load data ######
    if verbose:
        print(f"Loading dataset {hps['DATASET']}...")
    try:
        training_set,\
                validation_set,\
                test_set = dataset_split_handler[hps['DATASET']](hps)
    except KeyError:
        print(f"Dataset {hps['DATASET']} not known.")
        return

    if verbose:
        print(f"{len(training_set)} training files.")
        print(f"{len(validation_set)} validation files.")
        print(f"{len(test_set)} test files.")

    ###### Start training ######

    solver = Solver(optimizer=hps['OPTIMIZER'],
                    optim_params=hps['OPTIM_PARAMS'],
                    loss_func=hps['LOSS_FUNCTIONS'],
                    loss_func_weights =\
                      hps['LOSS_FUNCTION_WEIGHTS'],
                    save_path=experiment_dir)


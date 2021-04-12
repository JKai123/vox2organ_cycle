
""" Training procedure """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import json
import numpy as np

from utils.params import HyperPs

def training_routine(hps: dict, experiment_name=None):
    """
    A full training routine including setup of experiments etc.

    :param dict hps: Hyperparameters to use.
    :param str experiment_name (optional): The name of the experiment
    directory. If None, a name is created automatically.
    """
    ###### Prepare folder ######

    experiment_base_dir = hps[HyperPs.EXPERIMENT_BASE_DIR.name]
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

    # Store hyperparameters
    param_file = os.path.join(experiment_dir, "params.json")
    with open(param_file, 'w') as f:
        json.dump(hps, f)

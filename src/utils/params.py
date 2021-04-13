""" Documentation of project-wide parameters and default values 

Ideally, all occurring parameters should be documented here.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import Enum

import torch
import torch.nn.functional as F

hyper_ps_default={

    # The batch size used during training
    'BATCH_SIZE': 1,

    # The optimizer used for training
    'OPTIMIZER': torch.optim.Adam,

    # Parameters for the optimizer
    'OPTIM_PARAMS': {
        'lr': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.0},

    # The used loss functions
    'LOSS_FUNCTIONS': [F.cross_entropy],

    # The weights for the loss functions
    'LOSS_FUNCTION_WEIGHTS': [1.],

    # Seed for dataset splitting
    'DATASET_SEED': 1234,

    # Proportions of dataset splits
    'DATASET_SPLIT_PROPORTIONS': (80, 10, 10),

    # The directory where experiments are stored
    'EXPERIMENT_BASE_DIR': "../experiments/",

    # Directory of raw data
    'RAW_DATA_DIR': "/raw/data/dir", # <<<< Needs to set (e.g. in main.py)

    # Directory of preprocessed data
    'PREPROCESSED_DATA_DIR': "/preprocessed/data/dir", # <<<< Needs to set (e.g. in main.py)
}

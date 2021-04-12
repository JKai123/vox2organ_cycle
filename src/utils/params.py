""" Documentation of project-wide parameters and default values """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import Enum

import torch
import torch.nn.functional as F

class ExtendedEnum(Enum):
    """
    Extends an enum such that it can be converted to dict.
    """

    @classmethod
    def dict(cls):
        return {c.name: c.value for c in cls}

class HyperPs(ExtendedEnum):
    """
    List of hyperparameters. Use HyperPs.PARAM.name as an identifier and
    HyperPs.PARAM.value for the corresponding value.
    """

    # The batch size used during training
    BATCH_SIZE = 1

    # The optimizer used for training
    OPTIMIZER = torch.optim.Adam

    # Parameters for the optimizer
    OPTIM_PARAMS = {
        'lr': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.0}

    # The used loss functions
    LOSS_FUNCTIONS = [F.cross_entropy]

    # The weights for the loss functions
    LOSS_FUNCTION_WEIGHTS = [1.]

    # The directory where experiments are stored
    EXPERIMENT_BASE_DIR = "../experiments/"

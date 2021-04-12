""" Documentation of project-wide parameters and default values """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import Enum

class HyperPs(Enum):

    # The initial learning rate
    LEARNING_RATE = 1e-3

    # The batch size used during training
    BATCH_SIZE = 1

    # The directory where experiments are stored
    EXPERIMENT_BASE_DIR = "../experiments/"


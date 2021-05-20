
""" Put module information here """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import IntEnum

from data.hippocampus import Hippocampus
from data.cortex import Cortex

class SupportedDatasets(IntEnum):
    """ List supported datasets """
    Hippocampus = 1
    Cortex = 2

# Mapping supported datasets to split functions
dataset_split_handler = {
    SupportedDatasets.Hippocampus.name: Hippocampus.split,
    SupportedDatasets.Cortex.name: Cortex.split
}

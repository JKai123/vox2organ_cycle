
""" Put module information here """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import IntEnum

from data.hippocampus import Hippocampus
from data.cortex import Cortex

class SupportedDatasets(IntEnum):
    """ List supported datasets """
    Hippocampus = 1
    MALC_CSR = 2
    ADNI_CSR = 3

class CortexDatasets(IntEnum):
    """ List cortex datasets """
    MALC_CSR = SupportedDatasets.MALC_CSR.value
    ADNI_CSR = SupportedDatasets.ADNI_CSR.value

# Mapping supported datasets to split functions
dataset_split_handler = {
    SupportedDatasets.Hippocampus.name: Hippocampus.split,
    SupportedDatasets.MALC_CSR.name: Cortex.split,
    SupportedDatasets.ADNI_CSR.name: Cortex.split
}

dataset_paths = {
    SupportedDatasets.MALC_CSR.name: {
        'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/MALC_CSR/",
        'PREPROCESSED_DATA_DIR': "/home/fabianb/data/preprocessed/MALC_CSR/"
    },
    SupportedDatasets.ADNI_CSR.name: {
        'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/ADNI_CSR/",
        'PREPROCESSED_DATA_DIR': "/home/fabianb/data/preprocessed/ADNI_CSR/"
    },
    SupportedDatasets.Hippocampus.name: {
        'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/Task04_Hippocampus/"
    }
}

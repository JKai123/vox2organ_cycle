
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
        'PREPROCESSED_DATA_DIR': "/home/fabianb/data/preprocessed/MALC_CSR/",
        # 'FIXED_SPLIT': {
            # 'train': ['1010_3', '1007_3', '1003_3', '1104_3', '1015_3', '1001_3',
                      # '1018_3', '1014_3', '1122_3', '1000_3', '1008_3', '1128_3',
                      # '1017_3', '1113_3', '1011_3', '1125_3', '1005_3', '1107_3',
                      # '1019_3', '1013_3', '1006_3', '1012_3'],
            # 'validation': ['1036_3', '1110_3'],
            # 'test': ['1004_3', '1119_3', '1116_3', '1009_3', '1101_3', '1002_3']
        # },
        # split_challenge_mod
        'FIXED_SPLIT': {
            'train': ['1000_3', '1001_3', '1002_3', '1006_3', '1007_3', '1008_3',
                      '1009_3', '1010_3', '1011_3', '1012_3', '1013_3', '1014_3',
                      '1015_3', '1036_3', '1017_3'],
            'validation': ['1107_3', '1128_3'],
            'test': ['1101_3', '1003_3', '1004_3', '1125_3', '1005_3', '1122_3',
                     '1110_3', '1119_3', '1113_3', '1018_3', '1019_3', '1104_3',
                     '1116_3']
        }
    },
    SupportedDatasets.ADNI_CSR.name: {
        'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/ADNI_CSR/",
        'PREPROCESSED_DATA_DIR': "/home/fabianb/data/preprocessed/ADNI_CSR/",
        'FIXED_SPLIT': True # Read from files
    },
    SupportedDatasets.Hippocampus.name: {
        'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/Task04_Hippocampus/"
    }
}

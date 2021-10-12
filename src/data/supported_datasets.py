
""" Put module information here """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import sys
from enum import IntEnum

from pandas import read_csv

class SupportedDatasets(IntEnum):
    """ List supported datasets """
    Hippocampus = 1
    MALC_CSR = 2
    ADNI_CSR = 3

class CortexDatasets(IntEnum):
    """ List cortex datasets """
    MALC_CSR = SupportedDatasets.MALC_CSR.value
    ADNI_CSR = SupportedDatasets.ADNI_CSR.value

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

def valid_ids_MALC_CSR(candidates: list):
    """ Sort out non-valid ids of 'candidates' of samples in the MALC_CSR
    dataset and return adjusted list. """
    retest_ids = ('1023_3', '1024_3', '1025_3', '1038_3', '1039_3')
    valid = [c for c in candidates if (c[-1] == '3' and c not in retest_ids)]
    return valid

def valid_ids_ADNI_CSR(candidates: list):
    """ Sort out non-valid ids of 'candidates' of samples in the ADNI_CSR
    dataset and return adjusted list.

    !!!Note: This is a subset of ADNI at the moment!!!
    """
    raw_data_dir = "/mnt/nas/Data_Neuro/ADNI_CSR/"
    files_train = read_csv(os.path.join(
        raw_data_dir, 'train_small.csv'
    ), dtype=str)['IMAGEUID'].to_list()
    files_val = read_csv(os.path.join(
        raw_data_dir, 'val_small.csv'
    ), dtype=str)['IMAGEUID'].to_list()
    files_test = read_csv(os.path.join(
        raw_data_dir, 'test_small.csv'
    ), dtype=str)['IMAGEUID'].to_list()
    adni_valid = files_train + files_val + files_test
    valid = [c for c in candidates if c in adni_valid]
    return valid

def valid_ids(raw_data_dir: str):
    """ Get valid ids for supported datasets."""
    all_files = os.listdir(raw_data_dir)
    dataset = set(
        SupportedDatasets.__members__.keys()
    ).intersection(
        raw_data_dir.split("/")
    ).pop()
    this_module = sys.modules[__name__]
    return getattr(this_module, "valid_ids_" + dataset)(all_files)


""" Put module information here """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import re
import os
import sys
from enum import IntEnum

from pandas import read_csv

class SupportedDatasets(IntEnum):
    """ List supported datasets """
    Hippocampus = 1
    MALC_CSR = 2
    ADNI_CSR_small = 3
    ADNI_CSR_large = 4
    TRT_CSR_Data = 5

class CortexDatasets(IntEnum):
    """ List cortex datasets """
    MALC_CSR = SupportedDatasets.MALC_CSR.value
    ADNI_CSR_small = SupportedDatasets.ADNI_CSR_small.value
    ADNI_CSR_large = SupportedDatasets.ADNI_CSR_large.value
    TRT_CSR_Data = SupportedDatasets.TRT_CSR_Data.value

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
    SupportedDatasets.ADNI_CSR_small.name: {
        'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/ADNI_CSR/",
        'PREPROCESSED_DATA_DIR': "/home/fabianb/data/preprocessed/ADNI_CSR/",
        'FIXED_SPLIT': ["train_small.txt", "val_small.txt", "test_small.txt"] # Read from files
    },
    SupportedDatasets.ADNI_CSR_large.name: {
        'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/ADNI_CSR/",
        'PREPROCESSED_DATA_DIR': "/home/fabianb/data/preprocessed/ADNI_CSR/",
        'FIXED_SPLIT': ["ADNI_large_train_qc_pass.txt",
                        "ADNI_large_val_qc_pass.txt",
                        "ADNI_large_test_qc_pass.txt"] # Read from files
    },
    SupportedDatasets.TRT_CSR_Data.name: {
        'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/TRT_CSR_Data/",
        'PREPROCESSED_DATA_DIR': "/home/fabianb/data/preprocessed/TRT_CSR_Data/",
        'DATASET_SPLIT_PROPORTIONS': [0, 0, 100] # Test only
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
    """
    raw_data_dir = "/mnt/nas/Data_Neuro/ADNI_CSR/"
    convert = lambda x: str(int(x)) # 'x\n' --> 'x'
    train_split = os.path.join(raw_data_dir, 'train_large.txt')
    files_train = list(map(convert, list(open(train_split, 'r').readlines())))
    val_split = os.path.join(raw_data_dir, 'val_large.txt')
    files_val = list(map(convert, list(open(val_split, 'r').readlines())))
    test_split = os.path.join(raw_data_dir, 'test_large.txt')
    files_test = list(map(convert, list(open(test_split, 'r').readlines())))
    adni_valid_orig = files_train + files_val + files_test
    adni_valid_preprocessed =\
        os.listdir("/home/fabianb/data/preprocessed/ADNI_CSR/")
    adni_valid = set(adni_valid_orig).intersection(adni_valid_preprocessed)
    valid = [c for c in candidates if c in adni_valid]
    return valid

def valid_ids_TRT_CSR_Data(candidates: list):
    """ Sort out non-valid ids of 'candidates' of samples in the ADNI_CSR
    dataset and return adjusted list.
    """
    valid = [c for c in candidates if re.match(".*subject_.*/T1_.*", c)]
    return valid

def valid_ids(raw_data_dir: str):
    """ Get valid ids for supported datasets."""
    # IDs can be in the raw_data_dir or in subdirectories
    files_in_subdir = "TRT_CSR_Data" in raw_data_dir
    if files_in_subdir:
        all_files = ["/".join(x[0].split("/")[-2:])
                     for x in os.walk(raw_data_dir)]
    else:
        all_files = os.listdir(raw_data_dir)
    # Intersection of dataset dir and dataset name
    dataset = set(
        x for y in SupportedDatasets.__members__.keys()
        for x in raw_data_dir.split("/")
        if (x in y and x != "")
    ).pop()
    this_module = sys.modules[__name__]
    return getattr(this_module, "valid_ids_" + dataset)(all_files)

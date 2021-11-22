
""" Add supported datasets and their paths here. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import sys
from enum import IntEnum

class SupportedDatasets(IntEnum):
    """ List supported datasets """
    ADNI_CSR_large = 4

class CortexDatasets(IntEnum):
    """ List cortex datasets """
    ADNI_CSR_large = SupportedDatasets.ADNI_CSR_large.value

dataset_paths = {
    SupportedDatasets.ADNI_CSR_large.name: {
        'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/ADNI_CSR/",
        'PREPROCESSED_DATA_DIR': "/home/fabianb/data/preprocessed/ADNI_CSR/",
        'N_REF_POINTS_PER_STRUCTURE': 26800,
        'FIXED_SPLIT': [
            "ADNI_large_train_qc_pass.txt",
            "ADNI_large_val_qc_pass.txt",
            "ADNI_large_test_qc_pass.txt"
        ] # Read from files
        # General
        # 'RAW_DATA_DIR': "/path/to/raw/data",
        # 'PREPROCESSED_DATA_DIR': "/path/to/preprocessed/data",
        # 'N_REF_POINTS_PER_STRUCTURE': "<min. number of verts in training set>",
        # 'FIXED_SPLIT': [
            # "train_ids.txt",
            # "val_ids.txt",
            # "test_ids.txt"
        # ] # Read from files
    },
}

def valid_ids_ADNI_CSR(candidates: list):
    """ Sort out non-valid ids of 'candidates' of samples in the ADNI_CSR
    dataset and return adjusted list.
    """
    valid = [c for c in candidates if c.isdigit()]
    return valid


def valid_ids(raw_data_dir: str):
    """ Get valid ids for supported datasets."""

    # All files in directory are ID candidates
    all_files = os.listdir(raw_data_dir)

    # Intersection of dataset dir and dataset name
    dataset = set(
        x for y in SupportedDatasets.__members__.keys()
        for x in raw_data_dir.split("/")
        if (x in y and x != "")
    ).pop()
    this_module = sys.modules[__name__]
    return getattr(this_module, "valid_ids_" + dataset)(all_files)

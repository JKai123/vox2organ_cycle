
""" Mapping cortex label names and ids. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import IntEnum

import numpy as np
import torch

class CortexLabels(IntEnum):
    right_white_matter = 41
    left_white_matter = 2
    left_cerebral_cortex = 3
    right_cerebral_cortex = 42

def combine_labels(labels, names):
    """ Only consider labels in 'names' and set all those labels equally to 1
    """
    ids = [CortexLabels[n].value for n in names]
    combined_labels = np.isin(labels, ids).astype(int)

    if isinstance(labels, torch.Tensor):
        combined_labels = torch.from_numpy(combined_labels)

    return combined_labels

def valid_MALC_ids(candidates: list):
    """ Sort out non-valid ids of 'candidates' of samples in the MALC_CSR
    dataset and return adjusted list. """
    retest_ids = ('1023_3', '1024_3', '1025_3', '1038_3', '1039_3')
    valid = [c for c in candidates if (c[-1] == '3' and c not in retest_ids)]
    return valid

def valid_ADNI_ids(candidates: list):
    """ Sort out non-valid ids of 'candidates' of samples in the ADNI_CSR
    dataset and return adjusted list. """
    valid = [c for c in candidates if (c.isdigit())]
    return valid

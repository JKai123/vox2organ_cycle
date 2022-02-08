
""" Convenient dataset splitting. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from data.supported_datasets import SupportedDatasets
from data.hippocampus import Hippocampus
from data.cortex import Cortex

# Mapping supported datasets to split functions
dataset_split_handler = {
    SupportedDatasets.Hippocampus.name: Hippocampus.split,
    SupportedDatasets.MALC_CSR.name: Cortex.split,
    SupportedDatasets.TRT_CSR_Data.name: Cortex.split,
    SupportedDatasets.OASIS.name: Cortex.split,
    SupportedDatasets.OASIS_FS72.name: Cortex.split,
    SupportedDatasets.ADNI_CSR_fail.name: Cortex.split,
    SupportedDatasets.ADNI_CSR_orig.name: Cortex.split,
    SupportedDatasets.ADNI_CSR_small.name: Cortex.split,
    SupportedDatasets.ADNI_CSR_large.name: Cortex.split
}



""" Convenient dataset splitting. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from data.supported_datasets import SupportedDatasets
from data.hippocampus import Hippocampus
from data.cortex import CortexDataset

# Mapping supported datasets to split functions
dataset_split_handler = {
    SupportedDatasets.Hippocampus.name: Hippocampus.split,
    SupportedDatasets.MALC_CSR.name: CortexDataset.split,
    SupportedDatasets.TRT_CSR_Data.name: CortexDataset.split,
    SupportedDatasets.OASIS.name: CortexDataset.split,
    SupportedDatasets.OASIS_FS72.name: CortexDataset.split,
    SupportedDatasets.ADNI_CSR_fail.name: CortexDataset.split,
    SupportedDatasets.ADNI_CSR_orig.name: CortexDataset.split,
    SupportedDatasets.ADNI_CSR_small.name: CortexDataset.split,
    SupportedDatasets.ADNI_CSR_large.name: CortexDataset.split
}


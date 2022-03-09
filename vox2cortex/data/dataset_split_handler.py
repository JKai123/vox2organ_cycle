
""" Convenient dataset splitting. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from data.supported_datasets import SupportedDatasets
from data.hippocampus import Hippocampus
from data.cortex import CortexParcellationDataset

# Mapping supported datasets to split functions
dataset_split_handler = {
    SupportedDatasets.Hippocampus.name: Hippocampus.split,
    SupportedDatasets.MALC_CSR.name: CortexParcellationDataset.split,
    SupportedDatasets.TRT_CSR_Data.name: CortexParcellationDataset.split,
    SupportedDatasets.OASIS.name: CortexParcellationDataset.split,
    SupportedDatasets.OASIS_FS72.name: CortexParcellationDataset.split,
    SupportedDatasets.Mindboggle.name: CortexParcellationDataset.split,
    SupportedDatasets.ADNI_CSR_fail.name: CortexParcellationDataset.split,
    SupportedDatasets.ADNI_CSR_orig.name: CortexParcellationDataset.split,
    SupportedDatasets.ADNI_CSR_small.name: CortexParcellationDataset.split,
    SupportedDatasets.ADNI_CSR_large.name: CortexParcellationDataset.split
}

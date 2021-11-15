
""" Generate a visualization of regions affected by Alzheimer's based on
cortical thickness measurements. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import pandas
import trimesh
import numpy as np
from scipy.stats import ttest_ind

HEMI = "lh"
SURFACE_ID = 2
EPOCH = 38
THICKNESS_DIR = "/home/fabianb/work/cortex-parcellation-using-meshes/experiments/exp_576_v2/test_template_168058_ADNI_CSR_large/thickness/"
OUT_DIR = "/home/fabianb/work/cortex-parcellation-using-meshes/experiments/exp_576_v2/test_template_168058_ADNI_CSR_large/"
DATA_SPEC = "/mnt/nas/Data_Neuro/ADNI_CSR/ADNI_large_test_qc_pass.csv"

def get_thickness_fn(file_id, structure):
    return f"{file_id}_epoch{str(EPOCH)}_struc{str(structure)}_meshpred.thickness.reg.npy"

def load_thickness_values(ids):
    th_all = []
    for i in ids:
        fn = os.path.join(THICKNESS_DIR, get_thickness_fn(i, SURFACE_ID))
        try:
            thickness = np.load(fn)
        except FileNotFoundError:
            continue
        th_all.append(thickness)

    return np.stack(th_all)

# Find ids with and without AD
specs = pandas.read_csv(DATA_SPEC)
get_ids = lambda group: specs['IMAGEUID'][specs['DX'] == group]
ids_CN = get_ids('CN')
ids_Dem = get_ids('Dementia')

# Load thickness values
thickness_CN = load_thickness_values(ids_CN)
thickness_Dem = load_thickness_values(ids_Dem)

# Per-vertex t-test
stat, p = ttest_ind(thickness_CN, thickness_Dem, axis=0, alternative='less')
out_file = os.path.join(OUT_DIR, "p_thickness")
np.save(out_file, p)
print("Wrote p values to ", out_file)


""" Compute a per-parcel mean thickness. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import nibabel as nib
import pandas as pd
import numpy as np

EPOCH = 20
EXP_DIR = "/home/fabianb/work/vox2cortex-parc/experiments/quadro-exp_53/test_template_full_ADNI_CSR_large/"
SURFACES = ("lh_white", "rh_white", "lh_pial", "rh_pial")
DATA_SPEC = f"/mnt/nas/Data_Neuro/ADNI_CSR/ADNI_large_test_qc_pass.csv"
IMAGE_IDS = pd.read_csv(DATA_SPEC)['IMAGEUID']
THICKNESS_DIR = os.path.join(EXP_DIR, 'thickness')

for i, s in enumerate(SURFACES):
    t_annot_file = f"/mnt/nas/Data_Neuro/Parcellation_atlas/fsaverage/v2c_template/{s}.aparc.DKTatlas40.annot"
    parc, _, names = nib.freesurfer.io.read_annot(t_annot_file)
    for uid in IMAGE_IDS:
        thickness_fn = f"{uid}_epoch{EPOCH}_struc{i}_meshpred.thickness.npy"
        th = np.load(os.path.join(THICKNESS_DIR, thickness_fn))

        parcels = np.unique(parc)
        p_means = []
        for p in parcels:
            p_means.append(th[parc == p].mean())
        p_means = np.stack(p_means)
        out_file = os.path.join(
            THICKNESS_DIR,
            f"{uid}_epoch{EPOCH}_struc{i}_meshpred.mean_thickness.csv"
        )
        pd.DataFrame(data=np.stack((parcels, p_means)),
                     columns=np.array(names)[parcels]).to_csv(out_file)

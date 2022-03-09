
""" Evaluate the decline in cortical thickness due to Alzheimer's based on
a parcellation of the cortex."
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"


import os
import trimesh
import pandas as pd
import numpy as np
import nibabel as nib
import statsmodels.formula.api as smf
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind

from utils.fs_utils import fs_to_fsaverage, pred_to_fs_to_fsaverage

##### TO SET #####
# EXP_NAME = "quadro-exp_53"
EXP_NAME = "oasis_v2cparc_dropout_fsaverage"
EPOCH = 150
# DATASET = "OASIS_FS72"
DATASET = "ADNI_CSR_large"
SURFACE = "lh_white"
SIZE = "163842"
PRED = True # False for FreeSurfer
TEMPLATE = False # False for GNN parcellation approaches
#####
TEMPLATE_ANNOT_FILE = f"/mnt/nas/Data_Neuro/Parcellation_atlas/fsaverage/v2c_template/{SURFACE}.aparc.DKTatlas40.annot"
PARC_LABELS = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35] #32 total
SURFACES_ALL = ("lh_white", "rh_white", "lh_pial", "rh_pial")
TEST_DIR = f"/home/fabianb/work/vox2cortex-parc/experiments/{EXP_NAME}/test_template_{SIZE}_{DATASET}"
# TEST_DIR = f"/home/fabianb/work/vox2cortex-parc/experiments/{EXP_NAME}/test_template_{SIZE}_OASIS"
THICKNESS_DIR = os.path.join(TEST_DIR, "thickness")
if DATASET == 'OASIS_FS72':
    FS_DIR = f"/mnt/nas/Data_Neuro/OASIS/FS_full_72/"
elif DATASET == 'ADNI_CSR_large':
    FS_DIR = f"/mnt/nas/Data_Neuro/ADNI_CSR/FS72/FS72/"
else:
    raise ValueError("Unknown dataset")
OUT_DIR = os.path.join(TEST_DIR, "parc_thickness_gt")

if PRED:
    SURFACE_DIR = os.path.join(TEST_DIR, "meshes")
    OUT_DIR = os.path.join(TEST_DIR, "parc_thickness_pred")
else:
    SURFACE_DIR = FS_DIR

if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

if DATASET == "OASIS_FS72":
    DATA_SPEC = f"/mnt/nas/Data_Neuro/OASIS/OASIS_test.csv"
elif DATASET == "ADNI_CSR_large":
    DATA_SPEC = (
        f"/mnt/nas/Data_Neuro/ADNI_CSR/ADNI_large_test_qc_pass.csv"
    )

def load_thickness_values(ids):
    th_all = []
    for i in ids:
        if PRED:
            th_fn = f"{i}_epoch{str(EPOCH)}_struc{SURFACES_ALL.index(SURFACE)}_meshpred.thickness.npy"
        else:
            th_fn = f"{i}_struc{SURFACES_ALL.index(SURFACE)}_gt.thickness.npy"
        fn = os.path.join(THICKNESS_DIR, th_fn)
        thickness = np.load(fn)
        th_all.append(thickness)

    return th_all

def load_parc_labels(ids):
    parc_all = []
    for i in ids:
        suffix = ".aparc.DKTatlas.annot" if DATASET == "ADNI_CSR_large" else ".aparc.DKTatlas40.annot"

        if not PRED:
            fn = os.path.join(SURFACE_DIR, i, "label", SURFACE.split("_")[0] + suffix)
        elif not TEMPLATE:
            fn = os.path.join(SURFACE_DIR, f"{i}_epoch{str(EPOCH)}_struc{SURFACES_ALL.index(SURFACE)}_parcellation.annot")

        parc_all.append(nib.freesurfer.io.read_annot(fn)[0].astype(np.int32))

    return parc_all

def ttest(specs):
    if DATASET == "ADNI_CSR_large":
        get_ids = lambda group: specs['IMAGEUID'][specs['DX'] == group]
        ids_CN = list(map(str, get_ids('CN')))
        ids_Dem = list(map(str, get_ids('Dementia')))
    elif DATASET == "OASIS_FS72":
        get_ids = lambda group: specs['ID'][specs['CDR'] == group]
        ids_CN = get_ids(0.0) # Healthy
        ids_Dem = get_ids(1.0) # Dementia

    # Load thickness values
    thickness_CN = load_thickness_values(ids_CN)
    thickness_Dem = load_thickness_values(ids_Dem)

    # Map parcellation and thickness values to fsaverage for FreeSurfer
    if not PRED:
        # Load parcellation
        parc_CN = load_parc_labels(ids_CN)
        parc_Dem = load_parc_labels(ids_Dem)

        thickness_CN = np.stack(
            [fs_to_fsaverage(i, th, SURFACE_DIR, SURFACE)
             for i, th in zip(ids_CN, thickness_CN)]
        )
        thickness_Dem = np.stack(
            [fs_to_fsaverage(i, th, SURFACE_DIR, SURFACE)
             for i, th in zip(ids_Dem, thickness_Dem)]
        )
        parc_CN = np.stack(
            [fs_to_fsaverage(i, th, SURFACE_DIR, SURFACE)
             for i, th in zip(ids_CN, parc_CN)]
        )
        parc_Dem = np.stack(
            [fs_to_fsaverage(i, th, SURFACE_DIR, SURFACE)
             for i, th in zip(ids_Dem, parc_Dem)]
        )

    # Map parcellation and thickness values to fsaverage for gnn prediction
    if not TEMPLATE:
        # Load predicted meshes/point sets
        s_index = SURFACES_ALL.index(SURFACE)
        points_CN = []
        for i in ids_CN:
            points_CN.append(
                trimesh.load(
                    os.path.join(
                    SURFACE_DIR,
                    f"{i}_epoch{EPOCH}_struc{s_index}_meshpred.ply"
                    ),
                    process=False
                ).vertices
            )
        points_Dem = []
        for i in ids_Dem:
            points_Dem.append(
                trimesh.load(
                    os.path.join(
                    SURFACE_DIR,
                    f"{i}_epoch{EPOCH}_struc{s_index}_meshpred.ply"
                    ),
                    process=False
                ).vertices
            )

        # Load parcellation
        parc_CN = load_parc_labels(ids_CN)
        parc_Dem = load_parc_labels(ids_Dem)

        # Map to fsaverage
        thickness_CN = np.stack(
            [pred_to_fs_to_fsaverage(p, i, th, FS_DIR, SURFACE)
             for p, i, th in zip(points_CN, ids_CN, thickness_CN)]
        )
        thickness_Dem = np.stack(
            [pred_to_fs_to_fsaverage(p, i, th, FS_DIR, SURFACE)
             for p, i, th in zip(points_Dem, ids_Dem, thickness_Dem)]
        )
        parc_CN = np.stack(
            [pred_to_fs_to_fsaverage(p, i, th, FS_DIR, SURFACE)
             for p, i, th in zip(points_CN, ids_CN, parc_CN)]
        )
        parc_Dem = np.stack(
            [pred_to_fs_to_fsaverage(p, i, th, FS_DIR, SURFACE)
             for p, i, th in zip(points_Dem, ids_Dem, parc_Dem)]
        )

    # Load fsaverage template size
    fsparc, colors, names = nib.freesurfer.io.read_annot(TEMPLATE_ANNOT_FILE)

    p_vals_per_vertex = np.empty(fsparc.shape)
    p_vals_per_vertex[:] = np.NaN

    # Iterate over regions
    for label, name in zip(PARC_LABELS, np.array(names)[PARC_LABELS]):
        # t-test for the average thickness in the region
        vertex_mask = fsparc == label
        # To decide: mean or not
        region_th_CN = np.stack([th[vertex_mask].mean() for th in thickness_CN])
        region_th_Dem = np.stack([th[vertex_mask].mean() for th in thickness_Dem])

        # region_th_CN = np.stack([th[vertex_mask] for th in thickness_CN])
        # region_th_Dem = np.stack([th[vertex_mask] for th in thickness_Dem])
        stat, p = ttest_ind(
            region_th_CN, region_th_Dem, axis=None, alternative='greater'
        )
        # Correct pvalues
        p_corrected = p
        print(p_corrected)
        # p_corrected = np.log10(p)
        # reject, p_corrected, alphacSidak, alphacBonf = multipletests(
            # p, alpha=0.05, method='fdr_bh'
        # )
        # p_corrected[reject] = 2
        # p_corrected[~reject] = 0

        # Write
        out_file = os.path.join(OUT_DIR, f"p_thickness_region{label}_{SURFACE}.npy")
        np.save(out_file, p_corrected)
        # print(f"Wrote p value of region {label} ({name}) to {out_file}")

        p_vals_per_vertex[vertex_mask] = p_corrected

    out_file = os.path.join(OUT_DIR, f"p_thickness_region_summary_{SURFACE}.npy")
    np.save(out_file, p_vals_per_vertex)


if __name__ == '__main__':
    specs = pd.read_csv(DATA_SPEC)
    ttest(specs)


""" Generate a visualization of regions affected by Alzheimer's based on
cortical thickness measurements. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import pandas as pd
import trimesh
import numpy as np
import statsmodels.formula.api as smf
from tqdm import tqdm
from statsmodels.stats.multitest import multipletests
from scipy.stats import ttest_ind

from scripts.group_analysis.smooth_thickness import smooth_values

SURFACE = "rh_white"
HEMI = SURFACE.split("_")[0]
SURFACES = ("lh_white", "rh_white", "lh_pial", "rh_pial")
EPOCH = 38
THICKNESS_DIR = "/home/fabianb/work/cortex-parcellation-using-meshes/experiments/exp_576_v2/test_template_168058_ADNI_CSR_large/thickness/"
OUT_DIR = "/home/fabianb/work/cortex-parcellation-using-meshes/experiments/exp_576_v2/test_template_168058_ADNI_CSR_large/group_analysis/"
DATA_SPEC = "/mnt/nas/Data_Neuro/ADNI_CSR/ADNI_large_test_qc_pass.csv"

def get_thickness_fn(file_id, structure):
    return f"{file_id}_epoch{str(EPOCH)}_struc{str(structure)}_meshpred.thickness.reg.npy"

def load_thickness_values(ids):
    th_all = []
    for i in ids:
        fn = os.path.join(
            THICKNESS_DIR, get_thickness_fn(i, SURFACES.index(SURFACE))
        )
        try:
            thickness = np.load(fn)
        except FileNotFoundError:
            continue
        th_all.append(thickness)

    return np.stack(th_all)

def load_thickness_values_and_ids(ids):
    """ Return a dict with entries 'ID': thickness per vertex """
    th_all = {}
    for i in ids:
        fn = os.path.join(THICKNESS_DIR, get_thickness_fn(i, SURFACE_ID))
        try:
            thickness = np.load(fn)
        except FileNotFoundError:
            continue
        th_all[i] = thickness
        n_verts = len(thickness)

    return th_all, n_verts

def linear_model(specs):
    # Load thickness values per ID
    thickness_all, n_verts = load_thickness_values_and_ids(specs['IMAGEUID'])

    # Only take the required columns
    specs = specs[['AGE', 'PTGENDER', 'DX', 'IMAGEUID']]
    pvalues = []

    for vi in range(n_verts):
        specs_v = specs.copy()
        thickness_vi = {k: v[vi] for k, v in thickness_all.items()}
        thickness_vi_df = pd.DataFrame(
            {'IMAGEUID': [k for k, _ in thickness_vi.items()],
             'Thickness': [v for _, v in thickness_vi.items()]}
        )
        specs_v = specs_v.merge(thickness_vi_df, how='inner')

        # Only distinguish between CN and Dementia
        mask = specs_v['DX'].isin(('CN', 'Dementia'))
        specs_v = specs_v[mask]

        model = smf.ols(
            formula = "Thickness ~ C(DX) + C(PTGENDER) + AGE",
            data=specs_v
        )
        result = model.fit()
        pvalues.append(result.pvalues[1])

    # Correct pvalues
    reject, pvalues = multipletests(pvalues, alpha=0.05, method=fdr_bh)
    # Write
    out_file = os.path.join(OUT_DIR, "p_thickness")
    np.save(out_file, np.array(pvalues))
    print("Wrote p values to ", out_file)

def ttest(specs):
    get_ids = lambda group: specs['IMAGEUID'][specs['DX'] == group]
    ids_CN = get_ids('CN')
    ids_Dem = get_ids('Dementia')

    # Load thickness values
    thickness_CN = load_thickness_values(ids_CN)
    thickness_Dem = load_thickness_values(ids_Dem)

    # Smooth
    n_smooth = 0
    thickness_CN = np.stack(
        [smooth_values(th_CN, n_smooth, HEMI) for th_CN in tqdm(thickness_CN)]
    )
    np.stack(
        [smooth_values(th_Dem, n_smooth, HEMI) for th_Dem in
                         tqdm(thickness_Dem)]
    )

    # Per-vertex t-test
    stat, p = ttest_ind(thickness_CN, thickness_Dem, axis=0,
                        alternative='greater')
    # Correct pvalues
    p_corrected = p
    # p_corrected = np.log10(p)
    # reject, p_corrected, alphacSidak, alphacBonf = multipletests(
        # p, alpha=0.05, method='fdr_bh'
    # )
    # p_corrected[reject] = 2
    # p_corrected[~reject] = 0

    # Write
    out_file = os.path.join(OUT_DIR, f"p_thickness_{SURFACE}_ours")
    np.save(out_file, p_corrected)
    print("Wrote p values to ", out_file)

if __name__ == '__main__':
    specs = pd.read_csv(DATA_SPEC)
    ttest(specs)


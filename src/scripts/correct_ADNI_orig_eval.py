
""" Remove a couple of values from the ADNI_orig evaluation due to severe
failure of FreeSurfer. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

import numpy as np

from scripts.eval_meshes import ad_hd_output

EXP_NAME = "exp_581"
N_TEST_VERTICES = 168058
DATASET = "ADNI_CSR_orig"
EPOCH = 15

TEST_DIR = f"../experiments/{EXP_NAME}/test_template_{N_TEST_VERTICES}_{DATASET}"
AD_HD_DIR = os.path.join(TEST_DIR, "ad_hd")

STRUCTURES = ("lh_white", "rh_white", "lh_pial", "rh_pial")

IGNORE_IDS = ("118750", "118795", "444084", "59950")

all_files = os.listdir(AD_HD_DIR)
all_files = [f for f in all_files if str(EPOCH) in f]

ad_all = []
hd_all = []

new_summary_file = os.path.join(
    TEST_DIR, "eval_ad_hd_summary_corrected.csv"
)
for s_i, s in enumerate(STRUCTURES):
    new_struc_summary_file = os.path.join(
        TEST_DIR, f"{s}_eval_ad_hd_summary_corrected.csv"
    )
    struc_files = [f for f in all_files if f"struc{s_i}" in f]
    print(f"Found a total of {len(struc_files)} files.")
    struc_files = [f for f in struc_files
                   if all(ig not in f for ig in IGNORE_IDS)]
    print(f"Corrected length is a total of {len(struc_files)} files.")

    ad_struc = [np.load(os.path.join(AD_HD_DIR, f)).item()
                for f in struc_files if ".ad." in f]
    hd_struc = [np.load(os.path.join(AD_HD_DIR, f)).item()
                for f in struc_files if ".hd." in f]

    ad_all += ad_struc
    hd_all += hd_struc

    struc_arr = np.stack([ad_struc, hd_struc], axis=1)
    ad_hd_output(struc_arr, new_struc_summary_file)

    print(f"Wrote new summary file {new_struc_summary_file}.")

ad_hd_output(np.stack([ad_all, hd_all], axis=1), new_summary_file)
print(f"Wrote new summary file {new_summary_file}.")

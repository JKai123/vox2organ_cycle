
""" Get the max. number of vertices of a dataset """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import nibabel as nib
import os
import trimesh
from tqdm import tqdm

from data.supported_datasets import valid_ids

directory = "/mnt/nas/Data_Neuro/ADNI_CSR/"

files = valid_ids(directory)
files.sort()
print(f"{len(files)} files")

names = ("lh_pial_reduced_0.3.ply",
         "rh_pial_reduced_0.3.ply",
         "lh_white_reduced_0.3.ply",
         "rh_white_reduced_0.3.ply")

min_n = {k: 100000. for k in names}

for fn in tqdm(files, position=0, leave=True):
    for n in names:
        fn_full = os.path.join(directory, fn, n)
        mesh = trimesh.load_mesh(fn_full)
        n_V = len(mesh.vertices)
        if n_V < min_n[n]:
            min_n[n] = n_V

print("Min. number of vertices: ")
print(min_n)

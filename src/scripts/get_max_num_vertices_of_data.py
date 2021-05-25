
""" Get the max. number of vertices of a dataset """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import nibabel as nib
import os
import trimesh

directory = "/mnt/nas/Data_Neuro/MALC_CSR/"

files = [fn for fn in os.listdir(directory) if "meshes" not in fn]
files.sort()
print(f"{len(files)} files")

names = ("lh_pial.stl",
         "rh_pial.stl",
         "lh_white.stl",
         "rh_white.stl")

max_n = {k: 0. for k in names}

for fn in files:
    for n in names:
        fn_full = os.path.join(directory, fn, n)
        mesh = trimesh.load_mesh(fn_full)
        n_V = len(mesh.vertices)
        if n_V > max_n[n]:
            max_n[n] = n_V

print("Max. number of vertices: ")
print(max_n)


""" Get the max. number of vertices of a dataset """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import nibabel as nib
import os
import trimesh

directory = "/mnt/nas/Data_Neuro/MALC_CSR/"

files = [fn for fn in os.listdir(directory) if ("meshes" not in fn and
                                                "unregistered" not in fn)]
files.sort()
print(f"{len(files)} files")

names = ("lh_pial_reduced_0.3.stl",
         "rh_pial_reduced_0.3.stl",
         "lh_white_reduced_0.3.stl",
         "rh_white_reduced_0.3.stl")

min_n = {k: 100000. for k in names}

for fn in files:
    for n in names:
        fn_full = os.path.join(directory, fn, n)
        mesh = trimesh.load_mesh(fn_full)
        n_V = len(mesh.vertices)
        if n_V < min_n[n]:
            min_n[n] = n_V

print("Min. number of vertices: ")
print(min_n)

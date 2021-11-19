""" Get the max. image dimensions of data across dimensions """

import nibabel as nib
import os

directory = "/mnt/nas/Data_Neuro/Task04_Hippocampus/imagesTr/"

files = [fn for fn in os.listdir(directory) if "._" not in fn]
files.sort()
print(f"{len(files)} files")

max_dim = [0, 0, 0]

for fn in files:
    fn_full = os.path.join(directory, fn)
    img = nib.load(fn_full)
    for i, m in enumerate(max_dim):
        if img.shape[i] > m:
            max_dim[i] = img.shape[i]

print(f"Largest dimensions: {max_dim}")


""" Map values on a sphere to the fsaverage sphere. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from argparse import ArgumentParser

import numpy as np
import torch
import nibabel as nib
from pytorch3d.ops import knn_points

device = "cuda:1"

argparser = ArgumentParser()
argparser.add_argument('hemisphere',
                       type=str,
                       help="The hemisphere, 'lh' or 'rh'.")
argparser.add_argument('moving',
                       type=str,
                       help="Moving mesh.")
argparser.add_argument('values',
                       type=str,
                       help="The values corresponding to fixed vertices.")
argparser.add_argument('out',
                       type=str,
                       help="Out file for transferred values on fsaverage.")

args = argparser.parse_args()
moving = nib.freesurfer.io.read_geometry(args.moving)
moving_values = nib.freesurfer.io.read_morph_data(args.values)

average_sphere = nib.freesurfer.io.read_geometry(os.path.join(
    os.environ['FREESURFER_HOME'],
    "subjects",
    "fsaverage",
    "surf",
    f"{args.hemisphere}.sphere.reg.avg"
))

print(f"Transferring values from {args.moving} to fsaverage sphere of"
      f" {args.hemisphere}")


moving_verts = torch.tensor(moving[0]).to(device)[None]
fixed_verts = torch.tensor(average_sphere[0]).to(device)[None]

# Weighted sum of neighboring values
K = 5
dists, idx, _ = knn_points(fixed_verts, moving_verts, K=K)
dists = dists.squeeze()
weights = dists / dists.sum(axis=1, keepdim=True)
moving_values = torch.tensor(moving_values.astype(np.float32)).to(device)
moved_values = (moving_values[idx.squeeze()] * weights).sum(1).cpu().numpy()

np.save(args.out, moved_values)

print("Wrote values transferred to average sphere to ", args.out)

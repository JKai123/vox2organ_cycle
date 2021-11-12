
""" Register meshes using ICP algorithm and transfer per-vertex values to the
fixed mesh. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from argparse import ArgumentParser

import numpy as np
import torch
import trimesh
import nibabel as nib
from pytorch3d.ops import knn_points

device = "cuda:1"

argparser = ArgumentParser()
argparser.add_argument('moving',
                       type=str,
                       help="Moving mesh.")
argparser.add_argument('values',
                       type=str,
                       help="The values corresponding to fixed vertices.")
argparser.add_argument('fixed',
                       type=str,
                       help="Fixed mesh.")
argparser.add_argument('out',
                       type=str,
                       help="Out file for transferred values.")

args = argparser.parse_args()
moving = nib.freesurfer.io.read_geometry(args.moving)
moving_values = np.load(args.values)
fixed = nib.freesurfer.io.read_geometry(args.fixed)

print("Register mesh ", args.moving, " to ", args.fixed)

tri_moving = trimesh.Trimesh(moving[0], moving[1], process=False)
tri_fixed = trimesh.Trimesh(fixed[0], fixed[1], process=False)

to_fixed, cost = tri_moving.register(tri_fixed)
print("Registration cost: ", str(cost))

moved_verts = trimesh.transform_points(
    tri_moving.vertices, to_fixed
)

# Transfer values to fixed vertices
moved_verts = torch.tensor(moved_verts).to(device)[None]
fixed_verts = torch.tensor(fixed[0]).to(device)[None]

K = 1
dists, idx, _ = knn_points(fixed_verts, moved_verts, K=K)
# Weight with d / sum_neighbors(d) (only relevant if K > 1 which does probably
# suboptimal since k-th nearest neighbor may lie on another gyrus or sulcus and
# ICP has been applied)
weights = dists / dists.sum(axis=2, keepdim=True)
moving_values = torch.tensor(moving_values)[None, :, None].to(device)
idx_expanded = idx[:, :, :, None].expand(-1, -1, -1, 1)
moved_values = moving_values[:, :, None].expand(-1, -1, K, -1).gather(
    1, idx_expanded
).squeeze().cpu().numpy()

np.save(args.out, moved_values)

print("Wrote transferred values to  ", args.out)

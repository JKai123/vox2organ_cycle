
""" Checking implementation of inner volume """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch
from utils.utils_pca_loss import sample_inner_volume_in_voxel
from utils.utils_pca_loss import sample_outer_surface_in_voxel

t = torch.tensor([[[0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0]],

                  [[0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0]],

                  [[0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0]],

                  [[0, 1, 1, 1, 0],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [1, 1, 1, 1, 1],
                   [0, 1, 1, 1, 0]],

                  [[0, 0, 0, 0, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 1, 1, 1, 0],
                   [0, 0, 0, 0, 0]]])
print("t:")
print(t)

inner_vol = sample_inner_volume_in_voxel(t)
print("Inner vol:")
print(inner_vol)

assert (inner_vol + sample_outer_surface_in_voxel(inner_vol) == t).all(), "Test failed"

print("Test passed.")

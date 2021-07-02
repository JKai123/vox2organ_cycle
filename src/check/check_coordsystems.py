
""" Checking correctness of coordinate transformations. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch
import torch.nn.functional as F

from utils.utils import (
    normalize_vertices,
    unnormalize_vertices,
    normalize_vertices_per_max_dim,
    unnormalize_vertices_per_max_dim
)

def check_coords():
    batch_size = 2
    shape = (16, 32, 64)

    imgs = []
    verts = []
    for i in range(batch_size):
        # Input images
        img = torch.zeros(shape).float()
        img[12+i, 20, 4] = 1. # Nonzero point
        img[6, 25+i,  10] = 1. # Nonzero point
        imgs.append(img)

        # Coordinates
        # Img coords
        coo = torch.nonzero(img).float()
        # Normalized w.r.t. largest img dimension
        coo_n_max = normalize_vertices_per_max_dim(coo, shape)
        # Normalized w.r.t. each image dimension separately
        coo_n = normalize_vertices(
            unnormalize_vertices_per_max_dim(coo_n_max, shape), shape
        )
        verts.append(coo_n)
        # Further checks
        coo_un_max = unnormalize_vertices_per_max_dim(coo_n_max, shape)
        coo_un = unnormalize_vertices(coo_n, shape)
        assert torch.allclose(coo_un_max, coo)
        assert torch.allclose(coo_un, coo)

    imgs = torch.stack(imgs)
    verts = torch.stack(verts)

    features = F.grid_sample(imgs.unsqueeze(1), verts[:, :, None, None], mode='bilinear', padding_mode='border', align_corners=True)

    # Should be all ones
    assert torch.allclose(features, torch.tensor(1.))
    print(features)
    print("Successful.")

if __name__ == '__main__':
    check_coords()

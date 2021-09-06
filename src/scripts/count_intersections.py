
""" Convenience script to count the number of self-intersection. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from argparse import ArgumentParser

import numpy as np
import torch
import trimesh
from mesh_intersection.bvh_search_tree import BVH

def count_self_intersections(mesh: trimesh.Trimesh):
    """ Count the number of self-intersections in the provided mesh. """
    vertices = torch.tensor(mesh.vertices, dtype=torch.float32)
    faces = torch.tensor(
        mesh.faces.astype(np.int64), dtype=torch.long
    )
    triangles = vertices[faces].unsqueeze(0).cuda()

    m = BVH(max_collisions=8)
    out = m(triangles)
    out = out.detach().cpu().numpy().squeeze()

    collisions = out[out[:, 0] >= 0, :]

    return collisions.shape[0]

if __name__ == '__main__':
    argparser = ArgumentParser(description="Count intersections")
    argparser.add_argument('FILE',
                           type=str,
                           help="Filename of the mesh.")

    args = argparser.parse_args()
    file_name = args.FILE

    mesh = trimesh.load(file_name)

    n_intersections = count_self_intersections(mesh)

    print(f"Counted {n_intersections} intersections in the mesh.")


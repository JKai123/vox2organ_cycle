
""" Check if meshes fulfill the normal convention of pytorch3d.

The convention for a face consisting of vertices (v0, v1, v2) is 
n = (v1 - v0) x (v2 - v0)
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch
import numpy as np

from utils.eval_metrics import Jaccard
from utils.mesh import Mesh
from utils.file_handle import read_obj
from utils.utils import create_mesh_from_voxels, voxelize_mesh

def check_normal_convention(vertices, faces):
    assert len(vertices.shape) == 2 and len(faces.shape) == 2
    assert vertices.shape[1] == 3 and faces.shape[1] == 3

    # Assume that the center lies within the mesh
    center = vertices.mean(dim=0)
    vertices_centered = vertices - center
    vertices_faces = vertices_centered[faces]

    normals = torch.cross(vertices_faces[:,1] - vertices_faces[:,0],
                          vertices_faces[:,2] - vertices_faces[:,0])
    # Reference direction = direction to face center
    ref_dirs = 1/3 * (vertices_faces[:,0]
                      + vertices_faces[:,1]
                      + vertices_faces[:,2])

    sgn = torch.sign(torch.sum(normals * ref_dirs, dim=1)).int()
    return torch.all(sgn == 1)


def check_file(path):
    vertices, faces, _ = read_obj(path)
    vertices = torch.from_numpy(vertices)
    faces = torch.from_numpy(faces)
    res = check_normal_convention(vertices.view(-1,3), faces.view(-1,3))
    if res == True:
        print("Normal convention fulfilled in file.")
    else:
        print("[Error] Normal convention not fulfilled in file.")

def check_mc(volume):
    mesh = create_mesh_from_voxels(volume)
    vertices = mesh.vertices.view(-1,3)
    faces = mesh.faces.view(-1,3)
    res = check_normal_convention(mesh.vertices, mesh.faces)
    if res == True:
        print("Normal convention fulfilled in mc mesh.")
    else:
        print("[Error] Normal convention not fulfilled in mc mesh.")

    voxelized = voxelize_mesh(vertices, faces, volume.shape, 1)
    j_vox = Jaccard(volume, voxelized, 2)

    if np.isclose(j_vox, 1.0):
        print("Coordinates correct.")
    else:
        print(f"[Error] Jaccard of volume and voxelized mesh is {j_vox}, check"
              " coordinate spaces!")

if __name__ == '__main__':
    check_file("../supplementary_material/spheres/icosahedron_40962.obj")

    volume = torch.zeros(12,10,8)
    volume[2:4,2:5,2:6] = 1
    check_mc(volume)

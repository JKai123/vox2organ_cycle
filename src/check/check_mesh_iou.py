""" Check mesh iou computation """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import numpy as np
import torch

from utils.mesh import Mesh
from utils.evaluate import Jaccard_from_Coords

def get_std_cube(origin: np.ndarray):
    """ Get a cube mesh with side lengts 2 and lowest coordinates at 'origin'.
    """
    vertices = [origin + (0, 0, 0), # 0
                origin + (2, 0, 0), # 1
                origin + (2, 2, 0), # 2
                origin + (0, 2, 0), # 3
                origin + (0, 0, 2), # 4
                origin + (2, 0, 2), # 5
                origin + (2, 2, 2), # 6
                origin + (0, 2, 2)] # 7
    faces = [(0, 1, 3),
             (1, 2, 3),
             (0, 1, 4),
             (1, 4, 5),
             (1, 2, 5),
             (2, 5, 6),
             (4, 5, 7),
             (5, 6, 7),
             (0, 3, 4),
             (3, 4, 7),
             (2, 3, 7),
             (2, 6, 7)]

    return Mesh(vertices, faces)

def run_mesh_iou_check():
    """ Check the implementation of mesh IoU/Jaccard.
    """
    mesh1 = get_std_cube(origin=np.array([0, 0, 0]))
    mesh2 = get_std_cube(origin=np.array([1, 1, 0]))

    mesh1.store("../misc/mesh1.ply")
    mesh2.store("../misc/mesh2.ply")

    vox_occupied1 = mesh1.get_occupied_voxels()
    vox_occupied2 = mesh2.get_occupied_voxels()

    IoU = Jaccard_from_Coords([None, vox_occupied1], [None, vox_occupied2], 2)

    target_result = 12./42
    print(f"Smoothed IoU: {IoU:.4f}, correct is {target_result:.4f}")

if __name__ == '__main__':
    run_mesh_iou_check()

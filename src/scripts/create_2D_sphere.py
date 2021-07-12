
""" Create a 2D sphere template. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from math import atan2

import torch
import numpy as np
import matplotlib.pyplot as plt
from trimesh import Trimesh
from trimesh.scene.scene import Scene
from skimage import draw

def create_2D_sphere(side_length, save_path=None):
    """
    :param side_length: The side length of the square image into which the
    sphere is drawn. This defines the resolution of the sphere (coordinates are
    normalized but larger side_length means more accurate sphere).
    :param save_path: The path where to store the template.
    """

    radius = side_length // 2 - 2
    center_r, center_c = side_length // 2, side_length // 2
    arr = np.zeros((side_length, side_length))
    r, c = draw.circle_perimeter(center_r, center_c, radius=radius,
                                 shape=arr.shape)
    arr[r,c] = 1
    plt.imshow(arr)
    plt.savefig("../misc/circle.png")
    plt.close()

    # Sort unique vertices according to angle with 'r-axis'. In an
    # r-c-coordinate system, this is equal to clockwise-sorting.
    vertices_ = list(np.stack([r,c], axis=1))
    vertices_.sort(key=lambda c:atan2(c[0]-center_r, c[1]-center_c))
    vertices_ = torch.tensor(vertices_)
    vertices = vertices_.unique_consecutive(dim=0)
    assert len(vertices) == len(vertices_.unique(dim=0)) # ensure correctness
    # Edges = faces in 2D
    faces = [[len(vertices) - 1, 0]] # Connect end to beginning
    faces = torch.tensor(
        faces + [[i,i+1] for i in range(len(vertices) - 1)]
    )
    if save_path:
        structure = Trimesh(vertices, faces, process=False)
        template = Scene()
        template.add_geometry(structure, geom_name="sphere")
        template.export(save_path)

    return vertices, faces

if __name__ == '__main__':
    create_2D_sphere(
        128, "../supplementary_material/circles/icocircle_128.obj"
    )

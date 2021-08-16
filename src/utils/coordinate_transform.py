
""" Transformation of image/mesh coordinates. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from typing import Union, Tuple

import torch
import numpy as np

def normalize_vertices(vertices: Union[torch.Tensor, np.array],
                       shape: Tuple[int, int, int], faces=None):
    """ Normalize vertex coordinates from [0, patch size-1] into [-1, 1]
    treating each dimension separately and flip x- and z-axis.

    Flipping x- and z-axis also changes the direction of normals. If the
    ordering of faces follows a normal convention, they can be also provided
    and will modified appropriately.
    """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    if isinstance(vertices, torch.Tensor):
        shape = torch.tensor(
            shape
        ).float().to(vertices.device).flip(dims=[0]).unsqueeze(0)
        vertices = vertices.flip(dims=[1])
    elif isinstance(vertices, np.ndarray):
        shape = np.flip(np.array(shape, dtype=float), axis=0).unsqueeze(0)
        vertices = np.flip(vertices, axis=1)
    else:
        raise TypeError()

    new_verts = 2*(vertices/(shape-1) - 0.5)

    if faces is None:
        return new_verts

    assert len(faces.shape) == 2, "Faces should be packed."
    assert faces.shape[1] == 3, "Faces should be 3D."
    if isinstance(faces, torch.Tensor):
        new_faces = faces.flip(dims=[1])
    elif isinstance(faces, np.ndarray):
        new_faces = np.flip(faces, axis=1)
    else:
        raise TypeError()

    return new_verts, new_faces

def unnormalize_vertices(vertices: Union[torch.Tensor, np.array],
                         shape: Tuple[int, int, int], faces=None):
    """ Inverse of 'normalize vertices' """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    if isinstance(vertices, torch.Tensor):
        shape = torch.tensor(shape).float().to(vertices.device).unsqueeze(0)
        vertices = vertices.flip(dims=[1])
    elif isinstance(vertices, np.ndarray):
        shape = np.array(shape, dtype=float).unsqueeze(0)
        vertices = np.flip(vertices, axis=1)
    else:
        raise TypeError()

    new_verts = (0.5 * vertices + 0.5) * (shape - 1)

    if faces is None:
        return new_verts

    assert len(faces.shape) == 2, "Faces should be packed."
    assert faces.shape[1] == 3, "Faces should be 3D."
    if isinstance(faces, torch.Tensor):
        new_faces = faces.flip(dims=[1])
    elif isinstance(faces, np.ndarray):
        new_faces = np.flip(faces, axis=1)
    else:
        raise TypeError()

    return new_verts, new_faces

def normalize_vertices_per_max_dim(vertices: Union[torch.Tensor, np.array],
                                   shape: Tuple[int, int, int]):
    """ Normalize vertex coordinates w.r.t. the maximum input dimension.
    """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    return 2*(vertices/(np.max(shape)-1) - 0.5)

def unnormalize_vertices_per_max_dim(vertices: Union[torch.Tensor, np.array],
                                     shape: Tuple[int, int, int]):
    """ Inverse of 'normalize vertices_per_max_dim' """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    return (0.5 * vertices + 0.5) * (np.max(shape) - 1)

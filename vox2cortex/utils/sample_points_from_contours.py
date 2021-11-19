
""" This module implements utuility functions for sampling points from contours
and can be interpreted as a 2D version of
pytorch3d.ops.sample_points_from_meshes. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import sys

from typing import Tuple

import torch
import pytorch3d
from pytorch3d.ops.packed_to_padded import packed_to_padded

from utils.utils import edge_lengths_in_contours

def sample_points_from_contours(
    meshes: pytorch3d.structures.Meshes,
    n_points: int,
    return_normals: bool=False
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Uniformly sample n points from the contour defined by vertices and
    edges. Can be interpreted as a 2D version of
    pytorch3d.ops.sample_points_from_meshes.

    :param vertices: 2D vertex coordinates
    :param edges: Connections between vertices
    :param n_points: The number of points to sample.
    :param return_normals: Optionally return normal vectors

    :returns: Sampled points and normal vectors (or None if return_normals is
    False)

    Attention: The returned normals are not oriented (they do not define an
    inside or outside)!!!
    """

    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    # Faces = edges in 2D
    edges = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)  # Non empty meshes.

    # Initialize samples tensor with fill value 0 for empty meshes.
    samples = torch.zeros((num_meshes, n_points, 2), device=meshes.device)

    # Only compute samples for non empty meshes
    with torch.no_grad():
        lengths = edge_lengths_in_contours(verts, edges)
        max_edges = meshes.num_faces_per_mesh().max().item()
        lengths_padded = packed_to_padded(
            lengths, mesh_to_face[meshes.valid], max_edges
        )  # (M,F)

        sample_edge_idxs = lengths_padded.multinomial(
            n_points, replacement=True
        )  # (M, n_points)
        sample_edge_idxs += mesh_to_face[meshes.valid].view(num_valid_meshes, 1)

    # Get the vertex coordinates of the sampled edges.
    edge_verts = verts[edges]
    v0, v1 = edge_verts[:, 0], edge_verts[:, 1]

    # Randomly generate barycentric coords.
    w0, w1 = _rand_barycentric_coords(
        num_valid_meshes, n_points, verts.dtype, verts.device
    )

    # Use the barycentric coords to get a point on each sampled edge.
    a = v0[sample_edge_idxs]  # (M, n_points, 2)
    b = v1[sample_edge_idxs]
    samples[meshes.valid] = w0[:,:,None] * a + w1[:,:,None] * b

    if return_normals:
        # Initialize normals tensor with fill value 0 for empty meshes.
        # Normals for the sampled points are edge normals computed from
        # the vertices of the edge in which the sampled point lies in
        # counter-clockwise fashion.
        normals = torch.zeros((num_meshes, n_points, 2), device=meshes.device)
        vert_normals = - (v1 - v0)
        vert_normals = (v1 - v0)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_edge_idxs]
        normals[meshes.valid] = vert_normals

    if return_normals:
        return samples, normals

    return samples


def _rand_barycentric_coords(
    size1, size2, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Helper function to generate random barycentric coordinates which are uniformly
    distributed between two vertices.

    :params size1, size2: The number of coordinates generated will be size1*size2.
                      Output tensors will each be of shape (size1, size2).
    :param dtype: Datatype to generate.
    :param device: A torch.device object on which the outputs will be allocated.

    :returns:
        w0, w1: Tensors of shape (size1, size2) giving random barycentric
            coordinates
    """
    w0 = torch.rand(size1, size2, dtype=dtype, device=device)
    w1 = 1.0 - w0
    return w0, w1

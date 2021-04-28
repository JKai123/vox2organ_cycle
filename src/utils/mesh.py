
""" Mesh representation """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import numpy as np
import torch
from trimesh import Trimesh
from pytorch3d.structures import Meshes

class Mesh():
    """ Custom meshes """
    def __init__(self, vertices, faces, normals=None, values=None):
        self._vertices = vertices
        self._faces = faces
        self._normals = normals
        self._values = values

    @property
    def vertices(self):
        return self._vertices

    @property
    def faces(self):
        return self._faces

    @property
    def normals(self):
        return self._normals

    @property
    def values(self):
        return self._values

    def to_trimesh(self, process=False):
        if isinstance(self.vertices, torch.Tensor):
            vertices = self.vertices.squeeze().cpu()
        else:
            vertices = self.vertices
        if isinstance(self.faces, torch.Tensor):
            faces = self.faces.squeeze().cpu()
        else:
            faces = self.faces
        return Trimesh(vertices=vertices,
                       faces=faces,
                       normals=self.normals,
                       values=self.values,
                       process=process)
    def store(self, path: str):
        t_mesh = self.to_trimesh()
        t_mesh.export(path)

    def get_occupied_voxels(self, shape):
        "Get the occupied voxels of the mesh lying within 'shape'"""
        assert len(shape) == 3, "Shape should represent 3 dimensions."

        voxelized = self.to_trimesh().voxelized(1.0).fill()
        # Coords = trimesh coords + translation
        vox_occupied = np.around(voxelized.sparse_indices +\
            voxelized.translation).astype(int)

        # 0 <= coords < shape
        vox_occupied = [vo for vo in vox_occupied\
                        if (vo >= 0).all() and (vo < shape).all()]
        vox_occupied = np.asarray(vox_occupied)
        if vox_occupied.ndim < 2:
            # Bug?
            breakpoint()

        return vox_occupied

class MeshCollection():
    """ Collection of meshes with shape (S,C)
    """
    def __init__(self, n_steps, n_classes, vertices, faces):
        self.n_steps = n_steps # S
        self.n_classes = n_classes # C

        self.vertices = vertices
        self.faces = faces

    def __getitem__(self, index):
        if len(index) != 2:
            raise ValueError("Index should define two-dimensional"\
                             " coordinates but is of length {len(index)}.")
        s, c = index

        return Mesh(self.vertices[s][c], self.faces[s,c])

def verts_faces_to_Meshes(verts, faces, ndim):
    """ Convert lists of vertices and faces to lists of
    pytorch3d.structures.Meshes

    :param verts: Lists of vertices.
    :param faces: Lists of faces.
    :param ndim: The list dimensions.
    :returns: A list of Meshes of dimension n_dim.
    """
    meshes = []
    for v, f in zip(verts, faces):
        if ndim > 1:
            meshes.append(verts_faces_to_Meshes(v, f, ndim-1))
        else:
            meshes.append(Meshes(verts=list(v), faces=list(f)))

    return meshes

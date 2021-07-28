
""" Mesh representation """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from typing import Union

import numpy as np
import torch
from trimesh import Trimesh
from pytorch3d.structures import Meshes

class Mesh():
    """ Custom meshes
    The idea of this class is to hold vertices and faces of ONE mesh (which may
    consist of multiple structures) together very flexibly.
    For example, vertices may be represented by a 3D tensor (one
    dimenesion per mesh structure) or a 2D tensor of shape (V,3).

    :param vertices: torch.tensor or numpy.ndarray of vertices
    :param faces: torch.tensor or numpy.ndarray of faces
    :param normals: Vertex normals
    :param features: Vertex features
    """
    def __init__(self, vertices, faces, normals=None, features=None):
        self._vertices = vertices
        self._faces = faces
        self._normals = normals
        self._features = features
        self._ndims = vertices.shape[-1]

    @property
    def vertices(self):
        return self._vertices

    @vertices.setter
    def vertices(self, new_vertices):
        assert new_vertices.shape == self.vertices.shape
        self._vertices = new_vertices

    @property
    def ndims(self):
        return self._ndims

    @property
    def ndims(self):
        return self._ndims

    @property
    def faces(self):
        return self._faces

    @property
    def normals(self):
        return self._normals

    @property
    def features(self):
        return self._features

    def to_trimesh(self, process=False):
        assert type(self.vertices) == type(self.faces)
        if isinstance(self.vertices, torch.Tensor):
            if self.vertices.ndim == 3:
                # padded --> packed representation
                # Use process=True to remove padded vertices (padded vertices
                # can lead to problems with some trimesh functions).
                # On the other hand, this may also remove unoccupied vertices
                # leading to invalid faces. In the latter case, process=False
                # should be used.
                if process:
                    valid_ids = [np.unique(f) for f in self.faces.cpu()]
                    valid_ids = [i[i != -1] for i in valid_ids]

                    vertices_ = [v[valid_ids[i]] for i, v in
                                   enumerate(self.vertices.cpu())]
                    faces_ = [f for f in self.faces.cpu()]
                else:
                    vertices_ = self.vertices
                    faces_ = self.faces
                m = Meshes(vertices_, faces_)
                faces = m.faces_packed().cpu().numpy()
                vertices = m.verts_packed().cpu().numpy()
            else:
                vertices = self.vertices.cpu().numpy()
                faces = self.faces.cpu().numpy()
        else:
            # numpy
            vertices = self.vertices
            faces = self.faces

        return Trimesh(vertices=vertices,
                       faces=faces,
                       process=process)

    def to_pytorch3d_Meshes(self):
        assert self.vertices.ndim == self.faces.ndim
        # Note: With current pytorch3d version, vertex normals cannot be
        # handed to Meshes object
        if self.vertices.ndim == 3:
            # Avoid pytorch3d dimensionality check
            return Meshes([v for v in self.vertices],
                          [f for f in self.faces])
        if self.vertices.ndim == 2:
            return Meshes([self.vertices],
                          [self.faces])
        raise ValueError("Invalid dimension of vertices and/or faces.")

    def store(self, path: str):
        t_mesh = self.to_trimesh()
        t_mesh.export(path)

    def get_occupied_voxels(self, shape):
        """Get the occupied voxels of the mesh lying within 'shape'.

        Attention: 'shape' should be defined in the same coordinte system as
        the mesh.
        """
        assert len(shape) == 3, "Shape should represent 3 dimensions."

        voxelized = self.to_trimesh().voxelized(1.0).fill()
        # Coords = trimesh coords + translation
        vox_occupied = np.around(voxelized.sparse_indices +\
            voxelized.translation).astype(int)

        # 0 <= coords < shape
        vox_occupied = np.asarray(vox_occupied)
        mask = np.ones((vox_occupied.shape[0]), dtype=bool)
        for i, s in enumerate(shape):
            in_box = np.logical_and(vox_occupied[:,i] >= 0,
                                    vox_occupied[:,i] < s)
            mask = np.logical_and(mask, in_box)
        vox_occupied = vox_occupied[mask]

        if vox_occupied.size < 1:
            # No occupied voxels in the given shape
            vox_occupied = None

        return vox_occupied

    def transform_vertices(self, transformation_matrix: Union[np.ndarray,
                                                              torch.Tensor]):
        """ Transform the vertices of the mesh using a given transformation
        matrix. """
        if (tuple(transformation_matrix.shape) !=
            (self._ndims + 1, self._ndims + 1)):
            raise ValueError("Wrong shape of transformation matrix.")
        if isinstance(self.vertices, np.ndarray):
            vertices = torch.from_numpy(self.vertices)
        else:
            vertices = self.vertices
        if isinstance(transformation_matrix, np.ndarray):
            transformation_matrix =\
                torch.from_numpy(transformation_matrix).float()
        vertices = vertices.view(-1, self._ndims)
        coords = torch.cat(
            (vertices.T, torch.ones(1, vertices.shape[0])), dim=0
        )
        # Transform
        new_coords = (transformation_matrix @ coords)
        new_coords = new_coords.T[:,:-1].view(self.vertices.shape)

        # Correct data type
        if isinstance(self.vertices, np.ndarray):
            new_coords = new_coords.numpy()
        self.vertices = new_coords

class MeshesOfMeshes():
    """ Extending pytorch3d.structures.Meshes so that each mesh in a batch of
    meshes can consist of several distinguishable meshes (often individual
    structures in a scene). Basically, a new dimension 'M' is introduced
    to tensors of vertices and faces.

    Shapes of self.faces (analoguously for vertices and normals):
        - padded (N,M,F,3)
        - packed (N*M*F,3)
    where N is the batch size, M is the number of meshes per sample, and F
    is the number of faces per connected mesh. In general, M and F can be
    different for every mesh and their maximum is used in the padded
    representation.

    """
    def __init__(self, verts, faces, features=None):
        if verts.ndim != 4:
            raise ValueError("Vertices are required to be a 4D tensor.")
        if faces.ndim != 4:
            raise ValueError("Faces are required to be a 4D tensor.")
        if features is not None:
            self.contains_features = True
            if features.ndim != 4:
                raise ValueError("Features are required to be a 4D tensor.")
        else:
            self.contains_features = False

        self._verts_padded = verts
        self._faces_padded = faces
        self._edges_packed = None
        self.ndims = verts.shape[-1]

        if features is not None:
            self.update_features(features)
        else:
            self._features_padded = None

    def update_features(self, features):
        """ Add features to the mesh in padded representation """
        if features.shape[:-1] != self._verts_padded.shape[:-1]:
            raise ValueError("Invalid feature shape.")
        self._features_padded = features

    def verts_padded(self):
        return self._verts_padded

    def features_padded(self):
        return self._features_padded

    def faces_padded(self):
        return self._faces_padded

    def edges_packed(self):
        """ Based on pytorch3d.structures.Meshes.edges_packed()"""
        if self._edges_packed is None:
            if self.ndims == 3:
                # Calculate edges from faces
                faces = self.faces_packed()
                v0, v1, v2 = faces.chunk(3, dim=1)
                e01 = torch.cat([v0, v1], dim=1)  # (N*M*F), 2)
                e12 = torch.cat([v1, v2], dim=1)  # (N*M*F), 2)
                e20 = torch.cat([v2, v0], dim=1)  # (N*M*F), 2)
                # All edges including duplicates.
                self._edges_packed = torch.cat([e12, e20, e01], dim=0) # (N*M*F)*3, 2)
            else:
                # 2D equality of faces and edges
                self._edges_packed = self.faces_packed()

        return self._edges_packed

    def faces_packed(self):
        """ Packed representation of faces """
        N, M, V, _ = self._verts_padded.shape
        _, _, F, _ = self._faces_padded.shape
        # New face index = local index + Ni * Mi * V
        add_index = torch.cat(
            [torch.ones(F) * i * V for i in range(N*M)]
        ).long().to(self._faces_padded.device)
        return self._faces_padded.view(-1, self.ndims) + add_index.view(-1, 1)

    def verts_packed(self):
        """ Packed representation of vertices """
        return self._verts_padded.view(-1, self.ndims)

    def move_verts(self, offset):
        """ Move the vertex coordinates by offset """
        if offset.shape != self._verts_padded.shape:
            raise ValueError("Invalid offset.")
        self._verts_padded = self._verts_padded + offset

    def features_verts_packed(self):
        """ (features, verts) in packed representation """
        return torch.cat((self.features_packed(), self.verts_packed()), dim=1)

    def features_packed(self):
        """ Packed representation of features """
        if self.contains_features:
            _, _, _, C = self._features_padded.shape
            return self._features_padded.view(-1, C)
        return None

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

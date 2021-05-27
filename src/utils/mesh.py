
""" Mesh representation """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import numpy as np
import torch
import torch.nn.functional as F
from trimesh import Trimesh
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere

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

    def to_pytorch3d_Meshes(self):
        return Meshes([self.vertices],
                      [self.faces])

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

class MeshesOfMeshes():
    """ Extending pytorch3d.structures.Meshes so that each mesh in a batch of
    meshes can consist of several distinguishable meshes. Basically, a new
    dimension 'M' is introduced to tensors of vertices and faces.

    Shapes of self.faces (analoguously for vertices and normals):
        - padded (N,M,F,3)
        - packed (N*M*F,3)
    where N is the batch size, M is the number of meshes per sample, and F
    is the number of faces per connected mesh. In general, M and F can be
    different for every mesh and their maximum is used in the padded
    representation.

    Note: This implementation is quite inefficient and should in general be
    avoided since all vertices, faces etc. are stored twice. An efficient
    implementation of such a class should entirely replace
    pytorch3d.structures.Meshes and not use it internally.
    """
    def __init__(self, verts, faces):
        self._Meshes_list = []
        for v, f in zip(verts, faces):
            self._Meshes_list.append(Meshes(v,f))

        self._verts_packed = None
        self._faces_packed = None
        self._verts_normals_packed = None
        self._edges_packed = None

        self.N = self.get_N()
        self.M, self.V, self.F = self.get_M_V_F()

    def verts_list(self):
        return [m.verts_list() for m in self._Meshes_list]

    def get_N(self):
        return len(self._Meshes_list)

    def get_M_V_F(self):
        max_M = 0
        max_V = 0
        max_F = 0
        for m in self._Meshes_list:
            if m.verts_padded().shape[0] > max_M:
                max_M = m.verts_padded().shape[0]
            if m.verts_padded().shape[1] > max_V:
                max_V = m.verts_padded().shape[1]
            if m.faces_padded().shape[1] > max_F:
                max_F = m.faces_padded().shape[1]

        return max_M, max_V, max_F

    def verts_packed(self):
        if self._verts_packed is None:
            self._verts_packed = torch.cat(
                [m.verts_packed() for m in self._Meshes_list], dim=0
            )

        return self._verts_packed

    def verts_normals_packed(self):
        if self._verts_normals_packed is None:
            self._verts_normals_packed = torch.cat(
                [m.verts_normals_packed() for m in self._Meshes_list], dim=0
            )

        return self._verts_normals_packed

    def faces_packed(self):
        V_prev = 0
        if self._faces_packed is None:
            faces_packed = []
            for m in self._Meshes_list:
                faces_packed.append(m.faces_packed() + V_prev)
                V_pref += m.verts_packed().shape[0]

            self._faces_packed = torch.cat(faces_packed, dim=0)

        return self._faces_packed

    def edges_packed(self):
        V_prev = 0
        if self._edges_packed is None:
            edges_packed = []
            for m in self._Meshes_list:
                edges_packed.append(m.edges_packed() + V_prev)
                V_pref += m.verts_packed().shape[0]

            self._edges_packed = torch.cat(edges_packed, dim=0)

        return self._edges_packed

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

def generate_sphere_template(centers, radii, level=6):
    """ Generate a template with spheres centered at centers and corresponding
    radii
    - level 6: 40962 vertices
    - level 7: 163842 vertices
    """
    if len(centers) != len(radii):
        raise ValueError("Number of centroids and radii must be equal.")
    verts, faces = [], []
    for c, r in zip(centers, radii):
        # Get unit sphere
        sphere = ico_sphere(level)
        # Scale adequately
        v = sphere.verts_packed() * r + c
        verts.append(v)
        faces.append(sphere.faces_packed())

    return Meshes(verts=verts, faces=faces)

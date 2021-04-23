
""" Mesh representation """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from trimesh import Trimesh

class Mesh():
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
        return Trimesh(vertices=self.vertices,
                       faces=self.faces,
                       normals=self.normals,
                       values=self.values,
                       process=process)
    def store(self, path: str):
        t_mesh = self.to_trimesh()
        t_mesh.export(path)

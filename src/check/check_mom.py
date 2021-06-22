
""" Check implementation of MeshesOfMeshes """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch
from utils.mesh import MeshesOfMeshes

# Every mesh of a structure looks like
#    3
#   /|\
# 0/ | \2
#  \ | /
#   \|/
#    1

verts = torch.rand(2, 3, 4, 3)
faces = torch.tensor([[0,1,3], [1,2,3]])
faces = faces.repeat(2,3,1,1)
mom = MeshesOfMeshes(verts, faces)

print(mom.faces_packed())
print(mom.edges_packed())

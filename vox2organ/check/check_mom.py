
""" Check implementation of MeshesOfMeshes """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch
import numpy as np
import time
from pprint import pprint
from utils.mesh import MeshesOfMeshes, Meshes
from utils.utils_padded_packed import zero_pad_max_length
from utils.utils_padded_packed import pack, unpack, as_list


def predMoM_to_meshes(pred):
    """ Get the vertices and faces and features of shape (S,C)
    C as the number of classes
    S as the number of predictions
    """    
    C = pred[0][0].verts_padded().shape[1]

    meshes = pred[0][:] # Ignore template mesh at pos. 0
    ret_meshes = []
    for s, m in enumerate(meshes):
        vert_list = as_list(m.verts_padded(), m.verts_mask())
        face_list = as_list(m.faces_padded(), m.faces_mask())
        for c in range(C):
            ret_meshes.append(
                Meshes(
                    verts=vert_list[c],
                    faces=face_list[c],
                )
            )
    return ret_meshes




def pred_to_pred_meshes(pred):
    """ Create valid prediction meshes of shape (S,C) """
    pred_meshes = predMoM_to_Meshes(pred)
    return pred_meshes

    
# Every mesh of a structure looks like
#    3
#   /|\
# 0/ | \2
#  \ | /
#   \|/
#    1

# Control Parameters
reps = 100
batchsize = 100
M = 400

# Construct Template
num_verts = [6, 4, 4]
verts_all = []
for i in range(0, len(num_verts)):
    verts = torch.rand(num_verts[i], 3)
    verts_all.append(verts)
faces_big = torch.tensor([[0,1,3], [1,2,3], [2,3,4], [1,2,4], [0,3,5], [0,1,5]])
faces = torch.tensor([[0,1,3], [1,2,3]])
faces_all = [faces_big, faces, faces]


# Pad inputdata
verts_padded, verts_mask = zero_pad_max_length(verts_all, dimension=0)
faces_padded, faces_mask = zero_pad_max_length(faces_all, dimension=0)
verts_padded = torch.stack(verts_padded).float().unsqueeze(0)
faces_padded = torch.stack(faces_padded).long().unsqueeze(0)

# Construct Mesh
mom = MeshesOfMeshes(verts_padded, faces_padded, verts_mask=verts_mask, faces_mask=faces_mask)


# # Test for vertices
# verts_padded = mom.verts_padded()
# verts_padded = verts_padded.repeat(batchsize, 1, 1, 1)
# packed = pack(verts_padded, verts_mask)
# padded = unpack(packed, verts_mask, batchsize)

# # Test for faces
# faces_padded = mom.faces_padded()
# faces_padded = faces_padded.repeat(batchsize, 1, 1, 1)
# packed = pack(faces_padded, faces_mask)
# padded = unpack(packed, faces_mask, batchsize)
# # print(torch.eq(faces_padded, padded))


test = pred_to_pred_meshes(([mom, mom],1))

print("end")

""" Utility functions """

__author__ = "Johannes Kaiser"
__email__ = "johannes.kaiser@tum.de"


import numpy as np
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes



def zero_pad_max_length(data, dimension=0):
    """Pads along the zeroth dimension of a list of data with zeros such that the contents contained in 
    the list are of the same lenght
    :param data: list of torch.tensor elements of different length
    """
    lengths = torch.tensor([data_element.size(dim=dimension) for  data_element in data])
    max_lenght = max(lengths)
    pad = np.zeros((data[dimension].dim() - dimension - 1) * 2 + 1, dtype=int)
    padded = [F.pad(data_element, tuple(np.append(pad, max_lenght - lengths[i])), "constant", 0) for i, data_element in enumerate(data)]
    mask = lengths
    return padded, mask


def sequence_mask(lengths, maxlen=None, dtype=torch.bool):
        if maxlen is None:
            maxlen = lengths.max()
        row_vector = torch.arange(0, maxlen, 1)
        matrix = torch.unsqueeze(lengths, dim=-1)
        mask = row_vector < matrix
        count = torch.count_nonzero(mask[1])
        mask.type(dtype)
        return mask

def pack(padded, lengths):
    packed = []
    for i, batch_of_indiv_mesh in enumerate(torch.unbind(padded, dim=1)): # Results in M meshes of dimension N,F,3
        cut = batch_of_indiv_mesh[:, :lengths[i], :]  # Resutls in a cut version of the mesh with the unpadded number of faces
        cut = torch.flatten(cut, start_dim=0, end_dim=1) # Flattens the first two dimensions to result in N*F,3
        packed.append(cut) # Concatenates along the final dimension to get N*M*F, 3 in the Ordering M[N[FFF]] M[N[FFF]]
    packed = torch.cat(packed)
    return packed


def unpack(packed, lengths, batchsize):
    start = 0
    cut_list = []
    for length in lengths:
        cut = packed[start:start + batchsize * length, :] # Results in a slize of N[FFF]
        start += batchsize * length
        cut = cut.view(batchsize, length, -1) # N,F,3
        cut_list.append(cut)
    padded, _ = zero_pad_max_length(cut_list, dimension=1)
    padded = torch.stack(padded).float().squeeze(0).permute(1,0,2,3)
    return padded

def as_list(padded, lengths, dim=1, squeeze=False):
    mesh_list = [] # List with M entries of dimension NxFx3

    for i, batch_of_indiv_mesh in enumerate(torch.unbind(padded, dim=dim)): # Results in M meshes of dimension N,F,3
        cut = batch_of_indiv_mesh[:, :lengths[i], :]  # Resutls in a cut version of the mesh with the unpadded number of faces
        if squeeze:
            cut = torch.squeeze(cut)
        mesh_list.append(cut) # Concatenates along the final dimension to get N*M*F, 3 in the Ordering M[N[FFF]] M[N[FFF]]
    return mesh_list

def MoM_to_list(m, batchsize=1):
    
    final_list = []
    vert_list = as_list(m.verts_padded(), m.verts_mask())
    for organ_verts in vert_list:
        organ_list = [organ_verts for i in range(batchsize)]
        organ_tuple = (organ_list[0].cuda(), [], [], []) # points, norms, curfs, parcs
        final_list.append(organ_tuple)
    return final_list


def MoM_to_meshes(m, batchsize=1, packed=False):
    if packed:
        vert_list = as_list(m.verts_padded(), m.verts_mask())
        face_list = as_list(m.faces_padded(), m.faces_mask())
    else:
        vert_list = torch.unbind(m.verts_padded(), dim = 1)
        face_list = torch.unbind(m.faces_padded(), dim = 1)
    meshes_list = []
    for v, f in zip(vert_list, face_list):
        vert_organ_list = [torch.squeeze(v, dim=0) for i in range(batchsize)]
        face_organ_list = [torch.squeeze(f, dim=0) for i in range(batchsize)]
        meshes_list.append(Meshes(vert_organ_list, face_organ_list).cuda())
    return meshes_list



""" Utils for the statstical shape model loss """

__author__ = "Johannes Kaiser"
__email__ = "johannes.kaiser@tum.de"

import torch
import os
import numpy as np
from utils.coordinate_transform import transform_coords_affine_wo_shift



"""if any(isinstance(lf, PCA_loss) for lf in mesh_loss_func):
            try:
                ssm_path = ssm_path
            except:
                print("Path for SSM must be set with PCA_loss")"""


def load_ssm(ssm_path, num_organs, trans_affine):
    test = len(os.listdir(ssm_path))
    assert len(os.listdir(ssm_path)) == num_organs,\
            "The Number of SSM must be the same as predicted organs"
    ssm_mean_list = []
    ssm_eigvec_list = []
    ssm_eigval_list = []
    folder_names = []
    ssm_target = []
    for folder in sorted(os.listdir(ssm_path)):
        ssm_mean_list.append(os.path.join(ssm_path, folder, "mean.npy"))
        ssm_eigvec_list.append(os.path.join(ssm_path, folder, "eigenvectors.npy"))
        ssm_eigval_list.append(os.path.join(ssm_path, folder, "eigenvalues.npy"))
        folder_names.append(folder)

    for mean, eigenvector, eigenvalue, folder_name in zip(ssm_mean_list, ssm_eigvec_list, ssm_eigval_list, folder_names):
        eigvec = []
        organ_mean = torch.tensor(np.load(mean)).cuda().reshape(-1,3).float()
        organ_mean = transform_coords_affine_wo_shift(organ_mean, trans_affine).flatten()
        organ_eigvec = torch.tensor(np.load(eigenvector)).cuda().float()
        for single_vec in organ_eigvec:
            single_vec = single_vec.reshape(-1,3)
            single_vec = transform_coords_affine_wo_shift(single_vec, trans_affine).flatten()
            eigvec.append(single_vec)
        organ_eigvec = torch.stack(eigvec)

        organ_eigval = eigenvalue


        #organ_eigval = transform_coords_affine(organ_eigval, trans_affine) Are not used
        ssm_target.append((
            organ_mean,
            organ_eigvec,
            organ_eigval,
            folder_name
        ))
    return ssm_target


def gpa(mesh, path):
    center_origin(mesh)
    return mesh

def center_origin(shape_trimesh):
    center = torch.mean(shape_trimesh.verts_packed(), dim=0)
    shape_trimesh.offset_verts_(-center)

def get_subspace_dist(mesh, mean, eigenvectors, eigenvalues):
    # savepath = "../transformed_point_clouds/"
    # if not os.path.exists(savepath):
    #     os.mkdir(savepath)
    # np.save(savepath + "mesh_coor.npy", mesh.verts_packed().cpu().detach().numpy())
    # np.save(savepath + "mean_coor.npy", mean.cpu().detach().numpy())
    flattened_verts = mesh.verts_packed().flatten()
    data = flattened_verts - mean
    data = torch.matmul(data, eigenvectors.T)
    data = torch.matmul(data, eigenvectors)
    data += mean
    proj_verts = data.reshape(-1,3)
    unproj_verst = flattened_verts.reshape(-1,3)
    dist = torch.mean(torch.linalg.norm(proj_verts-unproj_verst, dim=1))
    return dist

""" Utility functions for the work with FreeSurfer data. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

import torch
import nibabel as nib
from pytorch3d.structures import Pointclouds, Meshes
from pytorch3d.ops import knn_gather, knn_points

def pred_to_fs_to_fsaverage(
    pred_points,
    img_id,
    values,
    fs_path,
    surf_name,
    template_path="/mnt/nas/Data_Neuro/Parcellation_atlas/fsaverage/",
    device="cuda:0"
):
    """ Map freesurfer vertex values to fsaverage via the registered
    *h.sphere.reg surfaces
    """
    fs_vertices, _ = nib.freesurfer.io.read_geometry(
        os.path.join(
            fs_path,
            img_id,
            'surf',
            surf_name.replace("_", ".") # *h.white/pial
        )
    )
    fs_pntcloud = torch.tensor(fs_vertices).float().to(device)[None]
    pred_pntcloud = torch.tensor(pred_points).float().to(device)[None]
    x_nn = knn_points(fs_pntcloud, pred_pntcloud, K=1)
    nn_values = knn_gather(
        torch.tensor(values, device=device)[None].unsqueeze(-1), x_nn.idx
    ).long().squeeze()

    return fs_to_fsaverage(
        img_id,
        nn_values.squeeze().cpu().numpy(),
        fs_path,
        surf_name,
        template_path=template_path,
        device=device
    )

def fs_to_fsaverage(img_id, values, fs_path, surf_name,
                    template_path="/mnt/nas/Data_Neuro/Parcellation_atlas/fsaverage/",
                    device="cuda:0"):
    """ Map freesurfer vertex values to fsaverage via the registered
    *h.sphere.reg surfaces
    """
    reg_sphere_v, reg_sphere_f = nib.freesurfer.io.read_geometry(
        os.path.join(
            fs_path,
            img_id,
            'surf',
            surf_name.split("_")[0] # hemi
            + ".sphere.reg"
        )
    )
    fs_reg_pntcloud = torch.from_numpy(
        reg_sphere_v
    )[None].float().to(device)
    fsaverage_sphere_v, fsaverage_sphere_f = nib.freesurfer.io.read_geometry(
        os.path.join(
            template_path,
            'surf',
            surf_name.split("_")[0] # hemi 
            + ".sphere.reg.avg"
        )
    )
    fs_avg_pntcloud = torch.from_numpy(
        fsaverage_sphere_v
    )[None].float().to(device)
    x_nn = knn_points(fs_avg_pntcloud, fs_reg_pntcloud, K=1)
    mapped_values = knn_gather(
        torch.tensor(values, device=device)[None].unsqueeze(-1), x_nn.idx
    ).squeeze()

    return mapped_values.cpu().numpy()

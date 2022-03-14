
""" Transfer the parcellation to the reduced surfaces. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

import numpy as np
import trimesh
import nibabel as nib
import torch
from pytorch3d.ops import knn_points

mesh_label_names = ("lh_white", "rh_white", "lh_pial", "rh_pial")

for mln in mesh_label_names:
    full_mesh_fn = mln + "_smoothed.ply"
    reduced_mesh_fn = mln + "_smoothed_reduced.ply"
    full_annot_fn = mln + ".aparc.DKTatlas40.annot"
    reduced_annot_fn = mln + "_reduced.aparc.DKTatlas40.annot"

    full_mesh = trimesh.load_mesh(full_mesh_fn, process=False)
    reduced_mesh = trimesh.load_mesh(reduced_mesh_fn, process=False)
    full_labels, colors, names = nib.freesurfer.io.read_annot(full_annot_fn)

    full_pnts = torch.from_numpy(full_mesh.vertices).float()[None]
    reduced_pnts = torch.from_numpy(reduced_mesh.vertices).float()[None]
    _, nn_idx,  _ = knn_points(reduced_pnts, full_pnts)

    reduced_labels = full_labels[nn_idx.squeeze().numpy()]
    nib.freesurfer.io.write_annot(reduced_annot_fn, reduced_labels, colors, names)
    np.save("labels.npy", reduced_labels)


""" Script for creation of voxel ground truth from meshes through voxelization. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

import torch
import numpy as np
import trimesh
import nibabel as nib
import matplotlib.pyplot as plt
from tqdm import tqdm

from data.supported_datasets import valid_ids
from utils.coordinate_transform import transform_mesh_affine
from utils.utils import voxelize_mesh
from utils.mesh import Mesh
from utils.visualization import show_slices

DEBUG = True

# Input data
RAW_DATA_DIR = "/mnt/nas/Data_Neuro/ADNI_CSR/"

# Output data
# OUT_DIR = "/mnt/nas/Data_Neuro/ADNI_CSR/"
OUT_DIR = "/home/fabianb/data/preprocessed/ADNI_CSR/"

# Surfaces
SURFACES = ("lh_white", "rh_white", "lh_pial", "rh_pial")

# File ids
filenames = valid_ids(RAW_DATA_DIR)

# Iterate over file ids
for fn in tqdm(filenames, position=0, leave=True,
               desc="Creating voxel gt from meshes."):

    mesh_folder = os.path.join(RAW_DATA_DIR, fn)

    # Voxel coords
    mri_file = os.path.join(mesh_folder, "mri.nii.gz")
    orig = nib.load(mri_file)
    vox2world_affine = orig.affine
    world2vox_affine = np.linalg.inv(vox2world_affine)

    # Read meshes
    voxelized_meshes = []
    for s in SURFACES:
        try:
            mesh_file = os.path.join(mesh_folder, s + ".ply")
            mesh = trimesh.load(mesh_file)
        except ValueError:
            mesh_file = os.path.join(mesh_folder, s + ".stl")
            mesh = trimesh.load(mesh_file)
        except:
            print("Cannot load mesh {mesh_file}, skipping.")
            continue
        # World --> voxel coordinates
        voxel_verts, voxel_faces = transform_mesh_affine(
            mesh.vertices, mesh.faces, world2vox_affine
        )
        # Voxelize
        vox = np.zeros_like(orig.get_fdata(), dtype=int)
        occ = Mesh(voxel_verts, voxel_faces).get_occupied_voxels(
            orig.get_fdata().shape
        )
        vox[occ[:,0], occ[:,1], occ[:,2]] = 1

        voxelized_meshes.append(vox)

        # Write
        out_img = nib.Nifti1Image(vox, vox2world_affine)
        out_file = os.path.join(OUT_DIR, fn, s + ".nii.gz")
        if not os.path.exists(out_file):
            nib.save(out_img, out_file)
        else:
            print(f"File {out_file} already exists, skipping.")

    if DEBUG:
        mri = orig.get_fdata()
        shape = mri.shape
        mri_slice1 = mri[int(shape[0]/4), :, :]
        mri_slice2 = mri[:, int(shape[1]/2), :]
        mri_slice3 = mri[:, :, int(shape[2]/2)]

        seg = np.sum(np.stack(voxelized_meshes), axis=0)
        seg_slice1 = seg[int(shape[0]/4), :, :]
        seg_slice2 = seg[:, int(shape[1]/2), :]
        seg_slice3 = seg[:, :, int(shape[2]/2)]

        show_slices(
            [mri_slice1, mri_slice2, mri_slice3],
            [seg_slice1, seg_slice2, seg_slice3],
            save_path="../misc/voxelized_and_mri.png",
            label_mode = 'fill'
        )
        show_slices(
            [seg_slice1, seg_slice2, seg_slice3],
            save_path="../misc/voxelized.png",
            label_mode = 'fill'
        )

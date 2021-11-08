
""" Create per-vertex cortical thickness ground truth for freesurfer meshes
by computing the orthogonal distance of each vertex to the respective other
surface."""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

import nibabel as nib
import torch
import trimesh
from pytorch3d.structures import Meshes, Pointclouds

from data.supported_datasets import valid_ids
from utils.cortical_thickness import _point_mesh_face_distance_unidirectional

structures = ("lh_white", "rh_white", "lh_pial", "rh_pial")
partner = {"lh_white": 2, "rh_white": 3, "lh_pial": 0, "rh_pial": 1}
suffix = "_reduced_0.3"
RAW_DATA_DIR = "/mnt/nas/Data_Neuro/ADNI_CSR/"
PREPROCESSED_DIR = "/home/fabianb/data/preprocessed/ADNI_CSR/"

files = valid_ids(RAW_DATA_DIR)

ignored = []

# Iterate over all files
for fn in files:
    prep_dir = os.path.join(PREPROCESSED_DIR, fn)
    if not os.path.isdir(prep_dir):
        os.makedirs(prep_dir)
    for struc in structures:
        red_th_name = os.path.join(
            PREPROCESSED_DIR, fn, struc + suffix + ".thickness"
        )
        # Do not overwrite
        if os.path.isfile(red_th_name):
            print(f"File {red_th_name} exists, skipping.")
            continue

        # Filenames
        try:
            red_mesh_name = os.path.join(
                RAW_DATA_DIR, fn, struc + suffix + ".stl"
            )
            red_partner_mesh_name = os.path.join(
                RAW_DATA_DIR, fn, structures[partner[struc]] + suffix + ".stl"
            )
            # Load meshes
            red_mesh = trimesh.load(red_mesh_name)
            red_mesh_partner = trimesh.load(red_partner_mesh_name)

        except ValueError:
            try:
                red_mesh_name = os.path.join(
                    RAW_DATA_DIR, fn, struc + suffix + ".ply"
                )
                red_partner_mesh_name = os.path.join(
                    RAW_DATA_DIR, fn, structures[partner[struc]] + suffix + ".ply"
                )
                # Load meshes
                red_mesh = trimesh.load(red_mesh_name)
                red_mesh_partner = trimesh.load(red_partner_mesh_name)

            except ValueError:
                ignored += [fn]
                continue

        # Compute thickness by nearest-neighbor distance for full meshes
        red_vertices = torch.from_numpy(red_mesh.vertices).float().cuda()
        red_faces = torch.from_numpy(red_mesh.faces).int().cuda()
        partner_vertices = torch.from_numpy(
            red_mesh_partner.vertices
        ).float().cuda()
        partner_faces = torch.from_numpy(red_mesh_partner.faces).long().cuda()

        pntcloud = Pointclouds([red_vertices])
        partner_mesh = Meshes([partner_vertices], [partner_faces])

        point_to_face = _point_mesh_face_distance_unidirectional(
            pntcloud, partner_mesh
        ).cpu().squeeze().numpy()

        # Write
        nib.freesurfer.io.write_morph_data(red_th_name, point_to_face)

        print("Created label for file ", fn + "/" + struc)

ignored = set(ignored) # Unique ids
ignored_file = os.path.join(PREPROCESSED_DIR, "ignored.txt")
with open(ignored_file, 'w') as f:
    f.write(",".join(ignored))
    f.write("\n")

if len(ignored) > 0:
    print(f"{len(ignored)} files ignored, stored respective ids in ", ignored_file)

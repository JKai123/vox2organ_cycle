
""" Create per-vertex cortical thickness with nearest correspondences from
FreeSurfer meshes."""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes, Pointclouds

from utils.cortical_thickness import _point_mesh_face_distance_unidirectional

# TO DEFINE
FS_BASE_DIR = "/mnt/nas/Data_Neuro/ADNI_CSR/"
OUT_DIR = "/home/fabianb/work/cortex-parcellation-using-meshes/experiments/FS/test_ADNI_CSR_large/thickness/"
IDS_FROM = "/home/fabianb/work/cortex-parcellation-using-meshes/experiments/exp_576_v2/test_template_168058_ADNI_CSR_large/thickness/"
#
structures = ("lh_white", "rh_white", "lh_pial", "rh_pial")
partner = (2, 3, 0, 1) # 0 --> 2, 1 --> 3 etc.
device = "cuda:0"

def get_mesh_fn(file_id, structure):
    return f"{file_id}/{structure}.ply"

if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

get_id = lambda x: x.split("_")[0]
ids = set(map(get_id, os.listdir(IDS_FROM)))

print("Using {} ids.".format(len(ids)))

# Iterate over ids
for i in list(ids):
    for struc_id, struc in enumerate(structures):
        mesh_fn = os.path.join(FS_BASE_DIR, get_mesh_fn(i, struc))
        partner_mesh_fn = os.path.join(
            FS_BASE_DIR, get_mesh_fn(i, structures[partner[struc_id]])
        )
        print("Mesh ", mesh_fn)
        print("Partner ", partner_mesh_fn)
        thickness_fn = os.path.join(
            OUT_DIR, get_mesh_fn(
                i, f"struc{struc_id}"
            ).replace("ply", "thickness").replace("/", "_")
        )
        # Do not overwrite
        if os.path.isfile(thickness_fn):
            print(f"File {thickness_fn} exists, skipping.")
            continue

        # Load meshes
        try:
            mesh = trimesh.load(mesh_fn, process=False)
            mesh_partner = trimesh.load(partner_mesh_fn, process=False)
        except:
            print("Could not process ", mesh_fn)
            continue

        # Compute thickness by nearest-neighbor distance
        vertices = torch.from_numpy(mesh.vertices).float().to(device)
        faces = torch.from_numpy(mesh.faces).int().to(device)

        partner_vertices = torch.from_numpy(
            mesh_partner.vertices
        ).float().to(device)
        partner_faces = torch.from_numpy(mesh_partner.faces).long().to(device)

        pntcloud = Pointclouds([vertices])
        partner_mesh = Meshes([partner_vertices], [partner_faces])

        point_to_face = _point_mesh_face_distance_unidirectional(
            pntcloud, partner_mesh
        ).cpu().squeeze().numpy()

        # Write
        np.save(thickness_fn, point_to_face)

        print("Created label ", thickness_fn)

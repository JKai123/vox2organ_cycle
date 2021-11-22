
""" Create per-vertex cortical thickness."""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes, Pointclouds

from utils.cortical_thickness import _point_mesh_face_distance_unidirectional

# TO DEFINE
MESH_DIR = "/home/fabianb/work/cortex-parcellation-using-meshes/experiments/exp_576_v2/test_template_168058/meshes/"
OUT_DIR = "/home/fabianb/work/cortex-parcellation-using-meshes/experiments/exp_576_v2/test_template_168058/thickness/"
EPOCH = 38
#
structures = ("lh_white", "rh_white", "lh_pial", "rh_pial")
partner = (2, 3, 0, 1) # 0 --> 2, 1 --> 3 etc.
device = "cuda:1"

def get_mesh_fn(file_id, structure_id):
    return f"{file_id}_epoch{str(EPOCH)}_struc{str(structure_id)}_meshpred.ply"

if not os.path.isdir(OUT_DIR):
    os.mkdir(OUT_DIR)

get_id = lambda x: x.split("_")[0]
ids = set(map(get_id, os.listdir(MESH_DIR)))

# Iterate over ids
for i in ids:
    for struc_id, struc in enumerate(structures):
        mesh_fn = os.path.join(MESH_DIR, get_mesh_fn(i, struc_id))
        partner_mesh_fn = os.path.join(MESH_DIR, get_mesh_fn(i, partner[struc_id]))
        thickness_fn = os.path.join(
            OUT_DIR, get_mesh_fn(i, struc_id).replace("ply", "thickness")
        )
        # Do not overwrite
        if os.path.isfile(thickness_fn):
            print(f"File {thickness_fn} exists, skipping.")
            continue

        # Load meshes
        mesh = trimesh.load(mesh_fn)
        mesh_partner = trimesh.load(partner_mesh_fn)

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

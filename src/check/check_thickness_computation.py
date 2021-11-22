
""" Check the validity of cortical thickness labels and their computation. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

import numpy as np
import torch
import trimesh
from trimesh import Trimesh
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import (
    knn_points,
    knn_gather,
)

from data.supported_datasets import valid_ids
from utils.cortical_thickness import _point_mesh_face_distance_unidirectional
from nibabel.freesurfer.io import read_morph_data, read_geometry
from utils.mesh import Mesh

structures = ("lh_white", "rh_white", "lh_pial", "rh_pial")
partner = {"lh_white": 2, "rh_white": 3, "lh_pial": 0, "rh_pial": 1}
suffix = "_reduced_0.3"
RAW_DATA_DIR = "/mnt/nas/Data_Neuro/MALC_CSR/"
RAW_FS_DIR = "/mnt/nas/Data_Neuro/MALC_CSR/FS/FS/"
PREPROCESSED_DIR = "/home/fabianb/data/preprocessed/MALC_CSR/"

files = valid_ids(PREPROCESSED_DIR)

# Compare thickness in stored files to thickness computed by orthogonal
# projection
for fn in files:
    f_dir = os.path.join("../misc/", fn)
    if not os.path.isdir(f_dir):
        os.mkdir(f_dir)
    for struc in structures:
        # Filenames
        red_mesh_name = os.path.join(
            RAW_DATA_DIR, fn, struc + suffix + ".stl"
        )
        red_th_name = os.path.join(
            PREPROCESSED_DIR, fn, struc + suffix + ".thickness"
        )
        full_mesh_name = os.path.join(
            RAW_FS_DIR, fn, "surf", ".".join(struc.split("_"))
        )
        full_mesh_partner_name = os.path.join(
            RAW_FS_DIR, fn, "surf",
            ".".join(structures[partner[struc]].split("_"))
        )
        full_th_name = os.path.join(
            RAW_FS_DIR, fn, "surf", struc.split("_")[0] + ".thickness"
        )

        # Load meshes
        red_mesh = trimesh.load(red_mesh_name)
        v, f = read_geometry(full_mesh_name)
        full_mesh = Trimesh(v, f, process=False)
        v, f = read_geometry(full_mesh_partner_name)
        full_mesh_partner = Trimesh(v, f, process=False)

        # Load thickness files
        red_thickness = read_morph_data(red_th_name)
        full_thickness = read_morph_data(full_th_name)

        # Compute thickness by orthogonal projection for full meshes
        full_vertices = torch.from_numpy(full_mesh.vertices).float().cuda()
        full_faces = torch.from_numpy(full_mesh.faces).int().cuda()
        partner_vertices = torch.from_numpy(
            full_mesh_partner.vertices
        ).float().cuda()
        partner_faces = torch.from_numpy(full_mesh_partner.faces).long().cuda()

        pntcloud = Pointclouds([full_vertices])
        partner_mesh = Meshes([partner_vertices], [partner_faces])

        point_to_face = _point_mesh_face_distance_unidirectional(
            pntcloud, partner_mesh
        ).cpu().squeeze().numpy()

        # Compare to stored FS thickness
        error_to_stored = np.abs(point_to_face - full_thickness)
        print("Mean/max error of computed to stored thickness for file"
              f" {fn + '/' + struc}: {error_to_stored.mean():.4f}/"
              f"{error_to_stored.max():.4f}")

        # Store for comparison
        m_red_stored = Mesh(
            red_mesh.vertices, red_mesh.faces, features=red_thickness
        )
        m_full_comp = Mesh(full_vertices, full_faces, features=point_to_face)
        m_full = Mesh(full_vertices, full_faces, features=full_thickness)

        m_red_stored.store_with_features(os.path.join(
            f_dir, f"m_red_stored_{struc}.ply"
        ))
        m_full_comp.store_with_features(os.path.join(
            f_dir, f"m_full_comp_{struc}.ply"
        ))
        m_full.store_with_features(os.path.join(
            f_dir, f"m_full_gt_{struc}.ply"
        ))

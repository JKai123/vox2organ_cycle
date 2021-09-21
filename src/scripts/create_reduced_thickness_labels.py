
""" Create per-vertex thickness labels for reduced FS meshes. For a bash script
that processes all MALC_CSR ids one aftet another see
scripts/create_reduced_thickness_labels.sh"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from argparse import ArgumentParser

import numpy as np
import torch
import trimesh
import nibabel as nib
from pytorch3d.ops import knn_points, knn_gather

if __name__ == '__main__':

    argparser = ArgumentParser(description="Create thicknes FS ground truth.")
    argparser.add_argument('IN_FILE_MESH',
                           type=str,
                           help="The name of the input file corresponding to"
                           " the thickness file.")
    argparser.add_argument('IN_FILE_THICKNESS',
                           type=str,
                           help="The name of the input thickness file.")
    argparser.add_argument('IN_FILE_REDUCED_MESH',
                           type=str,
                           help="The input file for which the ground truth"
                           " should be generated.")
    argparser.add_argument('OUT_FILE_THICKNESS',
                           type=str,
                           help="The output file that will be generated"
                           " (Contains the thickness values corresponding to"
                           " the reduced mesh).")
    argparser.add_argument('--transform',
                           default=None,
                           type=str,
                           nargs='*',
                           help="Transform the input mesh with the"
                           " transformation matrices in the given order.")

    # Parse args
    args = argparser.parse_args()
    in_mesh_fn = args.IN_FILE_MESH
    in_thickness_fn = args.IN_FILE_THICKNESS
    in_mesh_reduced_fn = args.IN_FILE_REDUCED_MESH
    out_thickness_fn = args.OUT_FILE_THICKNESS
    transform = args.transform

    # Reduced mesh
    reduced_mesh = trimesh.load(in_mesh_reduced_fn)
    reduced_verts = torch.from_numpy(reduced_mesh.vertices).float()

    # Full mesh + thickness
    full_verts, full_faces = nib.freesurfer.io.read_geometry(in_mesh_fn)
    full_thickness = nib.freesurfer.io.read_morph_data(in_thickness_fn)
    full_thickness = torch.from_numpy(full_thickness.astype(np.float32))

    # Transform full mesh
    if transform:
        full_verts_affine = np.concatenate(
            (full_verts.T, np.ones((1, full_verts.shape[0]))),
            axis=0
        )
        for t_fn in transform:
            t = np.loadtxt(t_fn)
            full_verts_affine = t @ full_verts_affine

        new_verts = full_verts_affine.T[:, :-1]
        assert new_verts.shape == full_verts.shape
        full_verts = new_verts

    full_mesh = trimesh.Trimesh(full_verts, full_faces, process=False)
    full_verts = torch.from_numpy(full_verts).float()

    # Get thickness of nearest point in full-resolution mesh
    _, knn_idx, _ = knn_points(reduced_verts[None], full_verts[None], K=1)
    reduced_thickness = knn_gather(
        full_thickness.unsqueeze(0).unsqueeze(2), knn_idx
    ).squeeze().numpy()

    # Write
    nib.freesurfer.io.write_morph_data(out_thickness_fn, reduced_thickness)

    # Debug
    reduced_mesh.export("../../misc/reduced.ply")
    full_mesh.export("../../misc/full.ply")

    print("Stored thickness file in ", out_thickness_fn)

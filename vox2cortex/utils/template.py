
""" Utility functions for templates. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from collections.abc import Sequence

import torch
import trimesh
import numpy as np
import nibabel as nib
from trimesh.scene.scene import Scene
from pytorch3d.utils import ico_sphere
from pytorch3d.structures import Meshes

from utils.coordinate_transform import transform_mesh_affine
from utils.mesh import Mesh

def generate_sphere_template(centers: dict, radii: dict, level=6):
    """ Generate a template with spheres centered at centers and corresponding
    radii
    - level 6: 40962 vertices
    - level 7: 163842 vertices

    :param centers: A dict containing {structure name: structure center}
    :param radii: A dict containing {structure name: structure radius}
    :param level: The ico level to use

    :returns: A trimesh.scene.scene.Scene
    """
    if len(centers) != len(radii):
        raise ValueError("Number of centroids and radii must be equal.")

    scene = Scene()
    for (k, c), (_, r) in zip(centers.items(), radii.items()):
        # Get unit sphere
        sphere = ico_sphere(level)
        # Scale adequately
        v = sphere.verts_packed() * r + c

        v = v.cpu().numpy()
        f = sphere.faces_packed().cpu().numpy()

        mesh = trimesh.Trimesh(v, f)

        scene.add_geometry(mesh, geom_name=k)

    return scene

def generate_ellipsoid_template(centers: dict, radii_x: dict, radii_y: dict,
                                radii_z: dict, level=6):
    """ Generate a template with ellipsoids centered at centers and corresponding
    radii
    - level 6: 40962 vertices
    - level 7: 163842 vertices

    :param centers: A dict containing {structure name: structure center}
    :param radii_x: A dict containing {structure name: structure radius}
    :param radii_y: A dict containing {structure name: structure radius}
    :param radii_z: A dict containing {structure name: structure radius}
    :param level: The ico level to use

    :returns: A trimesh.scene.scene.Scene
    """
    if (len(centers) != len(radii_x)
        or len(centers) != len(radii_y)
        or len(centers) != len(radii_z)):
        raise ValueError("Number of centroids and radii must be equal.")

    scene = Scene()
    for (k, c), (_, r_x), (_, r_y), (_, r_z) in zip(
        centers.items(), radii_x.items(), radii_y.items(), radii_z.items()):
        # Get unit sphere
        sphere = ico_sphere(level)
        # Scale adequately
        v = sphere.verts_packed() * torch.tensor((r_x, r_y, r_z)) + c

        v = v.cpu().numpy()
        f = sphere.faces_packed().cpu().numpy()

        mesh = trimesh.Trimesh(v, f)

        scene.add_geometry(mesh, geom_name=k)

    return scene


def load_mesh_template(
    path: str,
    mesh_label_names: Sequence,
    mesh_suffix: str="smoothed_reduced.ply",
    feature_suffix: str="_reduced.aparc.DKTatlas40.annot",
    trans_affine=torch.eye(4)
):
    vertices = []
    faces = []
    normals = []
    features = []

    # Load meshes and parcellation
    for mn in mesh_label_names:
        m = trimesh.load_mesh(
            os.path.join(path, mn + mesh_suffix),
            process=False
        )
        vertices.append(torch.from_numpy(m.vertices))
        faces.append(torch.from_numpy(m.faces))

        ft = torch.from_numpy(
            nib.freesurfer.io.read_annot(
               os.path.join(path, mn + feature_suffix)
            )[0].astype(np.int32)
        )
        # Combine -1 & 0 into one class
        ft[ft < 0] = 0
        features.append(ft)

    vertices = torch.stack(vertices).float()
    faces = torch.stack(faces).long()
    features = torch.stack(features).long().unsqueeze(-1)

    # Transform meshes
    vertices, faces = transform_mesh_affine(vertices, faces, trans_affine)

    # Compute normals
    normals = Meshes(vertices, faces).verts_normals_padded()

    return Mesh(vertices, faces, normals, features)


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
from utils.mesh import MeshesOfMeshes
from utils.utils import zero_pad_max_length


TEMPLATE_PATH = "../vox2organ/supplementary_material/"


# Specification of different templates
TEMPLATE_SPECS = {
    "fsaverage-smooth-reduced": {
        "path": os.path.join(TEMPLATE_PATH, "brain_template", "fsaverage"),
        "mesh_suffix": "_smoothed_reduced.ply",
        "feature_suffix": "_reduced.aparc.DKTatlas40.annot",
    },
    "fsaverage-smooth": {
        "path": os.path.join(TEMPLATE_PATH, "brain_template", "fsaverage"),
        "mesh_suffix": "_smoothed.ply",
        "feature_suffix": ".aparc.DKTatlas40.annot",
    },
    "abdomen-ellipses": {
        "path": os.path.join(TEMPLATE_PATH, "abdomen_template", "ellipses"),
        "mesh_suffix": ".ply",
        "feature_suffix": ""
    },
    "abdomen-case00017": {
        "path": os.path.join(TEMPLATE_PATH, "abdomen_template", "case_00017"),
        "mesh_suffix": ".ply",
        "feature_suffix": ""
    },
    "fsaverage-no-parc": {
        "path": os.path.join(TEMPLATE_PATH, "brain_template", "fsaverage"),
        "mesh_suffix": ".ply",
        "feature_suffix": "",
    },
}


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

        try:
            ft = torch.from_numpy(
                nib.freesurfer.io.read_annot(
                   os.path.join(path, mn + feature_suffix)
                )[0].astype(np.int32)
            )
            # Combine -1 & 0 into one class
            ft[ft < 0] = 0
            features.append(ft)
        except FileNotFoundError:
            # Insert dummy if no features (= vertex classes) could be found
            features.append(
                torch.zeros((m.vertices.shape[0]), dtype=torch.int32)
            )

    vertices_padded, _ = zero_pad_max_length(vertices)
    faces_padded, _ = zero_pad_max_length(faces)
    features_padded, _ = zero_pad_max_length(features)
    vertices_padded = torch.stack(vertices_padded).float().unsqueeze(0)
    faces_padded = torch.stack(faces_padded).long().unsqueeze(0)
    features_padded = torch.stack(features_padded).long()
    features_padded = features_padded.unsqueeze(-1).unsqueeze(0).permute((0,1,2,3))

    # Transform meshes
    vertices_padded, faces_padded = transform_mesh_affine(vertices_padded, faces_padded, trans_affine)

    # Compute normals
    normals = Meshes(vertices, faces).verts_normals_padded().unsqueeze(0)

    return MeshesOfMeshes(vertices_padded, faces_padded, normals, features_padded)

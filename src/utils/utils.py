""" Utility functions """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import collections.abc
from enum import Enum

import numpy as np
import nibabel as nib

from skimage import measure
from trimesh import Trimesh

from plyfile import PlyData

class ExtendedEnum(Enum):
    """
    Extends an enum such that it can be converted to dict.
    """

    @classmethod
    def dict(cls):
        return {c.name: c.value for c in cls}

def read_vertices_from_ply(filename: str) -> np.ndarray:
    """
    Read .ply file and return vertex coordinates

    :param str filename: The file that should be read.
    :return: Vertex data as numpy array
    :rtype: numpy.ndarray
    """

    plydata = PlyData.read(filename)
    vertex_data = plydata['vertex'].data # numpy array with fields ['x', 'y', 'z']
    pts = np.zeros([vertex_data.size, 3])
    pts[:, 0] = vertex_data['x']
    pts[:, 1] = vertex_data['y']
    pts[:, 2] = vertex_data['z']

    return pts

def find_label_to_img(base_dir: str, img_id: str, label_dir_id="label"):
    """
    Get the label file corresponding to an image file.

    :param str base_dir: The base directory containing the label directory.
    :param str img_id: The id of the image that is also cotained in the label
    file.
    :param str label_dir_id: The string that identifies the label directory.
    :return The label file name.
    """
    label_dir = None
    label_name = None
    for d in os.listdir(base_dir):
        d_full = os.path.join(base_dir, d)
        if (os.path.isdir(d_full) and (label_dir_id in d)):
            label_dir = d_full
            print(f"Found label directory '{label_dir}'.")
    if label_dir is None:
        print(f"No label directory found in {base_dir}, maybe adapt path"\
              " specification or search string.")
        return None
    else:
        # Label dir found
        for ln in os.listdir(label_dir):
            if img_id == ln.split('.')[0]:
                label_name = ln

        if label_name is None:
            print(f"No file with id '{img_id}' found in directory"\
                  " '{label_dir}'.")
            return None
    return os.path.join(label_dir, label_name)

def create_mesh_from_file(filename: str, output_dir: str=None, store=True):
    """
    Create a mesh from file using marching cubes.

    :param str filename: The name of the input file.
    :param str output_dir: The name of the output directory.
    :param bool store (optional): Store the created mesh.

    :return the created mesh
    """

    name = os.path.basename(filename) # filename without path
    name = name.split(".")[0]

    data = nib.load(filename)

    img3D = data.get_fdata() # get np.ndarray
    assert img3D.ndim == 3, "Image dimension not equal to 3."

    # Use marching cubes to obtain surface mesh
    outfile = os.path.join(output_dir, name + ".ply") # output .ply file
    verts, faces, normals, values = measure.marching_cubes(img3D)
    mesh = Trimesh(verts,
                   faces,
                   vertex_normals=normals,
                   vertex_colors=values)

    if (output_dir is not None and store):
        mesh.export(outfile)

    return mesh

def update_dict(d, u):
    """
    Recursive function for dictionary updating.

    :param d: The old dict.
    :param u: The dict that should be used for the update.

    :returns: The updated dict.
    """

    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update_dict(d.get(k, {}), v)
        else:
            d[k] = v
    return d

def serializable_dict(d: dict):
    """
    Convert all object references (classes, objects, functions etc.) to their name.

    :param dict d: The dict that should be made serializable.
    :returns: The dict with objects converted to their names.
    """
    u = d.copy()
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            # Dicts
            u[k] = serializable_dict(u.get(k, {}))
        elif isinstance(v, collections.abc.MutableSequence):
            # Lists
            seq = u.get(k, []).copy()
            for e in seq:
                if e.__class__.__name__ == 'type' or\
                     e.__class__.__name__ == 'function':
                    u[k].append(e.__name__)
                    u[k].remove(e)

        elif v.__class__.__name__ == 'type' or\
                v.__class__.__name__ == 'function':
            # Objects:
            u[k] = v.__name__
    return u

def crop_and_merge(tensor1, tensor2):

    slices = crop_slices(tensor1.size(), tensor2.size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    slices = tuple(slices)

    return torch.cat((tensor1[slices], tensor2), 1)

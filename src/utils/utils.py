""" Utility functions """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

import numpy as np

from plyfile import PlyData

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

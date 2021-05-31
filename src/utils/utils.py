""" Utility functions """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import copy
import inspect
import collections.abc
from enum import Enum

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F

from skimage import measure

from plyfile import PlyData

from utils.modes import ExecModes
from utils.mesh import Mesh

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

def create_mesh_from_file(filename: str, output_dir: str=None, store=True,
                          mc_step_size=1):
    """
    Create a mesh from file using marching cubes.

    :param str filename: The name of the input file.
    :param str output_dir: The name of the output directory.
    :param bool store (optional): Store the created mesh.
    :param int mc_step_size: The step size for marching cubes algorithm.

    :return the created mesh
    """

    name = os.path.basename(filename) # filename without path
    name = name.split(".")[0]

    data = nib.load(filename)

    img3D = data.get_fdata() # get np.ndarray
    assert img3D.ndim == 3, "Image dimension not equal to 3."

    # Use marching cubes to obtain surface mesh
    mesh = create_mesh_from_voxels(img3D, mc_step_size)

    # Store
    outfile = os.path.join(output_dir, name + ".ply") # output .ply file
    mesh = mesh.to_trimesh()

    if (output_dir is not None and store):
        mesh.export(outfile)

    return mesh

def normalize_vertices(vertices, shape):
    """ Normalize vertex coordinates from [0, patch size-1] into [-1, 1] """
    assert len(vertices.shape) == 2 and len(shape.shape) == 2, "Inputs must be 2 dim"
    assert shape.shape[0] == 1, "first dim of shape should be length 1"

    return 2*(vertices/(torch.max(shape)-1) - 0.5)

def unnormalize_vertices(vertices, shape):
    """ Inverse of 'normalize vertices' """
    assert len(vertices.shape) == 2 and len(shape.shape) == 2, "Inputs must be 2 dim"
    assert shape.shape[0] == 1, "first dim of shape should be length 1"

    return (0.5 * vertices + 0.5) * (torch.max(shape) - 1)

def create_mesh_from_voxels(volume, mc_step_size=1, flip=True):
    """ Convert a voxel volume to mesh using marching cubes

    :param volume: The voxel volume.
    :param mc_step_size: The step size for the marching cubes algorithm.
    :return: The generated mesh.
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().data.numpy()

    shape = torch.tensor(volume.shape)[None].float()

    vertices_mc, faces_mc, normals, values = measure.marching_cubes(
                                    volume,
                                    0,
                                    step_size=mc_step_size,
                                    allow_degenerate=False)

    if flip:
        vertices_mc = torch.flip(torch.from_numpy(vertices_mc), dims=[1]).float()  # convert z,y,x -> x, y, z
        normals = torch.flip(torch.from_numpy(normals), dims=[1]).float()  # convert z,y,x -> x, y, z
    vertices_mc = normalize_vertices(vertices_mc, shape)
    faces_mc = torch.from_numpy(faces_mc).long()

    return Mesh(vertices_mc, faces_mc, normals, values)

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

def string_dict(d: dict):
    """
    Convert classes and functions to their name and every object to its string.

    :param dict d: The dict that should be made serializable/writable.
    :returns: The dict with objects converted to their names.
    """
    u = copy.deepcopy(d)
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            # Dicts
            u[k] = string_dict(u.get(k, {}))
        elif isinstance(v, collections.abc.MutableSequence):
            # Lists
            seq = u.get(k, []).copy()
            for e in seq:
                # Class or function
                if inspect.isclass(e) or inspect.isfunction(e):
                    u[k].append(e.__name__)
                    u[k].remove(e)
                elif not all(isinstance(v_i, (int, float, str)) for v_i in v):
                    # Everything else than int, float, str to str
                    u[k].remove(e)
                    u[k].append(str(e))

        # Class or function
        elif inspect.isclass(v) or inspect.isfunction(v):
            u[k] = v.__name__
        elif isinstance(v, tuple):
            if not all(isinstance(v_i, (int, float, str)) for v_i in v):
                # Tuple with something else than int, float, str
                u[k] = str(v)
        elif not isinstance(v, (int, float, str)):
            # Everything else to string
            u[k] = str(v)
    return u

def crop_slices(shape1, shape2):
    """ From https://github.com/cvlab-epfl/voxel2mesh """
    slices = [slice((sh1 - sh2) // 2, (sh1 - sh2) // 2 + sh2) for sh1, sh2 in zip(shape1, shape2)]
    return slices

def crop_and_merge(tensor1, tensor2):
    """ Crops tensor1 such that it fits the shape of tensor2 and concatenates
    both along channel dimension.
    From https://github.com/cvlab-epfl/voxel2mesh """

    slices = crop_slices(tensor1.size(), tensor2.size())
    slices[0] = slice(None)
    slices[1] = slice(None)
    slices = tuple(slices)

    return torch.cat((tensor1[slices], tensor2), 1)

def sample_outer_surface_in_voxel(volume):
    """ Samples an outer surface in 3D given a volume representation of the
    objects. This is used in wickramasinghe 2020 as ground truth for mesh
    vertices.
    """
    if volume.ndim == 3:
        a = F.max_pool3d(volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))
        b = F.max_pool3d(volume[None,None].float(), kernel_size=(1,3,1), stride=1, padding=(0, 1, 0))
        c = F.max_pool3d(volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))
    elif volume.ndim == 4:
        a = F.max_pool3d(volume.unsqueeze(1).float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))
        b = F.max_pool3d(volume.unsqueeze(1).float(), kernel_size=(1,3,1), stride=1, padding=(0, 1, 0))
        c = F.max_pool3d(volume.unsqueeze(1).float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))
    else:
        raise NotImplementedError
    border, _ = torch.max(torch.cat([a,b,c], dim=1), dim=1)
    if volume.ndim == 3: # back to original shape
        border = border.squeeze()
    surface = border - volume.float()
    return surface.long()


def sample_inner_volume_in_voxel(volume):
    """ Samples an inner volume in 3D given a volume representation of the
    objects. This can be seen as 'stripping off' one layer of pixels.

    Attention: 'sample_inner_volume_in_voxel' and
    'sample_outer_surface_in_voxel' are not inverse to each other since
    several volumes can lead to the same inner volume.
    """
    neg_volume = -1 * volume # max --> min
    neg_volume_a = F.pad(neg_volume, (0,0,0,0,1,1)) # Zero-pad
    a = F.max_pool3d(neg_volume_a[None,None].float(), kernel_size=(3,1,1), stride=1)[0]
    neg_volume_b = F.pad(neg_volume, (0,0,1,1,0,0)) # Zero-pad
    b = F.max_pool3d(neg_volume_b[None,None].float(), kernel_size=(1,3,1), stride=1)[0]
    neg_volume_c = F.pad(neg_volume, (1,1,0,0,0,0)) # Zero-pad
    c = F.max_pool3d(neg_volume_c[None,None].float(), kernel_size=(1,1,3), stride=1)[0]
    border, _ = torch.max(torch.cat([a,b,c], dim=0), dim=0)
    border = -1 * border
    inner_volume = torch.logical_and(volume, border)
    # Seems to lead to problems if volume.dtype == torch.uint8
    return inner_volume.type(volume.dtype)

def normalize_max_one(data):
    """ Normalize the input such that the maximum value is 1. """
    max_value = float(data.max())
    return data / max_value

def normalize_plus_minus_one(data):
    """ Normalize the input such that the values are in [-1,1]. """
    max_value = float(data.max())
    assert data.min() >= 0 and max_value > 0, "Elements should be ge 0."
    return 2 * ((data / max_value) - 0.5)

def normalize_min_max(data):
    """ Min- max normalization into [0,1] """
    min_value = float(data.min())
    return (data - min_value) / (data.max() - min_value)

def Euclidean_weights(vertices, edges):
    """ Weights for all edges in terms of Euclidean length between vertices.
    """
    weights = torch.sqrt(torch.sum(
        (vertices[edges[:,0]] - vertices[edges[:,1]])**2,
        dim=1
    ))
    return weights

def score_is_better(old_value, new_value, name):
    """ Decide whether new_value is better than old_value based on the name of
    the score.
    """
    if old_value is None:
        if name in ('JaccardVoxel', 'JaccardMesh'):
            return True, 'max'
        elif name in ('Chamfer'):
            return True, 'min'
        else:
            raise ValueError("Unknown score name.")

    if name in ('JaccardVoxel', 'JaccardMesh'):
        return new_value > old_value, 'max'
    elif name in ('Chamfer'):
        return new_value < old_value, 'min'
    else:
        raise ValueError("Unknown score name.")

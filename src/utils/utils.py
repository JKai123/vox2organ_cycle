""" Utility functions """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
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

def get_cuda_device(number=0):
    device = torch.device(f"cuda:{number}"\
                          if torch.cuda.is_available()\
                          else "cpu")
    return device

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
    # Label dir found
    for ln in os.listdir(label_dir):
        if img_id == ln.split('.')[0]:
            label_name = ln

    if label_name is None:
        print(f"No file with id '{img_id}' found in directory"\
              " '{label_dir}'.")
        return None
    return os.path.join(label_dir, label_name)

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
    """ From https://github.com/cvlab-epfl/voxel2mesh """
    assert len(vertices.shape) == 2 and len(shape.shape) == 2, "Inputs must be 2 dim"
    assert shape.shape[0] == 1, "first dim of shape should be length 1"

    return 2*(vertices/(torch.max(shape)-1) - 0.5)

def create_mesh_from_voxels(volume, mc_step_size):
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

    vertices_mc = torch.flip(torch.from_numpy(vertices_mc), dims=[1]).float()  # convert z,y,x -> x, y, z
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
    u = d.copy()
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
                else:
                    # Everything else to string
                    u[k].remove(e)
                    u[k].append(str(e))

        # Class or function
        elif inspect.isclass(v) or inspect.isfunction(v):
            u[k] = v.__name__
        else:
            # Everything else to string
            u[k] = str(v)
    return u

def crop_slices(shape1, shape2):
    """ From https://github.com/cvlab-epfl/voxel2mesh """
    slices = [slice((sh1 - sh2) // 2, (sh1 - sh2) // 2 + sh2) for sh1, sh2 in zip(shape1, shape2)]
    return slices

def crop_and_merge(tensor1, tensor2):
    """ From https://github.com/cvlab-epfl/voxel2mesh """

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
    a = F.max_pool3d(volume[None,None].float(), kernel_size=(3,1,1), stride=1, padding=(1, 0, 0))[0]
    b = F.max_pool3d(volume[None,None].float(), kernel_size=(1,3, 1), stride=1, padding=(0, 1, 0))[0]
    c = F.max_pool3d(volume[None,None].float(), kernel_size=(1,1,3), stride=1, padding=(0, 0, 1))[0]
    border, _ = torch.max(torch.cat([a,b,c],dim=0),dim=0)
    surface = border - volume.float()
    return surface.long()

def convert_data_to_voxel2mesh_data(data, n_classes, mode, device_nr):
    """ Convert data such that it's compatible with the original voxel2mesh
    implementation. Code is an assembly of parts from
    https://github.com/cvlab-epfl/voxel2mesh
    """
    shape = torch.tensor(data[1].shape)[None]
    surface_points_normalized_all = []
    for c in range(1, n_classes):
        y_outer = sample_outer_surface_in_voxel((data[1]==c).long())
        surface_points = torch.nonzero(y_outer)
        surface_points = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z
        surface_points_normalized = normalize_vertices(surface_points, shape)

        perm = torch.randperm(len(surface_points_normalized))
        point_count = 3000
        surface_points_normalized_all += [surface_points_normalized[perm[:np.min([len(perm), point_count])]].cuda()]  # randomly pick 3000 points

    device = get_cuda_device(device_nr)

    if mode == ExecModes.TRAIN:
        voxel2mesh_data = {'x': data[0].float().to(device)[None][None],
                'y_voxels': data[1].long().to(device),
                'surface_points': surface_points_normalized_all,
                'unpool':[0, 1, 0, 1, 0]
                }
    elif mode == ExecModes.TEST:
        voxel2mesh_data = {'x': data[0].float().to(device)[None][None],
                   'y_voxels': data[1].long().to(device),
                   'vertices_mc': data[2].vertices.to(device),
                   'faces_mc': data[2].faces.to(device),
                   'surface_points': surface_points_normalized_all,
                   'unpool':[0, 1, 1, 1, 1]}
    else:
        raise ValueError("Unknown execution mode.")

    return voxel2mesh_data

def _box_in_bounds(box, image_shape):
    """ From https://github.com/cvlab-epfl/voxel2mesh """
    newbox = []
    pad_width = []

    for box_i, shape_i in zip(box, image_shape):
        pad_width_i = (max(0, -box_i[0]), max(0, box_i[1] - shape_i))
        newbox_i = (max(0, box_i[0]), min(shape_i, box_i[1]))

        newbox.append(newbox_i)
        pad_width.append(pad_width_i)

    needs_padding = any(i != (0, 0) for i in pad_width)

    return newbox, pad_width, needs_padding

def crop_indices(image_shape, patch_shape, center):
    """ From https://github.com/cvlab-epfl/voxel2mesh """
    box = [(i - ps // 2, i - ps // 2 + ps) for i, ps in zip(center, patch_shape)]
    box, pad_width, needs_padding = _box_in_bounds(box, image_shape)
    slices = tuple(slice(i[0], i[1]) for i in box)
    return slices, pad_width, needs_padding

def crop(image, patch_shape, center, mode='constant'):
    """ From https://github.com/cvlab-epfl/voxel2mesh """
    slices, pad_width, needs_padding = crop_indices(image.shape, patch_shape, center)
    patch = image[slices]

    if needs_padding and mode != 'nopadding':
        if isinstance(image, np.ndarray):
            if len(pad_width) < patch.ndim:
                pad_width.append((0, 0))
            patch = np.pad(patch, pad_width, mode=mode)
        elif isinstance(image, torch.Tensor):
            assert len(pad_width) == patch.dim(), "not supported"
            # [int(element) for element in np.flip(np.array(pad_width).flatten())]
            patch = F.pad(patch, tuple([int(element) for element in np.flip(np.array(pad_width), axis=0).flatten()]), mode=mode)

    return patch

def img_with_patch_size(img: np.ndarray, patch_size: int, is_label: bool) -> torch.tensor:
    """ Pad/interpolate an image such that it has a certain shape
    """
    # TODO: Not sure about the effect of this procedure --> analyse

    D, H, W = img.shape
    center_z, center_y, center_x = D // 2, H // 2, W // 2
    D, H, W = patch_size
    img = crop(img, (D, H, W), (center_z, center_y, center_x))

    img = torch.from_numpy(img).float()

    if is_label:
        img = F.interpolate(img[None, None].float(), patch_size, mode='nearest')[0, 0].long()
    else:
        img = F.interpolate(img[None, None], patch_size, mode='trilinear', align_corners=False)[0, 0]

    return img

""" Utility functions """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import copy
import inspect
import collections.abc
from enum import Enum
from typing import Union, Tuple

import numpy as np
import nibabel as nib
import torch
import torch.nn.functional as F
import scipy.ndimage as ndimage
from trimesh import Trimesh
from skimage import measure
from plyfile import PlyData

from utils.modes import ExecModes
from utils.mesh import Mesh
from utils.visualization import show_img_with_contour

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

def normalize_vertices(vertices: Union[torch.Tensor, np.array],
                       shape: Tuple[int, int, int]):
    """ Normalize vertex coordinates from [0, patch size-1] into [-1, 1]
    treating each dimension separately and flip x- and z-axis.
    """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    if isinstance(vertices, torch.Tensor):
        shape = torch.tensor(shape).float().to(vertices.device).flip(dims=[0])
        vertices = vertices.flip(dims=[1])
    if isinstance(vertices, np.ndarray):
        shape = np.flip(np.array(shape, dtype=float), axis=0)
        vertices = np.flip(vertices, axis=1)

    return 2*(vertices/(shape-1) - 0.5)

def unnormalize_vertices(vertices: Union[torch.Tensor, np.array],
                         shape: Tuple[int, int, int]):
    """ Inverse of 'normalize vertices' """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    if isinstance(vertices, torch.Tensor):
        shape = torch.tensor(shape).float().to(vertices.device)
        vertices = vertices.flip(dims=[1])
    if isinstance(vertices, np.ndarray):
        shape = np.array(shape, dtype=float)
        vertices = np.flip(vertices, axis=1)

    return (0.5 * vertices + 0.5) * (shape - 1)

def normalize_vertices_per_max_dim(vertices: Union[torch.Tensor, np.array],
                                   shape: Tuple[int, int, int]):
    """ Normalize vertex coordinates w.r.t. the maximum input dimension.
    """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    return 2*(vertices/(np.max(shape)-1) - 0.5)

def unnormalize_vertices_per_max_dim(vertices: Union[torch.Tensor, np.array],
                                     shape: Tuple[int, int, int]):
    """ Inverse of 'normalize vertices_per_max_dim' """
    assert len(vertices.shape) == 2, "Vertices should be packed."
    assert (len(shape) == 3 and vertices.shape[1] == 3
            or len(shape) == 2 and vertices.shape[1] ==2),\
            "Coordinates should be 2 or 3 dim."

    return (0.5 * vertices + 0.5) * (np.max(shape) - 1)

def create_mesh_from_voxels(volume, mc_step_size=1):
    """ Convert a voxel volume to mesh using marching cubes

    :param volume: The voxel volume.
    :param mc_step_size: The step size for the marching cubes algorithm.
    :return: The generated mesh.
    """
    if isinstance(volume, torch.Tensor):
        volume = volume.cpu().data.numpy()

    shape = volume.shape

    vertices_mc, faces_mc, normals, values = measure.marching_cubes(
                                    volume,
                                    0,
                                    step_size=mc_step_size,
                                    allow_degenerate=False)

    vertices_mc = normalize_vertices_per_max_dim(
        torch.from_numpy(vertices_mc).float(), shape
    )
    # measure.marching_cubes uses left-hand rule for normal directions, our
    # convention is right-hand rule
    faces_mc = torch.from_numpy(faces_mc).long().flip(dims=[1])

    # ! Normals are not valid anymore after normalization of vertices
    normals = None

    return Mesh(vertices_mc, faces_mc, normals, values)

def create_mesh_from_pixels(img):
    """ Convert an image to a 2D mesh (= a graph) using marching squares.

    :param img: The pixel input from which contours should be extracted.
    :return: The generated mesh.
    """
    if isinstance(img, torch.Tensor):
        img = img.cpu().data.numpy()

    shape = img.shape

    vertices_ms = measure.find_contours(img)
    # Only consider main contour
    vertices_ms = sorted(vertices_ms, key=lambda x: len(x))[-1]
    # Edges = faces in 2D
    faces_ms = []
    faces_ms = torch.tensor(
        faces_ms + [[i,i+1] for i in range(len(vertices_ms) - 1)]
    )

    vertices_ms = normalize_vertices_per_max_dim(
        torch.from_numpy(vertices_ms).float(), shape
    )

    return Mesh(vertices_ms, faces_ms)

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

def mirror_mesh_at_plane(mesh, plane_normal, plane_point):
    """ Mirror a mesh at a plane and return the mirrored mesh.
    The normal should point in the direction of the 'empty' side of the plane,
    i.e. the side where the mesh should be mirrored to.
    """
    # Normalize plane normal
    if not np.isclose(np.sqrt(np.sum(plane_normal ** 2)), 1):
        plane_normal = plane_normal / np.sqrt(np.sum(plane_normal ** 2))

    d = np.dot(plane_normal, plane_point)
    d_verts = -1 * (plane_normal @ mesh.vertices.T - d)
    mirrored_verts = mesh.vertices + 2 * (plane_normal[:,None] * d_verts).T

    # Preserve data type
    mirrored_mesh = Trimesh(mirrored_verts, mesh.faces)\
            if isinstance(mesh, Trimesh) else Mesh(mirrored_verts, mesh.faces)

    return mirrored_mesh

def voxelize_mesh(vertices, faces, shape, n_m_classes, strip=True):
    """ Voxelize the mesh and return a segmentation map of 'shape'. 

    :param vertices: The vertices of the mesh
    :param faces: Corresponding faces as indices to vertices
    :param shape: The shape the output image should have
    :param n_m_classes: The number of mesh classes, i.e., the number of
    different structures in the mesh. This is currently ignored but should be
    implemented at some time.
    :param strip: Whether to strip the outer layer of the voxelized mesh. This
    is often a more accurate representation of the discrete volume occupied by
    the mesh.
    """
    assert len(shape) == 3, "Shape should be 3D"
    voxelized_mesh = torch.zeros(shape, dtype=torch.long)
    vertices = vertices.view(n_m_classes, -1, 3)
    faces = faces.view(n_m_classes, -1, 3)
    unnorm_verts = unnormalize_vertices_per_max_dim(
        vertices.view(-1, 3), shape
    ).view(n_m_classes, -1, 3)
    pv = Mesh(unnorm_verts, faces).get_occupied_voxels(shape)
    if pv is not None:
        # Occupied voxels are considered to belong to one class
        voxelized_mesh[pv[:,0], pv[:,1], pv[:,2]] = 1
    else:
        # No mesh in the valid range predicted --> keep zeros
        pass

    # Strip outer layer of voxelized mesh
    if strip:
        voxelized_mesh = sample_inner_volume_in_voxel(voxelized_mesh)

    return voxelized_mesh

def voxelize_contour(vertices, shape):
    """ Voxelize the contour and return a segmentation map of shape 'shape'.
    See also
    https://stackoverflow.com/questions/39642680/create-mask-from-skimage-contour

    :param vertices: The vertices of the contour.
    :param edges: The connections between the vertices.
    :param shape: The target shape of the voxel map.
    """
    assert vertices.ndim == 3, "Vertices should be padded."
    assert vertices.shape[2] == 2 and len(shape) == 2,\
            "Method is dedicated to 2D data."
    if isinstance(vertices, torch.Tensor):
        vertices = vertices.cpu().numpy()
    voxelized_contour = np.zeros(shape, dtype=np.long)
    for vs in vertices:
        # Only consider points in valid range
        in_box = np.logical_and(np.logical_and(
            vs[:,0] >= 0, vs[:,0] < shape[0]
        ), np.logical_and(
            vs[:,1] >= 0, vs[:,1] < shape[1]
        ))
        vs_ = vs[in_box]
        # Round to voxel coordinates
        voxelized_contour[np.round(vs_[:,0]).astype('int'),
                          np.round(vs_[:,1]).astype('int')] = 1
        voxelized_contour = ndimage.binary_fill_holes(voxelized_contour)

    return torch.from_numpy(voxelized_contour)

def edge_lengths_in_contours(vertices, edges):
    """ Compute edge lengths for all edges in 'edges'."""
    if vertices.ndim != 2 or edges.ndim != 2:
        raise ValueError("Vertices and edges should be packed.")

    vertices_edges = vertices[edges]
    v1, v2 = vertices_edges[:,0], vertices_edges[:,1]

    return torch.norm(v1 - v2, dim=1)


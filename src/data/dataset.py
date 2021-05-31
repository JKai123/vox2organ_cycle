
""" Making datasets accessible

The file contains one base class for all datasets and a separate subclass for
each used dataset.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os

import numpy as np
import torch
import torch.nn.functional as F
import nibabel as nib
from tqdm import tqdm
from elasticdeform import deform_random_grid

from utils.modes import DataModes
from utils.mesh import Mesh
from utils.eval_metrics import Jaccard
from utils.logging import (
    write_scatter_plot_if_debug,
)
from utils.utils import (
    sample_outer_surface_in_voxel,
    sample_inner_volume_in_voxel,
    unnormalize_vertices,
    normalize_vertices
)

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

def augment_data(img, label):
    # Rotate 90
    if np.random.rand(1) > 0.5:
        img, label = np.rot90(img, 1, [0,1]), np.rot90(label, 1, [0,1])
    if np.random.rand(1) > 0.5:
        img, label = np.rot90(img, 1, [1,2]), np.rot90(label, 1, [1,2])
    if np.random.rand(1) > 0.5:
        img, label = np.rot90(img, 1, [2,0]), np.rot90(label, 1, [2,0])

    # Flip
    if np.random.rand(1) > 0.5:
        img, label = np.flip(img, 0), np.flip(label, 0)
    if np.random.rand(1) > 0.5:
        img, label = np.flip(img, 1), np.flip(label, 1)
    if np.random.rand(1) > 0.5:
        img, label = np.flip(img, 2), np.flip(label, 2)

    # Elastic deformation
    img, label = deform_random_grid([img, label], sigma=1, points=3,
                                    order=[3, 0])

    return img, label

def sample_surface_points(y_label, n_classes, point_count=3000):
    """ Sample outer surface points from a volume label """
    surface_points_normalized_all = []
    shape = torch.tensor(y_label.shape)
    for c in range(1, n_classes):
        y_label_outer = sample_outer_surface_in_voxel((y_label==c).long())
        surface_points = torch.nonzero(y_label_outer)
        # Point coordinates
        surface_points_normalized = normalize_vertices(surface_points, shape[None])
        # convert z,y,x -> x,y,z
        surface_points_normalized = torch.flip(surface_points_normalized,
                                               dims=[1]).float()
        # debug
        write_scatter_plot_if_debug(surface_points_normalized,
                                    "../misc/surface_points.png")
        n_points = len(surface_points_normalized)
        perm = torch.randperm(n_points)
        # randomly pick a maximum of point_count points
        surface_points_normalized = surface_points_normalized[
            perm[:np.min([n_points, point_count])]
        ].cuda()
        # pad s.t. a batch can be created
        if n_points < point_count:
            surface_points_normalized = F.pad(
                surface_points_normalized, (0, 0, 0, point_count-n_points)
            )
        surface_points_normalized_all.append(surface_points_normalized)

    return torch.stack(surface_points_normalized_all)


class DatasetHandler(torch.utils.data.Dataset):
    """
    Base class for all datasets. It implements a map-style dataset, see
    https://pytorch.org/docs/stable/data.html.

    :param list ids: The ids of the files the dataset split should contain
    :param DataModes datamode: TRAIN, VALIDATION, or TEST

    """

    def __init__(self, ids: list, mode: DataModes):
        self._mode = mode
        self._files = ids

    def __getitem__(self, key):
        if isinstance(key, slice):
            # get the start, stop, and step from the slice
            return [self[ii] for ii in range(*key.indices(len(self)))]
        if isinstance(key, int):
            # handle negative indices
            if key < 0:
                key += len(self)
            if key < 0 or key >= len(self):
                raise IndexError(f"The index {key} is out of range.")
            # get the data from direct index
            return self.get_item_from_index(key)

        raise TypeError("Invalid argument type.")

    def get_file_name_from_index(self, index):
        """ Return the filename corresponding to an index in the dataset """
        return self._files[index]

    @staticmethod
    def save_ids(train_ids, val_ids, test_ids, save_dir):
        """ Save ids to a file """
        filename = os.path.join(save_dir, "dataset_ids.txt")

        with open(filename, 'w') as f:
            f.write("##### Training ids #####\n\n")
            for idx, id_i in enumerate(train_ids):
                f.write(f"{idx}: {id_i}\n")
            f.write("\n\n")
            f.write("##### Validation ids #####\n\n")
            for idx, id_i in enumerate(val_ids):
                f.write(f"{idx}: {id_i}\n")
            f.write("\n\n")
            f.write("##### Test ids #####\n\n")
            for idx, id_i in enumerate(test_ids):
                f.write(f"{idx}: {id_i}\n")
            f.write("\n\n")


    def get_item_and_mesh_from_index(self, index):
        """ Return the 3D data plus a mesh """
        raise NotImplementedError

    def get_item_from_index(self, index: int):
        """
        An item consists in general of (data, labels)

        :param int index: The index of the data to access.
        """
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def check_data(self):
        """ Check if voxel and mesh data is consistent """
        for i in tqdm(range(len(self)),
                      desc="Checking IoU of voxel and mesh labels"):
            _, voxel_label, mesh = self.get_item_and_mesh_from_index(i)
            shape = torch.tensor(voxel_label.shape)[None]
            vertices, faces = mesh.vertices, mesh.faces
            voxelized_mesh = torch.zeros_like(voxel_label, dtype=torch.long)
            vertices = vertices.view(self.n_m_classes, -1, 3)
            faces = faces.view(self.n_m_classes, -1, 3)
            unnorm_verts = unnormalize_vertices(
                vertices.view(-1, 3), shape
            ).view(self.n_m_classes, -1, 3)
            pv = Mesh(unnorm_verts,
                      faces).get_occupied_voxels(shape.squeeze().cpu().numpy())
            if pv is not None:
                pv_flip = np.flip(pv, axis=1)  # convert x,y,z -> z, y, x
                # Occupied voxels are considered to belong to one class
                voxelized_mesh[pv_flip[:,0], pv_flip[:,1], pv_flip[:,2]] = 1
            else:
                # No mesh in the valid range predicted --> keep zeros
                pass

            # Strip outer layer of voxelized mesh
            strip = True
            if strip:
                voxelized_mesh = sample_inner_volume_in_voxel(voxelized_mesh)

            j_vox = Jaccard(voxel_label.cuda(), voxelized_mesh.cuda(), 2)

            assert j_vox > 0.85,\
                    "Voxelized mesh and voxel label should have a large IoU."

            img = nib.Nifti1Image(voxel_label.squeeze().cpu().numpy(), np.eye(4))
            nib.save(img, "../misc/data_voxel_label.nii.gz")
            img = nib.Nifti1Image(voxelized_mesh.squeeze().cpu().numpy(), np.eye(4))
            nib.save(img, "../misc/data_mesh_label.nii.gz")

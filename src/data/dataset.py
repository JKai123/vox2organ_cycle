
""" Making datasets accessible

The file contains one base class for all datasets and a separate subclass for
each used dataset.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import random
from enum import IntEnum

import numpy as np
import torch.utils.data
import torch.nn.functional as F
import nibabel as nib
from elasticdeform import deform_random_grid

import trimesh

from utils.modes import DataModes
from utils.utils import create_mesh_from_voxels

class SupportedDatasets(IntEnum):
    """ List supported datasets """
    Hippocampus = 1

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

    # Elastic deformation
    img, label = deform_random_grid([img, label], sigma=1, points=3,
                                    order=[3, 0])

    return img, label

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

class Hippocampus(DatasetHandler):
    """ Hippocampus dataset from
    https://drive.google.com/file/d/1RzPB1_bqzQhlWvU-YGvZzhx2omcDh38C/view

    It loads all data specified by 'ids' directly into memory. Only ids in
    'imagesTr' are considered (for 'imagesTs' no labels exist).

    :param list ids: The ids of the files the dataset split should contain, example:
        ['hippocampus_101', 'hippocampus_212',...]
    :param DataModes datamode: TRAIN, VALIDATION, or TEST
    :param str raw_data_dir: The raw base folder, contains e.g. subfolders
    imagesTr/ and labelsTr/
    :param str preprocessed_data_dir: Pre-processed data, e.g. meshes created with
    marching cubes.
    :param patch_size: The patch size of the images, e.g. (64, 64, 64)
    :param augment: Use image augmentation during training if 'True'
    :param bool load_mesh: Load a precomputed mesh. Possible values are
    'folder': Load meshes from a folder defined by 'preprocessed_data_dir',
    'create': create all meshes with marching cubes. 'no': Do not store meshes.
    :param mc_step_size: The step size for marching cubes generation, only
    relevant if load_mesh is 'create'.
    :param
    """

    def __init__(self, ids: list, mode: DataModes, raw_data_dir: str,
                 preprocessed_data_dir: str, patch_size, augment: bool,
                 load_mesh='no', mc_step_size=1):
        super().__init__(ids, mode)

        self._raw_data_dir = raw_data_dir
        self._preprocessed_data_dir = preprocessed_data_dir
        self._augment = augment
        self._mc_step_size = mc_step_size
        self.patch_size = patch_size

        self.data = self._load_data3D(folder="imagesTr")
        self.voxel_labels = self._load_data3D(folder="labelsTr")
        # Ignore distinction between anterior and posterior hippocampus
        for vl in self.voxel_labels:
            vl[vl > 1] = 1

        if load_mesh == 'folder':
            # Load all meshes from a folder
            self.mesh_labels = self._load_dataMesh(folder="meshlabelsTr")
        elif load_mesh == 'create':
            # Create all meshes with marching cubes
            self.mesh_labels = self._calc_mesh_labels_all()
        else:
            # Do not store mesh labels
            self.mesh_labels = None

        assert self.__len__() == len(self.data)
        assert self.__len__() == len(self.voxel_labels)
        if load_mesh != 'no':
            assert self.__len__() == len(self.mesh_labels)

    @staticmethod
    def split(raw_data_dir, preprocessed_data_dir, dataset_seed,
              dataset_split_proportions, patch_size, augment_train, save_dir, **kwargs):
        """ Create train, validation, and test split of the Hippocampus data"

        :param str raw_data_dir: The raw base folder, contains e.g. subfolders
        imagesTr/ and labelsTr/
        :param str preprocessed_data_dir: Pre-processed data, e.g. meshes created with
        marching cubes.
        :param dataset_seed: A seed for the random splitting of the dataset.
        :param dataset_split_proportions: The proportions of the dataset
        splits, e.g. (80, 10, 10)
        :patch_size: The patch size of the 3D images.
        :augment_train: Augment training data.
        :save_dir: A directory where the split ids can be saved.
        :overfit: All three splits are the same and contain only one element.
        :return: (Train dataset, Validation dataset, Test dataset)
        """

        overfit = kwargs.get("overfit", False)

        # Available files
        all_files = os.listdir(os.path.join(raw_data_dir, "imagesTr"))
        all_files = [fn for fn in all_files if "._" not in fn] # Remove invalid
        all_files = [fn.split(".")[0] for fn in all_files] # Remove file ext.

        # Shuffle with seed
        random.Random(dataset_seed).shuffle(all_files)

        # Split
        if overfit:
            # Only consider first element of available data
            indices_train = slice(0, 1)
            indices_val = slice(0, 1)
            indices_test = slice(0, 1)
        else:
            # No overfit
            assert np.sum(dataset_split_proportions) == 100, "Splits need to sum to 100."
            indices_train = slice(0, dataset_split_proportions[0] * len(all_files) // 100)
            indices_val = slice(indices_train.stop,
                                indices_train.stop +\
                                    (dataset_split_proportions[1] * len(all_files) // 100))
            indices_test = slice(indices_val.stop, len(all_files))

        # Create datasets
        train_dataset = Hippocampus(all_files[indices_train],
                                    DataModes.TRAIN,
                                    raw_data_dir,
                                    preprocessed_data_dir,
                                    patch_size,
                                    augment_train,
                                    load_mesh='no') # no meshes for train
        val_dataset = Hippocampus(all_files[indices_val],
                                  DataModes.VALIDATION,
                                  raw_data_dir,
                                  preprocessed_data_dir,
                                  patch_size,
                                  False, # no augment for val
                                  load_mesh='create') # create all mc meshes
        test_dataset = Hippocampus(all_files[indices_test],
                                  DataModes.TEST,
                                  raw_data_dir,
                                  preprocessed_data_dir,
                                  patch_size,
                                  False, # no augment for test
                                  load_mesh='create') # create all mc meshes

        # Save ids to file
        DatasetHandler.save_ids(all_files[indices_train], all_files[indices_val],
                         all_files[indices_test], save_dir)

        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        return len(self._files)

    def get_item_from_index(self, index: int):
        """
        One data item has the form
        (3D input image, 3D voxel label)
        with types
        (torch.tensor, torch.tensor)
        """
        img, voxel_label = self.data[index], self.voxel_labels[index]

        # Potentially augment
        if self._augment:
            img, voxel_label = augment_data(img, voxel_label)

        # Fit patch size
        img = img_with_patch_size(img, self.patch_size, False)
        voxel_label = img_with_patch_size(voxel_label, self.patch_size, True)

        return img, voxel_label

    def get_item_and_mesh_from_index(self, index: int):
        """ One data item and a corresponding mesh.
        Data is returned in the form
        (3D input image, 3D voxel label, 3D mesh)
        """
        img, voxel_label = self.get_item_from_index(index)
        mesh_label = self._get_mesh_from_index(index)

        return img, voxel_label, mesh_label

    def _load_data3D(self, folder: str):
        data_dir = os.path.join(self._raw_data_dir, folder)
        data = []
        for fn in self._files:
            d = nib.load(os.path.join(data_dir, fn + ".nii.gz")).get_fdata()
            data.append(d)

        return data

    def _calc_mesh_labels_all(self):
        meshes = []
        for v in self.voxel_labels:
            meshes.append(create_mesh_from_voxels(v, self._mc_step_size))

        return meshes

    def _get_mesh_from_index(self, index):
        if self.mesh_labels is not None: # read
            mesh_label = self.mesh_labels[index]
        else: # generate
            mesh_label = create_mesh_from_voxels(self.voxel_labels[index],
                                                   self._mc_step_size)

        return mesh_label

    def _load_dataMesh(self, folder):
        data_dir = os.path.join(self._preprocessed_data_dir, folder)
        data = []
        for fn in self._files:
            d = trimesh.load_mesh(os.path.join(data_dir, fn + ".ply"))
            data.append(d)

        return data

# Mapping supported datasets to split functions
dataset_split_handler = {
    SupportedDatasets.Hippocampus.name: Hippocampus.split
}

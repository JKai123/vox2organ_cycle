
""" Cortex dataset handler """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import random
import logging
from enum import IntEnum

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import trimesh
from pytorch3d.structures import Meshes

from utils.modes import DataModes, ExecModes
from utils.logging import measure_time
from utils.mesh import Mesh, generate_sphere_template
from utils.utils import (
    normalize_min_max,
    normalize_vertices
)
from data.dataset import (
    DatasetHandler,
    augment_data,
    img_with_patch_size
)

class CortexLabels(IntEnum):
    right_white_matter = 41
    left_white_matter = 2
    left_cerebral_cortex = 3
    right_cerebral_cortex = 42

def combine_labels(labels, names):
    """ Only consider labels in 'names' and set all those labels equally to 1
    """
    ids = [CortexLabels[n].value for n in names]

    return np.isin(labels, ids).astype(int)

class Cortex(DatasetHandler):
    """ Cortex dataset

    It loads all data specified by 'ids' directly into memory.

    :param list ids: The ids of the files the dataset split should contain, example:
        ['1000_3', '1001_3',...]
    :param DataModes datamode: TRAIN, VALIDATION, or TEST
    :param str raw_data_dir: The raw base folder, contains folders
    corresponding to sample ids
    :param patch_size: The patch size of the images, e.g. (256, 256, 256)
    :param augment: Use image augmentation during training if 'True'
    :param n_template_vertices: The number of vertices in a template
    :param seg_label_names: The segmentation labels to consider
    :param mesh_label_names: The mesh ground truth file names (can be multiple)
    """

    def __init__(self, ids: list, mode: DataModes, raw_data_dir: str,
                 patch_size, augment: bool, n_template_vertices: int=20000,
                 seg_label_names=("right_white_matter", "left_white_matter"),
                 mesh_label_names=("rh_white", "lh_white")):
        super().__init__(ids, mode)

        if augment:
            raise NotImplementedError("Cortex dataset does not support"\
                                      " augmentation at the moment.")
        self._raw_data_dir = raw_data_dir
        self._augment = augment
        self.patch_size = patch_size

        # Image data
        self.data = self._load_data3D(filename="mri.nii.gz")
        # NORMALIZE images
        for i, d in enumerate(self.data):
            self.data[i] = normalize_min_max(d)

        # Voxel labels
        self.voxel_labels = self._load_data3D(filename="aparc+aseg.nii.gz")
        self.voxel_labels = [
            combine_labels(l, seg_label_names) for l in self.voxel_labels
        ]

        # Mesh labels
        self.mesh_labels, (self.centers, self.radii) =\
                self._load_dataMesh(meshnames=mesh_label_names)
        self.mesh_label_names = mesh_label_names

        # Template for dataset
        self.template = generate_sphere_template(list(self.centers.values()),
                                                 list(self.radii.values()),
                                                 level=6)

        # Maximum number of reference vertices in the ground truth meshes
        # The maximum in the dataset is 146705 (calculated with
        # scripts.get_max_num_vertices_of_data)
        self.max_n_vertices = 40000 * len(mesh_label_names)

        assert self.__len__() == len(self.data)
        assert self.__len__() == len(self.voxel_labels)
        assert self.__len__() == len(self.mesh_labels)

    def store_template(self, path):
        verts = self.template.verts_packed()
        faces = self.template.faces_packed()

        Mesh(verts, faces).store(path)

    @staticmethod
    def split(raw_data_dir, dataset_seed, dataset_split_proportions,
              patch_size, augment_train, save_dir, **kwargs):
        """ Create train, validation, and test split of the cortex data"

        :param str raw_data_dir: The raw base folder, contains a folder for each
        sample
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
        all_files = os.listdir(raw_data_dir)
        all_files = [fn for fn in all_files if "meshes" not in fn] # Remove invalid

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
        train_dataset = Cortex(all_files[indices_train],
                               DataModes.TRAIN,
                               raw_data_dir,
                               patch_size,
                               augment_train) # no meshes for train
        val_dataset = Cortex(all_files[indices_val],
                             DataModes.VALIDATION,
                             raw_data_dir,
                             patch_size,
                             False) # create all mc meshes
        test_dataset = Cortex(all_files[indices_test],
                              DataModes.TEST,
                              raw_data_dir,
                              patch_size,
                              False) # create all mc meshes

        # Save ids to file
        DatasetHandler.save_ids(all_files[indices_train], all_files[indices_val],
                         all_files[indices_test], save_dir)

        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        return len(self._files)

    @measure_time
    def get_item_from_index(self, index: int):
        """
        One data item has the form
        (3D input image, 3D voxel label, points)
        with types
        (torch.tensor, torch.tensor, torch.tensor)
        """
        img = self.data[index]
        voxel_label = self.voxel_labels[index]
        points_label = self.mesh_labels[index].vertices
        # Pad/crop such that all pointclouds have the same number of points
        n_points = len(points_label)
        if self.max_n_vertices > len(points_label):
            points_label = F.pad(points_label,
                                 (0,0,0,self.max_n_vertices-n_points))
        else:
            perm = torch.randperm(self.max_n_vertices)
            points_label = points_label[perm]
        # For compatibility with multi-class pointclouds
        points_label = points_label[None]

        # TODO: implement augmentation

        # Fit patch size
        img = img_with_patch_size(img, self.patch_size, False)[None]
        voxel_label = img_with_patch_size(voxel_label, self.patch_size, True)

        logging.getLogger(ExecModes.TRAIN.name).debug("Dataset file %s",
                                                      self._files[index])

        return img, voxel_label, points_label

    def get_item_and_mesh_from_index(self, index):
        """ Get image, segmentation ground truth and reference mesh"""
        img, voxel_label, _ = self.get_item_from_index(index)
        mesh_label = self.mesh_labels[index]

        return img, voxel_label, mesh_label

    def _load_data3D(self, filename: str):
        data = []
        for fn in self._files:
            img = nib.load(os.path.join(self._raw_data_dir, fn, filename))

            d = img.get_fdata()
            data.append(d)

        return data

    def _load_dataMesh(self, meshnames):
        data = []
        for fn in self._files:
            # Voxel coords
            orig = nib.load(os.path.join(self._raw_data_dir, fn,
                                         'mri.nii.gz'))
            vox2world_affine = orig.affine
            world2vox_affine = np.linalg.inv(vox2world_affine)
            file_vertices = []
            file_faces = []
            centers_per_structure = {mn: [] for mn in meshnames}
            radii_per_structure = {mn: [] for mn in meshnames}
            for mn in meshnames:
                mesh = trimesh.load_mesh(os.path.join(
                    self._raw_data_dir, fn, mn + ".stl"
                ))
                vertices = mesh.vertices
                coords = np.concatenate((vertices.T,
                                          np.ones((1, vertices.shape[0]))),
                                         axis=0)
                new_verts = (world2vox_affine @ coords).T[:,:-1]
                new_verts = normalize_vertices(new_verts,
                                               torch.tensor(self.patch_size)[None])
                # Convert z,y,x --> x,y,z
                new_verts = torch.flip(new_verts, dims=[1])
                file_vertices.append(new_verts)
                file_faces.append(torch.from_numpy(mesh.faces))
                center = new_verts.mean(dim=0)
                radii = torch.sqrt(torch.sum((new_verts - center)**2, dim=1)).mean(dim=0)
                centers_per_structure[mn].append(center)
                radii_per_structure[mn].append(radii)
            # First treat as a batch of multiple meshes and then combine
            # into one mesh
            mesh_batch = Meshes(file_vertices, file_faces)
            mesh_single = Mesh(mesh_batch.verts_packed().float(),
                               mesh_batch.faces_packed().float())
            data.append(mesh_single)

            # Compute centroids and average radius per structure
            centroids = {k: torch.mean(torch.stack(v), dim=0)
                         for k, v in centers_per_structure.items()}
            radii = {k: torch.mean(torch.stack(v), dim=0)
                     for k, v in radii_per_structure.items()}

        return data, (centroids, radii)

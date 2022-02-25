
""" Cortex dataset handler """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import re
import os
import random
import logging
from collections.abc import Sequence
from typing import Union

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import trimesh
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes
from tqdm import tqdm

from utils.visualization import show_difference
from utils.eval_metrics import Jaccard
from utils.modes import DataModes, ExecModes
from utils.logging import measure_time, raise_warning
from utils.mesh import Mesh, curv_from_cotcurv_laplacian
from utils.utils import (
    voxelize_mesh,
    create_mesh_from_voxels,
    normalize_min_max,
)
from utils.coordinate_transform import (
    transform_mesh_affine,
    unnormalize_vertices_per_max_dim,
    normalize_vertices_per_max_dim,
)
from data.dataset import (
    DatasetHandler,
    flip_img,
    img_with_patch_size,
)
from data.supported_datasets import (
    valid_ids,
)
from data.cortex_labels import (
    combine_labels,
)

def _get_seg_and_mesh_label_names(structure_type, patch_mode, ndims):
    """ Helper function to map the structure type and the patch mode to the
    correct segmentation and mesh label names.

    For seg_label_names and voxelized_mesh_label_names entries can/should be
    grouped s.t. they represent one "voxel class" in the segmentation maps.
    """
    voxelized_mesh_label_names = None # Does not always exist
    if structure_type == "cerebral_cortex":
        if patch_mode=="single-patch":
            seg_label_names = (("right_cerebral_cortex",),)
            voxelized_mesh_label_names = (("rh_pial",),)
            mesh_label_names = ("rh_pial",)
        else: # not patch mode
            if ndims == 3: # 3D
                seg_label_names = (("left_cerebral_cortex",
                                   "right_cerebral_cortex"),)
                mesh_label_names = ("lh_pial", "rh_pial")
                voxelized_mesh_label_names = (("lh_pial", "rh_pial"),)
            else:
                raise NotImplementedError()

    elif structure_type == "white_matter":
        if patch_mode=="single-patch":
            seg_label_names = (("right_white_matter",),)
            mesh_label_names = ("rh_white",)
            voxelized_mesh_label_names = (("rh_white",),)
        else: # not patch mode
            if ndims == 3: # 3D
                seg_label_names = (("left_white_matter",
                                    "right_white_matter"),)
                mesh_label_names = ("lh_white", "rh_white")
                voxelized_mesh_label_names = (("lh_white", "rh_white"),)
            else:
                raise ValueError("Wrong dimensionality.")

    elif ("cerebral_cortex" in structure_type
          and "white_matter" in structure_type):
        if patch_mode == "single-patch":
            seg_label_names = (("right_white_matter",
                                "right_cerebral_cortex"),)
            mesh_label_names = ("rh_white", "rh_pial")
            voxelized_mesh_label_names = (("rh_white", "rh_pial"),)
        else:
            # Not patch mode
            seg_label_names = (("left_white_matter",
                               "right_white_matter",
                               "left_cerebral_cortex",
                               "right_cerebral_cortex"),)
            mesh_label_names = ("lh_white",
                                "rh_white",
                                "lh_pial",
                                "rh_pial")
            voxelized_mesh_label_names = (("lh_white",
                                           "rh_white",
                                           "lh_pial",
                                           "rh_pial"),)
    else:
        raise ValueError("Unknown structure type.")

    return seg_label_names, mesh_label_names, voxelized_mesh_label_names

class CortexDataset(DatasetHandler):
    """ Cortex dataset

    It loads all data specified by 'ids' directly into memory.

    :param list ids: The ids of the files the dataset split should contain, example:
        ['1000_3', '1001_3',...]
    :param DataModes datamode: TRAIN, VALIDATION, or TEST
    :param str raw_data_dir: The raw base folder, contains folders
    corresponding to sample ids
    :param augment: Use image augmentation during training if 'True'
    :param patch_size: The patch size of the images, e.g. (256, 256, 256)
    :param n_ref_points_per_structure: The number of ground truth points
    per 3D structure.
    :param structure_type: Either 'white_matter' or 'cerebral_cortex'
    :param patch_mode: "single-patch" or "no"
    :param patch_origin: The anker of an extracted patch, only has an effect if
    patch_mode is True.
    :param select_patch_size: The size of the cut out patches. Can be different
    to patch_size, e.g., if extracted patches should be resized after
    extraction.
    :param mc_step_size: The marching cubes step size.
    :param reduced_freesurfer: Use a freesurfer mesh with reduced number of
    vertices as mesh ground truth, e.g., 0.3. This has only an effect if patch_mode='no'.
    :param mesh_type: 'freesurfer' or 'marching cubes'
    :param preprocessed_data_dir: A directory that contains additional
    preprocessed data, e.g., thickness values.
    As a consequence, the ground truth can only consist of vertices and not of
    sampled surface points.
    :param seg_ground_truth: Either 'voxelized_meshes' or 'aseg'
    :param check_dir: An output dir where data is stored that should be
    checked.
    """

    img_filename = "mri.nii.gz"
    label_filename = "aseg.nii.gz" # For FS segmentations
    label_filename_Mb = "aparc+aseg_manual.nii.gz" # Manual Mindboggle segmentations

    def __init__(self,
                 ids: Sequence,
                 mode: DataModes,
                 raw_data_dir: str,
                 patch_size,
                 n_ref_points_per_structure: int,
                 structure_type: Union[str, Sequence]=('white_matter', 'cerebral_cortex'),
                 augment: bool=False,
                 patch_mode: str="no",
                 patch_origin=(0,0,0),
                 select_patch_size=None,
                 reduced_freesurfer: int=None,
                 mesh_type='freesurfer',
                 preprocessed_data_dir=None,
                 seg_ground_truth='voxelized_meshes',
                 check_dir='../to_check',
                 **kwargs):
        super().__init__(ids, mode)

        assert patch_mode in ("single-patch", "no"),\
                "Unknown patch mode."
        assert mesh_type in ("marching cubes", "freesurfer"),\
                "Unknown mesh type"

        if 'Mindboggle' in raw_data_dir:
            self.label_filename = self.label_filename_Mb

        (self.seg_label_names,
         self.mesh_label_names,
         self.voxelized_mesh_label_names) = _get_seg_and_mesh_label_names(
            structure_type, patch_mode, len(patch_size)
        )
        self.check_dir = check_dir
        self.structure_type = structure_type
        self._raw_data_dir = raw_data_dir
        self._preprocessed_data_dir = preprocessed_data_dir
        self._augment = augment
        self._patch_origin = patch_origin
        self._mc_step_size = kwargs.get('mc_step_size', 1)
        self.trans_affine = []
        self.ndims = len(patch_size)
        self.patch_mode = patch_mode
        self.patch_size = tuple(patch_size)
        self.select_patch_size = select_patch_size if (
            select_patch_size is not None) else patch_size
        self.n_m_classes = len(self.mesh_label_names)
        assert self.n_m_classes == kwargs.get(
            "n_m_classes", len(self.mesh_label_names)
        ), "Number of mesh classes incorrect."
        assert seg_ground_truth in ('voxelized_meshes', 'aseg')
        self.seg_ground_truth = seg_ground_truth
        self.n_ref_points_per_structure = n_ref_points_per_structure
        self.n_structures = len(self.mesh_label_names)
        self.mesh_type = mesh_type
        self.centers = None
        self.radii = None
        self.radii_x = None
        self.radii_y = None
        self.radii_z = None
        self.n_min_vertices, self.n_max_vertices = None, None
        # +1 for background
        self.n_v_classes = len(self.seg_label_names) + 1 if (
            self.seg_ground_truth == 'aseg'
        ) else len(self.voxelized_mesh_label_names) + 1

        # Sanity checks to make sure data is transformed correctly
        self.sanity_checks = kwargs.get("sanity_check_data", True)

        # Freesurfer meshes of desired resolution
        if reduced_freesurfer is not None:
            if reduced_freesurfer != 1.0:
                self.mesh_label_names = [
                    mn + "_reduced_" + str(reduced_freesurfer)
                    for mn in self.mesh_label_names
                ]

        if self.ndims == 3:
            self._prepare_data_3D()
        else:
            raise ValueError("Unknown number of dimensions ", self.ndims)

        # NORMALIZE images
        for i, img in enumerate(self.images):
            self.images[i] = normalize_min_max(img)

        # Do not store meshes in train split
        remove_meshes = self._mode == DataModes.TRAIN

        assert self.__len__() == len(self.images)
        assert self.__len__() == len(self.voxel_labels)
        assert self.__len__() == len(self.mesh_labels)

        if self.ndims == 3:
            assert self.__len__() == len(self.trans_affine)

        if self._augment:
            self.check_augmentation_normals()

    def _prepare_data_3D(self):
        """ Load 3D data """

        # Image data
        self.images, img_transforms = self._load_data3D_and_transform(
            self.img_filename, is_label=False
        )
        self.voxel_labels = None
        self.voxelized_meshes = None

        # Voxel labels
        if self.sanity_checks or self.seg_ground_truth == 'aseg':
            # Load 'aseg' segmentation maps from FreeSurfer
            self.voxel_labels, _ = self._load_data3D_and_transform(
                self.label_filename, is_label=True
            )
            # Combine labels as specified by groups (see
            # _get_seg_and_mesh_label_names)
            combine = lambda x: torch.sum(
                torch.stack([combine_labels(x, group, val) for val, group in
                enumerate(self.seg_label_names, 1)]),
                dim=0
            )
            self.voxel_labels = list(map(combine, self.voxel_labels))

        # Voxelized meshes
        if self.sanity_checks or self.seg_ground_truth == 'voxelized_meshes':
            try:
                self.voxelized_meshes = self._read_voxelized_meshes()
            except FileNotFoundError:
                self.voxelized_meshes = None # Compute later

        # Meshes
        self.mesh_labels = self._load_dataMesh_raw(meshnames=self.mesh_label_names)
        self._transform_meshes_as_images(img_transforms)

        # Voxelize meshes if voxelized meshes have not been created so far
        # and they are required (for sanity checks or as labels)
        if (self.voxelized_meshes is None and (
            self.sanity_checks or self.seg_ground_truth == 'voxelized_meshes')):
            self.voxelized_meshes = self._create_voxel_labels_from_meshes(
                self.mesh_labels
            )

        # Assert conformity of voxel labels and voxelized meshes
        if self.sanity_checks:
            for i, (vl, vm) in enumerate(zip(self.voxel_labels,
                                             self.voxelized_meshes)):
                iou = Jaccard(vl.cuda(), vm.cuda(), 2)
                if iou < 0.85:
                    out_fn = self._files[i].replace("/", "_")
                    show_difference(
                        vl,  vm,
                        os.path.join(
                            self.check_dir, f"diff_mesh_voxel_label_{out_fn}.png"
                        )
                    )
                    raise_warning(
                        f"Small IoU ({iou}) of voxel label and voxelized mesh"
                        f" label, check files at {self.check_dir}"
                    )

        # Use voxelized meshes as voxel ground truth
        if self.seg_ground_truth == 'voxelized_meshes':
            self.voxel_labels = self.voxelized_meshes

        self.thickness_per_vertex = self._get_morph_label(
            "thickness", subfolder=""
        )


    def _transform_meshes_as_images(self, img_transforms):
        """ Transform meshes according to image transformations
        (crops, resize) and normalize
        """
        for i, (m, t) in tqdm(
            enumerate(zip(self.mesh_labels, img_transforms)),
            position=0, leave=True, desc="Transform meshes accordingly..."
        ):
            # Transform vertices and potentially faces (to preserve normal
            # convention)
            new_vertices, new_faces = [], []
            for v, f in zip(m.verts_list(), m.faces_list()):
                new_v, new_f= transform_mesh_affine(v, f, t)
                new_v, norm_affine = normalize_vertices_per_max_dim(
                    new_v,
                    self.patch_size,
                    return_affine=True
                )
                new_vertices.append(new_v)
                new_faces.append(new_f)

            # Replace mesh with transformed one
            self.mesh_labels[i] = Meshes(new_vertices, new_faces)

            # Store affine transformations
            self.trans_affine[i] = norm_affine @ t @ self.trans_affine[i]

    def mean_area(self):
        """ Average surface area of meshes. """
        areas = []
        ndims = len(self.patch_size)
        for m in self.mesh_labels:
            m_unnorm = Mesh(
                unnormalize_vertices_per_max_dim(
                    m.verts_packed(), self.patch_size
                ),
                m.faces_packed()
            )
            areas.append(m_unnorm.to_trimesh().area)

        return np.mean(areas)

    def mean_edge_length(self):
        """ Average edge length in dataset.

        Code partly from pytorch3d.loss.mesh_edge_loss.
        """
        edge_lengths = []
        for m in self.mesh_labels:
            if self.ndims == 3:
                edges_packed = m.edges_packed()
            else:
                raise ValueError("Only 3D possible.")
            verts_packed = m.verts_packed()

            verts_edges = verts_packed[edges_packed]
            v0, v1 = verts_edges.unbind(1)
            edge_lengths.append(
                (v0 - v1).norm(dim=1, p=2).mean().item()
            )

        return torch.tensor(edge_lengths).mean()

    @classmethod
    def split(cls,
              raw_data_dir,
              save_dir,
              augment_train=False,
              dataset_seed=0,
              dataset_split_proportions=None,
              fixed_split: Union[dict, Sequence]=None,
              overfit=False,
              load_only=('train', 'validation', 'test'),
              **kwargs):
        """ Create train, validation, and test split of the cortex data"

        :param str raw_data_dir: The raw base folder, contains a folder for each
        sample
        :param dataset_seed: A seed for the random splitting of the dataset.
        :param dataset_split_proportions: The proportions of the dataset
        splits, e.g. (80, 10, 10)
        :param augment_train: Augment training data.
        :param save_dir: A directory where the split ids can be saved.
        :param fixed_split: A dict containing file ids for 'train',
        'validation', and 'test'. If specified, values of dataset_seed,
        overfit, and dataset_split_proportions will be ignored. Alternatively,
        a sequence of files containing ids can be given.
        :param overfit: Create small datasets for overfitting if this parameter
        is > 0.
        :param load_only: Only return the splits specified (in the order train,
        validation, test, while missing splits will be None)
        :param kwargs: Dataset parameters.
        :return: (Train dataset, Validation dataset, Test dataset)
        """

        # Decide between fixed and random split
        if fixed_split is not None:
            if isinstance(fixed_split, dict):
                files_train = fixed_split['train']
                files_val = fixed_split['validation']
                files_test = fixed_split['test']
            elif isinstance(fixed_split, Sequence):
                assert len(fixed_split) == 3,\
                        "Should contain one file per split"
                convert = lambda x: x[:-1] # 'x\n' --> 'x'
                train_split = os.path.join(raw_data_dir, fixed_split[0])
                try:
                    files_train = list(map(convert, open(train_split, 'r').readlines()))
                except:
                    files_train = []
                    raise_warning("No training files.")
                val_split = os.path.join(raw_data_dir, fixed_split[1])
                try:
                    files_val = list(map(convert, open(val_split, 'r').readlines()))
                except:
                    files_val = []
                    raise_warning("No validation files.")
                test_split = os.path.join(raw_data_dir, fixed_split[2])
                try:
                    files_test = list(map(convert, open(test_split, 'r').readlines()))
                except:
                    files_test = []
                    raise_warning("No test files.")

                # Choose valid
                # if "ADNI" in raw_data_dir:
                    # files_train = valid_ids_ADNI_CSR(files_train)
                    # files_val = valid_ids_ADNI_CSR(files_val)
                    # files_test = valid_ids_ADNI_CSR(files_test)
                # elif "OASIS" in raw_data_dir:
                    # files_train = valid_ids_OASIS(files_train)
                    # files_val = valid_ids_OASIS(files_val)
                    # files_test = valid_ids_OASIS(files_test)
                # else:
                    # raise NotImplementedError()
            else:
                raise TypeError("Wrong type of parameter 'fixed_split'."
                                f" Got {type(fixed_split)} but should be"
                                "'Sequence' or 'dict'")
        else: # Random split
            # Available files
            all_files = valid_ids(raw_data_dir) # Remove invalid

            # Shuffle with seed
            random.Random(dataset_seed).shuffle(all_files)

            # Split according to proportions
            assert np.sum(dataset_split_proportions) == 100, "Splits need to sum to 100."
            indices_train = slice(0, dataset_split_proportions[0] * len(all_files) // 100)
            indices_val = slice(indices_train.stop,
                                indices_train.stop +\
                                    (dataset_split_proportions[1] * len(all_files) // 100))
            indices_test = slice(indices_val.stop, len(all_files))

            files_train = all_files[indices_train]
            files_val = all_files[indices_val]
            files_test = all_files[indices_test]

        if overfit:
            # Consider the same splits for train validation and test
            files_train = files_train[:overfit]
            files_val = files_train[:overfit]
            files_test = files_train[:overfit]

        # Save ids to file
        DatasetHandler.save_ids(files_train, files_val, files_test, save_dir)

        assert (len(set(files_train) & set(files_val) & set(files_test)) == 0
                or overfit),\
                "Train, validation, and test set should not intersect!"

        # Create train, validation, and test datasets
        if 'train' in load_only:
            train_dataset = cls(
                ids=files_train,
                mode=DataModes.TRAIN,
                raw_data_dir=raw_data_dir,
                augment=augment_train,
                **kwargs
            )
        else:
            train_dataset = None
        if 'validation' in load_only:
            val_dataset = cls(
                ids=files_val,
                mode=DataModes.VALIDATION,
                raw_data_dir=raw_data_dir,
                augment=False,
                **kwargs
            )
        else:
            val_dataset = None
        if 'test' in load_only:
            test_dataset = cls(
                ids=files_test,
                mode=DataModes.TEST,
                raw_data_dir=raw_data_dir,
                augment=False,
                **kwargs
            )
        else:
            test_dataset = None

        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        return len(self._files)

    @measure_time
    def get_item_from_index(
        self,
        index: int
    ):
        """
        One data item for training.
        """
        # Raw data
        img = self.images[index]
        voxel_label = self.voxel_labels[index]
        mesh_label = list(self._get_mesh_target_no_faces(index))

        # Potentially augment
        if self._augment:
            target_points, target_normals = mesh_label[0], mesh_label[1]
            assert all(
                (np.array(img.shape) - np.array(self.patch_size)) % 2 == 0
            ), "Padding must be symmetric for augmentation."

            # Mesh coordinates --> image coordinates
            target_points = unnormalize_vertices_per_max_dim(
                target_points.view(-1, 3), self.patch_size
            ).view(self.n_m_classes, -1, 3)
            # Augment
            (img,
             voxel_label,
             target_points,
             target_normals) = self.augment_data(
                 img.numpy(),
                 voxel_label.numpy(),
                 target_points,
                 target_normals
             )
            # Image coordinates --> mesh coordinates
            target_points = normalize_vertices_per_max_dim(
                target_points.view(-1, 3), self.patch_size
            ).view(self.n_m_classes, -1, 3)

            img = torch.from_numpy(img)
            voxel_label = torch.from_numpy(voxel_label)

            # Insert back to mesh label
            mesh_label[0], mesh_label[1] = target_points, target_normals

        # Channel dimension
        img = img[None]

        logging.getLogger(ExecModes.TRAIN.name).debug(
            "Dataset file %s", self._files[index]
        )

        return (img,
                voxel_label,
                *mesh_label)

    def _get_full_mesh_target(self, index):
        verts = self.mesh_labels[index].verts_padded()
        normals = self.mesh_labels[index].verts_normals_padded()
        faces = self.mesh_labels[index].faces_padded()
        features = self.mesh_labels[index].verts_features_padded()
        mesh = Mesh(verts, faces, normals, features)

        return mesh

    def _get_mesh_target_no_faces(self, index):
        return [target[index] for target in self.mesh_targets]

    def get_item_and_mesh_from_index(self, index):
        """ Get image, segmentation ground truth and full reference mesh. In
        contrast to 'get_item_from_index', this function is usually used for
        evaluation where the full mesh is needed in contrast to training where
        different information might be required, e.g., no faces.
        """
        img = self.images[index][None]
        voxel_label = self.voxel_labels[index]
        thickness = self.get_thickness_from_index(index)
        mesh_label = self._get_full_mesh_target(index)
        # Mesh features given by thickness
        mesh_label.features = thickness
        trans_affine_label = self.trans_affine[index]

        return {
            "img": img,
            "voxel_label": voxel_label,
            "mesh_label": mesh_label,
            "trans_affine_label": trans_affine_label
        }

    def get_thickness_from_index(self, index: int):
        """ Return per-vertex thickness of the ith dataset element if possible."""
        return self.thickness_per_vertex[index] if (
            self.thickness_per_vertex is not None) else None

    def _read_voxelized_meshes(self):
        """ Read voxelized meshes stored in nifity files and set voxel classes
        as specified by _get_seg_and_mesh_label_names. """
        data = []
        # Iterate over sample ids
        for i, sample_id in tqdm(enumerate(self._files), position=0, leave=True,
                       desc="Loading voxelized meshes..."):
            voxelized_mesh_label = None
            # Iterate over voxel classes
            for group_id, voxel_group in enumerate(
                self.voxelized_mesh_label_names, 1
            ):
                # Iterate over files
                for vmln in voxel_group:
                    vm_file = os.path.join(
                        self._raw_data_dir, sample_id, vmln + ".nii.gz"
                    )
                    img = nib.load(vm_file).get_fdata().astype(int) * group_id
                    if voxelized_mesh_label is None:
                        voxelized_mesh_label = img
                    else:
                        np.putmask(voxelized_mesh_label, img>0, img)

            # Correct patch size
            voxelized_mesh_label, _ = self._get_single_patch(
                voxelized_mesh_label, is_label=True
            )
            data.append(voxelized_mesh_label)

        return data

    def _load_data3D_raw(self, filename: str):
        data = []
        for fn in tqdm(self._files, position=0, leave=True,
                       desc="Loading images..."):
            img = nib.load(os.path.join(self._raw_data_dir, fn, filename))

            d = img.get_fdata()
            data.append(d)

        return data

    def _load_data3D_and_transform(self, filename: str, is_label: bool):
        """ Load data and transform to correct patch size. """
        data = []
        transformations = []
        for fn in tqdm(self._files, position=0, leave=True,
                       desc="Loading images..."):
            img = nib.load(os.path.join(self._raw_data_dir, fn, filename))

            img_data = img.get_fdata()
            img_data, trans_affine = self._get_single_patch(img_data, is_label)
            data.append(img_data)
            transformations.append(trans_affine)

        return data, transformations

    def _get_single_patch(self, img, is_label):
        """ Extract a single patch from an image. """

        assert (tuple(self._patch_origin) == (0,0,0)
                or self.patch_mode != "no"),\
                "If patch mode is 'no', patch origin should be (0,0,0)"

        # Limits for patch selection
        lower_limit = np.array(self._patch_origin, dtype=int)
        upper_limit = np.array(self._patch_origin, dtype=int) +\
                np.array(self.select_patch_size, dtype=int)

        assert img.shape == (182, 218, 182),\
                "All images should have this shape."
        # Select patch from whole image
        img_patch, trans_affine_1 = img_with_patch_size(
            img, self.select_patch_size, is_label=is_label, mode='crop',
            crop_at=(lower_limit + upper_limit) // 2
        )
        # Zoom to certain size
        if self.patch_size != self.select_patch_size:
            img_patch, trans_affine_2 = img_with_patch_size(
                img_patch, self.patch_size, is_label=is_label, mode='interpolate'
            )
        else:
            trans_affine_2 = np.eye(self.ndims + 1) # Identity

        trans_affine = trans_affine_2 @ trans_affine_1

        return img_patch, trans_affine

    def _create_voxel_labels_from_meshes(self, mesh_labels):
        """ Return the voxelized meshes as 3D voxel labels. Here, individual
        structures are not distinguished. """
        data = []
        for m in tqdm(mesh_labels, position=0, leave=True,
                      desc="Voxelize meshes..."):
            vertices = m.verts_padded()
            faces = m.faces_padded()
            voxel_label = voxelize_mesh(
                vertices, faces, self.patch_size, self.n_m_classes
            ).sum(0).bool().long() # Treat as one class

            data.append(voxel_label)

        return data

    def _load_dataMesh_raw(self, meshnames):
        """ Load mesh such that it's registered to the respective 3D image. If
        a mesh cannot be found, a dummy is inserted if it is a test split.
        """
        data = []
        assert len(self.trans_affine) == 0, "Should be empty."
        for fn in tqdm(self._files, position=0, leave=True,
                       desc="Loading meshes..."):
            # Voxel coords
            orig = nib.load(os.path.join(self._raw_data_dir, fn,
                                         self.img_filename))
            vox2world_affine = orig.affine
            world2vox_affine = np.linalg.inv(vox2world_affine)
            self.trans_affine.append(world2vox_affine)
            file_vertices = []
            file_faces = []
            for mn in meshnames:
                try:
                    mesh = trimesh.load_mesh(
                        os.path.join(self._raw_data_dir, fn, mn + ".stl")
                    )
                except ValueError:
                    try:
                        mesh = trimesh.load_mesh(
                            os.path.join(self._raw_data_dir, fn, mn + ".ply"),
                            process=False
                        )
                    except Exception as e:
                        # Insert a dummy if dataset is test split
                        if self._mode != DataModes.TEST:
                            raise e
                        mesh = trimesh.creation.icosahedron()
                        raise_warning(
                            f"No mesh for file {fn}/{mn},"
                            " inserting dummy."
                        )
                # World --> voxel coordinates
                voxel_verts, voxel_faces = transform_mesh_affine(
                    mesh.vertices, mesh.faces, world2vox_affine
                )
                # Store min/max number of vertices
                self.n_max_vertices = np.maximum(
                    voxel_verts.shape[0], self.n_max_vertices) if (
                self.n_max_vertices is not None) else voxel_verts.shape[0]
                self.n_min_vertices = np.minimum(
                    voxel_verts.shape[0], self.n_min_vertices) if (
                self.n_min_vertices is not None) else voxel_verts.shape[0]
                # Add to structures of file
                file_vertices.append(torch.from_numpy(voxel_verts))
                file_faces.append(torch.from_numpy(voxel_faces))

            # Treat as a batch of meshes
            mesh = Meshes(file_vertices, file_faces)
            data.append(mesh)

        return data

    def _load_mc_dataMesh(self, voxel_labels):
        """ Create ground truth meshes from voxel labels."""
        data = []
        for vl in voxel_labels:
            assert tuple(vl.shape) == tuple(self.patch_size),\
                    "Voxel label should be of correct size."
            mc_mesh = create_mesh_from_voxels(
                vl, mc_step_size=self._mc_step_size,
            ).to_pytorch3d_Meshes()
            data.append(mc_mesh)

        return data

    def create_training_targets(self, remove_meshes=False):
        """ Sample surface points, normals and curvaturs from meshes.
        """
        points, normals, curvs = [], [], []

        # Iterate over mesh labels
        for i, m in tqdm(
            enumerate(self.mesh_labels),
            leave=True,
            position=0,
            desc="Get point labels from meshes..."
        ):
            # Create meshes with curvature
            curv_list = [curv_from_cotcurv_laplacian(v, f).unsqueeze(-1)
                         for v, f in zip(m.verts_list(), m.faces_list())]
            m_new = Meshes(
                m.verts_list(),
                m.faces_list(),
                verts_features=curv_list
            )
            p, n, c = sample_points_from_meshes(
                m_new,
                self.n_ref_points_per_structure,
                return_normals=True,
                interpolate_features='barycentric',
            )

            points.append(p)
            normals.append(n)
            curvs.append(c)

            # Remove meshes to save memory
            if remove_meshes:
                self.mesh_labels[i] = None
            else:
                self.mesh_labels[i] = m_new

        self.mesh_targets = (points, normals, curvs)

        return self.mesh_targets

    def augment_data(self, img, label, coordinates, normals):
        assert self._augment, "No augmentation in this dataset."
        return flip_img(img, label, coordinates, normals)

    def check_augmentation_normals(self):
        """ Assert correctness of the transformation of normals during
        augmentation.
        """
        py3d_mesh = self.mesh_labels[0]
        img_f, label_f, coo_f, normals_f = self.augment_data(
            self.images[0].numpy(), self.voxel_labels[0].numpy(),
            py3d_mesh.verts_padded(), py3d_mesh.verts_normals_padded()
        )
        py3d_mesh_aug = Meshes(coo_f, py3d_mesh.faces_padded())
        # Assert up to sign of direction
        assert (
            torch.allclose(normals_f, py3d_mesh_aug.verts_normals_padded(),
                           atol=7e-03)
            or torch.allclose(-normals_f, py3d_mesh_aug.verts_normals_padded(),
                             atol=7e-03)
        )

    def _get_morph_label(self, morphology, subfolder="surf"):
        """ Load per-vertex labels, e.g., thickness values, from a freesurfer
        morphology (curv) file in the preprocessed data (FS) directory for each
        dataset sample.
        :param morphology: The morphology to load, e.g., 'thickness'
        :param subfolder: The subfolder of a the sample folder where the
        morphology file could be found.
        :return: List of len(self) containing per-vertex morphology values for
        each sample.
        """
        if self._preprocessed_data_dir is None:
            return None

        morph_labels = []
        for fn in self._files:
            file_dir = os.path.join(
                self._preprocessed_data_dir, fn, subfolder
            )
            file_labels = []
            n_max = 0
            for mn in self.mesh_label_names:
                # Filenames have form 'lh_white_reduced_0.x.thickness'
                morph_fn = mn + "." + morphology
                morph_fn = os.path.join(file_dir, morph_fn)
                try:
                    morph_label = nib.freesurfer.io.read_morph_data(morph_fn)
                except Exception as e:
                    # Insert dummy if file could not
                    # be found
                    raise_warning(
                        f"File {morph_fn} could not be found, inserting dummy."
                    )
                    morph_label = np.zeros(self.mesh_labels[
                        self._files.index(fn)
                    ].verts_padded()[self.mesh_label_names.index(mn)].shape[0])

                file_labels.append(
                    torch.from_numpy(morph_label.astype(np.float32))
                )
                if morph_label.shape[0] > n_max:
                    n_max = morph_label.shape[0]

            # Pad values with 0
            file_labels_padded = []
            for fl in file_labels:
                file_labels_padded.append(
                    F.pad(fl, (0, n_max - fl.shape[0]))
                )

            morph_labels.append(torch.stack(file_labels_padded))

        return morph_labels


class CortexParcellationDataset(CortexDataset):
    """ CortexParcellationDataset extends CortexDataset to parcellation labels.
    """
    def __init__(
        self,
        ids: Sequence,
        mode: DataModes,
        raw_data_dir: str,
        patch_size,
        n_ref_points_per_structure: int,
        **kwargs
    ):

        super().__init__(
            ids,
            mode,
            raw_data_dir,
            patch_size,
            n_ref_points_per_structure,
            **kwargs
        )

        # Load parcellation labels
        (self.mesh_parc_labels,
         self.parc_colors,
         self.parc_info) = self.load_vertex_parc_labels()

    def create_training_targets(self, remove_meshes=False):
        """ Sample surface points, normals, curvatures and point classes
        from meshes.
        """
        point_classes = []
        # We need to sample twice from the meshes with the same seed, once for
        # curvature and once for point classes
        seed = torch.rand(1).item()
        torch.random.manual_seed(seed)
        points, normals, curvs = super().create_training_targets()

        # !Reset seed
        torch.random.manual_seed(seed)
        # Iterate over mesh labels
        for i, m in tqdm(
            enumerate(self.mesh_labels),
            leave=True,
            position=0,
            desc="Get point labels from meshes..."
        ):
            # Create meshes with parcellation
            m_new = Meshes(
                m.verts_list(),
                m.faces_list(),
                verts_features=self.mesh_parc_labels[i]
            )
            p, n, p_class = sample_points_from_meshes(
                m_new,
                self.n_ref_points_per_structure,
                return_normals=True,
                interpolate_features='nearest',
            )
            # The same points should have been sampled again
            assert torch.allclose(p, points[i])
            assert torch.allclose(n, normals[i])

            # Remove meshes to save memory
            if remove_meshes:
                self.mesh_labels[i] = None
            else:
                self.mesh_labels[i] = m_new

            point_classes.append(p_class)

        # Mesh targets for training
        self.mesh_targets = (points, normals, curvs, point_classes)

        return self.mesh_targets

    def load_vertex_parc_labels(self):
        """ Load labels (classes) for each vertex.
        """
        vertex_labels = []
        label_info = None
        label_colors = None
        V_max = 0

        # Iterate over subjects
        for fn in tqdm(
            self._files,
            leave=True,
            position=0,
            desc="Loading parcellation labels..."
        ):
            file_dir = os.path.join(
                self._raw_data_dir, fn
            )
            file_labels = []
            V_max = 0
            for mn in self.mesh_label_names:
                # Filenames have the form
                # 'lh_white_reduced.aparc.DKTatlas40.annot'
                # except for Mindboggle which has manual annotations
                label_fn = re.sub(r"_0\..", "", mn)
                if 'Mindboggle' in self._raw_data_dir:
                    label_fn += ".labels.DKT31.manual.annot"
                else:
                    label_fn += ".aparc.DKTatlas40.annot"
                label_fn = os.path.join(file_dir, label_fn)
                (label,
                 label_colors,
                 label_info) = nib.freesurfer.io.read_annot(label_fn)

                # Combine -1 & 0 into one class
                label[label < 0] = 0

                file_labels.append(
                    torch.from_numpy(label).unsqueeze(-1)
                )
                if label.shape[0] > V_max:
                    V_max = label.shape[0]

            vertex_labels.append(file_labels)

        return vertex_labels, label_colors, label_info

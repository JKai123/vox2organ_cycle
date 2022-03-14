
""" Cortex dataset handler """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import random
import logging
import warnings
import collections.abc as abc
from typing import Union, Sequence

import torch
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


class ImageAndMeshDataset(DatasetHandler):
    """ Base class for dataset handlers consisting of images, meshes, and
    segmentations.

    It loads all data specified by 'ids' directly into memory. The
    corresponding raw directory should contain a folder for each ID containing
    image and mesh data.

    :param list ids: The ids of the files the dataset split should contain,
    for example:
        ['1000_3', '1001_3',...]
    :param DataModes datamode: TRAIN, VALIDATION, or TEST
    :param str raw_data_dir: The raw base folder, contains folders
    corresponding to sample ids
    :param augment: Use image augmentation during training if 'True'
    :param patch_size: The patch size of the images, e.g. (256, 256, 256)
    :param n_ref_points_per_structure: The number of ground truth points
    per 3D structure.
    :param image_file_name: The name of the input images to load, should be
    equal for each ID.
    :param seg_file_name: The names of voxel mask segmentation files. If
    not specified, voxelized meshes will be used as segmentation ground truth.
    :param mesh_file_names: The names of the meshes to load.
    :param voxelized_mesh_file_names: Names of voxelized mesh files (optional).
    If not specified but needed, meshes are voxelized during preprocessing.
    :param patch_mode: "single-patch" or "no"
    :param patch_origin: The anker of an extracted patch, only has an effect if
    patch_mode is True.
    :param select_patch_size: The size of the cut-out patches. Can be different
    to patch_size, e.g., if extracted patches should be resized after
    extraction.
    :param seg_ground_truth: Either 'voxelized_meshes' or 'voxel_seg'
    :param check_dir: An output dir where data is stored that should be
    checked.
    :param sanity_check_data: Perform a sanity check regarding coherence of
    voxel segmentation and voxelized meshes.
    :param remove_meshes: If set to true, meshes will be removed after first
    creation of training targets. This is saves a lot of memory but disables
    the option of resampling the training targets.
    """

    # Generic names for voxel labels and mesh labels ignoring background
    voxel_label_names = (("foreground"), )
    mesh_label_names = ("foreground",)

    def __init__(
        self,
        ids: Sequence,
        mode: DataModes,
        raw_data_dir: str,
        patch_size,
        n_ref_points_per_structure: int,
        image_file_name: str,
        mesh_file_names: str,
        seg_file_name: Sequence[str]=None,
        voxelized_mesh_file_names: Sequence[str]=None,
        augment: bool=False,
        patch_mode: str="no",
        patch_origin=(0,0,0),
        select_patch_size=None,
        seg_ground_truth='voxelized_meshes',
        check_dir='../to_check',
        sanity_check_data=True,
        remove_meshes=False,
        **kwargs
    ):
        super().__init__(ids, mode)

        if patch_mode not in ("single-patch", "no"):
            raise ValueError(f"Unknown patch mode {patch_mode}")
        if seg_ground_truth not in ('voxelized_meshes', 'voxel_seg'):
            raise ValueError(f"Unknown seg_ground_truth {seg_ground_truth}")
        if len(mesh_file_names) != len(self.mesh_label_names):
            raise ValueError(
                "There should be one mesh file for each mesh label."
            )

        self._check_dir = check_dir
        self._raw_data_dir = raw_data_dir
        self._augment = augment
        self._patch_origin = patch_origin
        self.trans_affine = []
        self.ndims = len(patch_size)
        self.patch_mode = patch_mode
        self.patch_size = tuple(patch_size)
        # If not specified, select_patch_size is equal to patch_size
        self.select_patch_size = select_patch_size if (
            select_patch_size is not None
        ) else patch_size
        self.n_m_classes = len(mesh_file_names)
        self.seg_ground_truth = seg_ground_truth
        self.n_ref_points_per_structure = n_ref_points_per_structure
        self.n_min_vertices, self.n_max_vertices = None, None
        self.n_v_classes = len(self.voxel_label_names) + 1 # +1 for background

        # Sanity checks to make sure data is transformed correctly
        self.sanity_checks = sanity_check_data

        # Load/prepare data
        self._prepare_data_3D(
            image_file_name,
            seg_file_name,
            mesh_file_names,
            voxelized_mesh_file_names
        )

        # NORMALIZE images
        for i, img in enumerate(self.images):
            self.images[i] = normalize_min_max(img)

        assert self.__len__() == len(self.images)
        assert self.__len__() == len(self.voxel_labels)
        assert self.__len__() == len(self.mesh_labels)
        assert self.__len__() == len(self.trans_affine)

        if self._augment:
            self.check_augmentation_normals()

    def _prepare_data_3D(
        self,
        image_file_name,
        seg_file_name,
        mesh_file_names,
        voxelized_mesh_file_names
    ):
        """ Load 3D data """

        # Image data
        self.images, img_transforms = self._load_data3D_and_transform(
            image_file_name, is_label=False
        )
        self.voxel_labels = None
        self.voxelized_meshes = None

        # Voxel labels
        if self.sanity_checks or self.seg_ground_truth == 'voxel_seg':
            self.voxel_labels, _ = self._load_data3D_and_transform(
                seg_file_name, is_label=True
            )
            # Combine labels as specified by groups (see
            # _get_seg_and_mesh_file_names)
            combine = lambda x: torch.sum(
                torch.stack(
                    [DatasetHandler.combine_labels(x, self.seg_ids(group), val)
                    for val, group in enumerate(self.voxel_label_names, 1)]
                ),
                dim=0
            )
            self.voxel_labels = list(map(combine, self.voxel_labels))

        # Voxelized meshes
        if self.sanity_checks or self.seg_ground_truth == 'voxelized_meshes':
            try:
                self.voxelized_meshes = self._read_voxelized_meshes(
                    voxelized_mesh_file_names
                )
            except (FileNotFoundError, TypeError):
                self.voxelized_meshes = None # Compute later

        # Meshes
        self.mesh_labels = self._load_dataMesh_raw(meshnames=mesh_file_names)
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
                            self._check_dir, f"diff_mesh_voxel_label_{out_fn}.png"
                        )
                    )
                    raise_warning(
                        f"Small IoU ({iou}) of voxel label and voxelized mesh"
                        f" label, check files at {self._check_dir}"
                    )
                    img = nib.Nifti1Image(vl.squeeze().cpu().numpy(), np.eye(4))
                    nib.save(
                        img,
                        os.path.join(
                            self._check_dir,
                            "data_voxel_label.nii.gz"
                        )
                    )
                    img = nib.Nifti1Image(vm.squeeze().cpu().numpy(), np.eye(4))
                    nib.save(
                        img,
                        os.path.join(
                            self._check_dir,
                            "data_mesh_label.nii.gz"
                        )
                    )

        # Use voxelized meshes as voxel ground truth
        if self.seg_ground_truth == 'voxelized_meshes':
            self.voxel_labels = self.voxelized_meshes


    def seg_ids(self, names):
        """ Map voxel classes to IDs.
        """
        raise NotImplementedError()


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
    def split(
        cls,
        raw_data_dir,
        save_dir,
        augment_train=False,
        dataset_seed=0,
        dataset_split_proportions=None,
        fixed_split: Union[dict, Sequence]=None,
        overfit=False,
        load_only=('train', 'validation', 'test'),
        **kwargs
    ):
        """ Create train, validation, and test split of data"

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
            elif isinstance(fixed_split, abc.Sequence):
                assert len(fixed_split) == 3,\
                        "Should contain one file per split"
                convert = lambda x: x[:-1] # 'x\n' --> 'x'
                train_split = os.path.join(raw_data_dir, fixed_split[0])
                try:
                    files_train = list(map(convert, open(train_split, 'r').readlines()))
                except FileNotFoundError:
                    files_train = []
                    raise_warning("No training files.")
                val_split = os.path.join(raw_data_dir, fixed_split[1])
                try:
                    files_val = list(map(convert, open(val_split, 'r').readlines()))
                except FileNotFoundError:
                    files_val = []
                    raise_warning("No validation files.")
                test_split = os.path.join(raw_data_dir, fixed_split[2])
                try:
                    files_test = list(map(convert, open(test_split, 'r').readlines()))
                except FileNotFoundError:
                    files_test = []
                    raise_warning("No test files.")

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
        mesh_label = self._get_full_mesh_target(index)
        trans_affine_label = self.trans_affine[index]

        return {
            "img": img,
            "voxel_label": voxel_label,
            "mesh_label": mesh_label,
            "trans_affine_label": trans_affine_label
        }

    def _read_voxelized_meshes(self, voxelized_mesh_file_names):
        """ Read voxelized meshes stored in nifity files and set voxel classes
        according to groups (similar to voxel_label_names). """
        data = []
        # Iterate over sample ids
        for _, sample_id in tqdm(
            enumerate(self._files),
            position=0,
            leave=True,
            desc="Loading voxelized meshes..."
        ):
            voxelized_mesh_label = None
            # Iterate over voxel classes
            for group_id, voxel_group in enumerate(
                voxelized_mesh_file_names, 1
            ):
                # Iterate over files
                for vmln in voxel_group:
                    vm_file = os.path.join(
                        self._raw_data_dir, sample_id, vmln + ".nii.gz"
                    )
                    img = nib.load(vm_file).get_fdata()
                    img, _ = self._get_single_patch(img, is_label=True)
                    # Assign a group id
                    img = img.bool().long() * group_id
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
                                         self.image_file_name))
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

    def create_training_targets(self, remove_meshes=False):
        """ Sample surface points, normals and curvaturs from meshes.
        """
        if self.mesh_labels[0] is None:
            warnings.warn(
                "Mesh labels do not exist (anymore) and no new training"
                " targets can be created."
            )
            return self.mesh_labels

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

        # Placeholder for point labels
        point_classes = [torch.zeros_like(curvs[0])] * len(curvs)

        self.mesh_targets = (points, normals, curvs, point_classes)

        return self.mesh_targets

    def augment_data(self, img, label, coordinates, normals):
        assert self._augment, "No augmentation in this dataset."
        return flip_img(img, label, coordinates, normals)

    def check_augmentation_normals(self):
        """ Assert correctness of the transformation of normals during
        augmentation.
        """
        py3d_mesh = self.mesh_labels[0]
        _, _, coo_f, normals_f = self.augment_data(
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

    def check_data(self):
        """ Check if voxel and mesh data is consistent """
        for i in tqdm(range(len(self)),
                      desc="Checking IoU of voxel and mesh labels"):
            data = self.get_item_and_mesh_from_index(i)
            voxel_label = data[1]
            mesh = data[2]
            shape = voxel_label.shape
            vertices, faces = mesh.vertices, mesh.faces
            faces = faces.view(self.n_m_classes, -1, 3)
            voxelized_mesh = voxelize_mesh(
                vertices, faces, shape, self.n_m_classes
            ).cuda().sum(0).bool().long() # Treat as one class

            j_vox = Jaccard(voxel_label.cuda(), voxelized_mesh.cuda(), 2)

            if j_vox < 0.85:
                img = nib.Nifti1Image(voxel_label.squeeze().cpu().numpy(), np.eye(4))
                nib.save(
                    img,
                    os.path.join(
                        self._check_dir,
                        "data_voxel_label" + self._files[i] + ".nii.gz"
                    )
                )
                img = nib.Nifti1Image(voxelized_mesh.squeeze().cpu().numpy(), np.eye(4))
                nib.save(
                    img,
                    os.path.join(
                        self._check_dir,
                        "data_mesh_label" + self._files[i] + ".nii.gz"
                    )
                )
                print(f"[Warning] Small IoU ({j_vox}) of voxel label and"
                      f" voxelized mesh label, check files at {self._check_dir}")

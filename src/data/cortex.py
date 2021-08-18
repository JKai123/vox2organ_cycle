
""" Cortex dataset handler """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import random
import logging
from copy import deepcopy

import torch
import torch.nn.functional as F
import numpy as np
import nibabel as nib
import trimesh
from trimesh import Trimesh
from trimesh.scene.scene import Scene
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from utils.visualization import show_difference
from utils.eval_metrics import Jaccard
from utils.modes import DataModes, ExecModes
from utils.logging import measure_time
from utils.mesh import Mesh, curv_from_cotcurv_laplacian
from utils.ico_template import generate_sphere_template
from utils.utils import (
    choose_n_random_points,
    voxelize_mesh,
    voxelize_contour,
    create_mesh_from_voxels,
    create_mesh_from_pixels,
    mirror_mesh_at_plane,
    normalize_min_max,
)
from utils.coordinate_transform import (
    unnormalize_vertices_per_max_dim,
    normalize_vertices_per_max_dim,
)
from utils.sample_points_from_contours import sample_points_from_contours
from data.dataset import (
    DatasetHandler,
    flip_img,
    img_with_patch_size,
)
from data.cortex_labels import combine_labels

class Cortex(DatasetHandler):
    """ Cortex dataset

    It loads all data specified by 'ids' directly into memory.

    :param list ids: The ids of the files the dataset split should contain, example:
        ['1000_3', '1001_3',...]
    :param DataModes datamode: TRAIN, VALIDATION, or TEST
    :param str raw_data_dir: The raw base folder, contains folders
    corresponding to sample ids
    :param augment: Use image augmentation during training if 'True'
    :param patch_size: The patch size of the images, e.g. (256, 256, 256)
    :param mesh_target_type: 'mesh' or 'pointcloud'
    :param n_ref_points_per_structure: The number of ground truth points
    per 3D structure.
    :param structure_type: Either 'white_matter' or 'cerebral_cortex'
    :param patch_mode: "single-patch", "multi-patch", or "no"
    :param patch_origin: The anker of an extracted patch, only has an effect if
    patch_mode is True.
    :param select_patch_size: The size of the cut out patches. Can be different
    to patch_size, e.g., if extracted patches should be resized after
    extraction.
    :param mc_step_size: The marching cubes step size.
    :param reduced_freesurfer: Use a freesurfer mesh with reduced number of
    vertices as mesh ground truth, e.g., 0.3. This has only an effect if patch_mode='no'.
    :param mesh_type: 'freesurfer' or 'marching cubes'
    :param: provide_curvatures: Whether curvatures should be part of the items.
    As a consequence, the ground truth can only consist of vertices and not of
    sampled surface points.
    """

    img_filename = "mri.nii.gz"
    label_filename = "aseg.nii.gz"

    def __init__(self, ids: list, mode: DataModes, raw_data_dir: str,
                 augment: bool, patch_size, mesh_target_type: str,
                 n_ref_points_per_structure: int, structure_type: str,
                 patch_mode: str="no", patch_origin=(0,0,0), select_patch_size=None,
                 reduced_freesurfer: int=None, mesh_type='marching cubes',
                 provide_curvatures=False,
                 **kwargs):
        super().__init__(ids, mode)

        assert patch_mode in ("multi-patch", "single-patch", "no"),\
                "Unknown patch mode."
        assert mesh_type in ("marching cubes", "freesurfer"),\
                "Unknown mesh type"

        if structure_type == "cerebral_cortex":
            if patch_mode=="single-patch":
                seg_label_names = ("right_cerebral_cortex",)
                mesh_label_names = ("rh_pial",)
            elif patch_mode == "multi-patch":
                raise NotImplementedError()
            else: # not patch mode
                raise NotImplementedError()
        elif structure_type == "white_matter":
            if patch_mode=="single-patch":
                seg_label_names = ("right_white_matter",)
                mesh_label_names = ("rh_white",)
            elif patch_mode == "multi-patch":
                seg_label_names = ("left_white_matter", "right_white_matter")
                mesh_label_names = ("lh_white", "rh_white")
            else: # not patch mode
                if len(patch_size) == 3: # 3D
                    seg_label_names = ("left_white_matter", "right_white_matter")
                    mesh_label_names = ("lh_white", "rh_white")
                elif len(patch_size) == 2: # 2D
                    seg_label_names = ("right_white_matter",)
                    mesh_label_names = ("rh_white",)
                else:
                    raise ValueError("Wrong dimensionality.")
        else:
            raise ValueError("Unknown structure type.")

        self.structure_type = structure_type
        self._raw_data_dir = raw_data_dir
        self._augment = augment
        self._mesh_target_type = mesh_target_type
        self._patch_origin = patch_origin
        self._mc_step_size = kwargs.get('mc_step_size', 1)
        self.ndims = len(patch_size)
        self.patch_mode = patch_mode
        self.patch_size = tuple(patch_size)
        self.select_patch_size = select_patch_size if (
            select_patch_size is not None) else patch_size
        self.n_m_classes = len(mesh_label_names)
        self.provide_curvatures = provide_curvatures
        assert self.n_m_classes == kwargs.get("n_m_classes",
                                              len(mesh_label_names)),\
                "Number of mesh classes incorrect."
        self.n_ref_points_per_structure = n_ref_points_per_structure
        self.mesh_label_names = mesh_label_names
        self.seg_label_names = seg_label_names
        self.n_structures = len(mesh_label_names)
        self.mesh_type = mesh_type
        self.centers, self.radii = None, None
        # Vertex labels are combined into one class (and background)
        self.n_v_classes = 2

        # Freesurfer meshes of desired resolution
        if reduced_freesurfer is not None:
            if reduced_freesurfer != 1.0:
                self.mesh_label_names = [
                    mn + "_reduced_" + str(reduced_freesurfer)
                    for mn in self.mesh_label_names
                ]

        if self.ndims == 3:
            self._prepare_data_3D()
        elif self.ndims == 2:
            self._prepare_data_2D()
        else:
            raise ValueError("Unknown number of dimensions ", self.ndims)

        # NORMALIZE images
        for i, img in enumerate(self.images):
            self.images[i] = normalize_min_max(img)

        # Point, normal, and potentially curvature labels
        self.point_labels,\
                self.normal_labels,\
                self.curvatures = self._load_ref_points()

        assert self.__len__() == len(self.images)
        assert self.__len__() == len(self.voxel_labels)
        assert self.__len__() == len(self.mesh_labels)
        assert self.__len__() == len(self.point_labels)
        assert self.__len__() == len(self.normal_labels)
        if self.provide_curvatures:
            assert self.__len__() == len(self.curvatures)

        if self._augment:
            self.check_augmentation_normals()

    def _prepare_data_2D(self):
        """ Load 2D data """

        # Check constraints
        assert self.patch_mode == "no",\
                "Patch mode not supported for 2D data."
        assert 'voxelized_mesh' not in self.seg_label_names,\
                "Voxelization of mesh not possible for 2D data."

        # Load images
        self.images = self._load_single_data2D(
            filename=self.img_filename, is_label=False
        )

        # Load voxel labels
        self.voxel_labels = self._load_single_data2D(
            filename=self.label_filename, is_label=True
        )
        if self.seg_label_names == "all":
            for vl in self.voxel_labels:
                vl[vl > 1] = 1
        else:
            self.voxel_labels = [
                combine_labels(l, self.seg_label_names) for l in self.voxel_labels
            ]

        # Marching squares mesh labels
        self.mesh_labels = self._load_ms_dataMesh()

        # Update voxel labels as we chose only the main region for the mesh and
        # ignore small 'artifact' regions
        self.voxel_labels = self._create_voxel_labels_from_contours()

    def _prepare_data_3D(self):
        """ Load 3D data """

        # Check constraints
        assert (self.patch_mode != "single-patch"
                or len(self.seg_label_names) == 1),\
                "Can only use one segmentation class in single-patch mode."
        assert (self.mesh_type == 'freesurfer'
                or 'voxelized_mesh' not in self.seg_label_names),\
                "Voxelized mesh as voxel ground truth requires Freesurfer meshes."

        # Raw data
        raw_imgs = self._load_data3D_raw(self.img_filename)
        raw_voxel_labels = self._load_data3D_raw(self.label_filename)
        raw_mesh_labels = self._load_dataMesh_raw(meshnames=self.mesh_label_names)

        # Select desired voxel labels
        if self.seg_label_names == "all":
            for vl in raw_voxel_labels:
                vl[vl > 1] = 1
        else:
            raw_voxel_labels = [
                combine_labels(l, self.seg_label_names) for l in raw_voxel_labels
            ]

        # Preprocess routine
        self.images,\
                self.voxel_labels,\
                self.mesh_labels = self.preprocess_data_3D(
                    raw_imgs, raw_voxel_labels, raw_mesh_labels
                )

        self.centers, self.radii = self._get_centers_and_radii(
            self.mesh_labels
        )

    def preprocess_data_3D(self, raw_imgs: list, raw_voxel_labels: list,
                           raw_mesh_labels: list):
        """ Preprocess routine according to dataset parameters. """

        ### Multi-patch mode

        if self.patch_mode == "multi-patch":
            # Image data; attention: file names change in multi-patch mode
            # (each image leads to multiple patches)
            processed_imgs,\
                    processed_voxel_labels,\
                    self._files = self._get_multi_patches(
                        raw_imgs, raw_voxel_labels
                    )
            # Mesh data: marching cubes meshes (it's not straightforward to
            # extract patches from freesurfer meshes)
            assert self.mesh_type == 'marching cubes',\
                    "Multi-patch dataset requires marching cubes meshes."
            processed_mesh_labels = self._load_mc_dataMesh(processed_voxel_labels)

            # In multi-patch mode, different mesh structures cannot be
            # distinguished anymore after creation of patches
            self.n_m_classes = 1

            return (processed_imgs,
                    processed_voxel_labels,
                    processed_mesh_labels)

        ### Single-patch or no patch mode

        # Image data
        processed_imgs,\
                processed_voxel_labels,\
                trans_affine = self._get_single_patches(
                    raw_imgs, raw_voxel_labels
                )

        # Mesh data
        if self.mesh_type == "freesurfer":
            # Transform meshes according to image transformations
            # (crops, resize) and normalize
            processed_mesh_labels = deepcopy(raw_mesh_labels)
            for m, t in zip(processed_mesh_labels, trans_affine):
                m.transform_vertices(t)
                m.vertices = normalize_vertices_per_max_dim(
                    m.vertices.view(-1, self.ndims), self.patch_size
                ).view(self.n_m_classes, -1, self.ndims)

            # Voxelize meshes
            voxelized_meshes = self._create_voxel_labels_from_meshes(
                processed_mesh_labels
            )
            # Assert conformity of voxel labels and voxelized meshes
            for i, (vl, vm) in enumerate(zip(processed_voxel_labels,
                                             voxelized_meshes)):
                iou = Jaccard(vl.cuda(), vm.cuda(), 2)
                if iou < 0.85:
                    show_difference(
                        vl,  vm,
                        f"../to_check/diff_mesh_voxel_label_{self._files[i]}.png"
                    )
                    print(f"[Warning] Small IoU ({iou}) of voxel label and"
                          " voxelized mesh label, check files at ../to_check/")

            return (processed_imgs,
                    voxelized_meshes,
                    processed_mesh_labels)

        # Marching cubes mesh labels
        processed_mesh_labels = self._load_mc_dataMesh(
            processed_voxel_labels
        )

        return (processed_imgs,
                processed_voxel_labels,
                processed_mesh_labels)

    def mean_area(self):
        """ Average surface area of meshes. """
        areas = []
        ndims = len(self.patch_size)
        for m in self.mesh_labels:
            m_unnorm = Mesh(unnormalize_vertices_per_max_dim(
                m.vertices.view(-1, ndims), self.patch_size),
                m.faces.view(-1, ndims)
            )
            areas.append(m_unnorm.to_trimesh().area)

        return np.mean(areas)

    def mean_edge_length(self):
        """ Average edge length in dataset.

        Code partly from pytorch3d.loss.mesh_edge_loss.
        """
        edge_lengths = []
        for m in self.mesh_labels:
            m_ = m.to_pytorch3d_Meshes()
            if self.ndims == 3:
                edges_packed = m_.edges_packed()
            else: # 2D
                edges_packed = m_.faces_packed()
            verts_packed = m_.verts_packed()

            verts_edges = verts_packed[edges_packed]
            v0, v1 = verts_edges.unbind(1)
            edge_lengths.append(
                (v0 - v1).norm(dim=1, p=2).mean().item()
            )

        return torch.tensor(edge_lengths).mean()

    def store_sphere_template(self, path):
        """ Template for dataset. This can be stored and later used during
        training.
        """
        if self.centers is not None and self.radii is not None:
            template = generate_sphere_template(self.centers,
                                                self.radii,
                                                level=6)
            template.export(path)
        else:
            raise RuntimeError("Centers and/or radii are unknown, template"
                               " cannnot be created. ")
        return path

    def store_index0_template(self, path, n_max_points=41000):
        """ This template is the structure of dataset element at index 0,
        potentially mirrored at the hemisphere plane. """
        template = Scene()
        if len(self.mesh_label_names) == 2:
            label_1, label_2 = self.mesh_label_names
        else:
            label_1 = self.mesh_labels[0]
            label_2 = None
        # Select mesh to generate the template from
        vertices = self.mesh_labels[0].vertices[0]
        faces = self.mesh_labels[0].faces[0]

        # Remove padded vertices
        valid_ids = np.unique(faces)
        valid_ids = valid_ids[valid_ids != -1]
        vertices_ = vertices[valid_ids]

        structure_1 = Trimesh(vertices_, faces, process=False)

        # Increase granularity until desired number of points is reached
        while structure_1.subdivide().vertices.shape[0] < n_max_points:
            structure_1 = structure_1.subdivide()

        assert structure_1.is_watertight, "Mesh template should be watertight."
        print(f"Template structure has {structure_1.vertices.shape[0]}"
              " vertices.")
        template.add_geometry(structure_1, geom_name=label_1)

        # Second structure = mirror of first structure
        if label_2 is not None:
            plane_normal = np.array(self.centers[label_2] - self.centers[label_1])
            plane_point = 0.5 * np.array((self.centers[label_1] +
                                          self.centers[label_2]))
            structure_2 = mirror_mesh_at_plane(structure_1, plane_normal,
                                              plane_point)
            template.add_geometry(structure_2, geom_name=label_2)

        template.export(path)

        return path

    def store_convex_cortex_template(self, path, n_min_points=40000, n_max_points=41000):
        """ This template is created as follows:
            1. Take the convex hull of one of the two structures and subdivide
            faces until the required number of vertices is large enough
            2. Mirror this mesh on the plane that separates the two cortex
            hemispheres
            3. Store both meshes together in one template
        """
        n_points = 0
        i = 0
        while n_points > n_max_points or n_points < n_min_points:
            if i >= len(self):
                print("Template with the desired number of vertices could not"
                      " be created. Aborting.")
                return None
            template = Scene()
            if len(self.mesh_label_names) == 2:
                label_1, label_2 = self.mesh_label_names
            else:
                label_1 = self.mesh_labels[0]
                label_2 = None
            # Select mesh to generate the template from
            vertices = self.mesh_labels[i].vertices[0]
            faces = self.mesh_labels[i].faces[0]

            # Remove padded vertices
            valid_ids = np.unique(faces)
            valid_ids = valid_ids[valid_ids != -1]
            vertices_ = vertices[valid_ids]

            # Get convex hull of the mesh label
            structure_1 = Trimesh(vertices_, faces, process=False).convex_hull

            # Increase granularity until desired number of points is reached
            while structure_1.subdivide().vertices.shape[0] < n_max_points:
                structure_1 = structure_1.subdivide()

            assert structure_1.is_watertight, "Mesh template should be watertight."
            n_points = structure_1.vertices.shape[0]
            print(f"Template structure {i} has {n_points} vertices.")
            i += 1
        template.add_geometry(structure_1, geom_name=label_1)

        # Second structure = mirror of first structure
        if label_2 is not None:
            plane_normal = np.array(self.centers[label_2] - self.centers[label_1])
            plane_point = 0.5 * np.array((self.centers[label_1] +
                                          self.centers[label_2]))
            structure_2 = mirror_mesh_at_plane(structure_1, plane_normal,
                                              plane_point)
            template.add_geometry(structure_2, geom_name=label_2)

        template.export(path)

        return path

    @staticmethod
    def split(raw_data_dir, dataset_seed, dataset_split_proportions,
              augment_train, save_dir, overfit=False, **kwargs):
        """ Create train, validation, and test split of the cortex data"

        :param str raw_data_dir: The raw base folder, contains a folder for each
        sample
        :param dataset_seed: A seed for the random splitting of the dataset.
        :param dataset_split_proportions: The proportions of the dataset
        splits, e.g. (80, 10, 10)
        :param augment_train: Augment training data.
        :param save_dir: A directory where the split ids can be saved.
        :param overfit: Create small datasets for overfitting if this parameter
        is > 0.
        :param kwargs: Dataset parameters.
        :return: (Train dataset, Validation dataset, Test dataset)
        """

        # Available files
        all_files = os.listdir(raw_data_dir)
        all_files = [fn for fn in all_files if (
            "meshes" not in fn and "unregistered" not in fn
        )] # Remove invalid

        # Shuffle with seed
        random.Random(dataset_seed).shuffle(all_files)

        # Split
        if overfit:
            # Consider the same splits for train validation and test
            indices_train = slice(0, overfit)
            indices_val = slice(0, overfit)
            indices_test = slice(0, overfit)
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
                               augment=augment_train,
                               **kwargs)
        val_dataset = Cortex(all_files[indices_val],
                             DataModes.VALIDATION,
                             raw_data_dir,
                             augment=False,
                             **kwargs)
        test_dataset = Cortex(all_files[indices_test],
                              DataModes.TEST,
                              raw_data_dir,
                              augment=False,
                              **kwargs)

        # Save ids to file
        DatasetHandler.save_ids(all_files[indices_train], all_files[indices_val],
                         all_files[indices_test], save_dir)

        return train_dataset, val_dataset, test_dataset

    def __len__(self):
        return len(self._files)

    @measure_time
    def get_item_from_index(self, index: int, mesh_target_type: str=None,
                            *args, **kwargs):
        """
        One data item has the form
        (image, voxel label, points, faces, normals)
        with types all of type torch.Tensor
        """
        # Use mesh target type of object if not specified
        if mesh_target_type is None:
            mesh_target_type = self._mesh_target_type

        # Raw data
        img = self.images[index]
        voxel_label = self.voxel_labels[index]
        (target_points,
         target_faces,
         target_normals,
         target_curvs) = self._get_mesh_target(index, mesh_target_type)

        # Potentially augment
        if self._augment and self.ndims == 3:
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


        # Channel dimension
        img = img[None]

        logging.getLogger(ExecModes.TRAIN.name).debug("Dataset file %s",
                                                      self._files[index])

        return (img,
                voxel_label,
                target_points,
                target_faces,
                target_normals,
                target_curvs)

    def _get_mesh_target(self, index, target_type):
        """ Ground truth points and optionally normals """
        if target_type == 'pointcloud':
            points = self.point_labels[index]
            normals = np.array([]) # Empty, not used
            faces = np.array([]) # Empty, not used
            curvs = np.array([]) # Empty, not used
        elif target_type == 'mesh':
            points = self.point_labels[index]
            normals = self.normal_labels[index]
            faces = np.array([]) # Empty, not used
            curvs = self.curvatures[index]\
                    if self.provide_curvatures else np.array([])
        elif target_type == 'full_mesh':
            points = self.mesh_labels[index].vertices
            normals = self.mesh_labels[index].normals
            faces = self.mesh_labels[index].faces
            mesh = self.mesh_labels[index].to_pytorch3d_Meshes()
            curvs = curv_from_cotcurv_laplacian(
                mesh.verts_packed(),
                mesh.faces_packed()
            ).view(self.n_m_classes, -1, 1)
        else:
            raise ValueError("Invalid mesh target type.")

        return points, faces, normals, curvs

    def get_item_and_mesh_from_index(self, index):
        """ Get image, segmentation ground truth and reference mesh"""
        (img,
         voxel_label,
         vertices,
         faces,
         normals, _) = self.get_item_from_index(
            index, mesh_target_type='full_mesh'
        )
        mesh_label = Mesh(vertices, faces, normals)

        return img, voxel_label, mesh_label

    def _load_data3D_raw(self, filename: str):
        data = []
        for fn in self._files:
            img = nib.load(os.path.join(self._raw_data_dir, fn, filename))

            d = img.get_fdata()
            data.append(d)

        return data

    def _load_single_data2D(self, filename: str, is_label: bool):
        """Load the image data """

        data_3D = self._load_data3D_raw(filename)
        mode = 'nearest' if is_label else 'bilinear'
        align_corners = None if is_label else False

        data_2D = []
        for img in data_3D:
            img_2D = F.interpolate(
                torch.from_numpy(img[img.shape[0]//13*4, :, :])[None][None],
                size=self.patch_size,
                mode=mode,
                align_corners=align_corners
            ).squeeze()
            if is_label:
                data_2D.append(img_2D.long())
            else:
                data_2D.append(img_2D.float())

        return data_2D

    def _get_single_patches(self, raw_imgs, raw_labels):
        """ Extract a single patch from an image. """

        img_patches, label_patches, transformations = [], [], []

        # Limits for patch selection
        lower_limit = np.array(self._patch_origin, dtype=int)
        upper_limit = np.array(self._patch_origin, dtype=int) +\
                np.array(self.select_patch_size, dtype=int)

        # Iterate over images and labels
        for img, label in zip(raw_imgs, raw_labels):
            # Select patch from whole image
            img_patch, trans_affine_1 = img_with_patch_size(
                img, self.select_patch_size, is_label=False, mode='crop',
                crop_at=(lower_limit + upper_limit) // 2
            )
            label_patch, _ = img_with_patch_size(
                label, self.select_patch_size, is_label=True, mode='crop',
                crop_at=(lower_limit + upper_limit) // 2
            )
            # Zoom to certain size
            if self.patch_size != self.select_patch_size:
                img_patch, trans_affine_2 = img_with_patch_size(
                    img_patch, self.patch_size, is_label=False, mode='interpolate'
                )
                label_patch, _ = img_with_patch_size(
                    label_patch, self.patch_size, is_label=True, mode='interpolate'
                )
            else:
                trans_affine_2 = np.eye(self.ndims + 1) # Identity

            img_patches.append(img_patch)
            label_patches.append(label_patch)
            transformations.append(trans_affine_2 @ trans_affine_1)

        return img_patches, label_patches, transformations

    def _get_multi_patches(self, raw_imgs: list, raw_labels: list):
        """ Load multiple patches from each raw image and corresponding voxel
        label. """
        data, labels, ids = [], [], []
        for img, lab, fn in zip(raw_imgs, raw_labels, self._files):
            img_patches, label_patches = self._create_patches(
                torch.from_numpy(img).float(), torch.from_numpy(lab).long()
            )
            for i in range(len(img_patches)):
                ids.append(fn + "_patch_" + str(i))
            data += img_patches
            labels += label_patches

        return data, labels, ids

    def _create_patches(self, img, label, pad_width=2):
        """ Create 3D patches from an image and the respective voxel label """
        ndims = self.ndims
        assert ndims == 3
        # The relative volume that should be occupied in the patch by non-zero
        # labels. If this cannot be fulfilled, a smaller threshold is selected, see
        # below.
        occ_volume_max = 0.5

        pad_width_all = (pad_width,) * 2 * self.ndims

        shape = torch.tensor(label.shape)
        patch_size = torch.tensor(self.patch_size)
        idxs = [
            [-1,
             slice(int(shape[1] / 2 - patch_size[1] / 2 + pad_width),
                   int(shape[1] / 2 + patch_size[1] / 2 - pad_width)),
             slice(int(shape[2] / 2 - patch_size[2] / 2 + pad_width),
                   int(shape[2] / 2 + patch_size[2] / 2 - pad_width))
            ],
            [slice(int(shape[0] / 4 - patch_size[0] / 2 + pad_width),
                   int(shape[0] / 4 + patch_size[0] / 2 - pad_width)),
             -1,
             slice(int(shape[2] / 2 - patch_size[2] / 2 + pad_width),
                   int(shape[2] / 2 + patch_size[2] / 2 - pad_width))
            ],
            [slice(int(3 * shape[0] / 4 - patch_size[0] / 2 + pad_width),
                   int(3 * shape[0] / 4 + patch_size[0] / 2 - pad_width)),
             slice(int(shape[1] / 2 - patch_size[1] / 2 + pad_width),
                   int(shape[1] / 2 + patch_size[1] / 2 - pad_width)),
             -1
            ]
        ]
        w = torch.ones(tuple(patch_size - 2*pad_width)).float()[None][None]
        img_patches = []
        label_patches = []

        # Iterate over dimensions
        for idx_i in idxs:
            idx = deepcopy(idx_i)
            d = idx_i.index(-1) # -1 indicates the dimension to conv over
            # -->
            idx[d] = slice(0, label.shape[d])
            tmp_label = label[tuple(idx)].clone().float()[None][None]
            tmp_label_conv = F.conv3d(tmp_label, w).squeeze()

            # Try to extract a patch with highest possible occupied volume
            occ_volume = occ_volume_max
            while occ_volume >= 0.1:
                try:
                    pos = torch.min(torch.nonzero(
                        tmp_label_conv >
                        occ_volume * np.prod(self.patch_size)
                    ))
                    break
                except RuntimeError: # No volume found --> reduce threshold
                    occ_volume -= 0.1

            if occ_volume < 0.1:
                raise RuntimeError("No patch could be found.")

            idx[d] = slice(
                pos + pad_width, pos + self.patch_size[d] - pad_width
            )
            img_patches.append(F.pad(img[tuple(idx)], pad_width_all))
            label_patches.append(F.pad(label[tuple(idx)], pad_width_all))

            # <--
            idx[d] = slice(0, label.shape[d])
            tmp_label = label[tuple(idx)].clone().float().flip(dims=[d])[None][None]
            tmp_label_conv = F.conv3d(tmp_label, w).squeeze()

            # Try to extract a patch with highest possible occupied volume
            occ_volume = occ_volume_max
            while occ_volume >= 0.1:
                try:
                    pos = torch.min(torch.nonzero(
                        tmp_label_conv >
                        occ_volume * np.prod(self.patch_size)
                    ))
                    break
                except RuntimeError: # No volume found --> reduce threshold
                    occ_volume -= 0.1

            if occ_volume < 0.1:
                raise RuntimeError("No patch could be found.")

            idx[d] = slice(
                shape[d]-pos-1+pad_width-self.patch_size[d], shape[d]-pos-1-pad_width
            )
            img_patches.append(F.pad(img[tuple(idx)], pad_width_all))
            label_patches.append(F.pad(label[tuple(idx)], pad_width_all))

        return img_patches, label_patches


    def _create_voxel_labels_from_meshes(self, mesh_labels):
        """ Return the voxelized meshes as 3D voxel labels """
        data = []
        for m in mesh_labels:
            vertices = m.vertices.view(self.n_m_classes, -1, 3)
            faces = m.faces.view(self.n_m_classes, -1, 3)
            voxel_label = voxelize_mesh(
                vertices, faces, self.patch_size, self.n_m_classes
            )

            data.append(voxel_label)

        return data

    def _create_voxel_labels_from_contours(self):
        """ Return the voxelized contour as 2D voxel labels """
        data = []
        for m in self.mesh_labels:
            voxel_label = voxelize_contour(
                m.vertices, self.patch_size
            )
            data.append(voxel_label)

        return data

    def _get_centers_and_radii(self, meshes: list):
        """ Return average centers and radii of all meshes provided. """

        centers_per_structure = {mn: [] for mn in self.mesh_label_names}
        radii_per_structure = {mn: [] for mn in self.mesh_label_names}

        for m in meshes:
            for verts, mn in zip(m.vertices, self.mesh_label_names):
                center = verts.mean(dim=0)
                radius = torch.sqrt(torch.sum((verts - center)**2, dim=1)).mean(dim=0)
                centers_per_structure[mn].append(center)
                radii_per_structure[mn].append(radius)

        # Average radius per structure
        if self.__len__() > 0 and self.patch_mode != "multi-patch":
            centroids = {k: torch.mean(torch.stack(v), dim=0)
                         for k, v in centers_per_structure.items()}
            radii = {k: torch.mean(torch.stack(v), dim=0)
                     for k, v in radii_per_structure.items()}
        else:
            centroids, radii = None, None

        return centroids, radii

    def _load_dataMesh_raw(self, meshnames):
        """ Load mesh such that it's registered to the respective 3D image
        """
        data = []
        for fn in self._files:
            # Voxel coords
            orig = nib.load(os.path.join(self._raw_data_dir, fn,
                                         self.img_filename))
            vox2world_affine = orig.affine
            world2vox_affine = np.linalg.inv(vox2world_affine)
            file_vertices = []
            file_faces = []
            for mn in meshnames:
                mesh = trimesh.load_mesh(os.path.join(
                    self._raw_data_dir, fn, mn + ".stl"
                ))
                # World --> voxel coordinates
                coords = np.concatenate(
                    (mesh.vertices.T, np.ones((1, mesh.vertices.shape[0]))),
                    axis=0
                )
                voxel_verts = (world2vox_affine @ coords).T[:,:-1]
                # Add to structures of file
                file_vertices.append(torch.from_numpy(voxel_verts))
                # Keep normal convention if coordinates are mirrored an uneven
                # number of times
                if np.sum(np.sign(np.diag(world2vox_affine)) == -1) % 2 == 1:
                    file_faces.append(
                        torch.from_numpy(mesh.faces).flip(dims=[1])
                    )
                else: # no flips required
                    file_faces.append(torch.from_numpy(mesh.faces))

            # First treat as a batch of multiple meshes and then combine
            # into one mesh
            mesh_batch = Meshes(file_vertices, file_faces)
            mesh_single = Mesh(
                mesh_batch.verts_padded().float(),
                mesh_batch.faces_padded().long(),
                normals=mesh_batch.verts_normals_padded().float()
            )
            data.append(mesh_single)

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
            data.append(Mesh(
                mc_mesh.verts_padded(),
                mc_mesh.faces_padded(),
                mc_mesh.verts_normals_padded()
            ))

        return data

    def _load_ms_dataMesh(self):
        """ Create ground truth meshes from pixel labels."""
        data = []
        for vl in self.voxel_labels:
            assert tuple(vl.shape) == tuple(self.patch_size),\
                    "Voxel label should be of correct size."
            ms_mesh = create_mesh_from_pixels(vl).to_pytorch3d_Meshes()
            data.append(Mesh(
                ms_mesh.verts_padded(),
                ms_mesh.faces_padded() # faces = edges in 2D
            ))

        return data

    def _load_ref_points(self):
        """ Sample surface points from meshes """
        points, normals = [], []
        curvs = [] if self.provide_curvatures else None
        for m in self.mesh_labels:
            if self.ndims == 3:
                if self.provide_curvatures:
                    # Choose a certain number of vertices
                    m_ = m.to_pytorch3d_Meshes()
                    p, idx = choose_n_random_points(
                        m_.verts_padded(),
                        self.n_ref_points_per_structure,
                        return_idx=True
                    )
                    # Choose normals with the same indices as vertices
                    n = m_.verts_normals_padded()[idx.unbind(1)].view(
                        self.n_m_classes, -1, 3
                    )
                    # Choose curvatures with the same indices as vertices
                    c = curv_from_cotcurv_laplacian(
                        m_.verts_packed(), m_.faces_packed()
                    ).view(
                        self.n_m_classes, -1, 1
                    )[idx.unbind(1)].view(
                        self.n_m_classes, -1, 1
                    )
                else: # No curvatures
                    # Sample from mesh surface
                    p, n = sample_points_from_meshes(
                        m.to_pytorch3d_Meshes(),
                        self.n_ref_points_per_structure,
                        return_normals=True
                    )
            else:
                p, n = sample_points_from_contours(
                    m.to_pytorch3d_Meshes(),
                    self.n_ref_points_per_structure,
                    return_normals=True
                )
            points.append(p)
            normals.append(n)
            if self.provide_curvatures:
                curvs.append(c)

        return points, normals, curvs

    def augment_data(self, img, label, coordinates, normals):
        assert self._augment, "No augmentation in this dataset."
        return flip_img(img, label, coordinates, normals)

    def check_augmentation_normals(self):
        """ Assert correctness of the transformation of normals during
        augmentation.
        """
        py3d_mesh = self.mesh_labels[0].to_pytorch3d_Meshes()
        img_f, label_f, coo_f, normals_f = self.augment_data(
            self.images[0].numpy(), self.voxel_labels[0].numpy(),
            py3d_mesh.verts_padded(), py3d_mesh.verts_normals_padded()
        )
        py3d_mesh_aug = Meshes(coo_f, py3d_mesh.faces_padded())
        # Assert up to sign of direction
        assert (
            torch.allclose(normals_f, py3d_mesh_aug.verts_normals_padded(),
                           atol=2e-03)
            or torch.allclose(-normals_f, py3d_mesh_aug.verts_normals_padded(),
                             atol=2e-03)
        )

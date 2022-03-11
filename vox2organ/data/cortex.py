
""" Cortex dataset handler """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import re
import os
from typing import Union, Sequence
from enum import IntEnum

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from data.image_and_mesh_dataset import ImageAndMeshDataset
from utils.modes import DataModes
from utils.logging import raise_warning

class CortexLabels(IntEnum):
    """ Mapping IDs in segmentation masks to names.
    """
    right_white_matter = 41
    left_white_matter = 2
    left_cerebral_cortex = 3
    right_cerebral_cortex = 42


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
                seg_label_names = (
                    ("left_cerebral_cortex", "right_cerebral_cortex"),
                )
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
                seg_label_names = (
                    ("left_white_matter", "right_white_matter"),
                )
                mesh_label_names = ("lh_white", "rh_white")
                voxelized_mesh_label_names = (("lh_white", "rh_white"),)
            else:
                raise ValueError("Wrong dimensionality.")

    elif ("cerebral_cortex" in structure_type
          and "white_matter" in structure_type):
        if patch_mode == "single-patch":
            seg_label_names = (
                ("right_white_matter", "right_cerebral_cortex"),
            )
            mesh_label_names = ("rh_white", "rh_pial")
            voxelized_mesh_label_names = (("rh_white", "rh_pial"),)
        else:
            # Not patch mode
            seg_label_names = (
                (
                    "left_white_matter",
                    "right_white_matter",
                    "left_cerebral_cortex",
                    "right_cerebral_cortex"
                ),
            )
            mesh_label_names = ("lh_white", "rh_white", "lh_pial", "rh_pial")
            voxelized_mesh_label_names = (
                ("lh_white", "rh_white", "lh_pial", "rh_pial"),
            )
    else:
        raise ValueError("Unknown structure type.")

    return seg_label_names, mesh_label_names, voxelized_mesh_label_names


class CortexDataset(ImageAndMeshDataset):
    """ Cortex dataset

    This dataset contains images and meshes and has additional functionality
    specifically for cortex data.

    :param structure_type: Either 'cerebral_cortex' (outer cortex surfaces)
    or 'white_matter' (inner cortex surfaces) or both
    :param reduced_freesurfer: The factor of reduced freesurfer meshes, e.g.,
    0.3
    :param morph_data_dir: A directory containing morphological brain data,
    e.g. thickness values.
    :param kwargs: Parameters for ImageAndMeshDataset
    """

    image_file_name = "mri.nii.gz"
    label_filename = "aseg.nii.gz" # For FS segmentations
    label_filename_Mb = "aparc+aseg_manual.nii.gz" # Manual Mindboggle segmentations

    def __init__(
        self,
        structure_type: Union[str, Sequence[str]],
        reduced_freesurfer: int=None,
        morph_data_dir: str=None,
        **kwargs
    ):

        self._morph_data_dir = morph_data_dir

        # Map structure type to (file-)names
        (self.voxel_label_names,
         self.mesh_label_names,
         voxelized_mesh_label_names) = _get_seg_and_mesh_label_names(
             structure_type, kwargs['patch_mode'], kwargs['ndims']
         )
        if reduced_freesurfer is not None:
            if reduced_freesurfer != 1.0:
                self.mesh_label_names = [
                    mn + "_reduced_" + str(reduced_freesurfer)
                    for mn in self.mesh_label_names
                ]

        # Set file names
        seg_file_name=(
            self.label_filename_Mb
            if "Mindboggle" in kwargs['raw_data_dir']
            else self.label_filename
        )

        super().__init__(
            image_file_name=self.image_file_name,
            mesh_file_names=self.mesh_label_names,
            seg_file_name=seg_file_name,
            voxelized_mesh_file_names=voxelized_mesh_label_names,
            **kwargs
        )

        # Load morphology labels
        self.thickness_per_vertex = self._get_morph_label(
            "thickness", subfolder=""
        )


    def seg_ids(self, names):
        """ Map voxel classes to IDs.
        """
        return [CortexLabels[n].value for n in names]


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
        if self._morph_data_dir is None:
            return None

        morph_labels = []
        for fn in self._files:
            file_dir = os.path.join(
                self._morph_data_dir, fn, subfolder
            )
            file_labels = []
            n_max = 0
            for mn in self.mesh_label_names:
                # Filenames have form 'lh_white_reduced_0.x.thickness'
                morph_fn = mn + "." + morphology
                morph_fn = os.path.join(file_dir, morph_fn)
                try:
                    morph_label = nib.freesurfer.io.read_morph_data(morph_fn)
                except FileNotFoundError:
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

    :param kwargs: Parameters for base classes
    """
    def __init__(self, **kwargs):

        super().__init__(**kwargs)

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
                    torch.from_numpy(label.astype(np.int32)).unsqueeze(-1)
                )
                if label.shape[0] > V_max:
                    V_max = label.shape[0]

            vertex_labels.append(file_labels)

        return vertex_labels, label_colors, label_info

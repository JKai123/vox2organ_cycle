
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

class AbdomenLabels(IntEnum):
    """ Mapping IDs in segmentation masks to names.
    """
    liver = 1
    kidney = 2
    spleen = 3
    pancreas = 4


def _get_seg_and_mesh_label_names(structure_type, patch_mode, ndims):
    """ Helper function to map the structure type and the patch mode to the
    correct segmentation and mesh label names.

    For seg_label_names and voxelized_mesh_label_names entries can/should be
    grouped s.t. they represent one "voxel class" in the segmentation maps.
    """
    voxelized_mesh_label_names = None # Does not always exist
    if structure_type == "abdomen-all":
        if patch_mode=="no":
            seg_label_names = (
                ("liver",), ("kidney",), ("spleen",), ("pancreas",)
            )
            mesh_label_names = (
                "liver", "kidney_left", "kidney_right", "spleen", "pancreas"
            )
    elif structure_type == "abdomen-wo-pancreas":
        if patch_mode=="no":
            seg_label_names = (
                ("liver",), ("kidney",), ("spleen",)
            )
            mesh_label_names = (
                "liver", "kidney_left", "kidney_right", "spleen"
            )
        else:
            raise NotImplementedError()
    elif structure_type == "kidney-spleen":
            if patch_mode=="no":
                seg_label_names = (
                    ("kidney",), ("spleen",)
                )
                mesh_label_names = (
                    "kidney_right", "spleen"
                )
            else:
                raise NotImplementedError()

    else:
        raise NotImplementedError()

    return seg_label_names, mesh_label_names, voxelized_mesh_label_names


class AbdomenDataset(ImageAndMeshDataset):
    """ Abdomen dataset

    This dataset contains images and meshes and has additional functionality
    specifically for abdominal data.

    :param structure_type: A description of the structure(s) to segement, e.g.,
    'abdomen-all'
    :param kwargs: Parameters for ImageAndMeshDataset
    """

    image_file_name = "img.nii.gz"
    seg_file_name = "seg.nii.gz"

    def __init__(
        self,
        structure_type: Union[str, Sequence[str]],
        **kwargs
    ):

        # Map structure type to (file-)names
        (self.voxel_label_names,
         self.mesh_label_names,
         self.voxelized_mesh_label_names) = _get_seg_and_mesh_label_names(
             structure_type, kwargs['patch_mode'], kwargs['ndims']
         )

        super().__init__(
            image_file_name=self.image_file_name,
            mesh_file_names=self.mesh_label_names,
            seg_file_name=self.seg_file_name,
            voxelized_mesh_file_names=self.voxelized_mesh_label_names,
            **kwargs
        )


    def seg_ids(self, names):
        """ Map voxel classes to IDs.
        """
        return [AbdomenLabels[n].value for n in names]

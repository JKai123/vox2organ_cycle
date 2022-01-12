
""" Handling different architectures """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from utils.utils import ExtendedEnum

from models.vox2cortex import (
    Voxel2MeshPlusPlusGeneric,
)

class ModelHandler(ExtendedEnum):
    voxel2meshplusplusgeneric = Voxel2MeshPlusPlusGeneric

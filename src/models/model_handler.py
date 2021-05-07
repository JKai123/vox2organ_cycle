
""" Handling different architectures """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from utils.utils import ExtendedEnum

from models.voxel2mesh import Voxel2Mesh
from models.voxel2meshplusplus import Voxel2MeshPlusPlus

class ModelHandler(ExtendedEnum):
    voxel2mesh = Voxel2Mesh
    voxel2meshplusplus = Voxel2MeshPlusPlus

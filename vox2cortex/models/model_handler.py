
""" Handling different architectures """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from utils.utils import ExtendedEnum

from models.vox2cortex import Vox2Cortex
from models.corticalflow import CorticalFlow
from models.v2cflow import V2CFlow

class ModelHandler(ExtendedEnum):
    vox2cortex = Vox2Cortex
    corticalflow = CorticalFlow
    v2cflow = V2CFlow

""" Draw the network architecture based on PlotNeuralNet, see
https://github.com/HarisIqbal88/PlotNeuralNet"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import sys
sys.path.append("../../../PlotNeuralNet")

from pycore.tikzeng import *
from models.voxel2meshplusplus import Voxel2MeshPlusPlusGeneric

def plot_net(model):


def create_model():
    model = Voxel2MeshP(
        num_classes=2,
        patch_shape=(64, 64, 64),
        num_input_channels=1,
        encoder_channels: (256, 128, 64, 32, 16),
        decoder_channels: (64, 32, 16, 8),
        graph_channels: (256, 128, 64, 32, 16),
        batch_norm: True,
        mesh_template: str,
        unpool_indices: Union[list, tuple],
        use_adoptive_unpool: bool,
        deep_supervision: bool,
        weighted_edges: bool,
        voxel_decoder: bool,
        gc,
        propagate_coords: bool,


if __name__ == "__main__":
    model = create_model()
    plot_net(model)

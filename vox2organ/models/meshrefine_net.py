
""" Graph (sub-)networks for mesh refinement. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from torch import nn
from torch.cuda.amp import autocast
from pytorch3d.ops import GraphConv

from utils.utils.graph_conv import (
    Features2FeaturesResidual,
    zero_weight_init
)
from utils.logging import measure_time
from utils.mesh import MeshesOfMeshes

class MeshRefineNet(nn.Module):
    """ A graph net that takes vertex features (e.g. coordinates and
    features) as input and ouputs a deformation field per vertex.
    """
    def __init__(
        self,
        n_input_features: int,
        norm: str,
        latent_channels: int,
        GC,
        p_dropout: float=None,
        n_f2f_hidden_layer: int=2,
        ndims: int=3
    ):
        super().__init__()

        self.GC = GC
        self.ndims = ndims

        # Creation of latent features from input features (one residual block)
        self.f2f = Features2FeaturesResidual(
            n_input_features,
            latent_channels,
            n_f2f_hidden_layer,
            norm=norm,
            GC=GC,
            p_dropout=p_dropout,
            weighted_edges=False
        )

        # Feature to vertex layer
        self.f2v = GC(
            latent_channels,
            ndims,
            weighted_edges=False,
            init='zero'
        )

        # Init f2v layers to zero
        self.f2v.apply(zero_weight_init)

    @measure_time
    def forward(self, mesh: MeshesOfMeshes):
        """ Forward pass: take mesh features and predict displacement.
        """

        batch_size, M, V, _ = mesh.verts_padded().shape

        # No autocast for pytorch3d convs possible
        cast = not issubclass(self.GC, GraphConv)
        with autocast(enabled=cast):
            # Latent features
            edges_packed = mesh.edges_packed()
            features_packed = mesh.features_packed()
            latent_features_packed = self.f2f(features_packed, edges_packed)

            # Displacement
            deltaV_packed = self.f2v(latent_features_packed, edges_packed)
            deltaV_padded = deltaV_packed.view(
                batch_size, M, V, self.ndims
            )

        return deltaV_padded


""" Implementation of CorticalFlow model from Lebrat et al. 2021
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from typing import Union, Tuple

import torch
from torch.cuda.amp import autocast

from models.base_model import V2MModel
from models.u_net import ResidualUNet
from utils.mesh import MeshesOfMeshes
from utils.file_handle import read_obj
from utils.coordinate_transform import (
    unnormalize_vertices_per_max_dim,
    normalize_vertices
)
from utils.utils_voxel2meshplusplus.feature_aggregation import (
    aggregate_trilinear,
)

def DMDBlock(discrete_flow, mesh, n_integration, patch_size):
    """ Forward pass
    """
    # Flow has shape HxWxDxdim
    ndims = discrete_flow.shape[-1]

    # Stepsize
    h = 1 / float(n_integration)

    # Mesh vertices
    vertices_padded = mesh.verts_padded() # (B,M,V,3)
    B, M, V, _ = vertices_padded.shape

    # Integrate numerically
    for _ in range(n_integration):
        # Normalize vertices for F.grid_sample
        verts_img_co = normalize_vertices(
            unnormalize_vertices_per_max_dim(
                vertices_padded.view(-1, ndims),
                patch_size
            ),
            patch_size
        ).view(B, -1, ndims)

        # Get flow at vertex positions
        # Avoid bug related to automatic mixed precision, see also
        # https://github.com/pytorch/pytorch/issues/42218
        with autocast(enabled=False):
            # Trilinear interpolation of flow at vertex positions
            flow = aggregate_trilinear(
                discrete_flow, verts_img_co, mode='trilinear'
            ).view(B, M, V, ndims)

            # V <- V + h * Flow
            vertices_padded += h * flow

    return mesh


class CorticalFlow(V2MModel):
    """ CorticalFlow model from Lebrat et al. 2021

    :param n_m_classes: Number of mesh classes to distinguish
    :param num_input_channels: The number of image channels
    :param encoder_channels: Encoder channels for each UNet
    :param decoder_channels: Decoder channels for each UNet
    :param mesh_template: Path to the mesh template
    :param unpool_indices: Indices of stages where mesh unpooling is performed,
    starting with index 0
    :param p_dropout_unet: UNet dropout probability
    :param ndims: Dimension of the space (usually 3)
    :param uncertainty: A measure of uncertainty
    """
    def __init__(
        self,
        n_m_classes: int,
        num_input_channels: int,
        patch_size: Tuple[int, int, int],
        encoder_channels: Union[list, tuple],
        decoder_channels: Union[list, tuple],
        mesh_template: str,
        # unpool_indices: Union[list, tuple], # TODO: Implement
        p_dropout_unet: float,
        ndims: int,
        # uncertainty: Union[None, str], TODO: Implement
        **kwargs
    ):

        super().__init__()

        self.patch_size = patch_size

        # Set of UNets
        self.u_nets = []
        for s, (ec, dc) in enumerate(zip(encoder_channels, decoder_channels)):
            # First UNet only gets the image, all others also get the previous
            # deformation field
            if s == 0:
                n_c_input = num_input_channels
            else:
                n_c_input = num_input_channels + ndims

            # Add UNet
            self.u_nets.append(
                ResidualUNet(
                    num_classes=ndims, # n-dim. flow field
                    num_input_channels=n_c_input,
                    down_channels=ec,
                    up_channels=dc,
                    deep_supervision=False,
                    patch_shape=None, # Only required for deep sup.
                    voxel_decoder=True,
                    p_dropout=p_dropout_unet,
                    ndims=ndims,
                    init_last_zero=True
                )
            )

        # Template
        raw_sphere_vertices, raw_sphere_faces, _ = read_obj(mesh_template)
        self.sphere_vertices = torch.from_numpy(raw_sphere_vertices).cuda().float()
        self.sphere_faces = torch.from_numpy(raw_sphere_faces).cuda().long()

        # Batch size 1
        self.sphere_vertices = self.sphere_vertices.view_as(
            raw_sphere_vertices
        )[None]
        self.sphere_faces = self.sphere_faces.view_as(raw_sphere_faces)[None]


    def forward(self, x):
        """ Forward pass
        """
        B = x.shape[0]
        temp_vertices = torch.cat(B * [self.sphere_vertices], dim=0)
        temp_faces = torch.cat(B * [self.sphere_faces], dim=0)
        temp_meshes = MeshesOfMeshes(verts=temp_vertices, faces=temp_faces)

        # First input: only image
        u_net_input = x

        pred_meshes = [temp_meshes]

        for s, u_net in enumerate(self.u_nets):
            # Predicted flow field
            discrete_flow = u_net(u_net_input)

            # New input: image and previous flow
            u_net_input = torch.cat([x, discrete_flow], dim=1)

            # Heuristic: number of integration steps = 2 * maximum amplitude of
            # flow field + 1; this ensures hL < 1 for sure but could lead to
            # unnecessary many steps
            flow_magnitude = discrete_flow.norm(dim=-1)
            n_integration = int(torch.ceil(2 * flow_magnitude.max()).item())

            # Mesh deformation
            pred_meshes.append(
                DMDBlock(pred_meshes[s], discrete_flow, n_integration, self.patch_size)
            )

        return [pred_meshes]


    def save(self, path):
        """ Save model with its parameters to the given path.
        Conventionally the path should end with "*.model".

        :param str path: The path where the model should be saved.
        """

        torch.save(self.state_dict(), path)


    @staticmethod
    def pred_to_per_structure_uncertainty(pred):
        """ Get the average uncertainty per structure of shape (B, C)
        """
        return None

    @staticmethod
    def pred_to_displacements(pred):
        """ Get the magnitudes of vertex displacements of shape (S, B, C)
        """
        return None

    @staticmethod
    def pred_to_voxel_pred(pred):
        return None

    @staticmethod
    def pred_to_raw_voxel_pred(pred):
        return None

    @staticmethod
    def pred_to_verts_and_faces(pred):
        """ Get the vertices and faces of shape (S,C)
        """
        C = pred[0][0].verts_padded().shape[1]
        S = len(pred[0])

        vertices = []
        faces = []
        meshes = pred[0][1:] # Ignore template mesh at pos. 0
        for s, m in enumerate(meshes):
            v_s = []
            f_s = []
            for c in range(C):
                v_s.append(m.verts_padded()[:,c,:,:])
                f_s.append(m.faces_padded()[:,c,:,:])
            vertices.append(torch.stack(v_s))
            faces.append(torch.stack(f_s))

        return vertices, faces

    @staticmethod
    def pred_to_deltaV_and_faces(pred):
        """ Get the displacements and faces of shape (S,C)
        """
        return None

    @staticmethod
    def pred_to_pred_meshes(pred):
        """ Create valid prediction meshes of shape (S,C) """
        vertices, faces = self.__class__.pred_to_verts_and_faces(pred)
        pred_meshes = verts_faces_to_Meshes(vertices, faces, 2) # pytorch3d

        return pred_meshes

    @staticmethod
    def pred_to_pred_deltaV_meshes(pred):
        return None

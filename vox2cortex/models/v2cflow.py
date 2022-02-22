
""" Implementation of Vox2Cortex-Flow
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from typing import Sequence
from copy import  deepcopy

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast

from models.base_model import V2MModel
from models.u_net import ResidualUNet
from models.meshrefine_net import MeshRefineNet
from utils.mesh import MeshesOfMeshes, Mesh
from utils.mesh import vff_to_Meshes
from utils.coordinate_transform import (
    unnormalize_vertices_per_max_dim,
    normalize_vertices
)
from utils.utils_voxel2meshplusplus.feature_aggregation import (
    aggregate_structural_features,
    aggregate_trilinear,
)

def assemble_features(
    verts_padded: torch.tensor,
    group_structs: Sequence[Sequence[int]],
    k_struct_neighbors: int,
    verts_classes: torch.tensor
):
    """ Assemble vertex features.
    """
    struct_features = aggregate_structural_features(
        verts_padded,
        group_structs,
        True,
        k_struct_neighbors
    )
    class_features = F.one_hot(verts_classes)

    # Final features:
    # vertex coords | structural features | vertex classes
    return torch.cat([verts_padded, struct_features, class_features], dim=-1)


class DMDBlockRefined(nn.Module):
    """ Flow integration blocks.
    """
    def __init__(
        self,
        group_structs: Sequence[Sequence[int]],
        k_struct_neighbors: int,
        n_vertex_classes: int
    ):
        super().__init__()
        self.group_structs = group_structs
        self.k_struct_neighbors = k_struct_neighbors
        self.n_vertex_classes = n_vertex_classes

    def forward(
        self,
        discrete_flow,
        mesh,
        n_integration,
        patch_size,
        refine_net
    ):
        """ Forward pass: Integrate a flow defined on a discrete grid and refined
        by a graph net.
        """
        out_mesh = deepcopy(mesh)

        # Flow has shape BxCxHxWxD
        ndims = discrete_flow.shape[1]

        # Stepsize
        h = 1 / float(n_integration)

        # Mesh vertices
        vertices_padded = out_mesh.verts_padded() # (B,M,V,3)
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
            flow = aggregate_trilinear(
                discrete_flow, verts_img_co, mode='bilinear'
            ).view(B, M, V, ndims)

            # Get residual from gnn
            out_mesh.update_features(
                assemble_features(
                    out_mesh.verts_padded(),
                    self.group_structs,
                    self.k_struct_neighbors,
                    out_mesh.features_padded()
                )
            )
            residual_flow = refine_net(out_mesh)

            # Refine flow with residual
            refined_flow = flow + residual_flow

            # V <- V + h * Flow
            out_mesh.move_verts(h * refined_flow)

        return out_mesh


class V2CFlow(V2MModel):
    """ Vox2CortexFlow model.

    :param num_input_channels: The number of image channels
    :param patch_size: The size of the input images
    :param encoder_channels: Encoder channels for each UNet
    :param decoder_channels: Decoder channels for each UNet
    :param mesh_template: Mesh template to deform
    :param p_dropout_unet: UNet dropout probability
    :param ndims: Dimension of the space (usually 3)
    :param graph_channels: The number of graph latent features
    :param norm: The norm to apply in graph layers, e.g. 'batch'
    :param p_dropout_graph: Graph dropout probability
    :param group_structs: Group the structures in the graph network, e.g.,
    group left and right white matter hemisphere into group "white matter".
    During a graph net forward pass, features are exchanged between distinct
    groups but not within a group. For example, white surface vertex positions
    can be provided to the pial vertices and vice versa.
    :param k_struct_neighbors: K for the KNN features of other structures, only
    relevant if group_structs is specified and exchange_coords is True.
    :param exchange_coords: Whether to exchange coordinates between structure
    groups.
    :param n_vertex_classes: The number of vertex classes to distinguish
    """
    def __init__(
        self,
        num_input_channels: int,
        patch_size: Sequence[int],
        encoder_channels: Sequence[Sequence],
        decoder_channels: Sequence[Sequence],
        mesh_template: Mesh,
        p_dropout_unet: float,
        ndims: int,
        graph_channels: int,
        norm: str,
        gc,
        p_dropout_graph: float,
        group_structs: Sequence[Sequence[int]],
        k_struct_neighbors: int,
        n_vertex_classes: int,
        **kwargs
    ):

        super().__init__()

        self.eps = 1e-2
        self.uncertainty = None # Not implemented

        self.patch_size = patch_size

        # Set of GNNs and UNets
        self.u_nets = []
        self.graph_nets = []
        for s, (ec, dc) in enumerate(zip(encoder_channels, decoder_channels)):
            # First UNet only gets the image, all others also get the previous
            # deformation fields
            n_c_input = num_input_channels + s * ndims

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

            # Graph net gets vertex features as input, see assemble_features
            n_gc_input = (
                ndims
                + k_struct_neighbors * ndims
                + int(np.ceil(np.log2(len(group_structs))))
                + n_vertex_classes
            )

            # Add GNN
            self.graph_nets.append(
                MeshRefineNet(
                    n_input_features=n_gc_input,
                    norm=norm,
                    latent_channels=graph_channels,
                    GC=gc,
                    p_dropout=p_dropout_graph,
                    ndims=ndims
                )
            )

        self.u_nets = nn.ModuleList(self.u_nets)
        self.graph_nets = nn.ModuleList(self.graph_nets)

        # Integration block
        self.dmdblock = DMDBlockRefined(
            group_structs, k_struct_neighbors, n_vertex_classes
        )

        # Template (batch size 1)
        self.mesh_template = mesh_template
        self.sphere_vertices = mesh_template.vertices.cuda()[None]
        self.sphere_faces = mesh_template.faces.cuda()[None]


    def forward(self, x):
        """ Forward pass
        """
        B = x.shape[0]
        temp_vertices = torch.cat(B * [self.sphere_vertices], dim=0)
        temp_faces = torch.cat(B * [self.sphere_faces], dim=0)
        temp_meshes = MeshesOfMeshes(verts=temp_vertices, faces=temp_faces)

        _, M, V, _ = temp_vertices.shape

        # First input: only image
        u_net_input = x

        pred_meshes = [temp_meshes]

        # Iterate over deformation steps
        for s, (u_net, graph_net) in enumerate(
            zip(self.u_nets, self.graph_nets)
        ):
            # Predicted flow field
            discrete_flow = u_net(u_net_input)[2][-1]

            # New input: add predicted flow
            u_net_input = torch.cat([u_net_input, discrete_flow], dim=1)

            # Heuristic: number of integration steps = 2 * maximum amplitude of
            # flow field + 1; this ensures hL < 1 for sure but could lead to
            # unnecessary many steps
            flow_magnitude = discrete_flow.norm(dim=-1)
            n_integration = int(torch.ceil(2 * flow_magnitude.max() + self.eps).item())

            # Mesh deformation; autocast not possible due to F.grid_sample
            with autocast(enabled=False):
                # Predict discrete flow field
                discrete_flow_32 = discrete_flow.float()

                pred_meshes.append(
                    self.dmdblock(
                        discrete_flow_32,
                        pred_meshes[s],
                        n_integration,
                        self.patch_size,
                        graph_net
                    )
                )

        # Replace features with template features (e.g. vertex classes)
        for m in pred_meshes:
            m.update_features(
                self.mesh_template.features.expand(
                    B, M, V, -1
                ).cuda()
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
    def pred_to_vff(pred):
        """ Get the vertices and faces and features of shape (S,C)
        """
        C = pred[0][0].verts_padded().shape[1]
        S = len(pred[0])

        vertices = []
        faces = []
        features = []
        meshes = pred[0][1:] # Ignore template mesh at pos. 0
        for s, m in enumerate(meshes):
            v_s = []
            f_s = []
            vf_s = []
            for c in range(C):
                v_s.append(m.verts_padded()[:,c,:,:])
                f_s.append(m.faces_padded()[:,c,:,:])
                vf_s.append(m.features_padded()[:,c,:,:])
            vertices.append(torch.stack(v_s))
            faces.append(torch.stack(f_s))
            features.append(torch.stack(vf_s))

        return vertices, faces, features

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
        vertices, faces, features = CorticalFlow.pred_to_vff(pred)
        # To pytorch3d Meshes
        pred_meshes = vff_to_Meshes(vertices, faces, features, 2)

        return pred_meshes

    @staticmethod
    def pred_to_pred_deltaV_meshes(pred):
        return None

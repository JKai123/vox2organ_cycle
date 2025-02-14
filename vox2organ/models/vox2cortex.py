""" Voxel2Mesh++ """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from itertools import chain
from typing import Union, Tuple
from deprecated import deprecated

import torch

from utils.logging import measure_time
from utils.mesh import vff_to_Meshes, verts_faces_to_Meshes
from utils.utils_padded_packed import as_list

from models.u_net import ResidualUNet
from models.base_model import V2MModel
from models.graph_net import GraphDecoder


class Vox2Cortex(V2MModel):
    """ Implementation of the Vox2Cortex model.

    :param n_v_classes: Number of voxel classes to distinguish
    :param n_m_classes: Number of mesh classes to distinguish
    :param num_input_channels: The number of channels of the input image.
    :param encoder_channels: The number of channels of the encoder
    :param decoder_channels: The number of channels of the decoder
    :param graph_channels: The number of graph features per graph layer
    :param batch_norm: Whether or not to apply batch norm at graph layers.
    :param mesh_template: The mesh template that is deformed thoughout a
    forward pass.
    :param unpool_indices: Indicates the steps at which unpooling is performed. This
    has no impact on the model architecture and can be changed even after
    training.
    :param use_adoptive_unpool: Discard vertices that did not deform much to reduce
    number of vertices where appropriate (e.g. where curvature is low). Not
    implemented at the moment.
    :param weighted_edges: Whether or not to use graph convolutions with
    length-weighted edges.
    :param voxel_decoder: Whether or not to use a voxel decoder
    :param GC: The graph conv implementation to use
    :param propagate_coords: Whether to propagate coordinates in the graph conv
    :param patch_size: The used patch size of input images.
    :param aggregate_indices: Where to take the features from the UNet
    :param aggregate: 'trilinear', 'bilinear', or 'lns'
    :param p_dropout_unet: Dropout probability for UNet blocks
    :param p_dropout_graph: Dropout probability for  graph conv blocks
    :param ndims: Dimensionality of images
    :param group_structs: Group the structures in the graph network, e.g.,
    group left and right white matter hemisphere into group "white matter".
    During a graph net forward pass, features are exchanged between distinct
    groups but not within a group. For example, white surface vertex positions
    can be provided to the pial vertices and vice versa.
    :param k_struct_neighbors: K for the KNN features of other structures, only
    relevant if group_structs is specified and exchange_coords is True.
    :param exchange_coords: Whether to exchange coordinates between structure
    groups.
    :param n_vertex_classes: The number of vertex classes.
    """

    def __init__(self,
                 n_v_classes: int,
                 n_m_classes: int,
                 num_input_channels: int,
                 encoder_channels: Union[list, tuple],
                 decoder_channels: Union[list, tuple],
                 graph_channels: Union[list, tuple],
                 norm: str,
                 mesh_template: str,
                 unpool_indices: Union[list, tuple],
                 use_adoptive_unpool: bool,
                 deep_supervision: bool,
                 weighted_edges: bool,
                 voxel_decoder: bool,
                 gc,
                 propagate_coords: bool,
                 patch_size: Tuple[int, int, int],
                 aggregate: str,
                 aggregate_indices: Tuple[Tuple[int]],
                 p_dropout_unet: float,
                 p_dropout_graph: float,
                 ndims: int,
                 group_structs: Tuple[Tuple[int]],
                 k_struct_neighbors: int,
                 exchange_coords: bool,
                 n_vertex_classes: int,
                 **kwargs
                 ):
        super().__init__()

        # Voxel network
        self.voxel_net = ResidualUNet(
            num_classes=n_v_classes,
            num_input_channels=num_input_channels,
            patch_shape=patch_size,
            down_channels=encoder_channels,
            up_channels=decoder_channels,
            deep_supervision=deep_supervision,
            voxel_decoder=voxel_decoder,
            p_dropout=p_dropout_unet,
            ndims=ndims
        )
        # Graph network
        self.graph_net = GraphDecoder(
            norm=norm,
            mesh_template=mesh_template,
            unpool_indices=unpool_indices,
            use_adoptive_unpool=use_adoptive_unpool,
            graph_channels=graph_channels,
            skip_channels=encoder_channels+decoder_channels,
            weighted_edges=weighted_edges,
            propagate_coords=propagate_coords,
            patch_size=patch_size,
            aggregate_indices=aggregate_indices,
            aggregate=aggregate,
            k_struct_neighbors=k_struct_neighbors,
            exchange_coords=exchange_coords,
            GC=gc,
            p_dropout=p_dropout_graph,
            group_structs=group_structs,
            n_vertex_classes=n_vertex_classes,
            ndims=ndims
        )

    @measure_time
    def forward(self, x):

        encoder_skips, decoder_skips, seg_out = self.voxel_net(x)
        pred_meshes, pred_deltaV, cycle_pred_meshes = self.graph_net(encoder_skips + decoder_skips)

        # pred has the form
        #   - batch of predicted meshes
        #   - batch of voxel predictions,
        #   - batch of displacements (also stored as meshes but with
        #   coordinates representing displacements)
        pred = (pred_meshes, seg_out, pred_deltaV, cycle_pred_meshes)

        return pred

    def save(self, path):
        """ Save model with its parameters to the given path.
        Conventionally the path should end with "*.model".

        :param str path: The path where the model should be saved.
        """

        torch.save(self.state_dict(), path)

    @staticmethod
    def pred_to_displacements(pred):
        """ Get the magnitudes of vertex displacements of shape (S, B, C)
        """
        # No displacements for step 0
        displacement_meshes = pred[2][1:]
        # Magnitude (vertices in displacement meshes are equal to
        # displacements)
        d_norm = [d.verts_padded().norm(dim=-1) for d in displacement_meshes]
        # Mean over vertices since t|V| can vary among steps
        d_norm_mean = [d.mean(dim=-1) for d in d_norm]
        d_norm_mean = torch.stack(d_norm_mean)

        return d_norm_mean

    @staticmethod
    def pred_to_voxel_pred(pred):
        """ Get the final voxel prediction with argmax over classes applied """
        if pred[1]:
            return pred[1][-1].argmax(dim=1).squeeze()
        return None

    @staticmethod
    def pred_to_raw_voxel_pred(pred):
        """ Get the voxel prediction per class. May be a list if deep
        supervision is used. """
        return pred[1]

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
    def pred_to_verts_and_faces_list(pred):
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
    def pred_to_cycle_verts_and_faces(pred):
        """ Get the vertices and faces of shape (S,C)
        """
        C = pred[3][1].verts_padded().shape[1]
        S = len(pred[3])

        vertices = []
        faces = []
        meshes = pred[3][1:]
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
        C = pred[2][1].verts_padded().shape[1]
        S = len(pred[2])

        deltaV = []
        faces = []
        meshes = pred[2][1:] # Ignore step 0
        for s, m in enumerate(meshes):
            v_s = []
            f_s = []
            for c in range(C):
                v_s.append(m.verts_padded()[:,c,:,:])
                f_s.append(m.faces_padded()[:,c,:,:])
            deltaV.append(torch.stack(v_s))
            faces.append(torch.stack(f_s))

        return deltaV, faces

    @staticmethod
    def pred_to_pred_meshes(pred):
        """ Create valid prediction meshes of shape (S,C) """
        vertices, faces, features = Vox2Cortex.pred_to_vff(pred)
        # To pytorch3d Meshes
        pred_meshes = vff_to_Meshes(vertices, faces, features, 2)

        return pred_meshes

    @staticmethod
    def predMoM_to_meshes(pred):
        """ Get the vertices and faces and features of shape (S,C)
        C as the number of classes
        """    
        C = pred[0][0].verts_padded().shape[1]
        vert_list = []
        face_list = []
        feat_list = []
        meshes = pred[0][1:] # Ignore template mesh at pos. 0
        pred_meshes = []
        for s, m in enumerate(meshes):
            vert_list.append(as_list(m.verts_padded(), m.verts_mask()))
            face_list.append(as_list(m.faces_padded(), m.faces_mask()))
            feat_list.append(as_list(m.features_padded(), m.features_mask()))
        pred_meshes = vff_to_Meshes(vert_list, face_list, feat_list, 2)
        return pred_meshes

    @staticmethod
    def pred_to_pred_deltaV_meshes(pred):
        """ Create valid prediction meshes of shape (S,C) with RELATIVE
        coordinates, i.e., with vertices containing displacement vectors. """
        deltaV, faces = Vox2Cortex.pred_to_deltaV_and_faces(pred)
        pred_deltaV_meshes = verts_faces_to_Meshes(deltaV, faces, 2) # pytorch3d

        return pred_deltaV_meshes


    @staticmethod
    def pred_to_cycle_pred_meshes(pred):
        """ Create valid prediction meshes of shape (S,C) with RELATIVE
        coordinates, i.e., with vertices containing displacement vectors. """
        verts, faces = Vox2Cortex.pred_to_cycle_verts_and_faces(pred)
        pred_cycle_meshes = verts_faces_to_Meshes(verts, faces, 2) # pytorch3d

        return pred_cycle_meshes

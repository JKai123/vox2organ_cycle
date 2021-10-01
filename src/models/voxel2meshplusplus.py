""" Voxel2Mesh++ """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from itertools import chain
from typing import Union, Tuple
from deprecated import deprecated

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes, Pointclouds
from torch.cuda.amp import autocast

from utils.utils import (
    crop_and_merge,
    sample_outer_surface_in_voxel,
)
from utils.coordinate_transform import normalize_vertices_per_max_dim
from utils.utils_voxel2meshplusplus.graph_conv import (
    Feature2VertexLayer,
    Features2Features)
from utils.utils_voxel2mesh.feature_sampling import LearntNeighbourhoodSampling
from utils.utils_voxel2mesh.file_handle import read_obj
from utils.utils_voxel2mesh.unpooling import uniform_unpool, adoptive_unpool
from utils.modes import ExecModes
from utils.logging import measure_time, write_scatter_plot_if_debug
from utils.mesh import verts_faces_to_Meshes

from models.u_net import UNetLayer, ResidualUNet
from models.base_model import V2MModel
from models.graph_net import GraphDecoder

class Voxel2MeshPlusPlus(V2MModel):
    """ Voxel2MeshPlusPlus

    Based on Voxel2Mesh from
    https://github.com/cvlab-epfl/voxel2mesh

    :param ndims: Number of input dimensions, only tested with ndims=3
    :param n_v_classes: Number of vertex classes to distinguish
    :param n_m_classes: Number of mesh classes to distinguish. This is included
    for compatibility but has not effect since support for mesh classes has
    been added later in Voxel2MeshPlusPlusGeneric
    :param patch_shape: The shape of the input patches, e.g. (64, 64, 64)
    :param num_input_channels: The number of channels of the input image.
    :param first_layer_channels: The number of channels of the first encoder
    layer. The number of channels of many other channels is derived from this
    number.
    :param steps: The number of encoder/decoder steps - 1 (counting only the
    ones with skip connection)
    :param graph_conv_layer_count: The number of hidden layers in the
    feature-to-feature block
    :param batch_norm: Whether or not to apply batch norm at layers.
    :param mesh_template: The mesh template that is deformed thoughout a
    forward pass.
    :param unpool_indices: Indicates the steps at which unpooling is performed. This
    has no impact on the model architecture and can be changed even after
    training.
    :param adoptive_unpool: Discard vertices that did not deform much to reduce
    number of vertices where appropriate (e.g. where curvature is low)
    """

    def __init__(self,
                 ndims: int,
                 n_v_classes: int,
                 n_m_classes: int,
                 patch_shape,
                 num_input_channels,
                 first_layer_channels,
                 steps,
                 graph_conv_layer_count,
                 batch_norm,
                 mesh_template,
                 unpool_indices,
                 use_adoptive_unpool: bool,
                 **kwargs
                 ):
        super().__init__()

        self.steps = steps

        if ndims == 3:
            self.max_pool = nn.MaxPool3d(2)
        elif ndims == 2:
            self.max_pool = nn.MaxPool2d(2)
        else:
            raise ValueError("Invalid number of dimensions")

        self.n_v_classes = n_v_classes
        self.unpool_indices = unpool_indices
        self.use_adoptive_unpool = use_adoptive_unpool

        ConvLayer = nn.Conv3d if ndims == 3 else nn.Conv2d
        ConvTransposeLayer = nn.ConvTranspose3d if ndims == 3 else nn.ConvTranspose2d

        '''  Down layers '''
        down_layers = [UNetLayer(num_input_channels, first_layer_channels, ndims)]
        for i in range(1, steps + 1):
            conv_layer = UNetLayer(first_layer_channels * 2 ** (i - 1),
                                         first_layer_channels * 2 ** i, ndims)
            down_layers.append(conv_layer)
        self.down_layers = down_layers
        self.encoder = nn.Sequential(*down_layers)

        ''' Up layers '''
        self.skip_count = []
        self.latent_features_count = []
        for i in range(steps+1):
            self.skip_count += [first_layer_channels * 2 ** (steps-i)]
            self.latent_features_count += [32]

        # Dimensionality of vertex coordinates
        dim = 3

        up_std_conv_layers = []
        up_f2f_layers = []
        up_f2v_layers = []
        for i in range(steps+1):
            graph_unet_layers = [None] # No layer for class 0 needed
            feature2vertex_layers = [None] # No layer for class 0 needed
            skip = LearntNeighbourhoodSampling(patch_shape, steps, self.skip_count[i], i)
            if i == 0:
                # Lowest decoder level
                grid_upconv_layer = None
                grid_unet_layer = None
                for _ in range(1, self.n_v_classes):
                    graph_unet_layers += [Features2Features(self.skip_count[i] + dim,
                                                            self.latent_features_count[i],
                                                            hidden_layer_count=graph_conv_layer_count)]

            else:
                # All but lowest decoder levels
                grid_upconv_layer = ConvTransposeLayer(in_channels=first_layer_channels * 2 ** (steps - i + 1),
                                                       out_channels=first_layer_channels * 2**(steps-i),
                                                       kernel_size=2,
                                                       stride=2)
                grid_unet_layer = UNetLayer(first_layer_channels * 2 ** (steps - i + 1),
                                            first_layer_channels * 2**(steps-i),
                                            ndims)
                for _ in range(1, self.n_v_classes):
                    graph_unet_layers += [Features2Features(
                        self.skip_count[i] + self.latent_features_count[i-1] + dim,
                        self.latent_features_count[i],
                        hidden_layer_count=graph_conv_layer_count,
                        batch_norm=batch_norm)]

            for _ in range(1, self.n_v_classes):
                feature2vertex_layers +=\
                    [Feature2VertexLayer(self.latent_features_count[i],
                                         hidden_layer_count=3,
                                         batch_norm=batch_norm)]

            up_std_conv_layers.append((skip, grid_upconv_layer, grid_unet_layer))
            up_f2f_layers.append(graph_unet_layers)
            up_f2v_layers.append(feature2vertex_layers)

        self.up_std_conv_layers = up_std_conv_layers
        self.up_f2f_layers = up_f2f_layers
        self.up_f2v_layers = up_f2v_layers

        self.decoder_std_conv = nn.Sequential(*chain(*up_std_conv_layers))
        self.decoder_f2f = nn.Sequential(*chain(*up_f2f_layers))
        self.decoder_f2v = nn.Sequential(*chain(*up_f2v_layers))

        ''' Final layer (for voxel decoder)'''
        self.final_layer =\
            ConvLayer(in_channels=first_layer_channels,
                      out_channels=self.n_v_classes, kernel_size=1)

        sphere_path=mesh_template
        sphere_vertices, sphere_faces = read_obj(sphere_path)
        sphere_vertices = torch.from_numpy(sphere_vertices).cuda().float()
        self.sphere_vertices = sphere_vertices/torch.sqrt(torch.sum(sphere_vertices**2, dim=1)[:,None])[None]
        self.sphere_faces = torch.from_numpy(sphere_faces).cuda().long()[None]

    @property
    def unpool_indices(self):
        return self._unpool_indices

    @unpool_indices.setter
    def unpool_indices(self, indices):
        """ Set the unpool indices """
        if len(indices) != self.steps + 1:
            raise ValueError("Invalid unpool indices.")
        self._unpool_indices = indices

    @property
    def use_adoptive_unpool(self):
        return self._use_adoptive_unpool

    @use_adoptive_unpool.setter
    def use_adoptive_unpool(self, value: bool):
        """ Define adoptive unpooling """
        self._use_adoptive_unpool = value

    @measure_time
    def forward(self, x):

        batch_size = x.shape[0]

        # Batch of template meshes
        temp_vertices = torch.cat(batch_size * [self.sphere_vertices], dim=0)
        temp_faces = torch.cat(batch_size * [self.sphere_faces], dim=0)
        temp_meshes = Meshes(verts=list(temp_vertices), faces=list(temp_faces))

        # First encoder layer
        x = self.down_layers[0](x)
        down_outputs = [x]

        # Pass through encoder
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x)
            down_outputs.append(x)

        # A separate mesh prediction per class (None for background class 0)
        pred = [None] * self.n_v_classes
        for k in range(1, self.n_v_classes):
            # See definition of the prediction at the end of the function
            pred[k] = [[temp_meshes.clone(),
                        temp_meshes.clone(),
                        None,
                        None]]

        # Iterate over decoder steps
        for i, ((skip_connection, grid_upconv_layer, grid_unet_layer),
                up_f2f_layers,
                up_f2v_layers,
                down_output,
                skip_amount,
                do_unpool) in enumerate(zip(self.up_std_conv_layers,
                                            self.up_f2f_layers,
                                            self.up_f2v_layers,
                                            down_outputs[::-1],
                                            self.skip_count,
                                            self.unpool_indices)
                                        ):

            if grid_upconv_layer is not None and i > 0:
                x = grid_upconv_layer(x)
                x = crop_and_merge(down_output, x)
                x = grid_unet_layer(x)
            elif grid_upconv_layer is None:
                x = down_output
            else:
                raise ValueError("Unknown behavior")

            # Avoid bug related to automatic mixed precision, see also
            # https://github.com/pytorch/pytorch/issues/42218
            with autocast(enabled=False):
                # Iterate over classes ignoring background class 0
                for k in range(1, self.n_v_classes):

                    # Load mesh information from previous iteration for class k
                    prev_meshes = pred[k][i][0]
                    vertices_padded = prev_meshes.verts_padded()[:,:,-3:] # (B,V,3)
                    latent_features_padded = prev_meshes.verts_padded()[:,:,:-3] # (B,V,latent_features_count)
                    faces_padded = prev_meshes.faces_padded() # (B,F,3)

                    # Load template from previous step
                    temp_meshes = pred[k][i][1]
                    temp_vertices_padded = temp_meshes.verts_padded() # (B,V,3)
                    graph_unet_layer = up_f2f_layers[k]
                    feature2vertex = up_f2v_layers[k]

                    if do_unpool == 1:
                        faces_prev = faces_padded
                        _, N_prev, _ = vertices_padded.shape

                        # Get candidate vertices using uniform unpool
                        vertices_padded,\
                                faces_padded_new = uniform_unpool(vertices_padded,
                                                      faces_padded,
                                                      identical_face_batch=False)
                        latent_features_padded, _ = uniform_unpool(latent_features_padded,
                                                      faces_padded,
                                                      identical_face_batch=False)
                        temp_vertices_padded, _ = uniform_unpool(temp_vertices_padded,
                                                      faces_padded,
                                                      identical_face_batch=True)
                        faces_padded = faces_padded_new

                    # Latent features of vertices
                    skipped_features = skip_connection(
                        # Cast x to float32 (comes from autocast region)
                        x[:,:skip_amount].float(), vertices_padded
                    )
                    if latent_features_padded.nelement() > 0:
                        latent_features_padded = torch.cat([latent_features_padded,
                                                     skipped_features,
                                                     vertices_padded], dim=2)
                    else:
                        # First decoder step: No latent features from previous step
                        latent_features_padded = torch.cat([skipped_features,
                                                     vertices_padded], dim=2)

                    # New latent features
                    N_new = latent_features_padded.shape[1]
                    new_meshes = Meshes(latent_features_padded, faces_padded)
                    edges_packed = new_meshes.edges_packed()
                    latent_features_packed = new_meshes.verts_packed()
                    latent_features_packed =\
                        graph_unet_layer(latent_features_packed, edges_packed)

                    # Move vertices
                    deltaV_packed = feature2vertex(latent_features_packed,
                                                   edges_packed)
                    deltaV_padded = deltaV_packed.view(batch_size, N_new, -1)
                    vertices_packed = new_meshes.verts_packed()[:,-3:]
                    vertices_packed = vertices_packed + deltaV_packed

                    # Latent features = (latent features, vertex positions)
                    latent_features_packed = torch.cat([latent_features_packed,
                                                       vertices_packed], dim=1)
                    # !Requires all meshes to have the same number of vertices!
                    latent_features_padded =\
                        latent_features_packed.view(batch_size, N_new, -1)

                    # Final meshes
                    new_meshes = Meshes(latent_features_padded,
                                        new_meshes.faces_padded())

                    if do_unpool == 1 and self.use_adoptive_unpool:
                        raise NotImplementedError("Adoptive unpooling changes the"\
                                                  " number of vertices for each"\
                                                  " mesh which is currently"\
                                                  " expected to lead to problems.")
                        # Discard the vertices that were introduced from the uniform unpool and didn't deform much
                        # vertices, faces, latent_features, temp_vertices_padded = adoptive_unpool(vertices, faces_prev, sphere_vertices, latent_features, N_prev)

                    # Voxel prediction
                    voxel_pred = self.final_layer(x.float()) if i == len(self.up_std_conv_layers) - 1 else None

                    # Template meshes
                    new_temp_meshes = Meshes(temp_vertices_padded, faces_padded)

                # pred for one class at one decoder step has the form
                # [ - batch of pytorch3d prediction Meshes with the last 3 features
                #     being the actual coordinates
                #   - batch of pytorch3d template Meshes (primarily for debugging)
                #   - batch of voxel predictions,
                #   - batch of displacements]
                pred[k] += [[new_meshes, new_temp_meshes, voxel_pred,
                             deltaV_padded]]

        return pred

    def save(self, path):
        """ Save model with its parameters to the given path.
        Conventionally the path should end with "*.model".

        :param str path: The path where the model should be saved.
        """

        torch.save(self.state_dict(), path)

    @staticmethod
    @measure_time
    @deprecated
    def convert_data(data, n_v_classes, mode):
        """ Convert data such that it's compatible with the above voxel2mesh
        implementation.
        """
        x, y = data[0].cuda(), data[1].cuda() # chop
        if x.ndim == 3:
            x = x[None]
        if y.ndim == 3:
            y = y[None]
        shape = torch.tensor(y.shape[1:]) # (D, H, W)
        batch_size = y.shape[0]
        surface_points_normalized_all = []
        for c in range(1, n_v_classes):
            y_outer = sample_outer_surface_in_voxel((y==c).long())
            surface_points = torch.nonzero(y_outer)
            # Coord. 0 = index of data within batch
            batch_ids = surface_points[:,0]
            # Point coordinates
            surface_points_normalized = normalize_vertices_per_max_dim(
                surface_points[:,1:], shape
            )

            surface_points_normalized_batch = []
            # Iterate over minibatch
            for b in range(batch_size):
                points = surface_points_normalized[batch_ids == b]

                # debug
                write_scatter_plot_if_debug(points,
                                            "../misc/surface_points.png")
                perm = torch.randperm(len(points))
                point_count = 3000
                # randomly pick 3000 points
                surface_points_normalized_batch +=\
                    [points[perm[:np.min([len(perm), point_count])]].cuda()]

            surface_points_normalized_all +=\
                [Pointclouds(surface_points_normalized_batch)]

        if mode == ExecModes.TRAIN:
            voxel2mesh_data = {'x': x.float().unsqueeze(1),
                    'y_voxels': y.long(),
                    'surface_points': surface_points_normalized_all
                    }
        elif mode == ExecModes.TEST:
            voxel2mesh_data = {'x': x.float().unsqueeze(1),
                       'y_voxels': y.long(),
                       'vertices_mc': data[2].vertices.cuda(),
                       'faces_mc': data[2].faces.cuda(),
                       'surface_points': surface_points_normalized_all
                       }
        else:
            raise ValueError("Unknown execution mode.")

        return voxel2mesh_data

    @staticmethod
    def pred_to_displacements(pred):
        """ Get the vertex displacements of shape (S,C)
        """
        C = len(pred)
        S = len(pred[1])

        displacements = []
        for s in range(S):
            if s > 0: # No displacements for step 0
                ds = []
                for c in range(C):
                    # No vertices for background
                    if c != 0:
                        _, _, _, disps = pred[c][s]
                        # Mean over vertices since t|V| can vary among steps
                        ds.append(disps.mean(dim=1, keepdim=True))
                displacements.append(torch.stack(ds))
        displacements = torch.stack(displacements)

        return displacements

    @staticmethod
    def pred_to_voxel_pred(pred):
        """ Get the voxel prediction with argmax over classes applied """
        return pred[-1][-1][2].argmax(dim=1).squeeze()

    @staticmethod
    def pred_to_raw_voxel_pred(pred):
        """ Get the voxel prediction per class """
        return pred[-1][-1][2]

    @staticmethod
    def pred_to_verts_and_faces(pred):
        """ Get the vertices and faces of shape (S,C)
        """
        C = len(pred)
        S = len(pred[1])

        vertices = np.empty((S,C), object)
        faces = np.empty((S,C), object)
        for s in range(S):
            for c in range(C):
                # No vertices and faces for background
                if c != 0:
                    meshes, _, _, _ = pred[c][s]
                    vertices[s,c] = meshes.verts_padded()[:,:,-3:]
                    faces[s,c] = meshes.faces_padded()

        return vertices, faces

    @staticmethod
    def pred_to_pred_meshes(pred):
        """ Create valid prediction meshes """
        vertices, faces = Voxel2MeshPlusPlus.pred_to_verts_and_faces(pred)
        # Ignore step 0 and class 0
        pred_meshes = verts_faces_to_Meshes(vertices[1:,1:], faces[1:,1:], 2) # pytorch3d

        return pred_meshes


class Voxel2MeshPlusPlusGeneric(V2MModel):
    """ Voxel2MeshPlusPlus with features taken either from the encoder or the
    decoder. The primary reference for this implementation is
    https://arxiv.org/abs/2102.07899.

    :param n_v_classes: Number of voxel classes to distinguish
    :param n_m_classes: Number of mesh classes to distinguish
    :param patch_shape: The shape of the input patches, e.g. (64, 64, 64)
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
    :param p_dropout: Dropout probability for UNet blocks
    :param ndims: Dimensionality of images
    :param group_structs: Group the structures in the graph network, e.g.,
    group left and right white matter hemisphere into group "white matter".
    During a graph net forward pass, features are exchanged between distinct
    groups but not within a group. For example, white surface vertex positions
    can be provided to the pial vertices and vice versa.
    :param k_struct_neighbors: K for the KNN features of other structures, only
    relevant if group_structs is specified.
    """

    def __init__(self,
                 n_v_classes: int,
                 n_m_classes: int,
                 patch_shape: Union[list, tuple],
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
                 aggregate_indices: Tuple[Tuple[int]],
                 p_dropout: float,
                 ndims: int,
                 group_structs: Tuple[Tuple[int]],
                 k_struct_neighbors: int,
                 **kwargs
                 ):
        super().__init__()

        # Voxel network
        self.voxel_net = ResidualUNet(num_classes=n_v_classes,
                                      num_input_channels=num_input_channels,
                                      patch_shape=patch_shape,
                                      down_channels=encoder_channels,
                                      up_channels=decoder_channels,
                                      deep_supervision=deep_supervision,
                                      voxel_decoder=voxel_decoder,
                                      p_dropout=p_dropout,
                                      ndims=ndims)
        # Graph network
        aggregate = 'trilinear' if ndims == 3 else 'bilinear'
        self.graph_net = GraphDecoder(norm=norm,
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
                                      GC=gc,
                                      group_structs=group_structs,
                                      ndims=ndims)

    @measure_time
    def forward(self, x):

        encoder_skips, decoder_skips, seg_out = self.voxel_net(x)
        pred_meshes, pred_deltaV = self.graph_net(encoder_skips + decoder_skips)

        # pred has the form
        # ( - batch of pytorch3d prediction Meshes with the last 3 features
        #     being the actual coordinates
        #   - batch of voxel predictions,
        #   - batch of displacements)
        pred = (pred_meshes, seg_out, pred_deltaV)

        return pred

    def save(self, path):
        """ Save model with its parameters to the given path.
        Conventionally the path should end with "*.model".

        :param str path: The path where the model should be saved.
        """

        torch.save(self.state_dict(), path)

    @staticmethod
    @measure_time
    @deprecated
    def convert_data(data, n_v_classes, mode):
        """ Convert data such that it's compatible with the above voxel2mesh
        implementation. Currently, it's the same as for Voxel2MeshPlusPlus but
        it may change in the future.
        """
        return Voxel2MeshPlusPlus.convert_data(data, n_v_classes, mode)

    @staticmethod
    def pred_to_displacements(pred):
        """ Get the magnitudes of vertex displacements of shape (S, B, C)
        """
        # No displacements for step 0
        displacements = pred[2][1:]
        # Magnitude
        d_norm = [d.norm(dim=-1) for d in displacements]
        # Mean over vertices since t|V| can vary among steps
        d_norm_mean = [d.mean(dim=-1) for d in d_norm]
        d_norm_mean = torch.stack(d_norm_mean)

        return d_norm_mean

    @staticmethod
    def pred_to_voxel_pred(pred):
        """ Get the final voxel prediction with argmax over classes applied """
        if pred[1] is not None:
            return pred[1][-1].argmax(dim=1).squeeze()
        return None

    @staticmethod
    def pred_to_raw_voxel_pred(pred):
        """ Get the voxel prediction per class. May be a list if deep
        supervision is used. """
        return pred[1]

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
        vertices, faces = Voxel2MeshPlusPlusGeneric.pred_to_verts_and_faces(pred)
        pred_meshes = verts_faces_to_Meshes(vertices, faces, 2) # pytorch3d

        return pred_meshes

    @staticmethod
    def pred_to_pred_deltaV_meshes(pred):
        """ Create valid prediction meshes of shape (S,C) with RELATIVE
        coordinates, i.e., with vertices containing displacement vectors. """
        deltaV, faces = Voxel2MeshPlusPlusGeneric.pred_to_deltaV_and_faces(pred)
        pred_deltaV_meshes = verts_faces_to_Meshes(deltaV, faces, 2) # pytorch3d

        return pred_deltaV_meshes

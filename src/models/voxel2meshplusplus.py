""" Voxel2Mesh++ """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from itertools import chain

import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import GraphConv

from utils.utils import (
    crop_and_merge,
    sample_outer_surface_in_voxel,
    normalize_vertices)
from utils.utils_voxel2meshplusplus.graph_conv import (
    Features2Features,
    Feature2VertexLayer)
from utils.utils_voxel2mesh.feature_sampling import LearntNeighbourhoodSampling
from utils.utils_voxel2mesh.file_handle import read_obj
from utils.utils_voxel2mesh.unpooling import uniform_unpool, adoptive_unpool
from utils.modes import ExecModes
from utils.logging import measure_time, write_scatter_plot_if_debug
from utils.mesh import verts_faces_to_Meshes

from models.u_net import UNetLayer
from models.base_model import V2MModel

class Voxel2MeshPlusPlus(V2MModel):
    """ Voxel2MeshPlusPlus

    Based on Voxel2Mesh from
    https://github.com/cvlab-epfl/voxel2mesh

    :param ndims: Number of input dimensions, only tested with ndims=3
    :param num_classes: Number of classes to distinguish
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
                 num_classes: int,
                 patch_shape,
                 num_input_channels,
                 first_layer_channels,
                 steps,
                 graph_conv_layer_count,
                 batch_norm,
                 mesh_template,
                 unpool_indices,
                 use_adoptive_unpool: bool
                 ):
        super().__init__()

        self.steps = steps

        if ndims == 3:
            self.max_pool = nn.MaxPool3d(2)
        elif ndims == 2:
            self.max_pool = nn.MaxPool2d(2)
        else:
            raise ValueError("Invalid number of dimensions")

        self.num_classes = num_classes
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
                for _ in range(1, self.num_classes):
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
                for _ in range(1, self.num_classes):
                    graph_unet_layers += [Features2Features(self.skip_count[i] + self.latent_features_count[i-1] + dim,
                                                            self.latent_features_count[i],
                                                            hidden_layer_count=graph_conv_layer_count,
                                                            graph_conv=GraphConv)]

            for _ in range(1, self.num_classes):
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
                      out_channels=self.num_classes, kernel_size=1)

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
    def forward(self, data):

        x = data['x']
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
        pred = [None] * self.num_classes
        for k in range(1, self.num_classes):
            # See definition of the prediction at the end of the function
            pred[k] = [[temp_meshes.clone(),
                        temp_meshes.clone(),
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

            # Iterate over classes ignoring background class 0
            for k in range(1, self.num_classes):

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
                skipped_features = skip_connection(x[:, :skip_amount],
                                                   vertices_padded)
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
                voxel_pred = self.final_layer(x) if i == len(self.up_std_conv_layers) - 1 else None

                # Template meshes
                new_temp_meshes = Meshes(temp_vertices_padded, faces_padded)

                # pred for one class at one decoder step has the form
                # [ - batch of pytorch3d prediction Meshes with the last 3 features
                #     being the actual coordinates
                #   - batch of pytorch3d template Meshes (primarily for debugging)
                #   - batch of voxel predictions]
                pred[k] += [[new_meshes, new_temp_meshes, voxel_pred]]

        return pred

    def save(self, path):
        """ Save model with its parameters to the given path.
        Conventionally the path should end with "*.model".

        :param str path: The path where the model should be saved.
        """

        torch.save(self.state_dict(), path)

    @staticmethod
    @measure_time
    def convert_data(data, n_classes, mode):
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
        for c in range(1, n_classes):
            y_outer = sample_outer_surface_in_voxel((y==c).long())
            surface_points = torch.nonzero(y_outer)
            # Coord. 0 = index of data within batch
            batch_ids = surface_points[:,0]
            # Point coordinates
            surface_points_normalized = normalize_vertices(surface_points[:,1:], shape[None])

            surface_points_normalized_batch = []
            # Iterate over minibatch
            for b in range(batch_size):
                points = surface_points_normalized[batch_ids == b]
                points = torch.flip(points, dims=[1]).float() # convert z,y,x -> x, y, z

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
                    meshes, _, _ = pred[c][s]
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

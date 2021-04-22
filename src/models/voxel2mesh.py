""" Voxel2Mesh model from https://github.com/cvlab-epfl/voxel2mesh """

import torch.nn as nn
import torch 
import torch.nn.functional as F 

from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import (chamfer_distance,  mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency)

import numpy as np
from itertools import product, combinations, chain
from scipy.spatial import ConvexHull
from deprecated import deprecated

# from IPython import embed 
import time 

from utils.utils import (
    crop_and_merge,
    sample_outer_surface_in_voxel,
    normalize_vertices)
from utils.utils_voxel2mesh.graph_conv import adjacency_matrix, Features2Features, Feature2VertexLayer 
from utils.utils_voxel2mesh.feature_sampling import LearntNeighbourhoodSampling 
from utils.utils_voxel2mesh.file_handle import read_obj 
from utils.utils_voxel2mesh.unpooling import uniform_unpool, adoptive_unpool
from utils.modes import ExecModes

from models.u_net import UNetLayer


  
 
 
class Voxel2Mesh(nn.Module):
    """ Voxel2Mesh  """
 
    def __init__(self, ndims, num_classes, patch_shape, config):
        super(Voxel2Mesh, self).__init__()

        self.config = config
          
        self.max_pool = nn.MaxPool3d(2) if ndims == 3 else nn.MaxPool2d(2) 
        self.num_classes = num_classes

        ConvLayer = nn.Conv3d if ndims == 3 else nn.Conv2d
        ConvTransposeLayer = nn.ConvTranspose3d if ndims == 3 else nn.ConvTranspose2d
 

        '''  Down layers '''
        down_layers = [UNetLayer(config['NUM_INPUT_CHANNELS'], config['FIRST_LAYER_CHANNELS'], ndims)]
        for i in range(1, config['STEPS'] + 1):
            graph_conv_layer = UNetLayer(config['FIRST_LAYER_CHANNELS'] * 2 ** (i - 1), config['FIRST_LAYER_CHANNELS'] * 2 ** i, ndims)
            down_layers.append(graph_conv_layer)
        self.down_layers = down_layers
        self.encoder = nn.Sequential(*down_layers)
 

        ''' Up layers ''' 
        self.skip_count = []
        self.latent_features_coount = []
        for i in range(config['STEPS']+1):
            self.skip_count += [config['FIRST_LAYER_CHANNELS'] * 2 ** (config['STEPS']-i)] 
            self.latent_features_coount += [32]

        dim = 3

        up_std_conv_layers = []
        up_f2f_layers = []
        up_f2v_layers = []
        for i in range(config['STEPS']+1):
            graph_unet_layers = []
            feature2vertex_layers = []
            skip = LearntNeighbourhoodSampling(patch_shape, config['STEPS'], self.skip_count[i], i)
            # lyr = Feature2VertexLayer(self.skip_count[i])
            if i == 0:
                grid_upconv_layer = None
                grid_unet_layer = None
                for k in range(self.num_classes-1):
                    graph_unet_layers += [Features2Features(self.skip_count[i] + dim, self.latent_features_coount[i], hidden_layer_count=config['GRAPH_CONV_LAYER_COUNT'])] # , graph_conv=GraphConv

            else:
                grid_upconv_layer = ConvTransposeLayer(in_channels=config['FIRST_LAYER_CHANNELS']   * 2**(config['STEPS'] - i+1), out_channels=config['FIRST_LAYER_CHANNELS'] * 2**(config['STEPS']-i), kernel_size=2, stride=2)
                grid_unet_layer = UNetLayer(config['FIRST_LAYER_CHANNELS'] * 2**(config['STEPS'] - i + 1), config['FIRST_LAYER_CHANNELS'] * 2**(config['STEPS']-i), ndims, config['BATCH_NORM'])
                for k in range(self.num_classes-1):
                    graph_unet_layers += [Features2Features(self.skip_count[i] + self.latent_features_coount[i-1] + dim, self.latent_features_coount[i], hidden_layer_count=config['GRAPH_CONV_LAYER_COUNT'])] #, graph_conv=GraphConv if i < config['STEPS'] else GraphConvNoNeighbours

            for k in range(self.num_classes-1):
                feature2vertex_layers += [Feature2VertexLayer(self.latent_features_coount[i], 3)] 
 

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
            ConvLayer(in_channels=config['FIRST_LAYER_CHANNELS'],
                      out_channels=self.num_classes, kernel_size=1)

        sphere_path=config['MESH_TEMPLATE']
        sphere_vertices, sphere_faces = read_obj(sphere_path)
        sphere_vertices = torch.from_numpy(sphere_vertices).cuda().float()
        self.sphere_vertices = sphere_vertices/torch.sqrt(torch.sum(sphere_vertices**2,dim=1)[:,None])[None]
        self.sphere_faces = torch.from_numpy(sphere_faces).cuda().long()[None]


 
  
    def forward(self, data):
         
        x = data['x'] 
        unpool_indices = data['unpool'] 

        sphere_vertices = self.sphere_vertices.clone()
        vertices = sphere_vertices.clone()
        faces = self.sphere_faces.clone() 
 
        # first layer
        x = self.down_layers[0](x)
        down_outputs = [x]

        # down layers
        for unet_layer in self.down_layers[1:]:
            x = self.max_pool(x)
            x = unet_layer(x) 
            down_outputs.append(x)

  
        A, D = adjacency_matrix(vertices, faces)
        pred = [None] * self.num_classes 
        for k in range(self.num_classes-1):
            pred[k] = [[vertices.clone(), faces.clone(), None, None, sphere_vertices.clone()]]

 
        for i, ((skip_connection, grid_upconv_layer, grid_unet_layer), up_f2f_layers, up_f2v_layers, down_output, skip_amount, do_unpool) in enumerate(zip(self.up_std_conv_layers, self.up_f2f_layers, self.up_f2v_layers, down_outputs[::-1], self.skip_count, unpool_indices)):
            if grid_upconv_layer is not None and i > 0:
                x = grid_upconv_layer(x)
                x = crop_and_merge(down_output, x)
                x = grid_unet_layer(x)
            elif grid_upconv_layer is None:
                x = down_output
          

            for k in range(self.num_classes-1):

            	# load mesh information from previous iteratioin for class k
                vertices = pred[k][i][0]
                faces = pred[k][i][1]
                latent_features = pred[k][i][2]
                sphere_vertices = pred[k][i][4]
                graph_unet_layer = up_f2f_layers[k]
                feature2vertex = up_f2v_layers[k]
 
                if do_unpool == 1: # changed do_unpool[0] -> do_unpool
                    faces_prev = faces
                    _, N_prev, _ = vertices.shape 

                    # Get candidate vertices using uniform unpool
                    vertices, faces_ = uniform_unpool(vertices, faces)  
                    latent_features, _ = uniform_unpool(latent_features, faces)  
                    sphere_vertices, _ = uniform_unpool(sphere_vertices, faces) 
                    faces = faces_  

                
                A, D = adjacency_matrix(vertices, faces)
                skipped_features = skip_connection(x[:, :skip_amount], vertices)      
                      
                latent_features = torch.cat([latent_features, skipped_features, vertices], dim=2) if latent_features is not None else torch.cat([skipped_features, vertices], dim=2)
 
                latent_features = graph_unet_layer(latent_features, A, D, vertices, faces)
                deltaV = feature2vertex(latent_features, A, D, vertices, faces)
                vertices = vertices + deltaV 
                
                if do_unpool == 1: # changed do_unpool[0] -> do_unpool
                    # Discard the vertices that were introduced from the uniform unpool and didn't deform much
                    vertices, faces, latent_features, sphere_vertices = adoptive_unpool(vertices, faces_prev, sphere_vertices, latent_features, N_prev)

                

                voxel_pred = self.final_layer(x) if i == len(self.up_std_conv_layers)-1 else None

                pred[k] += [[vertices, faces, latent_features, voxel_pred, sphere_vertices]]
 
        return pred


    @deprecated # Use model-independent loss calculation instead
    def loss(self, data, epoch):

         
        pred = self.forward(data)  
        # embed()
        

         
        CE_Loss = nn.CrossEntropyLoss() 
        ce_loss = CE_Loss(pred[0][-1][3], data['y_voxels'])


        chamfer_loss = torch.tensor(0).float().cuda()
        edge_loss = torch.tensor(0).float().cuda()
        laplacian_loss = torch.tensor(0).float().cuda()
        normal_consistency_loss = torch.tensor(0).float().cuda()  

        for c in range(self.num_classes-1):
            target = data['surface_points'][c].cuda() 
            for k, (vertices, faces, _, _, _) in enumerate(pred[c][1:]):
      
                pred_mesh = Meshes(verts=list(vertices), faces=list(faces))
                pred_points = sample_points_from_meshes(pred_mesh, 3000)
                
                chamfer_loss +=  chamfer_distance(pred_points, target)[0]
                laplacian_loss +=   mesh_laplacian_smoothing(pred_mesh, method="uniform")
                normal_consistency_loss += mesh_normal_consistency(pred_mesh) 
                edge_loss += mesh_edge_loss(pred_mesh) 

        
        
 
        loss = 1 * chamfer_loss + 1 * ce_loss + 0.1 * laplacian_loss + 1 * edge_loss + 0.1 * normal_consistency_loss
 
        log = {"loss": loss.detach(),
               "chamfer_loss": chamfer_loss.detach(), 
               "ce_loss": ce_loss.detach(),
               "normal_consistency_loss": normal_consistency_loss.detach(),
               "edge_loss": edge_loss.detach(),
               "laplacian_loss": laplacian_loss.detach()}
        return loss, log

    def save(self, path):
        """ Save model with its parameters to the given path.
        Conventionally the path should end with "*.model".

        :param str path: The path where the model should be saved.
        """

        torch.save(self.state_dict(), path)


    @staticmethod
    def convert_data_to_voxel2mesh_data(data, n_classes, mode):
        """ Convert data such that it's compatible with the original voxel2mesh
        implementation from above. Code is an assembly of parts from
        https://github.com/cvlab-epfl/voxel2mesh
        """
        shape = torch.tensor(data[1].shape)[None]
        surface_points_normalized_all = []
        for c in range(1, n_classes):
            y_outer = sample_outer_surface_in_voxel((data[1]==c).long())
            surface_points = torch.nonzero(y_outer)
            surface_points = torch.flip(surface_points, dims=[1]).float()  # convert z,y,x -> x, y, z
            surface_points_normalized = normalize_vertices(surface_points, shape)

            perm = torch.randperm(len(surface_points_normalized))
            point_count = 3000
            # randomly pick 3000 points
            surface_points_normalized_all +=\
                [Pointclouds([surface_points_normalized[\
                                    perm[:np.min([len(perm), point_count])]].cuda()])]
        if mode == ExecModes.TRAIN:
            voxel2mesh_data = {'x': data[0].float().cuda()[None][None],
                    'y_voxels': data[1].long().cuda()[None],
                    'surface_points': surface_points_normalized_all,
                    'unpool':[0, 1, 0, 1, 0]
                    }
        elif mode == ExecModes.TEST:
            voxel2mesh_data = {'x': data[0].float().cuda()[None][None],
                       'y_voxels': data[1].long().cuda()[None],
                       'vertices_mc': data[2].vertices.cuda(),
                       'faces_mc': data[2].faces.cuda(),
                       'surface_points': surface_points_normalized_all,
                       'unpool':[0, 1, 1, 1, 1]}
        else:
            raise ValueError("Unknown execution mode.")

        return voxel2mesh_data

    @staticmethod
    def pred_to_voxel_pred(pred):
        """ Get the voxel prediction """
        return pred[0][-1][3].argmax(dim=1).squeeze()

    @staticmethod
    def pred_to_verts_and_faces(pred):
        """ Get the vertices and faces of shape (S,C)
        """
        C = len(pred) - 1 # ignore background class
        S = len(pred[0]) - 1 # ignore step 1

        vertices = []
        faces = []
        for s in range(S):
            step_verts = []
            step_faces = []
            for c in range(C):
                v, f, _, _, _ = pred[c][s+1]
                step_verts.append(v)
                step_faces.append(f)

            vertices.append(step_verts)
            faces.append(step_faces)

        return vertices, faces



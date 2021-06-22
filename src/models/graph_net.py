
""" Graph (sub-)networks """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from typing import Union

import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from pytorch3d.ops import GraphConv

from utils.utils_voxel2mesh.unpooling import uniform_unpool
from utils.utils_voxel2meshplusplus.graph_conv import (
    Features2FeaturesSimpleResidual,
    GraphIdLayer,
    Features2FeaturesResidual,
    zero_weight_init
)
from utils.utils_voxel2meshplusplus.feature_aggregation import (
    aggregate_from_indices
)
from utils.file_handle import read_obj
from utils.logging import measure_time
from utils.utils_voxel2meshplusplus.custom_layers import IdLayer
from utils.mesh import MeshesOfMeshes

class GraphDecoder(nn.Module):
    """ A graph decoder that takes a template mesh and voxel features as input.
    """
    def __init__(self,
                 norm: str,
                 mesh_template: str,
                 unpool_indices: Union[list, tuple],
                 use_adoptive_unpool: bool,
                 graph_channels: Union[list, tuple],
                 skip_channels: Union[list, tuple],
                 weighted_edges: bool,
                 GC,
                 propagate_coords: bool,
                 dim: int=3,
                 aggregate: str='trilinear',
                 n_residual_blocks: int=3,
                 n_f2f_hidden_layer: int=2,
                 aggregate_indices=((3,4,5,6),
                                    (2,3,6,7),
                                    (1,2,7,8),
                                    (0,1,7,8))): # 8 = last decoder skip
        super().__init__()

        assert (len(graph_channels) - 1 ==\
                len(aggregate_indices) ==\
                len(unpool_indices)),\
                "Graph channels, aggregation indices, and unpool indices must"\
                " match the number of mesh decoder steps."

        if weighted_edges and not propagate_coords:
            raise ValueError("Edge weighing requires propagation of vertex"
                             " coordinates to the graph convs.")

        # Number of vertex latent features (1D)
        self.latent_features_count = graph_channels
        # The initial creation of latent features from vertex coordinates
        # does not count as a decoder step
        self.num_steps = len(graph_channels) - 1
        self.aggregate_indices = aggregate_indices
        self.unpool_indices = unpool_indices
        self.use_adoptive_unpool = use_adoptive_unpool
        self.GC = GC

        # Aggregation of voxel features
        self.aggregate = aggregate

        # Initial creation of latent features from coordinates
        self.graph_conv_first = Features2FeaturesResidual(
            dim, graph_channels[0], n_f2f_hidden_layer, norm=norm,
            GC=GC, weighted_edges=weighted_edges
        )

        # Graph decoder
        f2f_res_layers = [] # Residual feature to feature blocks
        f2v_layers = [] # Features to vertices

        # Whether or not to add the vertex coordinates to the features again
        # after each step
        self.propagate_coords = propagate_coords
        if propagate_coords:
            add_n = 3
        else:
            add_n = 0

        for i in range(self.num_steps):
            # Multiple sequential graph residual blocks
            indices = aggregate_indices[i]
            skip_features_count =\
                torch.sum(torch.tensor(skip_channels)[indices, None]).item()
            res_blocks = [Features2FeaturesResidual(
                skip_features_count + self.latent_features_count[i] + add_n,
                self.latent_features_count[i+1],
                hidden_layer_count=n_f2f_hidden_layer,
                norm=norm,
                GC=GC,
                weighted_edges=weighted_edges
            )]
            for _ in range(n_residual_blocks - 1):
                res_blocks.append(Features2FeaturesResidual(
                    self.latent_features_count[i+1],
                    self.latent_features_count[i+1],
                    hidden_layer_count=n_f2f_hidden_layer,
                    norm=norm,
                    GC=GC,
                    weighted_edges=False # No weighted edges here
                ))

            # Cannot be nn.Sequential because graph convs take two inputs but
            # provide only one output. Maybe try torch_geometric.nn.Sequential
            res_blocks = nn.ModuleList(res_blocks)
            f2f_res_layers.append(res_blocks)

            # Feature to vertex layer, edge weighing never used
            f2v_layers.append(GC(
                self.latent_features_count[i+1], dim, weighted_edges=False,
                init='zero'
            ))

        self.f2f_res_layers = nn.ModuleList(f2f_res_layers)
        self.f2v_layers = nn.ModuleList(f2v_layers)

        # Init f2v layers to zero
        self.f2v_layers.apply(zero_weight_init)

        # Template (batch size 1)
        sphere_path = mesh_template
        sphere_vertices, sphere_faces, _ = read_obj(sphere_path)
        sphere_vertices = torch.from_numpy(sphere_vertices).cuda().float()

        # Normalize template only if coords not in [-1, 1]
        if sphere_vertices.max() > 1 or sphere_vertices.min() < -1:
            self.sphere_vertices = sphere_vertices/torch.sqrt(torch.sum(sphere_vertices**2, dim=1)[:,None])[None]
        else:
            self.sphere_vertices = sphere_vertices[None]

        self.sphere_faces = torch.from_numpy(sphere_faces).cuda().long()[None]

    @property
    def unpool_indices(self):
        return self._unpool_indices

    @unpool_indices.setter
    def unpool_indices(self, indices):
        """ Set the unpool indices """
        if len(indices) != self.num_steps:
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
    def forward(self, skips):

        # Batch of template meshes
        batch_size = skips[0].shape[0]
        temp_vertices = torch.cat(batch_size * [self.sphere_vertices], dim=0)
        temp_faces = torch.cat(batch_size * [self.sphere_faces], dim=0)
        temp_meshes = MeshesOfMeshes(verts=temp_vertices, faces=temp_faces)

        _, M, V, _ = temp_vertices.shape

        skips = [s.float() for s in skips]

        # No autocast for pytorch3d convs possible
        cast = not issubclass(self.GC, GraphConv)
        with autocast(enabled=cast):
            # First graph conv: Vertex coords --> latent features
            edges_packed = temp_meshes.edges_packed()
            verts_packed = temp_meshes.verts_packed()
            latent_features = self.graph_conv_first(verts_packed, edges_packed)
            temp_meshes.update_features(
                latent_features.view(batch_size, M, V, -1)
            )

            pred_meshes = [temp_meshes]
            # No delta V for initial step
            pred_deltaV = [None]

            # Iterate over decoder steps
            for i, (f2f_res,
                    f2v,
                    agg_indices,
                    do_unpool) in enumerate(zip(
                        self.f2f_res_layers,
                        self.f2v_layers,
                        self.aggregate_indices,
                        self.unpool_indices)):

                # Load mesh information from previous iteration for class k
                prev_meshes = pred_meshes[i]
                vertices_padded = prev_meshes.verts_padded() # (N,M,V,3)
                latent_features_padded = prev_meshes.features_padded() # (N,M,V,latent_features_count)
                faces_padded = prev_meshes.faces_padded() # (N,M,F,3)

                if do_unpool == 1:
                    faces_prev = faces_padded
                    _, _, V_prev, _ = vertices_padded.shape

                    # Get candidate vertices using uniform unpool
                    vertices_padded,\
                            faces_padded_new = uniform_unpool(vertices_padded,
                                                  faces_padded,
                                                  identical_face_batch=False)
                    latent_features_padded, _ = uniform_unpool(latent_features_padded,
                                                  faces_padded,
                                                  identical_face_batch=False)
                    faces_padded = faces_padded_new
                V_new = latent_features_padded.shape[2]

                # Latent features of vertices from voxels
                # Avoid bug related to automatic mixed precision, see also
                # https://github.com/pytorch/pytorch/issues/42218
                with autocast(enabled=False):
                    skipped_features = aggregate_from_indices(
                        skips,
                        vertices_padded.view(batch_size, -1, 3),
                        agg_indices,
                        mode=self.aggregate
                    ).view(batch_size, M, V_new, -1)

                latent_features_padded = torch.cat(
                    (latent_features_padded, skipped_features), dim=3
                )

                # New latent features
                new_meshes = MeshesOfMeshes(vertices_padded, faces_padded,
                                            latent_features_padded)
                edges_packed = new_meshes.edges_packed()
                if self.propagate_coords:
                    latent_features_packed = new_meshes.features_verts_packed()
                else:
                    latent_features_packed = new_meshes.features_packed()
                for f2f in f2f_res:
                    latent_features_packed =\
                        f2f(latent_features_packed, edges_packed)
                new_meshes.update_features(
                    latent_features_packed.view(batch_size, M, V_new, -1)
                )

                # Move vertices
                deltaV_packed = f2v(latent_features_packed, edges_packed)
                deltaV_padded = deltaV_packed.view(batch_size, M, V_new, 3)
                new_meshes.move_verts(deltaV_padded)

                # New latent features
                if self.propagate_coords:
                    latent_features_packed = new_meshes.features_verts_packed()

                if do_unpool == 1 and self.use_adoptive_unpool:
                    raise NotImplementedError("Adoptive unpooling changes the"\
                                              " number of vertices for each"\
                                              " mesh which is currently"\
                                              " expected to lead to problems.")
                    # Discard the vertices that were introduced from the uniform unpool and didn't deform much
                    # vertices, faces, latent_features, temp_vertices_padded = adoptive_unpool(vertices, faces_prev, sphere_vertices, latent_features, V_prev)

                pred_meshes.append(new_meshes)
                pred_deltaV.append(deltaV_padded)

        return pred_meshes, pred_deltaV

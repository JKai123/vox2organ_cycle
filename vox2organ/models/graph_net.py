
""" Graph (sub-)networks """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from typing import Union, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from pytorch3d.ops import GraphConv

from utils.graph_conv import (
    Features2FeaturesResidual,
    zero_weight_init
)
from utils.feature_aggregation import (
    aggregate_structural_features,
    aggregate_from_indices
)
from utils.logging import measure_time
from utils.utils_padded_packed import pack, unpack
from utils.mesh import MeshesOfMeshes, Mesh
from utils.coordinate_transform import (
    normalize_vertices,
    unnormalize_vertices,
    unnormalize_vertices_per_max_dim,
    normalize_vertices_per_max_dim
)

class GraphDecoder(nn.Module):
    """ A graph decoder that takes a template mesh and voxel features as input.
    """
    def __init__(self,
                 norm: str,
                 mesh_template: MeshesOfMeshes,
                 unpool_indices: Union[list, tuple],
                 use_adoptive_unpool: bool,
                 graph_channels: Union[list, tuple],
                 skip_channels: Union[list, tuple],
                 weighted_edges: bool,
                 GC,
                 propagate_coords: bool,
                 patch_size: Tuple[int, int, int],
                 aggregate_indices: Tuple[Tuple[int]],
                 exchange_coords: bool,
                 group_structs: Tuple[Tuple[int]]=None,
                 p_dropout: float=None,
                 k_struct_neighbors=5,
                 ndims: int=3,
                 aggregate: str='trilinear',
                 n_vertex_classes: int=36,
                 n_residual_blocks: int=3,
                 n_f2f_hidden_layer: int=2):
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
        self.p_dropout = p_dropout
        self.patch_size = patch_size
        self.ndims = ndims
        self.group_structs = group_structs
        self.k_struct_neighbors = k_struct_neighbors
        self.exchange_coords = exchange_coords

        # Aggregation of voxel features
        self.aggregate = aggregate

        # Initial creation of latent features from coordinates and vertex
        # classes
        self.graph_conv_first = Features2FeaturesResidual(
            ndims + n_vertex_classes,
            graph_channels[0],
            n_f2f_hidden_layer,
            norm=norm,
            GC=GC,
            p_dropout=None, # no dropout here
            weighted_edges=weighted_edges
        )

        # Graph decoder
        f2f_res_layers = [] # Residual feature to feature blocks
        f2v_layers = [] # Features to vertices
        lns_layers = [] # Learnt neighborhood sampling (optional)
        f2f_res_layers_cycle = [] # Residual feature to feature blocks
        f2v_layers_cycle = [] # Features to vertices

        # Whether or not to add the vertex coordinates to the features again
        # after each step
        self.propagate_coords = propagate_coords
        if propagate_coords:
            add_n = ndims
        else:
            add_n = 0

        # Additional structural information added
        if group_structs:
            if exchange_coords:
                # Neighbor coordinates + surface id
                add_n += k_struct_neighbors * ndims +\
                        int(np.ceil(np.log2(len(group_structs))))
            else:
                # Surface id
                add_n += int(np.ceil(np.log2(len(group_structs))))

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
                p_dropout=p_dropout,
                weighted_edges=weighted_edges
            )]
            for _ in range(n_residual_blocks - 1):
                res_blocks.append(Features2FeaturesResidual(
                    self.latent_features_count[i+1],
                    self.latent_features_count[i+1],
                    hidden_layer_count=n_f2f_hidden_layer,
                    norm=norm,
                    GC=GC,
                    p_dropout=p_dropout,
                    weighted_edges=False # No weighted edges here
                ))

            # Cannot be nn.Sequential because graph convs take two inputs but
            # provide only one output. Maybe try torch_geometric.nn.Sequential
            res_blocks = nn.ModuleList(res_blocks)
            f2f_res_layers.append(res_blocks)

            # Feature to vertex layer, edge weighing never used
            f2v_layers.append(GC(
                self.latent_features_count[i+1], ndims, weighted_edges=False,
                init='zero'
            ))

            # Optionally create lns layers
            if self.aggregate == 'lns':
                raise NotImplementedError("LNS not implemented, see"
                                          " Voxel2Mesh repo.")


            # Create layers for cycle graph
            res_blocks_cycle = [
                Features2FeaturesResidual(
                    add_n,
                    self.latent_features_count[i+1],
                    hidden_layer_count=n_f2f_hidden_layer,
                    norm=norm,
                    GC=GC,
                    p_dropout=p_dropout,
                    weighted_edges=weighted_edges 
                )
            ]
            for _ in range(n_residual_blocks - 1): # TODO
                res_blocks_cycle.append(Features2FeaturesResidual(
                    self.latent_features_count[i+1],
                    self.latent_features_count[i+1],
                    hidden_layer_count=n_f2f_hidden_layer,
                    norm=norm,
                    GC=GC,
                    p_dropout=p_dropout,
                    weighted_edges=False # No weighted edges here
                ))


            # Cannot be nn.Sequential because graph convs take two inputs but
            # provide only one output. Maybe try torch_geometric.nn.Sequential
            res_blocks_cycle = nn.ModuleList(res_blocks_cycle)
            f2f_res_layers_cycle.append(res_blocks_cycle)

            # Feature to vertex layer, edge weighing never used
            f2v_layers_cycle.append(GC(
                self.latent_features_count[i+1], ndims, weighted_edges=False,
                init='normal'
            ))


        self.f2f_res_layers_cycle = nn.ModuleList(f2f_res_layers_cycle)
        self.f2v_layers_cycle = nn.ModuleList(f2v_layers_cycle)


        self.f2f_res_layers = nn.ModuleList(f2f_res_layers)
        self.f2v_layers = nn.ModuleList(f2v_layers)
        self.lns_layers = nn.ModuleList(lns_layers)

        # Init f2v layers to zero
        self.f2v_layers.apply(zero_weight_init)

        # Template (batch size 1)
        # TODO maybe recomment this
        self.mesh_template = mesh_template
        self.sphere_vertices = mesh_template.verts_padded().cuda()
        self.sphere_faces = mesh_template.faces_padded().cuda()
        self.sphere_features = mesh_template.features_padded().cuda()
        assert self.sphere_features.max().cpu().item() == n_vertex_classes - 1

        # # Assert correctness of the structure grouping
        if group_structs:
            n_m_classes = self.sphere_vertices.shape[1]
            gs_tensor = torch.tensor(group_structs)
            gs_unique = set(torch.unique(gs_tensor).tolist())
            assert len(set(range(n_m_classes)) - gs_unique) == 0,\
                    "Missing structure IDs in structure groups."
            assert torch.bincount(gs_tensor.flatten()).max() == 1,\
                    "Structure IDs must only occur once."

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

        # Batch of template meshes --> increases the first dimension to fit the batch size
        batch_size = skips[0].shape[0]
        temp_vertices = torch.cat(batch_size * [self.sphere_vertices], dim=0)
        temp_faces = torch.cat(batch_size * [self.sphere_faces], dim=0)
        temp_features = torch.cat(batch_size * [self.sphere_features], dim=0)
        temp_meshes = MeshesOfMeshes(
            verts=temp_vertices, 
            faces=temp_faces, 
            features=temp_features,
            verts_mask= self.mesh_template.verts_mask(), 
            faces_mask=self.mesh_template.faces_mask() , 
            normals_mask=self.mesh_template.normals_mask(), 
            features_mask=self.mesh_template.features_mask()
        )

        _, M, V, _ = temp_vertices.shape

        skips = [s.float() for s in skips]

        # No autocast for pytorch3d convs possible
        cast = not issubclass(self.GC, GraphConv)
        with autocast(enabled=cast):
            # First graph conv: Vertex coords & classes --> latent features
            edges_packed = temp_meshes.edges_packed().cuda()
            verts_packed = temp_meshes.verts_packed().cuda()
            features_packed = temp_meshes.features_packed().cuda()
            test_cat = torch.cat(
                    # Classes one-hot encoded
                    [verts_packed, F.one_hot(
                        features_packed.squeeze()
                    ).float().cuda()],
                    dim=-1
                )
            latent_features = self.graph_conv_first(
                torch.cat(
                    # Classes one-hot encoded
                    [verts_packed, F.one_hot(
                        features_packed.squeeze()
                    ).float().cuda()],
                    dim=-1
                ),
                edges_packed
            )
            temp_meshes.update_features(unpack(latent_features,temp_meshes.features_mask(), batch_size)
            )

            pred_meshes = [temp_meshes]
            # No delta V for initial step
            pred_deltaV = [None]
            cycle_pred_meshes = [None]

            # Iterate over decoder steps
            for i, (f2f_res,
                    f2f_res_cycle,
                    f2v,
                    f2v_cycle,
                    agg_indices,
                    do_unpool) in enumerate(zip(
                        self.f2f_res_layers,
                        self.f2f_res_layers_cycle,
                        self.f2v_layers,
                        self.f2v_layers_cycle,
                        self.aggregate_indices,
                        self.unpool_indices)):

                # Load mesh information from previous iteration for class k
                prev_meshes = pred_meshes[i]
                vertices_padded = prev_meshes.verts_padded().cuda() # (N,M,V,3)
                latent_features_padded = prev_meshes.features_padded().cuda() # (N,M,V,latent_features_count)
                faces_padded = prev_meshes.faces_padded().cuda() # (N,M,F,3)

                if do_unpool == 1:
                    raise NotImplementedError("Unpooling not implemented.")

                # Number of vertices might have changed
                V_new = latent_features_padded.shape[2]

                # Mesh coordinates for F.grid_sample
                verts_img_co = normalize_vertices(
                    unnormalize_vertices_per_max_dim(
                        vertices_padded.view(-1, self.ndims),
                        self.patch_size
                    ),
                    self.patch_size
                ).view(batch_size, -1, self.ndims)

                # Latent features of vertices from voxels
                # Avoid bug related to automatic mixed precision, see also
                # https://github.com/pytorch/pytorch/issues/42218
                with autocast(enabled=False):
                    if self.aggregate in ('trilinear', 'bilinear'):
                        skipped_features = aggregate_from_indices(
                            skips,
                            verts_img_co,
                            agg_indices,
                            mode=self.aggregate
                        ).view(batch_size, M, V_new, -1)
                    elif self.aggregate == 'lns':
                        # Learnt neighborhood sampling
                        assert len(agg_indices) == 1,\
                                "Does only work with single feature map."
                        skipped_features = self.lns_layers[i](
                            skips[agg_indices[0]], verts_img_co
                        ).view(batch_size, M, V_new, -1)
                    else:
                        raise ValueError("Unknown aggregation scheme ",
                                         self.aggregate)

                # Latent features related to structural information (ID of
                # structure, geometry of nearby structure)
                if self.group_structs:
                    struct_features = aggregate_structural_features(
                        vertices_padded,
                        self.group_structs,
                        self.exchange_coords,
                        self.k_struct_neighbors
                    )
                    # Concatenate along feature dimension
                    latent_features_padded = torch.cat(
                        (latent_features_padded,
                         skipped_features,
                         struct_features),
                        dim=3
                    )
                else: # No struct features
                    # Concatenate along feature dimension
                    latent_features_padded = torch.cat(
                        (latent_features_padded, skipped_features), dim=3
                    )

                # New latent features
                # TODO maybe masks are wrong if number of vertices has changed
                new_meshes = MeshesOfMeshes(
                    vertices_padded, 
                    faces_padded, 
                    features=latent_features_padded,
                    verts_mask= prev_meshes.verts_mask(), 
                    faces_mask=prev_meshes.faces_mask() , 
                    normals_mask=prev_meshes.normals_mask(), 
                    features_mask=prev_meshes.features_mask()
                )
                edges_packed = new_meshes.edges_packed()
                if self.propagate_coords:
                    latent_features_packed = new_meshes.features_verts_packed()
                else:
                    latent_features_packed = new_meshes.features_packed()
                for f2f in f2f_res:
                    latent_features_packed =\
                        f2f(latent_features_packed, edges_packed)
                # TODO Falsch wenn sich V geändert hat
                new_meshes.update_features(unpack(latent_features_packed,temp_meshes.features_mask(), batch_size)
                )

                # Move vertices
                deltaV_packed = f2v(latent_features_packed, edges_packed)
                deltaV_padded = unpack(deltaV_packed, new_meshes.verts_mask(), batch_size)
                new_deltaV_mesh = MeshesOfMeshes(
                    deltaV_padded, 
                    faces_padded, 
                    features=latent_features_padded,
                    verts_mask= prev_meshes.verts_mask(), 
                    faces_mask=prev_meshes.faces_mask() , 
                    normals_mask=prev_meshes.normals_mask(), 
                    features_mask=prev_meshes.features_mask())
                    
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



                # Cycle Mesh
                cycle_vertices_padded = new_meshes.verts_padded().cuda() # (N,M,V,3)
                cycle_latent_features_padded = new_meshes.features_padded().cuda() # (N,M,V,latent_features_count)
                cycle_faces_padded = new_meshes.faces_padded().cuda() # (N,M,F,3)
                latent_features_packed_cycle = new_meshes.verts_packed()
                edges_packed = new_meshes.edges_packed()


                cycle_new_meshes = MeshesOfMeshes(
                    cycle_vertices_padded, 
                    cycle_faces_padded, 
                    features=cycle_latent_features_padded,
                    verts_mask= new_meshes.verts_mask(), 
                    faces_mask=new_meshes.faces_mask() , 
                    normals_mask=new_meshes.normals_mask(), 
                    features_mask=new_meshes.features_mask()
                )

                for f2f in f2f_res_cycle:
                    latent_features_packed_cycle =\
                        f2f(latent_features_packed_cycle, edges_packed)
                deltaV_packed_cycle = f2v_cycle(latent_features_packed_cycle, edges_packed)
                deltaV_padded_cycle = unpack(deltaV_packed_cycle, new_meshes.verts_mask(), batch_size)
                cycle_new_meshes.move_verts(deltaV_padded_cycle)

                cycle_pred_meshes.append(cycle_new_meshes)
                pred_meshes.append(new_meshes)
                pred_deltaV.append(new_deltaV_mesh)

            # Replace features with template features (e.g. vertex classes)
            for m in pred_meshes:
                m.update_features(
                    self.mesh_template.features_padded().expand(
                        batch_size, M, V, -1
                    ).cuda()
                )

        return pred_meshes, pred_deltaV, cycle_pred_meshes

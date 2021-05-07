
""" Graph conv blocks for Voxel2MeshPlusPlus.

Implementation based on https://github.com/cvlab-epfl/voxel2mesh.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import GraphConv
from torch_geometric.nn import GCNConv, ChebConv

class GraphConvNorm(GraphConv):
    """ Wrapper for pytorch3d.ops.GraphConv that normalizes the features
    w.r.t. the degree of the vertices.
    """
    def __init__(self, input_dim: int, output_dim: int, init: str='normal',
                 directed: bool=False):
        super().__init__(input_dim, output_dim, init, directed)

    def forward(self, verts, edges):
        D_inv = 1.0 / torch.unique(edges, return_counts=True)[1].unsqueeze(1)
        return D_inv * super().forward(verts, edges)

class PTGeoConvWrapped(GCNConv):
    """ Wrapper for torch_geometric.nn graph convolution s.t. inputs are the same as for
    pytorch3d convs.
    """
    def __init__(self, input_dim, output_dim, **kwargs):
        super().__init__(input_dim, output_dim, **kwargs)
        nn.init.normal_(self.weight, mean=0.0, std=0.1)

    def forward(self, verts, edges):
        edges_both_dir = torch.cat([edges.T, torch.flip(edges.T, dims=[0])],
                                   dim=1)
        return super().forward(verts, edges_both_dir)

class Features2Features(nn.Module):
    """ A graph conv block """

    def __init__(self, in_features, out_features, hidden_layer_count,
                 GC=GraphConv):
        # TODO: batch norm?
        super().__init__()

        self.gconv_first = GC(in_features, out_features)
        gconv_hidden = []
        for _ in range(hidden_layer_count):
            gconv_hidden += [GC(out_features, out_features)]
        self.gconv_hidden = nn.Sequential(*gconv_hidden)
        self.gconv_last = GC(out_features, out_features)

    def forward(self, features, edges):
        features = F.relu(self.gconv_first(features, edges))
        for gconv_hidden in self.gconv_hidden:
            features = F.relu(gconv_hidden(features, edges))
        return self.gconv_last(features, edges)

class Feature2VertexLayer(nn.Module):
    """ Neural mapping from vertex features to vertex coordinates (can be
    absolute or relative coordinates"""

    def __init__(self, in_features, hidden_layer_count, batch_norm=False,
                 GC=GraphConvNorm):
        super().__init__()
        self.batch_norm = batch_norm
        self.layers = []
        for i in range(hidden_layer_count, 1, -1):
            layer_in_features = i * in_features // hidden_layer_count
            layer_out_features = (i-1) * in_features // hidden_layer_count
            self.layers += [GC(layer_in_features, layer_out_features)]
            if self.batch_norm:
                self.layers += [nn.BatchNorm1d(layer_out_features)]

        self.gconv_layers = nn.Sequential(*self.layers)

        # Output of F2V should be 3D coordinates
        self.gconv_last = GC(in_features // hidden_layer_count, 3)

    def forward(self, features, edges):
        for layer in self.layers:
            if isinstance(layer, (GraphConv, GCNConv, ChebConv)):
                features = layer(features, edges)
            elif isinstance(layer, nn.BatchNorm1d):
                features = layer(features)
            else:
                raise ValueError("Unknown layer type.")
            features = F.relu(features)
        return self.gconv_last(features, edges)

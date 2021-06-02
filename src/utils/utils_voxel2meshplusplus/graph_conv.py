
""" Graph conv blocks for Voxel2MeshPlusPlus.

Implementation based on https://github.com/cvlab-epfl/voxel2mesh.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import GraphConv
from torch_geometric.nn import GCNConv, ChebConv, GINConv
from torch_sparse import SparseTensor

from utils.utils_voxel2meshplusplus.custom_layers import IdLayer
from utils.utils import Euclidean_weights

class GraphConvNorm(GraphConv):
    """ Wrapper for pytorch3d.ops.GraphConv that normalizes the features
    w.r.t. the degree of the vertices.
    """
    def __init__(self, input_dim: int, output_dim: int, init: str='normal',
                 directed: bool=False, **kwargs):
        super().__init__(input_dim, output_dim, init, directed)
        # Bug in GraphConv: bias is not initialized to zero
        if init == 'zero':
            self.w0.bias.data.zero_()
            self.w1.bias.data.zero_()
        if kwargs.get('weighted_edges', False) == True:
            raise ValueError(
                "pytorch3d.ops.GraphConv cannot be edge-weighted."
            )

    def forward(self, verts, edges):
        D_inv = 1.0 / torch.unique(edges, return_counts=True)[1].unsqueeze(1)
        return D_inv * super().forward(verts, edges)

class GCNConvWrapped(GCNConv):
    """ Wrapper for torch_geometric.nn graph convolution s.t. inputs are the same as for
    pytorch3d convs.
    """
    def __init__(self, input_dim, output_dim, weighted_edges, **kwargs):
        # Maybe cached?
        super().__init__(input_dim, output_dim, improved=True, **kwargs)

        # Zero initialization to avoid large non-sense displacements at the
        # beginning
        nn.init.constant_(self.weight, 0.0)
        self.weighted_edges = weighted_edges

    def forward(self, verts, edges):
        edges_both_dir = torch.cat([edges.T, torch.flip(edges.T, dims=[0])],
                                   dim=1)
        if self.weighted_edges:
            # Assume that verts contains the absolute coordinates at verts[-3:] and
            # feature sizes are multiples of two
            assert (verts.shape[1] - 3) % 2 == 0,\
                    "Features should be (vertex features, vertex coordinates)"
            coords = verts[:,-3:]
            weights = Euclidean_weights(coords, edges_both_dir.T)
        else:
            weights = None

        return super().forward(verts, edges_both_dir, weights)

class GINConvWrapped(GINConv):
    """ Wrapper for torch_geometric.nn graph convolution s.t. inputs are the same as for
    pytorch3d convs.
    """
    def __init__(self, input_dim, output_dim, weighted_edges, **kwargs):
        super().__init__(nn=nn.Linear(input_dim, output_dim), **kwargs)

        # Zero initialization to avoid large non-sense displacemnets at the
        # beginning
        nn.init.constant_(self.nn.weight, 0.0)

    def forward(self, verts, edges):
        edges_both_dir = torch.cat([edges.T, torch.flip(edges.T, dims=[0])],
                                   dim=1)
        # A_sparse = SparseTensor(row=edges_both_dir[0], col=edges_both_dir[1]).t()

        return super().forward(verts, edges_both_dir)

class Features2Features(nn.Module):
    """ A graph conv block """

    def __init__(self, in_features, out_features, hidden_layer_count,
                 batch_norm=False, GC=GraphConv):
        super().__init__()

        # Only make a residual layer if #input features = #output features
        self.is_residual = in_features == out_features

        self.gconv_first = GC(in_features, out_features)
        if batch_norm:
            self.bn_first = nn.BatchNorm1d(out_features)
        else:
            self.bn_first = IdLayer()
        gconv_hidden = []
        for _ in range(hidden_layer_count):
            gc_layer = GC(out_features, out_features)
            if batch_norm:
                bn_layer = nn.BatchNorm1d(out_features)
            else:
                bn_layer = IdLayer() # Id

            gconv_hidden += [nn.Sequential(gc_layer, bn_layer)]

        self.gconv_hidden = nn.Sequential(*gconv_hidden)
        self.gconv_last = GC(out_features, out_features)

    def forward(self, features, edges):
        res = features
        # Conv --> Norm --> ReLU
        features = F.relu(self.bn_first(self.gconv_first(features, edges)))
        for i, (gconv, bn) in enumerate(self.gconv_hidden, 1):
            # Adding residual before last relu
            if i == len(self.gconv_hidden) and self.is_residual:
                features = bn(gconv(features, edges)) + res
                features = F.relu(features)
            else:
                features = F.relu(bn(gconv(features, edges)))

        return self.gconv_last(features, edges)

class Feature2VertexLayer(nn.Module):
    """ Neural mapping from vertex features to vertex coordinates (can be
    absolute or relative coordinates"""

    def __init__(self, in_features, hidden_layer_count, batch_norm=False,
                 GC=GraphConvNorm):
        super().__init__()
        self.feature_layers = []
        for i in range(hidden_layer_count, 1, -1):
            layer_in_features = i * in_features // hidden_layer_count
            layer_out_features = (i-1) * in_features // hidden_layer_count
            gc_layer = GC(layer_in_features, layer_out_features)
            if batch_norm:
                bn_layer = nn.BatchNorm1d(layer_out_features)
            else:
                bn_layer = IdLayer() # Id

            layer = nn.Sequential(gc_layer, bn_layer)
            self.feature_layers.append(layer)

        self.feature_layers = nn.Sequential(*self.feature_layers)

        # Output of F2V should be 3D coordinates
        self.gconv_last = GC(in_features // hidden_layer_count, 3)

    def forward(self, features, edges):
        for conv_layer, bn in self.feature_layers:
            # Conv --> Norm --> ReLU
            features = conv_layer(features, edges)
            features = F.relu(bn(features))

        return self.gconv_last(features, edges)

class Features2FeaturesResidual(nn.Module):
    """ A residual graph conv block consisting of 'hidden_layer_count' many graph convs """

    def __init__(self, in_features, out_features, hidden_layer_count,
                 batch_norm=False, GC=GraphConv, weighted_edges=False):
        super().__init__()

        self.out_features = out_features

        self.gconv_first = GC(in_features, out_features, weighted_edges=weighted_edges)
        if batch_norm:
            self.bn_first = nn.BatchNorm1d(out_features)
        else:
            self.bn_first = IdLayer()

        gconv_hidden = []
        for _ in range(hidden_layer_count):
            # No weighted edges and no propagated coordinates in hidden layers
            gc_layer = GC(out_features, out_features, weighted_edges=False)
            if batch_norm:
                bn_layer = nn.BatchNorm1d(out_features)
            else:
                bn_layer = IdLayer() # Id

            gconv_hidden += [nn.Sequential(gc_layer, bn_layer)]

        self.gconv_hidden = nn.Sequential(*gconv_hidden)

    def forward(self, features, edges):
        if features.shape[-1] == self.out_features:
            res = features
        else:
            res = F.interpolate(features.unsqueeze(1), self.out_features,
                                mode='nearest').squeeze(1)

        # Conv --> Norm --> ReLU
        features = F.relu(self.bn_first(self.gconv_first(features, edges)))
        for i, (gconv, bn) in enumerate(self.gconv_hidden, 1):
            # Adding residual before last relu
            if i == len(self.gconv_hidden):
                features = bn(gconv(features, edges)) + res
                features = F.relu(features)
            else:
                features = F.relu(bn(gconv(features, edges)))

        return features

class Features2FeaturesSimple(nn.Module):
    """ A simple graph conv + batch norm (optional) + ReLU """

    def __init__(self, in_features, out_features,
                 batch_norm=False, GC=GraphConv,
                 weighted_edges=False):
        super().__init__()

        self.gconv = GC(in_features, out_features, weighted_edges=weighted_edges)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_features)
        else:
            self.bn = IdLayer()

    def forward(self, features, edges):
        # Conv --> Norm --> ReLU
        return F.relu(self.bn(self.gconv(features, edges)))

class Features2FeaturesSimpleResidual(nn.Module):
    """ A simple residual graph conv + batch norm (optional) + ReLU """

    def __init__(self, in_features, out_features,
                 batch_norm=False, GC=GraphConv,
                 weighted_edges=False):
        super().__init__()

        self.out_features = out_features
        self.gconv = GC(in_features, out_features, weighted_edges=weighted_edges)
        if batch_norm:
            self.bn = nn.BatchNorm1d(out_features)
        else:
            self.bn = IdLayer()

    def forward(self, features, edges):
        if features.shape[-1] == self.out_features:
            res = features
        else:
            res = F.interpolate(features.unsqueeze(1), self.out_features,
                                mode='nearest').squeeze(1)
        # Conv --> Norm --> ReLU
        return F.relu(self.bn(self.gconv(features, edges)) + res)

class GraphIdLayer(nn.Module):
    """ Graph identity layer """

    def __init__(self):
        super().__init__()

    def forward(self, features, edges):
        return features

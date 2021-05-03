
""" Graph conv blocks for Voxel2MeshPlusPlus.

Implementation based on https://github.com/cvlab-epfl/voxel2mesh.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch.nn as nn
import torch.nn.functional as F
from pytorch3d.ops import GraphConv

class Features2Features(nn.Module):
    """ A graph conv block """

    def __init__(self, in_features, out_features, hidden_layer_count,
                 graph_conv=GraphConv):
        # TODO: batch norm?
        super().__init__()

        self.gconv_first = graph_conv(in_features, out_features)
        gconv_hidden = []
        for _ in range(hidden_layer_count):
            gconv_hidden += [graph_conv(out_features, out_features)]
        self.gconv_hidden = nn.Sequential(*gconv_hidden)
        self.gconv_last = graph_conv(out_features, out_features)

    def forward(self, features, edges):
        features = F.relu(self.gconv_first(features, edges))
        for gconv_hidden in self.gconv_hidden:
            features = F.relu(gconv_hidden(features, edges))
        return self.gconv_last(features, edges)

class Feature2VertexLayer(nn.Module):
    """ Neural mapping from vertex features to vertex coordinates (can be
    absolute or relative coordinates"""

    def __init__(self, in_features, hidden_layer_count, batch_norm=False):
        super().__init__()
        self.batch_norm = batch_norm
        self.layers = []
        for i in range(hidden_layer_count, 1, -1):
            layer_in_features = i * in_features // hidden_layer_count
            layer_out_features = (i-1) * in_features // hidden_layer_count
            self.layers += [GraphConv(layer_in_features, layer_out_features)]
            if self.batch_norm:
                self.layers += [nn.BatchNorm1d(layer_out_features)]

        self.gconv_layers = nn.Sequential(*self.layers)

        # Output of F2V should be 3D coordinates
        self.gconv_last = GraphConv(in_features // hidden_layer_count, 3)

    def forward(self, features, edges):
        for layer in self.layers:
            if isinstance(layer, GraphConv):
                features = layer(features, edges)
            elif isinstance(layer, nn.BatchNorm1d):
                features = layer(features)
            else:
                raise ValueError("Unknown layer type.")
            features = F.relu(features)
        return self.gconv_last(features, edges)

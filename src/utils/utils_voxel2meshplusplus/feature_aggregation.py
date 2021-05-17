
""" Aggregation of voxel features at vertex locations """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch
import torch.nn.functional as F

def aggregate_trilinear(voxel_features, vertices):
    """ Trilinear aggregation of voxel features at vertex locations """
    features = F.grid_sample(voxel_features, vertices[:, :, None, None], mode='bilinear', padding_mode='border', align_corners=True)
    features = features[:, :, :, 0, 0].transpose(2, 1)

    return features

def aggregate_from_indices(voxel_features, vertices, skip_indices,
                           mode='trilinear'):
    """ Aggregation of voxel features at different encoder/decoder indices """
    features = []
    for i in skip_indices:
        if mode=='trilinear':
            features.append(aggregate_trilinear(voxel_features[i], vertices))

    return torch.cat(features, dim=2)





""" Aggregation of voxel features at vertex locations """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch
import torch.nn.functional as F

def aggregate_trilinear(voxel_features, vertices, mode='bilinear'):
    """ Trilinear/bilinear aggregation of voxel features at vertex locations """
    if vertices.shape[-1] == 3:
        vertices_ = vertices[:, :, None, None]
    elif vertices.shape[-1] == 2:
        vertices_ = vertices[:, :, None]
    else:
        raise ValueError("Wrong dimensionality of vertices.")
    features = F.grid_sample(voxel_features, vertices_, mode=mode,
                             padding_mode='border', align_corners=True)
    # Channel dimension <--> V dimension
    if vertices.shape[-1] == 3:
        features = features[:, :, :, 0, 0].transpose(2, 1)
    else: # 2D
        features = features[:, :, :, 0].transpose(2, 1)

    return features

def aggregate_from_indices(voxel_features, vertices, skip_indices,
                           mode='bilinear'):
    """ Aggregation of voxel features at different encoder/decoder indices """
    features = []
    for i in skip_indices:
        if mode == 'bilinear' or mode == 'trilinear':
            # Trilinear = bilinear
            mode = 'bilinear' if mode == 'trilinear' else 'bilinear'
            features.append(aggregate_trilinear(
                voxel_features[i], vertices, mode
            ))

    return torch.cat(features, dim=2)




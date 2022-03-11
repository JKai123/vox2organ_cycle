
""" Check implementation of structural feature aggregation. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch

from utils.feature_aggregation import (
    aggregate_structural_features,
)

a = torch.rand(2, 4, 5, 3)
groups = ((0, 2), (1, 3))

features = aggregate_structural_features(a, groups, K=2)

breakpoint()

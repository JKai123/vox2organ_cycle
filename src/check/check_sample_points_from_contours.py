
""" Unit test for sampling points from contours """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import unittest

import torch
from pytorch3d.structures import Meshes

from utils.sample_points_from_contours import sample_points_from_contours

class CheckPointSampling(unittest.TestCase):
    def test_point_sampling(self):
        vertices = torch.tensor([[0,1], [1,0], [0,0]]).float()
        edges = torch.tensor([[0,1], [1,2], [2,0]]).float()
        meshes = Meshes([vertices], [edges])

        p, n = sample_points_from_contours(meshes, 10, True)

        for p_ in p:
            for pp_ in p_:
                self.assertTrue(
                    torch.isclose(pp_[0], torch.tensor(0.0))
                    or torch.isclose(pp_[1], torch.tensor(0.0))
                    or torch.isclose(pp_[0] + pp_[1], torch.tensor(1.0))
                )

if __name__ == '__main__':
    unittest.main()

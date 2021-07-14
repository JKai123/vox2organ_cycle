
""" Check loss implementations (for 2D) """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import unittest

import torch
from pytorch3d.structures import Meshes

from scripts.create_2D_sphere import create_2D_sphere
from utils.losses import NormalConsistencyLoss, EdgeLoss

class TestLossMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        v, f = create_2D_sphere(8)
        cls.m = Meshes([v.float()], [f])

    def test_NormalConsistencyLoss2D(self):
        loss = NormalConsistencyLoss().get_loss(self.m)

        target_loss = (1 - torch.cosine_similarity(
            torch.tensor([[1,0]]).float(), torch.tensor([[1,1]]).float()
        )) * 8 / 12

        self.assertAlmostEqual(loss.item(), target_loss.item())

    def test_EdgeLoss2D(self):
        loss = EdgeLoss(0.0).get_loss(self.m)

        target_loss = (8 * 1.0 + 4 * 2.0) / 12

        self.assertAlmostEqual(loss, target_loss)

if __name__ == '__main__':
    unittest.main()

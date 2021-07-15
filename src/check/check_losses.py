
""" Check loss implementations (for 2D) """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import unittest

import torch
from pytorch3d.structures import Meshes

from scripts.create_2D_sphere import create_2D_sphere
from utils.coordinate_transform import unnormalize_vertices
from utils.losses import (
    NormalConsistencyLoss,
    EdgeLoss,
    ChamferAndNormalsLoss,
    meshes_to_vertex_normals_2D_packed,
)

class TestLossMethods(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        v, f = create_2D_sphere(8)
        cls.m = Meshes([unnormalize_vertices(v.float(), (8,8))], [f])

    def test_NormalConsistencyLoss2D(self):
        loss = NormalConsistencyLoss().get_loss(self.m)

        target_loss = (1 - torch.cosine_similarity(
            torch.tensor([[1,0]]).float(), torch.tensor([[1,1]]).float()
        )) * 8 / 12

        self.assertAlmostEqual(loss.item(), target_loss.item())

    def test_EdgeLoss2D(self):
        loss = EdgeLoss(0.0).get_loss(self.m).item()

        target_loss = (8 * 1.0 + 4 * 2.0) / 12

        self.assertAlmostEqual(loss, target_loss)

    def test_ChamferAndNormalsLoss2D(self):
        m2 = Meshes([self.m.verts_packed() + 0.1], [self.m.faces_packed()])
        loss = ChamferAndNormalsLoss().get_loss(
            self.m, (m2.verts_padded(), meshes_to_vertex_normals_2D_packed(m2)[None])
        )

        chamfer_target_loss = (torch.tensor([0.1, 0.1]).norm() ** 2) * 2
        cos_target_loss = 0.0

        self.assertAlmostEqual(loss[0].item(), chamfer_target_loss.item(), 6)
        self.assertAlmostEqual(loss[1].item(), cos_target_loss)

if __name__ == '__main__':
    unittest.main()

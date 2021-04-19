
""" Loss function collection for convenient calculation over multiple classes
and/or instances of meshes.

    Notation:
        - B: batch size
        - S: number of instances, e.g. steps/positions where the loss is computed in
        the model
        - C: number of channels (= number of classes usually)
        architecture
        - V: number of vertices
        - F: number of faces
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from abc import ABC, abstractmethod

import torch
import pytorch3d.structures
from pytorch3d.loss import (chamfer_distance,
                            mesh_edge_loss,
                            mesh_laplacian_smoothing,
                            mesh_normal_consistency)
from pytorch3d.ops import sample_points_from_meshes

class MeshLoss(ABC):
    def __str__(self):
        return self.__class__.__name__ + "()"
    def __call__(self, pred_meshes, target):
        """ Chamfer loss calculation

        :param pred_meshes: A multidimensional array of predicted meshes of shape
        (S, C), each of type pytorch3d.structures.Meshes
        :param target: A multidimensional array of target points of shape (C)
        i.e. one tensor per class
        :return: The calculated loss.
        """
        mesh_loss = torch.tensor(0).float().cuda()

        S = len(pred_meshes)
        C = len(pred_meshes[0])

        for s in range(S):
            for c in range(C):
                mesh_loss += self.get_loss(pred_meshes[s][c], target[c])

        return mesh_loss

    @abstractmethod
    def get_loss(self,
                 pred_meshes: pytorch3d.structures.Meshes,
                 target: pytorch3d.structures.Pointclouds):
        pass

class ChamferLoss(MeshLoss):
    def get_loss(self,
                 pred_meshes: pytorch3d.structures.Meshes,
                 target: pytorch3d.structures.Pointclouds):
        n_points = torch.min(target.num_points_per_cloud())
        pred_points = sample_points_from_meshes(pred_meshes, n_points)
        return chamfer_distance(pred_points, target)[0]

class LaplacianLoss(MeshLoss):
    def get_loss(self,
                 pred_meshes: pytorch3d.structures.Meshes,
                 target: pytorch3d.structures.Pointclouds=None):
        return mesh_laplacian_smoothing(pred_meshes, method='uniform')

class NormalConsistencyLoss(MeshLoss):
    def get_loss(self,
                 pred_meshes: pytorch3d.structures.Meshes,
                 target: pytorch3d.structures.Pointclouds=None):
        return mesh_normal_consistency(pred_meshes)

class EdgeLoss(MeshLoss):
    def get_loss(self,
                 pred_meshes: pytorch3d.structures.Meshes,
                 target: pytorch3d.structures.Pointclouds=None):
        return mesh_edge_loss(pred_meshes)

def linear_loss_combine(losses, weights):
    """ Compute the losses in a linear manner, e.g.
    a1 * loss1 + a2 * loss2 + ...

    :param losses: The individual losses.
    :param weights: The weights for the losses.
    :returns: The overall (weighted) loss.
    """
    loss_total = 0
    for loss, weight in zip(losses, weights):
        loss_total += weight * loss

        # wandb.log({loss_func.__name__: loss})

    # wandb.log({'loss_total': loss_total})

    return loss_total

def geometric_loss_combine(losses, weights):
    """ Compute the losses in a geometric manner, e.g.
    (loss1^a1) * (loss2^a2) * ...

    :param losses: The individual losses.
    :param weights: The weights for the losses.
    :returns: The overall (weighted) loss.
    """
    loss_total = 1
    for loss, weight in zip(losses, weights):
        loss_total *= torch.pow(loss, weight)

        # wandb.log({loss_func.__name__: loss})

    # wandb.log({'loss_total': loss_total})

    return loss_total

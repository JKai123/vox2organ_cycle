
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
from typing import Union

import torch
import pytorch3d.structures
from pytorch3d.structures import Meshes
from pytorch3d.loss import (chamfer_distance,
                            mesh_edge_loss,
                            mesh_laplacian_smoothing,
                            mesh_normal_consistency)
from pytorch3d.ops import sample_points_from_meshes
from torch.cuda.amp import autocast

class MeshLoss(ABC):
    def __str__(self):
        return self.__class__.__name__ + "()"
    def __call__(self, pred_meshes, target):
        """ Mesh loss calculation

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
                 target: Union[pytorch3d.structures.Pointclouds,
                               pytorch3d.structures.Meshes,
                               tuple,
                               list]):
        pass

class ChamferLoss(MeshLoss):
    """ Chamfer distance between the predicted mesh and randomly sampled
    surface points or a reference mesh. """
    def get_loss(self, pred_meshes, target):
        if isinstance(target, pytorch3d.structures.Pointclouds):
            n_points = torch.min(target.num_points_per_cloud())
        if isinstance(target, pytorch3d.structures.Meshes):
            n_points = torch.min(target.num_verts_per_mesh())
            target = sample_points_from_meshes(target, n_points)
        pred_points = sample_points_from_meshes(pred_meshes, n_points)
        return chamfer_distance(pred_points, target)[0]

class ChamferAndNormalsLoss(MeshLoss):
    """ Chamfer distance + cosine distance between the vertices and normals of
    the predicted mesh and a reference mesh. """
    def get_loss(self, pred_meshes, target):
        if len(target) != 2:
            raise TypeError("ChamferAndNormalsLoss requires vertices and"\
                            " normals.")
        target_points, target_normals = target
        n_points = target_points.shape[1]
        pred_points, pred_normals = sample_points_from_meshes(
            pred_meshes, n_points, return_normals=True
        )
        d_chamfer, d_cosine = chamfer_distance(pred_points, target_points,
                                               x_normals=pred_normals,
                                               y_normals=target_normals)
        return d_chamfer + 0.1 * d_cosine

class LaplacianLoss(MeshLoss):
    def get_loss(self, pred_meshes, target=None):
        # Method does not support autocast
        with autocast(enabled=False):
            loss = mesh_laplacian_smoothing(
                Meshes(pred_meshes.verts_padded().float(),
                       pred_meshes.faces_padded().float()),
                method='uniform'
            )
        return loss

class NormalConsistencyLoss(MeshLoss):
    def get_loss(self, pred_meshes, target=None):
        return mesh_normal_consistency(pred_meshes)

class EdgeLoss(MeshLoss):
    def get_loss(self, pred_meshes, target=None):
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

    return loss_total

def all_linear_loss_combine(voxel_loss_func, voxel_loss_func_weights,
                            voxel_pred, voxel_target,
                            mesh_loss_func, mesh_loss_func_weights,
                            mesh_pred, mesh_target):
    """ Linear combination of all losses. """
    losses = {}
    # Voxel losses
    if voxel_pred is not None:
        if not isinstance(voxel_pred, list):
            # If deep supervision is used, voxel predicition is a list. Therefore,
            # non-list predictions are made compatible
            voxel_pred = [voxel_pred]
        for lf in voxel_loss_func:
            losses[str(lf)] = 0.0
            for vp in voxel_pred:
                losses[str(lf)] += lf(vp, voxel_target)
    # Mesh losses
    for lf in mesh_loss_func:
        losses[str(lf)] = lf(mesh_pred, mesh_target)

    # Merge loss weights into one list
    combined_loss_weights = voxel_loss_func_weights +\
            mesh_loss_func_weights

    loss_total = linear_loss_combine(losses.values(),
                                     combined_loss_weights)

    return losses, loss_total

def voxel_linear_mesh_geometric_loss_combine(voxel_loss_func, voxel_loss_func_weights,
                            voxel_pred, voxel_target,
                            mesh_loss_func, mesh_loss_func_weights,
                            mesh_pred, mesh_target):
    """ Linear combination of voxel losses, geometric combination of mesh losses
    at each step and linear combination of steps and voxel and mesh losses.
    Reference: Kong et al. 2021 """

    # Voxel losses
    voxel_losses = {}
    if voxel_pred is not None:
        if not isinstance(voxel_pred, list):
            # If deep supervision is used, voxel predicition is a list. Therefore,
            # non-list predictions are made compatible
            voxel_pred = [voxel_pred]
        for lf in voxel_loss_func:
            voxel_losses[str(lf)] = 0.0
            for vp in voxel_pred:
                voxel_losses[str(lf)] += lf(vp, voxel_target)
        # Linear combination of voxel_losses
        voxel_loss = linear_loss_combine(voxel_losses.values(),
                                         voxel_loss_func_weights)
    else:
        voxel_loss = 0.0

    # Mesh losses
    mesh_losses_combined = []
    mesh_losses = {str(lf): [] for lf in mesh_loss_func}
    S = len(mesh_pred)
    C = len(mesh_pred[0])
    for s in range(S):
        for c in range(C):
            mesh_losses_sc = {}
            for lf in mesh_loss_func:
                ml = lf([[mesh_pred[s][c]]], [mesh_target[c]])
                mesh_losses_sc[str(lf)] = ml
                mesh_losses[str(lf)].append(ml.detach().cpu()) # log
            # Geometric combination of losses at a certain step and for a
            # certain class
            mesh_loss_sc = geometric_loss_combine(mesh_losses_sc.values(),
                                                  mesh_loss_func_weights)
            mesh_losses_combined.append(mesh_loss_sc)

    # Final loss = voxel_loss + mesh_loss[s1][c1] + mesh_loss[s1][c2] + ...
    loss_total = voxel_loss
    for ml in mesh_losses_combined:
        loss_total += ml

    # Log mean over S and C per loss
    mesh_losses= {k: torch.mean(torch.tensor(v)) for k, v in mesh_losses.items()}

    # All losses in one dictionary
    losses = {**voxel_losses, **mesh_losses}

    return losses, loss_total

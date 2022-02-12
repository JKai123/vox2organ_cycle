
""" Loss function collection for convenient calculation over multiple classes
and/or instances of meshes.

    Notation:
        - B: batch size
        - S: number of instances, e.g. steps/positions where the loss is computed in
        the model
        - C: number of channels (= number of classes usually)
        - V: number of vertices
        - F: number of faces
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from abc import ABC, abstractmethod
from typing import Union
from collections.abc import Sequence

import torch
import pytorch3d.structures
from pytorch3d.structures import Meshes
from pytorch3d.loss import (chamfer_distance,
                            mesh_edge_loss,
                            mesh_laplacian_smoothing,
                            mesh_normal_consistency)
from pytorch3d.ops import sample_points_from_meshes, laplacian
from torch.cuda.amp import autocast

from utils.utils import choose_n_random_points
from utils.mesh import curv_from_cotcurv_laplacian

def point_weigths_from_curvature(curvatures: torch.Tensor,
                                 points: torch.Tensor,
                                 max_weight: Union[float, int, torch.Tensor],
                                 padded_coordinates=(-1.0, -1.0, -1.0)):
    """ Calculate Chamfer weights from curvatures such that they are in
    [1, max_weight]. In addition, the weight of padded points is set to zero."""

    if not isinstance(max_weight, torch.Tensor):
        max_weight = torch.tensor(max_weight).float()

    # Weights in [1, max_weight]
    weights = torch.minimum(1 + curvatures, max_weight.cuda())

    # Set weights of padded vertices to 0
    padded_coordinates = torch.Tensor(padded_coordinates).to(points.device)
    weights[torch.isclose(points, padded_coordinates).all(dim=2)] = 0.0

    return weights

class MeshLoss(ABC):
    """ Abstract base class for all mesh losses. """

    def __init__(self, ignore_coordinates=(-1.0, -1.0, -1.0)):
        self.ignore_coordinates = torch.tensor(ignore_coordinates)

    def __str__(self):
        return self.__class__.__name__ + "()"

    def __call__(self, pred_meshes, target, weights=None):
        """ Mesh loss calculation

        :param pred_meshes: A multidimensional array of predicted meshes of shape
        (S, C), each of type pytorch3d.structures.Meshes
        :param target: A multidimensional array of target points of shape (C)
        i.e. one tensor per class
        :param weights: Losses are weighed per class.
        :return: The calculated loss.
        """
        if isinstance(self, ChamferAndNormalsLoss):
            mesh_loss = torch.tensor([0,0]).float().cuda()
        else:
            mesh_loss = torch.tensor(0).float().cuda()

        S = len(pred_meshes)
        C = len(pred_meshes[0])

        if weights is not None:
            if len(weights) != C:
                raise ValueError("Weights should be specified per class.")
        else: # no per-class-weights provided
            weights = torch.tensor([1.0] * C).float().cuda()

        for s in range(S):
            for c, w in zip(range(C), weights):
                mesh_loss += self.get_loss(pred_meshes[s][c], target[c]) * w

        return mesh_loss

    @abstractmethod
    def get_loss(self,
                 pred_meshes: pytorch3d.structures.Meshes,
                 target: Union[pytorch3d.structures.Pointclouds,
                               pytorch3d.structures.Meshes,
                               tuple,
                               list]):
        raise NotImplementedError()


class ChamferLoss(MeshLoss):
    """ Chamfer distance between the predicted mesh and randomly sampled
    surface points or a reference mesh. """

    def __init__(self, curv_weight_max=None):
        super().__init__()
        self.curv_weight_max = curv_weight_max

    def __str__(self):
        return f"ChamferLoss(curv_weight_max={self.curv_weight_max})"

    def get_loss(self, pred_meshes, target):
        if isinstance(target, pytorch3d.structures.Pointclouds):
            n_points = torch.min(target.num_points_per_cloud())
            target_ = target
            if self.curv_weight_max is not None:
                raise RuntimeError("Can only apply curvature weights if they"
                                   " are provided in the target.")

        if isinstance(target, pytorch3d.structures.Meshes):
            n_points = torch.min(target.num_verts_per_mesh())
            target_ = sample_points_from_meshes(target, n_points)
            if self.curv_weight_max is not None:
                raise RuntimeError("Cannot apply curvature weights for"
                                   " targets of type 'Meshes'.")

        if isinstance(target, Sequence):
            # target = (verts, normals, curvatures)
            target_ = target[0] # Only vertices relevant
            assert target_.ndim == 3 # padded
            n_points = target_.shape[1]
            target_curvs = target[2]
            point_weights = point_weigths_from_curvature(
                target_curvs, target_points, self.curv_weight_max
            ) if self.curv_weight_max else None

        pred_points = sample_points_from_meshes(pred_meshes, n_points)

        pred_lengths = (~torch.isclose(
            pred_points, self.ignore_coordinates.to(pred_points.device)
        ).all(dim=2)).sum(dim=1)
        target_lengths = (~torch.isclose(
            target_, self.ignore_coordinates.to(pred_points.device)
        ).all(dim=2)).sum(dim=1)
        return chamfer_distance(
            pred_points,
            target_,
            point_weights=point_weights,
            x_lengths=pred_lengths,
            y_lengths=target_lengths
        )[0]


class ChamferAndNormalsLoss(MeshLoss):
    """ Chamfer distance & cosine distance between the vertices and normals of
    the predicted mesh and a reference mesh.
    """

    def __init__(self, curv_weight_max=None):
        super().__init__()
        self.curv_weight_max = curv_weight_max

    def __str__(self):
        return f"ChamferAndNormalsLoss(curv_weight_max={self.curv_weight_max})"

    def get_loss(self, pred_meshes, target):
        if len(target) < 2:
            raise TypeError("ChamferAndNormalsLoss requires vertices and"\
                            " normals.")
        target_points, target_normals = target[0], target[1]
        assert target_points.ndim == 3 and target_normals.ndim == 3
        n_points = target_points.shape[1]
        pred_points, pred_normals = sample_points_from_meshes(
            pred_meshes, n_points, return_normals=True
        )
        target_curvs = target[2]
        point_weights = point_weigths_from_curvature(
            target_curvs, target_points, self.curv_weight_max
        ) if self.curv_weight_max else None

        pred_lengths = (~torch.isclose(
            pred_points, self.ignore_coordinates.to(pred_points.device)
        ).all(dim=2)).sum(dim=1)
        target_lengths = (~torch.isclose(
            target_points, self.ignore_coordinates.to(pred_points.device)
        ).all(dim=2)).sum(dim=1)
        losses = chamfer_distance(
            pred_points,
            target_points,
            x_normals=pred_normals,
            x_lengths=pred_lengths,
            y_normals=target_normals,
            y_lengths=target_lengths,
            point_weights=point_weights,
            oriented_cosine_similarity=True
        )
        d_chamfer = losses[0]
        d_cosine = losses[1]

        return torch.stack([d_chamfer, d_cosine])


class HackyChamferAndNormalsLoss(ChamferAndNormalsLoss):
    """ Preliminary implementation of class-specific ChamferAndNormalsLoss.
    """

    def __init__(self, curv_weight_max=None):
        super().__init__()
        self.curv_weight_max = curv_weight_max

    def __str__(self):
        return f"ChamferAndNormalsLoss(curv_weight_max={self.curv_weight_max})"

    def get_loss(self, pred_meshes, target):
        # Chop
        target_points, target_normals, target_curvs, target_classes = target
        batch_size, n_points, dim = target_points.shape
        # Normals contain classes
        pred_classes = pred_meshes.verts_normals_padded()[:, :, 0].long()
        # TODO: Implement sampling with classes; Attention when giving
        # pred_meshes to such a method, might use verts_normals which are hacky
        # here!!!
        # pred_points, pred_normals = sample_points_from_meshes(
            # pred_meshes, n_points, return_normals=True
        # )
        # Tmp solution: Don't sample but use vertices
        pred_points = pred_meshes.verts_padded()
        # True normals
        pred_normals = Meshes(
            pred_meshes.verts_padded(),
            pred_meshes.faces_padded()
        ).verts_normals_padded()

        point_weights = point_weigths_from_curvature(
            target_curvs, target_points, self.curv_weight_max
        ) if self.curv_weight_max else None

        # Ignore here
        # pred_lengths = (~torch.isclose(
            # pred_points, self.ignore_coordinates.to(pred_points.device)
        # ).all(dim=2)).sum(dim=1)
        # target_lengths = (~torch.isclose(
            # target_points, self.ignore_coordinates.to(pred_points.device)
        # ).all(dim=2)).sum(dim=1)

        d_chamfer = 0.0
        d_cosine = 0.0
        weight_sum = 0.0
        # Iterate over vertex classes
        classes= target_classes[target_classes != -1].unique()
        for cls in classes:
            pred_cls = pred_classes == cls
            pred_p = pred_points[pred_cls].view((batch_size, -1, dim))
            pred_n = pred_normals[pred_cls].view((batch_size, -1, dim))

            target_cls = (target_classes == cls).squeeze(-1)
            target_p = target_points[target_cls].view((batch_size, -1, dim))
            target_n = target_normals[target_cls].view((batch_size, -1, dim))
            point_w = point_weights[target_cls].view((batch_size, -1, 1))

            losses = chamfer_distance(
                pred_p,
                target_p,
                x_normals=pred_n,
                y_normals=target_n,
                point_weights=point_w,
                return_point_weight_sum=True,
                oriented_cosine_similarity=True,
                batch_reduction='sum',
                point_reduction='sum'
            )
            d_chamfer += losses[0]
            d_cosine += losses[1]
            weight_sum += losses[2]

        # Reduction
        # TODO: Use reduction by point weights
        # d_chamfer /= weight_sum
        d_chamfer /= (pred_points.shape[1] + target_points.shape[1])
        d_chamfer /= batch_size
        # Cosine distance not weighted
        d_cosine /= (pred_points.shape[1] + target_points.shape[1])
        d_cosine /= batch_size

        return torch.stack([d_chamfer, d_cosine])


class LaplacianLoss(MeshLoss):
    def __init(self):
        super().__init__()
    # Method does not support autocast
    @autocast(enabled=False)
    def get_loss(self, pred_meshes, target=None):
        # pytorch3d loss for 3D
        loss = mesh_laplacian_smoothing(
            Meshes(pred_meshes.verts_padded().float(),
                   pred_meshes.faces_padded().float()),
            method='uniform'
        )

        return loss


class NormalConsistencyLoss(MeshLoss):
    def __init(self):
        super().__init__()
    def get_loss(self, pred_meshes, target=None):
        return mesh_normal_consistency(pred_meshes)


class EdgeLoss(MeshLoss):
    def __init__(self, target_length):
        super().__init__()
        self.target_length = target_length

    def __str__(self):
        return f"EdgeLoss({self.target_length})"

    def get_loss(self, pred_meshes, target=None):
        return mesh_edge_loss(pred_meshes, target_length=self.target_length)


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

    return loss_total


def _add_MultiLoss_to_dict(loss_dict, loss_func, mesh_pred,
                            mesh_target, weights, names):
    """ Add a multi-loss (i.e. a loss that returns multiple values like
    ChamferAndNormalsLoss) to the loss_dict.
    """
    # All weights a Sequence or none of them
    assert (all(map(lambda x: isinstance(x, Sequence), weights))
            or
            not any(map(lambda x: isinstance(x, Sequence), weights)))
    # Same length of weights and names
    assert len(weights) == len(names)
    # Either per-class weight or single weight for all classes
    if isinstance(weights[0], Sequence):
        # Reorder weights by exchanging class and loss function dimension
        weights = torch.tensor(weights).cuda().T
        # Weights processed by loss function
        ml = loss_func(mesh_pred, mesh_target, weights)
        for i, n in enumerate(names):
            loss_dict[n] = ml[i]
    else:
        ml = loss_func(mesh_pred, mesh_target)
        # Weights multiplied here
        for i, (n, w) in enumerate(zip(names, weights)):
            loss_dict[n] = ml[i] * w

def all_linear_loss_combine(voxel_loss_func, voxel_loss_func_weights,
                            voxel_pred, voxel_target,
                            mesh_loss_func, mesh_loss_func_weights,
                            mesh_pred, deltaV_mesh_pred, mesh_target):
    """ Linear combination of all losses. In contrast to geometric averaging,
    this also allows for per-class mesh loss weights. """
    losses = {}
    # Voxel losses
    if voxel_pred is not None:
        if not isinstance(voxel_pred, Sequence):
            # If deep supervision is used, voxel prediction is a list. Therefore,
            # non-list predictions are made compatible
            voxel_pred = [voxel_pred]
        for lf, w in zip(voxel_loss_func, voxel_loss_func_weights):
            losses[str(lf)] = 0.0
            for vp in voxel_pred:
                losses[str(lf)] += lf(vp, voxel_target) * w

    # Mesh losses
    mesh_loss_weights_iter = iter(mesh_loss_func_weights)
    for lf in mesh_loss_func:
        weight = next(mesh_loss_weights_iter)
        if isinstance(lf, ChamferAndNormalsLoss):
            w1 = weight
            w2 = next(mesh_loss_weights_iter)
            _add_MultiLoss_to_dict(
                losses, lf, mesh_pred, mesh_target, (w1, w2),
                ("ChamferLoss()", "CosineLoss()")
            )
        else: # add single loss to dict
            if isinstance(weight, Sequence):
                if isinstance(lf, LaplacianLoss):
                    # Use relative coordinates
                    ml = lf(deltaV_mesh_pred, mesh_target, weight)
                else:
                    ml = lf(mesh_pred, mesh_target, weight)
                losses[str(lf)] = ml
            else:
                if isinstance(lf, LaplacianLoss):
                    # Use relative coordinates
                    ml = lf(deltaV_mesh_pred, mesh_target)
                else:
                    ml = lf(mesh_pred, mesh_target)
                losses[str(lf)] = ml * weight

    loss_total = sum(losses.values())

    return losses, loss_total


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
import torch.nn.functional as F
import pytorch3d.structures
import numpy as np
import os
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss import (
    chamfer_distance,
    mesh_edge_loss,
    mesh_laplacian_smoothing,
    mesh_normal_consistency
)

from utils.utils_padded_packed import MoM_to_list, MoM_to_meshes
from pytorch3d.ops import sample_points_from_meshes, laplacian
from torch.cuda.amp import autocast
from utils.mesh import curv_from_cotcurv_laplacian
from utils.utils_pca_loss.gpa import gpa
from utils.utils_pca_loss.eval import mesh_vert_sqr_dist
from utils.utils_pca_loss.pca import project_subspace

def point_weigths_from_curvature(curvatures: torch.Tensor,
                                 points: torch.Tensor,
                                 max_weight: Union[float, int, torch.Tensor],
                                 padded_coordinates=(0.0, 0.0, 0.0)):
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

    def __init__(self, ignore_coordinates=(0.0, 0.0, 0.0)):
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
                target_curvs, target_, self.curv_weight_max
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


class ClassAgnosticChamferAndNormalsLoss(ChamferAndNormalsLoss):
    """ Preliminary implementation of class-specific ChamferAndNormalsLoss.
    """

    def __init__(self, curv_weight_max=None, class_weights=None):
        super().__init__()
        self.curv_weight_max = curv_weight_max
        self.class_weights = class_weights

    def __str__(self):
        return f"ClassAgnosticChamferAndNormalsLoss(curv_weight_max={self.curv_weight_max})"

    def get_loss(self, pred_meshes, target):
        """ ChamferAndNormalsLoss per vertex class, i.e., vertices of a certain
        class only 'see' ground truth points of the same class.
        """
        d_chamfer = 0.0
        d_cosine = 0.0

        # Chop
        target_points, target_normals, target_curvs, target_classes = target
        batch_size, n_points, _ = target_points.shape
        # Sample points on meshes. Predicted classes are assigned to sampled
        # points accoring to neares vertex
        pred_points, pred_normals, pred_classes = sample_points_from_meshes(
            pred_meshes,
            n_points,
            return_normals=True,
            interpolate_features='nearest'
        )
        # Curvature weights
        point_weights = point_weigths_from_curvature(
            target_curvs, target_points, self.curv_weight_max
        ) if self.curv_weight_max else None

        # Vertex classes
        classes= target_classes[target_classes != -1].unique()
        if self.class_weights is not None:
            assert len(self.class_weights) == len(classes),\
                    "There must be exactly one weight per class."
        # Iterate over vertex classes
        for c, cls in enumerate(classes):
            pred_cls = (pred_classes == cls).view(batch_size, n_points)
            # Batches of points and normals of a certain class
            pred_p = Pointclouds(
                [pred_points[i][pred_cls[i]]
                 for i in range(batch_size)],
                normals=[pred_normals[i][pred_cls[i]]
                         for i in range(batch_size)],
            )
            target_cls = (target_classes == cls).view(batch_size, n_points)
            target_p = Pointclouds(
                [target_points[i][target_cls[i]]
                 for i in range(batch_size)],
                normals=[target_normals[i][target_cls[i]]
                         for i in range(batch_size)],
            )
            # Pad point weights with 0
            n_target = target_p.num_points_per_cloud()
            n_target_max = n_target.max()
            if point_weights is not None:
                point_w = torch.stack(
                    [F.pad(
                        point_weights[i][target_cls[i]],
                        (0, 0, 0, n_target_max - n_target[i])
                    ) for i in range(batch_size)]
                )
            else:
                point_w = None

            losses = chamfer_distance(
                pred_p, # Contains also normals
                target_p, # Contains also normals
                point_weights=point_w,
                oriented_cosine_similarity=True,
                batch_reduction='mean',
                point_reduction='mean',
            )

            # Skip if losses are invalid; this happens if the current class is
            # not present in all target pointclouds of the batch
            if torch.isnan(losses[0]) or torch.isnan(losses[1]):
                continue

            if self.class_weights is not None:
                d_chamfer += losses[0] * self.class_weights[c]
                d_cosine += losses[1] * self.class_weights[c]
            else:
                d_chamfer += losses[0]
                d_cosine += losses[1]

        # Class reduction
        if self.class_weights is not None:
            d_chamfer /= torch.sum(torch.tensor(self.class_weights))
            d_cosine /= torch.sum(torch.tensor(self.class_weights))
        else:
            d_chamfer /= len(classes)
            d_cosine /= len(classes)

        return torch.stack([d_chamfer, d_cosine])



class CycleLossChamfer(ChamferLoss):
    def __init__(self):
        super().__init__()

class CycleLoss(MeshLoss):
    def __init__(self):
        super().__init__()
    def get_loss(self, pred_meshes, target):
        n_points = 30000 # TODO make this accesable
        # target_points = sample_points_from_meshes(target, n_points)
        # pred_points = sample_points_from_meshes(pred_meshes, n_points)
        pred_verts = pred_meshes.verts_padded().float()
        target_verts = target.verts_padded().float()
        MSEloss = torch.nn.MSELoss()
        loss = MSEloss(pred_verts, target_verts)
        test = 1
        return loss
    

class LaplacianLoss(MeshLoss):
    def __init__(self):
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
    def __init__(self):
        super().__init__()
    def get_loss(self, pred_meshes, target=None):
        test = mesh_normal_consistency(pred_meshes)
        return mesh_normal_consistency(pred_meshes)


class AverageEdgeLoss(MeshLoss):
    def __init__(self):
        super().__init__()

    def get_loss(self, pred_meshes, target=None):
        edges = pred_meshes.edges_packed()
        e0 = edges[:,0]
        e1 = edges[:,1]
        verts = pred_meshes.verts_packed()
        v0 = verts[e0]
        v1 = verts[e1]
        vec = v1 - v0
        edge_lengths = torch.linalg.norm(vec, dim=1, ord=2)
        avg_edge_length = torch.mean(edge_lengths)
        return mesh_edge_loss(pred_meshes, avg_edge_length)

class PCA_loss(MeshLoss):
    def __init__(self):
        super().__init__()

    def get_loss(self, pred_meshes, target=None):
        mean, eigenvectors, eigenvalues, folder_name = target
        pred_meshes = gpa([pred_meshes], None)
        meshes_proj = project_subspace(pred_meshes, mean, eigenvectors, eigenvalues)
        dist, _ = mesh_vert_sqr_dist(pred_meshes, meshes_proj)
        return dist
        
        


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
                            mesh_pred, deltaV_mesh_pred, mesh_pred_unpadded, 
                            mesh_pred_cycle, template_MoM, mesh_target,
                            ssm_target):

    """ Linear combination of all losses. In contrast to geometric averaging,
    this also allows for per-class mesh loss weights. 
    mesh_pred_unpadded: List of Meshes - each Meshes contains N Meshes - the list is of length M
    """

    template = MoM_to_list(template_MoM)
    template_meshes = MoM_to_meshes(template_MoM)

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
                elif isinstance(lf, NormalConsistencyLoss):
                    ml = lf(mesh_pred_unpadded, mesh_target, weight)
                elif isinstance(lf, CycleLoss):
                    ml = lf(mesh_pred_cycle, template_meshes, weight)
                    # ml = lf(mesh_pred, mesh_target, weight)
                elif isinstance(lf, PCA_loss):
                    ml = lf(mesh_pred, ssm_target, weight)
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

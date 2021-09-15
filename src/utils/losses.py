
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
from geomloss import SamplesLoss

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

def meshes_to_edge_normals_2D_packed(meshes: Meshes):
    """ Helper function to get normals of 2D meshes (contours) for every edge.
    The normals have the same length as the respective edge they belong
    to."""
    verts_packed = meshes.verts_packed()
    edges_packed = meshes.faces_packed() # edges=faces

    v0_idx, v1_idx = edges_packed[:, 0], edges_packed[:, 1]

    # Normal of edge [x,y] is [-y,x]
    normals = (verts_packed[v1_idx] - verts_packed[v0_idx]).flip(
        dims=[1]) * torch.tensor([-1.0, 1.0]).to(meshes.device)

    # Length of normals is automatically equal to the edge length as the normals
    # are just rotated edges.

    return normals, v0_idx, v1_idx

def meshes_to_vertex_normals_2D_packed(meshes: Meshes):
    """ Helper function to get normals of 2D meshes (contours) for every vertex.
    The normals are defined as the sum of the normals of the two adjacent edges
    of the vertex. """
    # Normals of edges
    edge_normals, v0_idx, v1_idx = meshes_to_edge_normals_2D_packed(
        meshes
    )
    # (Normals at vertices) = (sum of normals at adjacent edges of the
    # respective edge length)
    normals_next = edge_normals[torch.argsort(v0_idx)]
    normals_prev = edge_normals[torch.argsort(v1_idx)]

    return normals_next + normals_prev

class MeshLoss(ABC):
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
        elif isinstance(self, ChamferAndNormalsAndCurvatureLoss):
            mesh_loss = torch.tensor([0,0,0]).float().cuda()
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

class WassersteinLoss(MeshLoss):
    """ Wasserstein loss: it is approximated using Sinkhorn's iteration. The
    resulting distance is also frequently called 'Sinkhorn divergence'.

    Note: In contrast to the Wasserstein evaluation score, this function treats
    every scene of multiple structures as one pointcloud. However, losses are
    averaged over the batch.
    """

    def __init__(self):
        self.loss_func = SamplesLoss(loss="sinkhorn", p=2, diameter=2, blur=0.05)

    def get_loss(self, pred_meshes, target):
        pred_ = pred_meshes.verts_padded()
        if isinstance(target, pytorch3d.structures.Pointclouds):
            target_ = target.points_padded()
        if isinstance(target, pytorch3d.structures.Meshes):
            target_ = target.verts_padded()
        if isinstance(target, Sequence):
            # target = (verts, normals)
            target_ = target[0] # Only vertices relevant
        # Select an equal number of points
        assert pred_.ndim == 3 and target_.ndim == 3
        if pred_.shape[1] < target_.shape[1]:
            n_points = pred_.shape[1]
            target_ = choose_n_random_points(target_, n_points,
                                             ignore_padded=True)
        elif target_.shape[1] < pred_.shape[1]:
            n_points = target_.shape[1]
            pred_ = choose_n_random_points(pred_, n_points, ignore_padded=True)

        w_loss = torch.tensor(0.0, requires_grad=True).cuda()
        for p, t in zip(pred_, target_):
            w_loss += self.loss_func(p, t)

        # Average over batch
        return w_loss / float(pred_.shape[0])

class ChamferLoss(MeshLoss):
    """ Chamfer distance between the predicted mesh and randomly sampled
    surface points or a reference mesh. """

    def __init__(self, curv_weight_max=None):
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
            if target.verts_padded().shape[1] == 3:
                target_ = sample_points_from_meshes(target, n_points)
            else: # 2D --> choose vertex points
                target_ = choose_n_random_points(
                    target.verts_padded(), n_points
                )
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

        if pred_meshes.verts_packed().shape[-1] == 3:
            pred_points = sample_points_from_meshes(pred_meshes, n_points)
        else: # 2D
            pred_points = choose_n_random_points(
                pred_meshes.verts_padded(), n_points
            )

        return chamfer_distance(
            pred_points, target_, point_weights=point_weights
        )[0]

class ChamferAndNormalsAndCurvatureLoss(MeshLoss):
    """ Chamfer distance, cosine distance & difference in curvature
    """

    def __init__(self, curv_weight_max=None):
        self.curv_weight_max = curv_weight_max

    def __str__(self):
        return "ChamferAndNormalsAndCurvatureLoss"\
               f"(curv_weight_max={self.curv_weight_max})"

    # Method does not support autocast (due to sparse operations in cotangent
    # laplacian)
    @autocast(enabled=False)
    def get_loss(self, pred_meshes, target):
        if len(target) < 3:
            raise ValueError("ChamferAndNormalsAndCurvatureLoss requires"
                             " vertices, normals, and curvatures.")
        target_points= target[0]
        target_normals = target[1]
        target_curvs = target[2]
        assert (target_points.ndim == 3 and target_normals.ndim == 3 and
                target_curvs.ndim == 3)
        n_points = target_points.shape[1]
        if target_points.shape[-1] == 3: # 3D
            orig_shape = pred_meshes.verts_padded().shape
            pred_points, idx = choose_n_random_points(
                pred_meshes.verts_padded(), n_points, return_idx=True
            ) # (N,L,3)
            pred_normals = pred_meshes.verts_normals_padded()[idx.unbind(1)]
            pred_normals = pred_normals.view_as(pred_points) # (N,L,3)
            pred_curvs = curv_from_cotcurv_laplacian(
                pred_meshes.verts_packed(), pred_meshes.faces_packed()
            ).view(
                orig_shape[0], orig_shape[1], 1
            )[idx.unbind(1)].view(
                pred_points.shape[0], pred_points.shape[1], 1
            ) # (N,L,1)
            point_weights = point_weigths_from_curvature(
                target_curvs, target_points, self.curv_weight_max
            ) if self.curv_weight_max else None
        else: # 2D
            raise NotImplementedError()

        losses = chamfer_distance(
            pred_points,
            target_points,
            x_normals=pred_normals,
            y_normals=target_normals,
            x_curvatures=pred_curvs,
            y_curvatures=target_curvs,
            point_weights=point_weights,
            oriented_cosine_similarity=True
        )

        d_chamfer = losses[0]
        d_cosine = losses[1]
        d_curv = losses[2]

        return torch.stack([d_chamfer, d_cosine, d_curv])

class ChamferAndNormalsLoss(MeshLoss):
    """ Chamfer distance & cosine distance between the vertices and normals of
    the predicted mesh and a reference mesh.
    """

    def __init__(self, curv_weight_max=None):
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
        if target_points.shape[-1] == 3: # 3D
            pred_points, pred_normals = sample_points_from_meshes(
                pred_meshes, n_points, return_normals=True
            )
            target_curvs = target[2]
            point_weights = point_weigths_from_curvature(
                target_curvs, target_points, self.curv_weight_max
            ) if self.curv_weight_max else None
        else: # 2D
            pred_points, idx = choose_n_random_points(
                pred_meshes.verts_padded(), n_points, return_idx=True
            )
            pred_normals = meshes_to_vertex_normals_2D_packed(pred_meshes)
            # Select the normals of the corresponding points
            pt_shape = pred_points.shape
            pred_normals = pred_normals.view(
                pt_shape)[idx.unbind(1)].view(pt_shape)
            # Normals are required to be 3 dim. for chamfer function
            N, V, _ = target_normals.shape
            target_normals = torch.cat(
                [target_normals,
                 torch.zeros((N,V,1)).to(target_normals.device)], dim=2
            )
            pred_normals = torch.cat(
                [pred_normals,
                 torch.zeros((N,V,1)).to(pred_normals.device)], dim=2
            )
            if self.curv_weight_max is not None:
                raise RuntimeError("Cannot apply curvature weights in 2D.")

        losses = chamfer_distance(
            pred_points,
            target_points,
            x_normals=pred_normals,
            y_normals=target_normals,
            point_weights=point_weights,
            oriented_cosine_similarity=True
        )
        d_chamfer = losses[0]
        d_cosine = losses[1]

        return torch.stack([d_chamfer, d_cosine])

class LaplacianLoss(MeshLoss):
    # Method does not support autocast
    @autocast(enabled=False)
    def get_loss(self, pred_meshes, target=None):
        # pytorch3d loss for 3D
        if pred_meshes.verts_padded().shape[-1] == 3:
            loss = mesh_laplacian_smoothing(
                Meshes(pred_meshes.verts_padded().float(),
                       pred_meshes.faces_padded().float()),
                method='uniform'
            )
        # 2D
        else:
            verts_packed = pred_meshes.verts_packed()
            edges_packed = pred_meshes.faces_packed() # faces = edges
            V = len(verts_packed)
            # Uniform Laplacian
            with torch.no_grad():
                L = laplacian(verts_packed, edges_packed)
            loss = L.mm(verts_packed).norm(dim=1).sum() / V

        return loss

class NormalConsistencyLoss(MeshLoss):
    def get_loss(self, pred_meshes, target=None):
        # 2D: assumes clock-wise ordering of vertex indices in each edge
        if pred_meshes.verts_padded().shape[-1] == 2:
            normals, v0_idx, v1_idx = meshes_to_edge_normals_2D_packed(pred_meshes)
            loss = 1 - torch.cosine_similarity(normals[v0_idx],
                                               normals[v1_idx])
            return loss.sum() / len(v0_idx)

        # 3D: pytorch3d
        return mesh_normal_consistency(pred_meshes)

class EdgeLoss(MeshLoss):
    def __init__(self, target_length):
        self.target_length = target_length

    def __str__(self):
        return f"EdgeLoss({self.target_length})"

    def get_loss(self, pred_meshes, target=None):
        # 2D
        if pred_meshes.verts_padded().shape[2] == 2:
            verts_packed = pred_meshes.verts_packed()
            edges_packed = pred_meshes.faces_packed() # edges=faces
            verts_edges = verts_packed[edges_packed]
            v0, v1 = verts_edges.unbind(1)
            loss = ((v0 - v1).norm(dim=1, p=2) - self.target_length) ** 2.0
            return loss.sum() / len(edges_packed)
        # 3D
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
                            mesh_pred, mesh_target):
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
        elif isinstance(lf, ChamferAndNormalsAndCurvatureLoss):
            w1 = weight
            w2 = next(mesh_loss_weights_iter)
            w3 = next(mesh_loss_weights_iter)
            _add_MultiLoss_to_dict(
                losses, lf, mesh_pred, mesh_target, (w1, w2, w3),
                ("ChamferLoss()", "CosineLoss()", "CurvatureLoss()")
            )
        else: # add single loss to dict
            if isinstance(weight, Sequence):
                ml = lf(mesh_pred, mesh_target, weight)
                losses[str(lf)] = ml
            else:
                ml = lf(mesh_pred, mesh_target)
                losses[str(lf)] = ml * weight

    loss_total = sum(losses.values())

    return losses, loss_total

def voxel_linear_mesh_geometric_loss_combine(voxel_loss_func, voxel_loss_func_weights,
                            voxel_pred, voxel_target,
                            mesh_loss_func, mesh_loss_func_weights,
                            mesh_pred, mesh_target):
    """ Linear combination of voxel losses, geometric combination of mesh losses
    at each step and linear combination of steps and voxel and mesh losses.
    Reference: Kong et al. 2021 """

    if any(map(lambda x: isinstance(x, Sequence), mesh_loss_func_weights)):
        raise ValueError("Per-class-weights not supported.")

    # Voxel losses
    voxel_losses = {}
    if voxel_pred is not None:
        if not isinstance(voxel_pred, Sequence):
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
    # Init
    mesh_losses = {}
    for lf in mesh_loss_func:
        if isinstance(lf, ChamferAndNormalsLoss):
            mesh_losses['ChamferLoss()'] = []
            mesh_losses['CosineLoss()'] = []
        elif isinstance(lf, ChamferAndNormalsAndCurvatureLoss):
            mesh_losses['ChamferLoss()'] = []
            mesh_losses['CosineLoss()'] = []
            mesh_losses['CurvatureLoss()'] = []
        else:
            mesh_losses[str(lf)] = []

    S = len(mesh_pred)
    C = len(mesh_pred[0])
    for s in range(S):
        for c in range(C):
            mesh_losses_sc = {}
            for lf in mesh_loss_func:
                ml = lf([[mesh_pred[s][c]]], [mesh_target[c]])
                if isinstance(lf, ChamferAndNormalsLoss):
                    assert len(ml) == 2
                    mesh_losses_sc['ChamferLoss()'] = ml[0]
                    mesh_losses['ChamferLoss()'].append(
                        ml[0].detach().cpu()) # log
                    mesh_losses_sc['CosineLoss()'] = ml[1]
                    mesh_losses['CosineLoss()'].append(
                        ml[1].detach().cpu()) # log
                elif isinstance(lf, ChamferAndNormalsAndCurvatureLoss):
                    assert len(ml) == 3
                    mesh_losses_sc['ChamferLoss()'] = ml[0]
                    mesh_losses['ChamferLoss()'].append(
                        ml[0].detach().cpu()) # log
                    mesh_losses_sc['CosineLoss()'] = ml[1]
                    mesh_losses['CosineLoss()'].append(
                        ml[1].detach().cpu()) # log
                    mesh_losses_sc['CurvatureLoss()'] = ml[2]
                    mesh_losses['CurvatureLoss()'].append(
                        ml[2].detach().cpu()) # log
                else:
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
    mesh_losses = {k: torch.mean(torch.tensor(v)) for k, v in mesh_losses.items()}

    # All losses in one dictionary
    losses = {**voxel_losses, **mesh_losses}

    return losses, loss_total

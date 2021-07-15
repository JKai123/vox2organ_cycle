
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
from pytorch3d.ops import sample_points_from_meshes, laplacian
from torch.cuda.amp import autocast
from geomloss import SamplesLoss

from utils.utils import choose_n_random_points

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

    return (normals_next + normals_prev)

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
        if isinstance(self, ChamferAndNormalsLoss):
            mesh_loss = torch.tensor([0,0]).float().cuda()
        else:
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
        if isinstance(target, (tuple, list)):
            # target = (verts, normals)
            target_ = target[0] # Only vertices relevant
        # Select an equal number of points
        assert pred_.ndim == 3 and target._ndim == 3
        if pred_.shape[1] < target_.shape[1]:
            n_points = pred_.shape[1]
            perm = torch.randperm(target_.shape[1])
            perm = perm[:n_points]
            target_ = target_[:, perm, :]
        else:
            n_points = target_.shape[1]
            perm = torch.randperm(pred_.shape[1])
            perm = perm[:n_points]
            pred_ = pred_[:, perm, :]

        w_loss = torch.tensor(0.0, requires_grad=True).cuda()
        for p, t in zip(pred_, target_):
            w_loss += self.loss_func(p, t)

        # Average over batch
        return w_loss / float(pred_.shape[0])

class ChamferLoss(MeshLoss):
    """ Chamfer distance between the predicted mesh and randomly sampled
    surface points or a reference mesh. """
    def get_loss(self, pred_meshes, target):
        if isinstance(target, pytorch3d.structures.Pointclouds):
            n_points = torch.min(target.num_points_per_cloud())
            target_ = target
        if isinstance(target, pytorch3d.structures.Meshes):
            n_points = torch.min(target.num_verts_per_mesh())
            if target.verts_padded().shape[1] == 3:
                target_ = sample_points_from_meshes(target, n_points)
            else: # 2D --> choose vertex points
                target_ = choose_n_random_points(
                    target.verts_padded(), n_points
                )
        if isinstance(target, (tuple, list)):
            # target = (verts, normals)
            target_ = target[0] # Only vertices relevant
            assert target_.ndim == 3 # padded
            n_points = target_.shape[1]
        if pred_meshes.verts_packed().shape[-1] == 3:
            pred_points = sample_points_from_meshes(pred_meshes, n_points)
        else: # 2D
            pred_points = choose_n_random_points(
                pred_meshes.verts_padded(), n_points
            )
        return chamfer_distance(pred_points, target_)[0]

class ChamferAndNormalsLoss(MeshLoss):
    """ Chamfer distance + cosine distance between the vertices and normals of
    the predicted mesh and a reference mesh.

    Attention: When using this loss function, it should be assured that the
    prediction and the target follow the same normal convention.
    """
    def get_loss(self, pred_meshes, target):
        if len(target) != 2:
            raise TypeError("ChamferAndNormalsLoss requires vertices and"\
                            " normals.")
        target_points, target_normals = target
        assert target_points.ndim == 3 and target_normals.ndim == 3
        n_points = target_points.shape[1]
        if target_points.shape[-1] == 3: # 3D
            pred_points, pred_normals = sample_points_from_meshes(
                pred_meshes, n_points, return_normals=True
            )
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
        d_chamfer, d_cosine = chamfer_distance(pred_points, target_points,
                                               x_normals=pred_normals,
                                               y_normals=target_normals)

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
        ml = lf(mesh_pred, mesh_target)
        if isinstance(lf, ChamferAndNormalsLoss):
            assert len(ml) == 2
            losses['ChamferLoss()'] = ml[0]
            losses['CosineLoss()'] = ml[1]
        else:
            losses[str(lf)] = ml

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
    # Init
    mesh_losses = {}
    for lf in mesh_loss_func:
        if isinstance(lf, ChamferAndNormalsLoss):
            mesh_losses['ChamferLoss()'] = []
            mesh_losses['CosineLoss()'] = []
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

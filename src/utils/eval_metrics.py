
""" Evaluation metrics """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import IntEnum

import numpy as np
import torch
from pytorch3d.loss import chamfer_distance
from scipy.spatial.distance import directed_hausdorff
from geomloss import SamplesLoss

from utils.mesh import Mesh
from utils.utils import (
    sample_inner_volume_in_voxel,
    voxelize_mesh,
    voxelize_contour,
)
from utils.logging import (
    write_img_if_debug,
    measure_time)

class EvalMetrics(IntEnum):
    """ Supported evaluation metrics """
    # Jaccard score/ Intersection over Union from voxel prediction
    JaccardVoxel = 1

    # Chamfer distance between ground truth mesh and predicted mesh
    Chamfer = 2

    # Jaccard score/ Intersection over Union from mesh prediction
    JaccardMesh = 3

    # Symmetric Hausdorff distance between two meshes
    SymmetricHausdorff = 4

    # Wasserstein distance between point clouds
    Wasserstein = 5

@measure_time
def WassersteinScore(pred, data, n_v_classes, n_m_classes, model_class):
    """ Approximate Wasserstein distance with Sinkhorn iteration.
    If multiple structures are present, the average is returned.
    """
    # Ground truth
    mesh_gt = data[2]
    ndims = mesh_gt.ndims
    gt_vertices = mesh_gt.vertices.view(n_m_classes, -1, ndims).cuda()

    # Prediction: Only consider mesh of last step
    pred_vertices, _ = model_class.pred_to_verts_and_faces(pred)
    pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims)

    # Loss criterion: Sinkhorn divergence with L2 ground distance
    dist = SamplesLoss(loss="sinkhorn", p=2, diameter=2, blur=0.05)

    wds = []

    for p, gt in zip(pred_vertices, gt_vertices):
        # Select an equal number of points
        if len(p) < len(gt):
            p_ = p
            perm = torch.randperm(len(gt))
            perm = perm[:len(p)]
            gt_ = gt[perm, :]
        else:
            gt_ = gt
            perm = torch.randperm(len(p))
            perm = perm[:len(gt)]
            p_ = p[perm, :]

        wds.append(dist(p_, gt_).cpu().item())

    return wds

@measure_time
def SymmetricHausdorffScore(pred, data, n_v_classes, n_m_classes, model_class):
    """ Symmetric Hausdorff distance between predicted point clouds. If
    multiple structures are present, the maximum over the structures is
    returned.
    """
    # Ground truth
    mesh_gt = data[2]
    ndims = mesh_gt.ndims
    gt_vertices = mesh_gt.vertices.view(n_m_classes, -1, ndims)

    # Prediction: Only consider mesh of last step
    pred_vertices, _ = model_class.pred_to_verts_and_faces(pred)
    pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims).cpu().numpy()

    hds = []
    for pred, gt in zip(pred_vertices, gt_vertices):
        d = max(directed_hausdorff(pred, gt)[0],
                directed_hausdorff(gt, pred)[0])
        hds.append(d)

    return hds

@measure_time
def JaccardMeshScore(pred, data, n_v_classes, n_m_classes, model_class,
                     strip=True, compare_with='mesh_gt'):
    """ Jaccard averaged over classes ignoring background. The mesh prediction
    is compared against the voxel ground truth or against the mesh ground truth.
    """
    assert compare_with in ("voxel_gt", "mesh_gt")
    input_img = data[0].cuda()
    voxel_gt = data[1].cuda()
    mesh_gt = data[2]
    ndims = mesh_gt.ndims
    shape = voxel_gt.shape
    if compare_with == 'mesh_gt':
        vertices, faces = mesh_gt.vertices, mesh_gt.faces
        if ndims == 3:
            voxel_target = voxelize_mesh(
                vertices, faces, shape, n_m_classes
            ).cuda()
        else: # 2D
            voxel_target = voxelize_contour(
                vertices, shape
            ).cuda()
    else: # voxel gt
        voxel_target = voxel_gt
    vertices, faces = model_class.pred_to_verts_and_faces(pred)
    # Only mesh of last step considered and batch dimension squeezed out
    vertices = vertices[-1].view(n_m_classes, -1, ndims)
    faces = faces[-1].view(n_m_classes, -1, ndims)
    if ndims == 3:
        voxel_pred = voxelize_mesh(
            vertices, faces, shape, n_m_classes
        ).cuda()
    else: # 2D
        voxel_pred = voxelize_contour(
            vertices, shape
        ).cuda()

    if voxel_target.ndim == 3:
        voxel_target = voxel_target.unsqueeze(0)
        # Combine all structures into one voxelization
        voxel_pred = voxel_pred.sum(0).bool().long().unsqueeze(0)

    # Debug
    write_img_if_debug(input_img.squeeze().cpu().numpy(),
                       "../misc/voxel_input_img_eval.nii.gz")
    for i, (vp, vt) in enumerate(zip(voxel_pred, voxel_target)):
        write_img_if_debug(vp.squeeze().cpu().numpy(),
                           f"../misc/mesh_pred_img_eval_{i}.nii.gz")
        write_img_if_debug(vt.squeeze().cpu().numpy(),
                           f"../misc/voxel_target_img_eval_{i}.nii.gz")

    # Jaccard per structure
    j_vox_all = []
    for vp, vt in zip(voxel_pred.cuda(), voxel_target.cuda()):
        j_vox_all.append(
            Jaccard(vp.cuda(), vt.cuda(), 2)
        )

    return j_vox_all

@measure_time
def JaccardVoxelScore(pred, data, n_v_classes, n_m_classes, model_class, *args):
    """ Jaccard averaged over classes ignoring background """
    voxel_pred = model_class.pred_to_voxel_pred(pred)
    voxel_label = data[1].cuda()

    return Jaccard(voxel_pred, voxel_label, n_v_classes)

@measure_time
def Jaccard_from_Coords(pred, target, n_v_classes):
    """ Jaccard/ Intersection over Union from lists of occupied voxels. This
    necessarily implies that all occupied voxels belong to one class.

    Attention: This function is usally a lot slower than 'Jaccard' (probably
    because it does not exploit cuda).

    :param pred: Shape (C, V, 3)
    :param target: Shape (C, V, 3)
    :param n_v_classes: C
    """
    ious = []
    # Ignoring background class 0
    for c in range(1, n_v_classes):
        if isinstance(pred[c], torch.Tensor):
            pred[c] = pred[c].cpu().numpy()
        if isinstance(target[c], torch.Tensor):
            target[c] = target[c].cpu().numpy()
        intersection = 0
        for co in pred[c]:
            if any(np.equal(target[c], co).all(1)):
                intersection += 1

        union = pred[c].shape[0] + target[c].shape[0] - intersection

        # +1 for smoothing (no division by 0)
        ious.append(float(intersection + 1) / float(union + 1))

    return np.sum(ious)/(n_v_classes - 1)

@measure_time
def Jaccard(pred, target, n_classes):
    """ Jaccard/Intersection over Union """
    ious = []
    # Ignoring background class 0
    for c in range(1, n_classes):
        pred_idxs = pred == c
        target_idxs = target == c
        intersection = pred_idxs[target_idxs].long().sum().data.cpu()
        union = pred_idxs.long().sum().data.cpu() + \
                    target_idxs.long().sum().data.cpu() -\
                    intersection
        # +1 for smoothing (no division by 0)
        ious.append(float(intersection + 1) / float(union + 1))

    # Return average iou over classes ignoring background
    return np.sum(ious)/(n_classes - 1)

def ChamferScore(pred, data, n_v_classes, n_m_classes, model_class, *args):
    """ Chamfer distance averaged over classes

    Note: In contrast to the ChamferLoss, where the Chamfer distance may be computed
    between the predicted loss and randomly sampled surface points, here the
    Chamfer distance is computed between the predicted mesh and the ground
    truth mesh. """
    pred_vertices, _ = model_class.pred_to_verts_and_faces(pred)
    gt_vertices = data[2].vertices.cuda()
    if gt_vertices.ndim == 2:
        gt_vertices = gt_vertices.unsqueeze(0)
    chamfer_scores = []
    for c in range(n_m_classes):
        pv = pred_vertices[-1][c] # only consider last mesh step
        chamfer_scores.append(
            chamfer_distance(pv, gt_vertices[c][None])[0].cpu().item()
        )

    # Average over classes
    return np.sum(chamfer_scores) / float(n_m_classes)

EvalMetricHandler = {
    EvalMetrics.JaccardVoxel.name: JaccardVoxelScore,
    EvalMetrics.JaccardMesh.name: JaccardMeshScore,
    EvalMetrics.Chamfer.name: ChamferScore,
    EvalMetrics.SymmetricHausdorff.name: SymmetricHausdorffScore,
    EvalMetrics.Wasserstein.name: WassersteinScore
}


""" Evaluation metrics. Those metrics are typically computed directly from the
model prediction, i.e., in normalized coordinate space unless specified
otherwise."""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import IntEnum
import pymeshlab as pyml
import numpy as np
import open3d as o3d
import torch
from pytorch3d.loss import chamfer_distance, mesh_normal_consistency
from pytorch3d.ops import (
    knn_points,
    knn_gather,
    sample_points_from_meshes
)
from pytorch3d.structures import Meshes, Pointclouds
from scipy.spatial.distance import directed_hausdorff

from utils.utils import (
    voxelize_mesh,
    voxelize_contour,
)
from utils.logging import (
    write_img_if_debug,
    measure_time)
from utils.cortical_thickness import cortical_thickness
from utils.coordinate_transform import transform_mesh_affine
from utils.mesh import Mesh
from utils.cortical_thickness import _point_mesh_face_distance_unidirectional
from utils.utils_padded_packed import as_list

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
    # Wasserstein = 5

    # Difference in cortical thickness compared to ground truth
    CorticalThicknessError = 6

    # Average distance of predicted and groun truth mesh in terms of
    # point-to-mesh distance
    AverageDistance = 7

    
    # Number of self intersections
    SelfIntersections = 8

    # Normal Consistency
    NormalConsistency = 9


def SelfIntersectionsScore_o3d(
    pred,
    data,
    n_v_classes,
    n_m_classes,
    model_class,
) :
    """ Compute the relative number of self intersections. """
    # Prediction: Only consider mesh of last step
    pred_vertices, pred_faces = model_class.pred_to_verts_and_faces(pred)
    ndims = pred_vertices[-1].shape[-1]
    pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims)
    pred_faces = pred_faces[-1].view(n_m_classes, -1, ndims)

    isect_all = []

    for v, f in zip(pred_vertices, pred_faces):
        ms = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(v), o3d.utility.Vector3dVector(f))
        tri_index = np.asarray(ms.get_self_intersecting_triangles())
        num_self_intersections = np.shape(tri_index)[0]
        isect_all.append(num_self_intersections)
    return isect_all




def SelfIntersectionsScore(
    pred,
    data,
    n_v_classes,
    n_m_classes,
    model_class,
) :
    """ Compute the relative number of self intersections. """
    # Prediction: Only consider mesh of last step
    pred_vertices, pred_faces = model_class.pred_to_verts_and_faces(pred)
    ndims = pred_vertices[-1].shape[-1]
    pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims)
    pred_faces = pred_faces[-1].view(n_m_classes, -1, ndims)

    isect_all = []

    for v, f in zip(pred_vertices, pred_faces):
        ms = pyml.MeshSet()
        ms.add_mesh(pyml.Mesh(v.cpu().numpy(), f.cpu().numpy()))
        faces = ms.compute_topological_measures()['faces_number']
        ms.select_self_intersecting_faces()
        ms.delete_selected_faces()
        nnSI_faces = ms.compute_topological_measures()['faces_number']
        SI_faces = faces-nnSI_faces
        fracSI = (SI_faces/faces)*100
        isect_all.append(fracSI)

    return isect_all

def AverageDistanceScore(
    pred,
    data,
    n_v_classes,
    n_m_classes,
    model_class,
) :
    """ Compute point-to-mesh distance between prediction and ground truth. """

    # Ground truth
    gt_mesh = data['mesh_label']
    trans_affine = data['trans_affine_label']
    # Back to original coordinate space
    gt_vertices, gt_faces= gt_mesh.vertices, gt_mesh.faces
    ndims = gt_vertices.shape[-1]

    # Prediction: Only consider mesh of last step
    pred_vertices, pred_faces = model_class.pred_to_verts_and_faces(pred)
    pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims)
    pred_faces = pred_faces[-1].view(n_m_classes, -1, ndims)

    device = pred_vertices.device

    # Iterate over structures
    assd_all = []
    for pred_v, pred_f, gt_v, gt_f in zip(
        pred_vertices,
        pred_faces,
        gt_vertices.to(device),
        gt_faces.to(device)
    ):

        # Prediction in original space
        pred_v_t, pred_f_t = transform_mesh_affine(
            pred_v, pred_f, np.linalg.inv(trans_affine)
        )
        pred_mesh = Meshes([pred_v_t], [pred_f_t])
        pred_pcl = sample_points_from_meshes(pred_mesh, 100000)
        pred_pcl = Pointclouds(pred_pcl)

        # Ground truth in original space
        gt_v_t, gt_f_t = transform_mesh_affine(
            gt_v, gt_f, np.linalg.inv(trans_affine)
        )
        gt_mesh = Meshes([gt_v_t], [gt_f_t])
        gt_pcl = sample_points_from_meshes(gt_mesh, 100000)
        gt_pcl = Pointclouds(gt_pcl)

        # Compute distance
        P2G_dist = _point_mesh_face_distance_unidirectional(
            gt_pcl, pred_mesh
        ).cpu().numpy()
        G2P_dist = _point_mesh_face_distance_unidirectional(
            pred_pcl, gt_mesh
        ).cpu().numpy()

        assd2 = (P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])

        assd_all.append(assd2)

    return assd_all

def CorticalThicknessScore(pred, data, n_v_classes, n_m_classes, model_class):
    """ Compare cortical thickness to ground truth in terms of average absolute
    difference per vertex. In order for this measure to be meaningful, predited
    and ground truth meshes are transformed into the original coordinate space."""

    if n_m_classes not in (2, 4):
        raise ValueError("Cortical thickness score requires 2 or 4 surface meshes.")

    gt_mesh = data['mesh_label']
    trans_affine = data['trans_affine_label']
    # Back to original coordinate space
    new_vertices, new_faces = transform_mesh_affine(
        gt_mesh.vertices, gt_mesh.faces, np.linalg.inv(trans_affine)
    )
    gt_mesh_transformed = Mesh(new_vertices, new_faces, features=gt_mesh.features)
    gt_thickness = gt_mesh_transformed.features.view(n_m_classes, -1).cuda()
    ndims = gt_mesh_transformed.ndims
    gt_vertices = gt_mesh_transformed.vertices.view(n_m_classes, -1, ndims).cuda()

    # Prediction: Only consider mesh of last step
    pred_vertices, pred_faces = model_class.pred_to_verts_and_faces(pred)
    pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims)
    pred_faces = pred_faces[-1].view(n_m_classes, -1, ndims)
    pred_vertices, pred_faces = transform_mesh_affine(
        pred_vertices, pred_faces, np.linalg.inv(trans_affine)
    )

    # Thickness prediction
    pred_meshes = cortical_thickness(pred_vertices, pred_faces)

    # Compare white surface thickness prediction to thickness of nearest
    # gt point
    th_all = []
    for p_mesh, gt_v, gt_th in zip(pred_meshes, gt_vertices, gt_thickness):
        pred_v = p_mesh.vertices.view(1, -1, ndims)
        pred_th = p_mesh.features.view(-1)
        _, knn_idx, _ = knn_points(pred_v, gt_v.view(1, -1, ndims))
        nearest_thickness = knn_gather(gt_th.view(1, -1, 1), knn_idx).squeeze()

        thickness_score = torch.abs(pred_th - nearest_thickness).mean()

        th_all.append(thickness_score.cpu().item())

    return th_all

@measure_time
def SymmetricHausdorffScore(pred, data, n_v_classes, n_m_classes, model_class,
                           padded_coordinates=(0.0, 0.0, 0.0)):
    """ Symmetric Hausdorff distance between predicted point clouds.
    """
    # Ground truth
    mesh_gt = data['mesh_label']
    ndims = mesh_gt.ndims
    gt_vertices = mesh_gt.vertices.view(n_m_classes, -1, ndims)

    # Prediction: Only consider mesh of last step
    pred_vertices, _ = model_class.pred_to_verts_and_faces(pred)
    pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims).cpu().numpy()

    hds = []
    for p, gt_ in zip(pred_vertices, gt_vertices):
        # Remove padded vertices from gt
        gt = gt_[~np.isclose(gt_, padded_coordinates).all(axis=1)]
        d = max(directed_hausdorff(p, gt)[0],
                directed_hausdorff(gt, p)[0])
        hds.append(d)

    return hds

@measure_time
def JaccardMeshScore(pred, data, n_v_classes, n_m_classes, model_class,
                     strip=True, compare_with='mesh_gt'):
    """ Jaccard averaged over classes ignoring background. The mesh prediction
    is compared against the voxel ground truth or against the mesh ground truth.
    """
    assert compare_with in ("voxel_gt", "mesh_gt")
    input_img = data['img'].cuda()
    voxel_gt = data['voxel_label'].cuda()
    mesh_gt = data['mesh_label']
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
    voxel_label = data['voxel_label'].cuda()

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


def NormalConsistency(pred,
    data,
    n_v_classes,
    n_m_classes,
    model_class,
    ):
    "Normal consistency for each mesh"
    pred_vertices, pred_faces = model_class.pred_to_verts_and_faces(pred)
    ncs = []
    pv = pred_vertices[-1] # only consider last mesh step
    pf = pred_faces[-1]
    vmask = pred[0][0]._verts_mask
    fmask = pred[0][0]._faces_mask
    # pv = torch.squeeze(pv, dim=1)
    # pf = torch.squeeze(pf, dim=1)
    pv_l = as_list(pv, vmask, dim=0, squeeze=True)
    pf_l = as_list(pf, fmask, dim=0, squeeze=True)
    for pv, pf in zip(pv_l, pf_l):
        meshes = Meshes([pv], [pf])
        ncs.append(mesh_normal_consistency(meshes).cpu().item())
    return ncs
    

def ChamferScore(pred, data, n_v_classes, n_m_classes, model_class,
                 padded_coordinates=(0.0, 0.0, 0.0), **kwargs):
    """ Chamfer distance averaged over classes

    Note: In contrast to the ChamferLoss, where the Chamfer distance may be computed
    between the predicted loss and randomly sampled surface points, here the
    Chamfer distance is computed between the predicted mesh and the ground
    truth mesh. """
    gt_vertices, gt_faces = data['mesh_label'].vertices.cuda(), data['mesh_label'].faces.cuda()
    #gt_vertices = torch.permute(gt_vertices, [2, 1, 0])
    ndims = gt_vertices.shape[-1]
    pred_vertices, pred_faces = model_class.pred_to_verts_and_faces(pred)
    pred_vertices = pred_vertices[-1].view(n_m_classes, -1, ndims)
    pred_faces = pred_faces[-1].view(n_m_classes, -1, ndims)
    device = pred_vertices.device
    trans_affine = data['trans_affine_label']
    inv_trans_affine = np.linalg.inv(trans_affine)
    padded_coordinates = torch.Tensor(padded_coordinates).cuda()
    if gt_vertices.ndim == 2:
        gt_vertices = gt_vertices.unsqueeze(0)
    chamfer_scores = []
    for pv, pf, gv, gf in zip(
        pred_vertices,
        pred_faces,
        gt_vertices.to(device),
        gt_faces.to(device)
    ):


        pv_t, pf_t = transform_mesh_affine(
            pv, pf, inv_trans_affine
        )
        pv_t = pv_t[pv_t[:, 2] != 0]
        gv_t, gf_t = transform_mesh_affine(
            gv, gf, inv_trans_affine
        )
        gv_t = gv_t[gv_t[:, 2] != gv_t[-1, 2]]
        chamfer_scores.append(
            chamfer_distance(pv_t[None, :], gv_t[None, :])[0].cpu().item()
        )

    # Average over classes
    return chamfer_scores

EvalMetricHandler = {
    EvalMetrics.JaccardVoxel.name: JaccardVoxelScore,
    EvalMetrics.JaccardMesh.name: JaccardMeshScore,
    EvalMetrics.Chamfer.name: ChamferScore,
    EvalMetrics.SymmetricHausdorff.name: SymmetricHausdorffScore,
    EvalMetrics.CorticalThicknessError.name: CorticalThicknessScore,
    EvalMetrics.AverageDistance.name: AverageDistanceScore,
    EvalMetrics.SelfIntersections.name: SelfIntersectionsScore,
    EvalMetrics.NormalConsistency.name: NormalConsistency
}

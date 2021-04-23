""" Evaluation of a model """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import IntEnum
import os
import logging

import numpy as np
import torch
from tqdm import tqdm
from pytorch3d.loss import chamfer_distance

from utils.modes import ExecModes
from utils.mesh import Mesh
from utils.utils import create_mesh_from_voxels, verts_faces_to_Meshes
from models.voxel2mesh import Voxel2Mesh

class EvalMetrics(IntEnum):
    """ Supported evaluation metrics """
    # Jaccard score/ Intersection over Union
    Jaccard = 1

    # Chamfer distance between ground truth mesh and predicted mesh
    Chamfer = 2

def Jaccard_score(pred, data, n_classes):
    """ Jaccard averaged over classes ignoring background """
    voxel_pred = Voxel2Mesh.pred_to_voxel_pred(pred)
    _, voxel_label, _ = data # chop
    ious = []
    for c in range(1, n_classes):
        pred_idxs = voxel_pred == c
        target_idxs = voxel_label == c
        intersection = pred_idxs[target_idxs].long().sum().data.cpu()
        union = pred_idxs.long().sum().data.cpu() + \
                    target_idxs.long().sum().data.cpu() -\
                    intersection
        # +1 for smoothing (no division by 0)
        ious.append(float(intersection + 1) / float(union + 1))

    # Return average iou over classes ignoring background
    return np.sum(ious)/(n_classes - 1)

def Chamfer_score(pred, data, n_classes):
    """ Chamfer distance averaged over classes

    Note: In contrast to the ChamferLoss, where the Chamfer distance is computed
    between the predicted loss and randomly sampled surface points, here the
    Chamfer distance is computed between the predicted mesh and the ground
    truth mesh. """
    pred_vertices, _ = Voxel2Mesh.pred_to_verts_and_faces(pred)
    _, _, gt_mesh = data # chop
    gt_vertices = gt_mesh.vertices.cuda()[None] # currently only one class
    chamfer_scores = []
    for c in range(n_classes - 1):
        pred_vertices = pred_vertices[-1][c] # only consider last mesh step
        chamfer_scores.append(chamfer_distance(pred_vertices,
                                               gt_vertices)[0].cpu().item())

    # Average over classes
    return np.sum(chamfer_scores) / float(n_classes - 1)

class ModelEvaluator():
    """ Class for evaluation of models.

    :param eval_dataset: The dataset split that should be used for evaluation.
    :param save_dir: The experiment directory where data can be saved.
    :param n_classes: Number of classes.
    :param eval_metrics: A list of metrics to use for evaluation.
    :param mc_step_size: Marching cubes step size.
    """
    def __init__(self, eval_dataset, save_dir, n_classes, eval_metrics,
                 mc_step_size=1, **kwargs):
        self._dataset = eval_dataset
        self._save_dir = save_dir
        self._n_classes = n_classes
        self._eval_metrics = eval_metrics
        self._mc_step_size = mc_step_size

        self._mesh_dir = os.path.join(self._save_dir, "meshes")
        if not os.path.isdir(self._mesh_dir):
            os.mkdir(self._mesh_dir)

        self._metricHandler = {
            EvalMetrics.Jaccard.name: Jaccard_score,
            EvalMetrics.Chamfer.name: Chamfer_score
        }

    def evaluate(self, model, epoch, save_meshes=5):
        results_all = {}
        for m in self._eval_metrics:
            results_all[m] = []
        # Iterate over data split
        with torch.no_grad():
            for i in tqdm(range(len(self._dataset)), desc="Evaluate..."):
                data = self._dataset.get_item_and_mesh_from_index(i)
                data_voxel2mesh = Voxel2Mesh.convert_data_to_voxel2mesh_data(data,
                                                                  self._n_classes,
                                                                  ExecModes.TEST)
                pred = model(data_voxel2mesh)

                for metric in self._eval_metrics:
                    res = self._metricHandler[metric](pred, data, self._n_classes)
                    results_all[metric].append(res)

                if i < save_meshes: # Store meshes for visual inspection
                    filename =\
                            self._dataset.get_file_name_from_index(i).split(".")[0]
                    self.store_meshes(pred, data, filename, epoch)

        # Just consider means over evaluation set
        results = {k: np.mean(v) for k, v in results_all.items()}

        return results

    def store_meshes(self, pred, data, filename, epoch):
        """ Save predicted meshes and ground truth created with marching
        cubes
        """
        _, voxel_label, _ = data # chop
        for c in range(self._n_classes-1):
            # Label
            gt_filename = filename + "_class" + str(c + 1) + "_gt.ply"
            if not os.path.isfile(gt_filename):
                # gt file does not exist yet
                voxel_label_class = voxel_label.cpu()
                voxel_label_class[voxel_label != c + 1] = 0
                gt_mesh = create_mesh_from_voxels(voxel_label_class,
                                                  self._mc_step_size)
                gt_mesh.store(os.path.join(self._mesh_dir, gt_filename))

            # Mesh prediction
            pred_mesh_filename = filename + "_epoch" + str(epoch) +\
                "_class" + str(c + 1) + "_meshpred.ply"
            vertices, faces = Voxel2Mesh.pred_to_verts_and_faces(pred)
            vertices, faces = vertices[-1][c], faces[-1][c]
            pred_mesh = Mesh(vertices.squeeze().cpu(),
                             faces.squeeze().cpu())
            pred_mesh.store(os.path.join(self._mesh_dir, pred_mesh_filename))

            # Voxel prediction
            pred_voxel_filename = filename + "_epoch" + str(epoch) +\
                "_class" + str(c + 1) + "_voxelpred.ply"
            pred_voxel = Voxel2Mesh.pred_to_voxel_pred(pred)
            try:
                mc_pred_mesh = create_mesh_from_voxels(pred_voxel,
                                                  self._mc_step_size).to_trimesh(process=True)
                mc_pred_mesh.export(os.path.join(self._mesh_dir, pred_voxel_filename))
            except RuntimeError as e:
                logging.getLogger(ExecModes.TEST.name).warning(\
                       "In voxel prediction for file: %s: %s ", filename, e)

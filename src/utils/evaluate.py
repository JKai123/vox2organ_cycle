""" Evaluation of a model """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import logging
import glob

import numpy as np
import torch
from tqdm import tqdm

from utils.modes import ExecModes
from utils.eval_metrics import EvalMetricHandler
from utils.utils import (
    create_mesh_from_voxels,
)
from utils.coordinate_transform import (
    unnormalize_vertices_per_max_dim,
)
from utils.mesh import Mesh
from utils.logging import (
    write_img_if_debug
)
from utils.visualization import show_slices, show_img_with_contour

class ModelEvaluator():
    """ Class for evaluation of models.

    :param eval_dataset: The dataset split that should be used for evaluation.
    :param save_dir: The experiment directory where data can be saved.
    :param n_v_classes: Number of vertex classes.
    :param n_m_classes: Number of mesh classes.
    :param eval_metrics: A list of metrics to use for evaluation.
    :param mc_step_size: Marching cubes step size.
    """
    def __init__(self, eval_dataset, save_dir, n_v_classes, n_m_classes, eval_metrics,
                 mc_step_size=1, **kwargs):
        self._dataset = eval_dataset
        self._save_dir = save_dir
        self._n_v_classes = n_v_classes
        self._n_m_classes = n_m_classes
        self._eval_metrics = eval_metrics
        self._mc_step_size = mc_step_size

        self._mesh_dir = os.path.join(self._save_dir, "meshes")
        if not os.path.isdir(self._mesh_dir):
            os.mkdir(self._mesh_dir)

    def evaluate(self, model, epoch, save_meshes=5):
        results_all = {}
        model_class = model.__class__
        for m in self._eval_metrics:
            results_all[m] = []

        # Iterate over data split
        for i in tqdm(range(len(self._dataset)), desc="Evaluate..."):
            data = self._dataset.get_item_and_mesh_from_index(i)
            write_img_if_debug(data[1].squeeze().cpu().numpy(),
                               "../misc/raw_voxel_target_img_eval.nii.gz")
            write_img_if_debug(data[0].squeeze().cpu().numpy(),
                               "../misc/raw_voxel_input_img_eval.nii.gz")
            with torch.no_grad():
                pred = model(data[0][None].cuda())

            for metric in self._eval_metrics:
                res = EvalMetricHandler[metric](pred, data,
                                                self._n_v_classes,
                                                self._n_m_classes,
                                                model_class)
                results_all[metric].append(res)

            if i < save_meshes: # Store meshes for visual inspection
                filename =\
                        self._dataset.get_file_name_from_index(i).split(".")[0]
                self.store_meshes(pred, data, filename, epoch,
                                  model_class)

        # Just consider means over evaluation set
        results = {k: np.mean(v) for k, v in results_all.items()}

        return results

    def store_meshes(self, pred, data, filename, epoch, model_class,
                     show_all_steps=False, remove_previous=True):
        """ Save predicted meshes and ground truth created with marching
        cubes
        """
        # Remove previously stored files to avoid dumping storage
        if remove_previous:
            for suffix in ("*_meshpred.ply", "*_voxelpred.ply",
                           "*_meshpred.png", "*_voxelpred.png"):
                files_to_delete = glob.glob(os.path.join(
                    self._mesh_dir, filename + suffix
                ))
                for f in files_to_delete:
                    try:
                        os.remove(f)
                    except:
                        print("Error while deleting file ", f)
        # Data
        img = data[0].squeeze()
        # Label
        gt_mesh = data[2]
        ndims = gt_mesh.ndims
        logging.getLogger(ExecModes.TEST.name).debug(
            "%d vertices in ground truth mesh",
            len(gt_mesh.vertices.view(-1, ndims))
        )
        # Store ground truth if it does not exist yet
        if ndims == 3:
            gt_filename = filename + "_gt.ply"
            gt_filename = os.path.join(self._mesh_dir, gt_filename)
            if not os.path.isfile(gt_filename):
                gt_mesh.store(gt_filename)
        else: # 2D
            gt_filename = filename + "_gt.png"
            gt_filename = os.path.join(self._mesh_dir, gt_filename)
            gt_mesh = gt_mesh.to_pytorch3d_Meshes()
            if not os.path.isfile(gt_filename):
                pass
                show_img_with_contour(
                    img,
                    unnormalize_vertices_per_max_dim(
                        gt_mesh.verts_packed(),
                        img.shape
                    ),
                    gt_mesh.faces_packed(),
                    gt_filename
                )

        # Mesh prediction
        vertices, faces = model_class.pred_to_verts_and_faces(pred)
        if show_all_steps:
            # Visualize meshes of all steps
            for s, (v, f) in enumerate(zip(vertices, faces)):
                pred_mesh = Mesh(v.squeeze().cpu(), f.squeeze().cpu())
                logging.getLogger(ExecModes.TEST.name).debug(
                    "%d vertices in predicted mesh", len(v.view(-1, ndims))
                )
                if ndims == 3:
                    pred_mesh_filename = filename + "_epoch" + str(epoch) +\
                        "_step" + str(s) + "_meshpred.ply"
                    pred_mesh_filename = os.path.join(self._mesh_dir,
                                                      pred_mesh_filename)
                    pred_mesh.store(pred_mesh_filename)
                else: # 2D
                    pred_mesh_filename = filename + "_epoch" + str(epoch) +\
                        "_step" + str(s) + "_meshpred.png"
                    pred_mesh_filename = os.path.join(self._mesh_dir,
                                                      pred_mesh_filename)
                    pred_mesh = pred_mesh.to_pytorch3d_Meshes()
                    show_img_with_contour(
                        img,
                        unnormalize_vertices_per_max_dim(
                            pred_mesh.verts_packed(), img.shape
                        ),
                        pred_mesh.faces_packed(),
                        pred_mesh_filename
                    )
        else:
            # Only visualize last step
            v, f = vertices[-1], faces[-1]
            pred_mesh = Mesh(v.squeeze().cpu(), f.squeeze().cpu())
            logging.getLogger(ExecModes.TEST.name).debug(
                "%d vertices in predicted mesh", len(v.view(-1, ndims))
            )
            if ndims == 3:
                pred_mesh_filename = filename + "_epoch" + str(epoch) +\
                    "_meshpred.ply"
                pred_mesh_filename = os.path.join(self._mesh_dir,
                                                  pred_mesh_filename)
                pred_mesh.store(pred_mesh_filename)
            else: # 2D
                pred_mesh_filename = filename + "_epoch" + str(epoch) +\
                    "_meshpred.png"
                pred_mesh_filename = os.path.join(self._mesh_dir,
                                                  pred_mesh_filename)
                pred_mesh = pred_mesh.to_pytorch3d_Meshes()
                show_img_with_contour(
                    img,
                    unnormalize_vertices_per_max_dim(
                        pred_mesh.verts_packed(),
                        img.shape
                    ),
                    pred_mesh.faces_packed(),
                    pred_mesh_filename
                )

        # Voxel prediction
        voxel_pred = model_class.pred_to_voxel_pred(pred)
        for c in range(1, self._n_v_classes):
            voxel_pred_class = voxel_pred.squeeze()
            voxel_pred_class[voxel_pred_class != c] = 0
            if ndims == 3:
                pred_voxel_filename = filename + "_epoch" + str(epoch) +\
                    "_class" + str(c) + "_voxelpred.ply"
                pred_voxel_filename = os.path.join(self._mesh_dir,
                                                   pred_voxel_filename)
                try:
                    mc_pred_mesh = create_mesh_from_voxels(
                        voxel_pred_class, self._mc_step_size
                    ).to_trimesh(process=True)
                    mc_pred_mesh.export()
                except ValueError as e:
                    logging.getLogger(ExecModes.TEST.name).warning(
                           "In voxel prediction for file: %s: %s."
                           " This means usually that the prediction"
                           " is all 1.", filename, e)
                except RuntimeError as e:
                    logging.getLogger(ExecModes.TEST.name).warning(
                           "In voxel prediction for file: %s: %s ",
                           filename, e)
                except AttributeError:
                    # No voxel prediction exists
                    pass
            else: # 2D
                pred_voxel_filename = filename + "_epoch" + str(epoch) +\
                    "_class" + str(c) + "_voxelpred.png"
                pred_voxel_filename = os.path.join(self._mesh_dir,
                                                   pred_voxel_filename)
                show_slices(
                    [img.cpu()],
                    [voxel_pred_class.cpu()],
                    pred_voxel_filename
                )

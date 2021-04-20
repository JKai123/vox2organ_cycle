
""" Training procedure """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import logging

import json
import torch
import wandb
import numpy as np
from pytorch3d.structures import Meshes
from torch.utils.data import DataLoader

from utils.utils import (string_dict,
                         verts_faces_to_Meshes)
from utils.logging import init_logging, log_losses, get_log_dir
from utils.modes import ExecModes
from utils.evaluate import ModelEvaluator
from utils.losses import linear_loss_combine, geometric_loss_combine
from data.dataset import dataset_split_handler
from models.model_handler import ModelHandler
from models.voxel2mesh import Voxel2Mesh

class Solver():
    """
    Solver class for optimizing the weights of neural networks.

    :param int n_classes: The number of classes to distinguish (including
    background)
    :param torch.optim optimizer_class: The optimizer to use, e.g. Adam.
    :param dict optim_params: The parameters for the optimizer. If empty,
    default values are used.
    :param evaluator: Evaluator for the optimized model.
    :param list voxel_loss_func: A list of loss functions to apply for the 3D voxel
    prediction.
    :param list voxel_loss_func_weights: A list of the same length of 'voxel_loss_func'
    with weights for the losses.
    :param list mesh_loss_func: A list of loss functions to apply for the mesh
    prediction.
    :param list mesh_loss_func_weights: A list of the same length of 'mesh_loss_func'
    with weights for the losses.
    :param str loss_averaging: The way the weighted average of the losses is
    computed, e.g. 'linear' weighted average, 'geometric' mean
    :param str save_path: The path where results and stats are saved.
    :param log_every: Log the stats every n iterations.
    :param n_sample_points: The number of points sampled for mesh loss
    computation.
    :param str device: The device for execution, e.g. 'cuda:0'.

    """

    def __init__(self,
                 n_classes,
                 optimizer_class,
                 optim_params,
                 evaluator,
                 voxel_loss_func,
                 voxel_loss_func_weights,
                 mesh_loss_func,
                 mesh_loss_func_weights,
                 loss_averaging,
                 save_path,
                 log_every,
                 n_sample_points,
                 device,
                 **kwargs):

        self.n_classes = n_classes
        self.optim_class = optimizer_class
        self.optim_params = optim_params
        self.optim = None # defined for each training separately
        self.evaluator = evaluator
        self.voxel_loss_func = voxel_loss_func
        self.voxel_loss_func_weights = voxel_loss_func_weights
        assert len(voxel_loss_func) == len(voxel_loss_func_weights),\
                "Number of weights must be equal to number of 3D seg. losses."

        self.mesh_loss_func = mesh_loss_func
        self.mesh_loss_func_weights = mesh_loss_func_weights
        assert len(mesh_loss_func) == len(mesh_loss_func_weights),\
                "Number of weights must be equal to number of mesh losses."

        self.loss_averaging = loss_averaging
        self.save_path = save_path
        self.log_every = log_every
        self.n_sample_points = n_sample_points
        self.device = device

    def training_step(self, model, data, iteration):
        """ One training step.

        :param model: Current pytorch model.
        :param data: The minibatch.
        :param iteration: The training iteration (used for logging)
        :returns: The overall (weighted) loss.
        """
        self.optim.zero_grad()
        loss_total = self.compute_loss(model, data, iteration)
        loss_total.backward()
        self.optim.step()

        logging.getLogger(ExecModes.TRAIN.name).debug("Completed backward"\
                                                      " pass.")

        return loss_total

    def compute_loss(self, model, data, iteration) -> torch.tensor:
        # make data compatible
        data = Voxel2Mesh.convert_data_to_voxel2mesh_data(data,
                                               self.n_classes,
                                               ExecModes.TRAIN)
        pred = model(data)

        losses = {}
        # Voxel losses
        for lf in self.voxel_loss_func:
            losses[str(lf)] = lf(pred[0][-1][3], data['y_voxels'])

        # Mesh losses
        vertices, faces = Voxel2Mesh.pred_to_verts_and_faces(pred)
        pred_meshes = verts_faces_to_Meshes(vertices, faces, 2) # pytorch3d
        targets = data['surface_points']
        for lf in self.mesh_loss_func:
            losses[str(lf)] = lf(pred_meshes, targets)

        # Merge loss weights into one list
        combined_loss_weights = self.voxel_loss_func_weights +\
                self.mesh_loss_func_weights

        if self.loss_averaging == 'linear':
            loss_total = linear_loss_combine(losses.values(),
                                             combined_loss_weights)
        elif self.loss_averaging == 'geometric':
            loss_total = geometric_loss_combine(losses.values(),
                                                combined_loss_weights)
        else:
            raise ValueError("Unknown loss averaging.")

        losses['TotalLoss'] = loss_total

        # log
        if iteration % self.log_every == 0:
            log_losses(losses, iteration)

        return loss_total

    def train(self,
              model: torch.nn.Module,
              training_set: torch.utils.data.Dataset,
              validation_set: torch.utils.data.Dataset,
              n_epochs: int,
              batch_size: int,
              early_stop: bool,
              eval_every: int):
        """
        Training procedure

        :param model: The model to train.
        :param training_set: The training dataset.
        :param validation_set: The validation dataset.
        :param n_epochs: The number of training epochs.
        :param batch_size: The minibatch size.
        :param early_stop: Enable early stopping.
        :param eval_every: Evaluate the model every n epochs.
        """

        model.float().to(self.device)

        trainLogger = logging.getLogger(ExecModes.TRAIN.name)
        trainLogger.info("Training on device %s", self.device)

        self.optim = self.optim_class(model.parameters(), **self.optim_params)

        training_loader = DataLoader(training_set, batch_size=batch_size)
        trainLogger.info("Created training loader of length %d",
                    len(training_loader))

        iteration = 1

        for epoch in range(1, n_epochs+1):
            model.train()

            # TODO: Change training_set -> training_loader for batch size > 1
            for iter_in_epoch, data in enumerate(training_set):
                # Step
                loss = self.training_step(model, data, iteration)

                iteration += 1

            # Evaluate
            if epoch % eval_every == 0:
                model.eval()
                # self.evaluator.eval(model)
                best_state = model.state_dict()
                best_epoch = epoch

            # TODO: Early stopping

        # Save models
        model.eval()
        model.save(os.path.join(self.save_path, "final.model"))
        model.load_state_dict(best_state)
        model.save(os.path.join(self.save_path, "best.model"))

        # Save epochs corresponding to models
        epochs_file = os.path.join(self.save_path, "models_to_epochs.json")
        with open(epochs_file, 'w') as f:
            json.dump({"final.model": n_epochs, "best.model": best_epoch}, f)


        logging.getLogger(ExecModes.TRAIN.name).info("Saved models at"\
                                                     " %s", self.save_path)
        logging.getLogger(ExecModes.TRAIN.name).info("Training finished.")

def training_routine(hps: dict, experiment_name=None, loglevel='INFO'):
    """
    A full training routine including setup of experiments etc.

    :param dict hps: Hyperparameters to use.
    :param str experiment_name (optional): The name of the experiment
    directory. If None, a name is created automatically.
    :param lovlevel: The loglevel of the standard logger to use.
    """

    ###### Prepare training experiment ######

    experiment_base_dir = hps['EXPERIMENT_BASE_DIR']

    if experiment_name is not None:
        experiment_dir = os.path.join(experiment_base_dir, experiment_name)
    else:
        # Automatically enumerate experiments exp_i
        ids_exist = []
        for n in os.listdir(experiment_base_dir):
            try:
                ids_exist.append(int(n.split("_")[-1]))
            except ValueError:
                pass
        if len(ids_exist) > 0:
            new_id = np.max(ids_exist) + 1
        else:
            new_id = 1

        experiment_name = "exp_" + str(new_id)
        hps['EXPERIMENT_NAME'] = experiment_name

        experiment_dir = os.path.join(experiment_base_dir, experiment_name)

    # Lower case param names as input to constructors/functions
    hps_lower = dict((k.lower(), v) for k, v in hps.items())

    # Create directories
    log_dir = get_log_dir(experiment_dir)
    if experiment_name=="debug":
        # Overwrite
        os.makedirs(log_dir, exist_ok=True)
    else:
        # Throw error if directory exists already
        os.makedirs(log_dir)

    # Store hyperparameters
    param_file = os.path.join(experiment_dir, "params.json")
    hps_to_write = string_dict(hps)
    with open(param_file, 'w') as f:
        json.dump(hps_to_write, f)

    # Configure logging
    init_logging(logger_name=ExecModes.TRAIN.name,
                 exp_name=experiment_name,
                 log_dir=log_dir,
                 loglevel=loglevel,
                 mode=ExecModes.TRAIN,
                 proj_name=hps['PROJ_NAME'],
                 group_name=hps['GROUP_NAME'],
                 params=hps_to_write)
    trainLogger = logging.getLogger(ExecModes.TRAIN.name)
    trainLogger.info("Start training '%s'...", experiment_name)

    ###### Load data ######
    trainLogger.info("Loading dataset %s...", hps['DATASET'])
    try:
        training_set,\
                validation_set,\
                test_set = dataset_split_handler[hps['DATASET']](**hps_lower)
    except KeyError:
        print(f"Dataset {hps['DATASET']} not known.")
        return

    trainLogger.info("%d training files.", len(training_set))
    trainLogger.info("%d validation files.", len(validation_set))
    trainLogger.info("%d test files.", len(test_set))

    ###### Training ######

    model = ModelHandler[hps['ARCHITECTURE']].value(\
                                        ndims=hps['N_DIMS'],
                                        num_classes=hps['N_CLASSES'],
                                        patch_shape=hps['PATCH_SIZE'],
                                        config=hps['MODEL_CONFIG'])
    evaluator = ModelEvaluator(eval_dataset=validation_set,
                               save_dir=experiment_dir, **hps_lower)

    solver = Solver(evaluator=evaluator, save_path=experiment_dir, **hps_lower)

    solver.train(model=model,
                 training_set=training_set,
                 validation_set=validation_set,
                 n_epochs=hps['N_EPOCHS'],
                 batch_size=hps['BATCH_SIZE'],
                 early_stop=hps['EARLY_STOP'],
                 eval_every=hps['EVAL_EVERY'])

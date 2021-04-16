
""" Training procedure """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import logging

import json
import torch
import wandb
import numpy as np
from torch.utils.data import DataLoader

from utils.utils import serializable_dict, convert_data_to_voxel2mesh_data
from utils.logging import init_logging
from utils.modes import ExecModes
from utils.evaluate import ModelEvaluator
from data.dataset import dataset_split_handler
from models.model_handler import ModelHandler

class Solver():
    """
    Solver class for optimizing the weights of neural networks.

    :param int n_classes: The number of classes to distinguish (including
    background)
    :param torch.optim optimizer: The optimizer to use, e.g. Adam.
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

    """

    def __init__(self,
                 n_classes,
                 optimizer,
                 optim_params,
                 evaluator,
                 voxel_loss_func,
                 voxel_loss_func_weights,
                 mesh_loss_func,
                 mesh_loss_func_weights,
                 loss_averaging,
                 save_path,
                 log_every):

        self.n_classes = n_classes
        self.optim = optimizer
        self.optim_params = optim_params
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

        return loss_total

    def compute_loss(self, model, data, iteration) -> torch.tensor:
        loss_total = 0
        data = convert_data_to_voxel2mesh_data(data, self.n_classes,
                                               ExecModes.TRAIN) # compatibility of code
        breakpoint()
        pred = model(data)
        return loss_total

    def linear_loss(self, losses, weights):
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

    def train(self,
              model: torch.nn.Module,
              training_set: torch.utils.data.Dataset,
              validation_set: torch.utils.data.Dataset,
              n_epochs: int,
              early_stop: bool,
              eval_every: int):
        """
        Training procedure

        :param model: The model to train.
        :param training_set: The training dataset.
        :param validation_set: The validation dataset.
        :param n_epochs: The number of training epochs.
        :param early_stop: Enable early stopping.
        :param eval_every: Evaluate the model every n epochs.
        """

        breakpoint()
        # optim = self.optim(model.parameters(), **self.optim_params)
        optim = self.optim(filter(lambda p: p.requires_grad,
                                  model.parameters(),
                                  **self.optim_params))
        training_loader = DataLoader(training_set)

        iteration = 1

        for epoch in range(1, n_epochs+1):
            for iter_in_epoch, data in enumerate(training_set):

                # Step
                loss = self.training_step(model, data, iteration)

                iteration += 1

            # Evaluate
            if epoch % eval_every == 0:
                self.evaluator.eval(model, validation_set, epoch)

            # TODO: Early stopping




def training_routine(hps: dict, experiment_name=None, loglevel='INFO'):
    """
    A full training routine including setup of experiments etc.

    :param dict hps: Hyperparameters to use.
    :param str experiment_name (optional): The name of the experiment
    directory. If None, a name is created automatically.
    """

    ###### Prepare training experiment ######

    experiment_base_dir = hps['EXPERIMENT_BASE_DIR']
    experiment_name = hps.get('EXPERIMENT_NAME', None)

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

    # Store hyperparameters
    param_file = os.path.join(experiment_dir, "params.json")
    hps_to_write = serializable_dict(hps)
    with open(param_file, 'w') as f:
        json.dump(hps_to_write, f)

    # Create directories
    log_dir = os.path.join(experiment_dir, "logs")
    if experiment_name=="debug":
        # Overwrite
        os.makedirs(log_dir, exist_ok=True)
    else:
        # Throw error if directory exists already
        os.makedirs(log_dir)

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
                test_set = dataset_split_handler[hps['DATASET']](hps)
    except KeyError:
        print(f"Dataset {hps['DATASET']} not known.")
        return

    trainLogger.info("%d training files.", len(training_set))
    trainLogger.info("%d validation files.", len(validation_set))
    trainLogger.info("%d test files.", len(test_set))

    breakpoint()

    ###### Training ######

    model = ModelHandler[hps['ARCHITECTURE']].value(\
                                        batch_size=hps['BATCH_SIZE'],
                                        ndims=hps['N_DIMS'],
                                        num_classes=hps['N_CLASSES'],
                                        patch_shape=hps['PATCH_SHAPE'],
                                        config=hps['VOXEL2MESH_ORIG_CONFIG'])
    evaluator = ModelEvaluator()

    solver = Solver(
        n_classes=hps['N_CLASSES'],
        optimizer=hps['OPTIMIZER'],
        optim_params=hps['OPTIM_PARAMS'],
        evaluator=evaluator,
        voxel_loss_func=hps['VOXEL_LOSS_FUNCTIONS'],
        voxel_loss_func_weights=hps['VOXEL_LOSS_FUNCTION_WEIGHTS'],
        mesh_loss_func=hps['MESH_LOSS_FUNCTIONS'],
        mesh_loss_func_weights=hps['MESH_LOSS_FUNCTION_WEIGHTS'],
        loss_averaging=hps['LOSS_FUNCTION_AVERAGING'],
        save_path=experiment_dir,
        log_every=hps['LOG_EVERY'])

    solver.train(model=model,
                 training_set=training_set,
                 validation_set=validation_set,
                 n_epochs=hps['EPOCHS'],
                 early_stop=hps['EARLY_STOP'],
                 eval_every=hps['EVAL_EVERY'])

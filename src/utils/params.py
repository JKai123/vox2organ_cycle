""" Documentation of project-wide parameters and default values 

Ideally, all occurring parameters should be documented here.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import Enum

import torch
import torch.nn.functional as F

hyper_ps_default={

    # The cuda device
    'DEVICE_NR': 0,

    # The number of classes to distinguish (including background)
    'N_CLASSES': 2,

    # The batch size used during training
    'BATCH_SIZE': 1,

    # The number of training epochs
    'EPOCHS': 1,

    # The optimizer used for training
    'OPTIMIZER': torch.optim.Adam,

    # Parameters for the optimizer
    'OPTIM_PARAMS': {#
        'lr': 1e-4,
        'betas': (0.9, 0.999),
        'eps': 1e-8,
        'weight_decay': 0.0},

    # The used loss functions for the voxel segmentation
    'VOXEL_LOSS_FUNCTIONS': [F.cross_entropy],

    # The weights for the voxel loss functions
    'VOXEL_LOSS_FUNCTION_WEIGHTS': [1.],

    # The used loss functions for the mesh
    'MESH_LOSS_FUNCTIONS': [], # TODO

    # The weights for the mesh loss functions
    'MESH_LOSS_FUNCTION_WEIGHTS': [], # TODO

    # The way the weighted average of the losses is computed,
    # e.g. 'linear' weighted average, 'geometric' mean
    'LOSS_FUNCTION_AVERAGING': 'linear',

    # Log losses etc. every n iterations
    'LOG_EVERY': 1,

    # Evaluate model every n epochs
    'EVAL_EVERY': 1,

    # Use early stopping
    'EARLY_STOP': False,

    # The number of image dimensions
    'N_DIMS': 3,

    # Voxel2Mesh original parameters
    # (from https://github.com/cvlab-epfl/voxel2mesh)
    'VOXEL2MESH_ORIG_CONFIG': {
        'FIRST_LAYER_CHANNELS': 16,
        'NUM_INPUT_CHANNELS': 1,
        'STEPS': 4,
        'BATCH_NORM': True,
        'GRAPH_CONV_LAYER_COUNT': 4,
        'MESH_TEMPLATE': '../supplementary_material/spheres/icosahedron_162.obj'},

    # input should be cubic. Otherwise, input should be padded accordingly.
    'PATCH_SIZE': (64, 64, 64),

    # Seed for dataset splitting
    'DATASET_SEED': 1234,

    # Proportions of dataset splits
    'DATASET_SPLIT_PROPORTIONS': (80, 10, 10),

    # The directory where experiments are stored
    'EXPERIMENT_BASE_DIR': "../experiments/",

    # Directory of raw data
    'RAW_DATA_DIR': "/raw/data/dir", # <<<< Needs to set (e.g. in main.py)

    # Directory of preprocessed data
    'PREPROCESSED_DATA_DIR': "/preprocessed/data/dir", # <<<< Needs to set (e.g. in main.py)
}

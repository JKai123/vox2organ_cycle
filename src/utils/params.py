""" Documentation of project-wide parameters and default values 

Ideally, all occurring parameters should be documented here.
"""

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from enum import Enum

import torch

from utils.losses import (
    ChamferLoss,
    LaplacianLoss,
    NormalConsistencyLoss,
    EdgeLoss
)
from utils.utils_voxel2meshplusplus.graph_conv import (
    GraphConvNorm
)

hyper_ps_default={

    # >>> Note: Using tuples (...) instead of lists [...] may lead to problems
    # when resuming broken trainings (json converts tuples to lists when dumping).
    # Therefore, it is recommended to use lists for parameters here.

    # The number of vertex classes to distinguish (including background)
    'N_V_CLASSES': 2,

    # The number of mesh classes. This is usually the number of non-connected
    # components/structures
    'N_M_CLASSES': 2,

    # The number of vertices in a single template structure
    'N_TEMPLATE_VERTICES': 162,

    # The number of reference points in a cortex structure
    'N_REF_POINTS_PER_STRUCTURE': 40962,

    # The batch size used during training
    'BATCH_SIZE': 1,

    # Accumulate n gradients before doing a backward pass
    'ACCUMULATE_N_GRADIENTS': 1,

    # The number of training epochs
    'N_EPOCHS': 5,

    # The optimizer used for training
    'OPTIMIZER_CLASS': torch.optim.Adam,

    # Parameters for the optimizer
    'OPTIM_PARAMS': {#
        'lr': 1e-4,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'weight_decay': 0.0},

    # Data augmentation
    'AUGMENT_TRAIN': False,

    # Whether or not to use Pytorch's automatic mixed precision
    'MIXED_PRECISION': False,

    # The used loss functions for the voxel segmentation
    'VOXEL_LOSS_FUNC': [torch.nn.CrossEntropyLoss()],

    # The weights for the voxel loss functions
    'VOXEL_LOSS_FUNC_WEIGHTS': [1.],

    # The used loss functions for the mesh
    'MESH_LOSS_FUNC': [ChamferLoss(),
                       LaplacianLoss(),
                       NormalConsistencyLoss(),
                       EdgeLoss()],

    # The weights for the mesh loss functions, given are the values from
    # Wickramasinghe et al. Kong et al. used a geometric averaging and weights
    # [0.3, 0.05, 0.46, 0.16]
    'MESH_LOSS_FUNC_WEIGHTS': [1.0, 0.1, 0.1, 1.0],

    # The number of sample points for the mesh loss computation if done as by
    # Wickramasinghe 2020, i.e. sampling n random points from the outer surface
    # of the voxel ground truth
    'N_SAMPLE_POINTS': 3000,

    # The way the weighted average of the losses is computed,
    # e.g. 'linear' weighted average, 'geometric' mean
    'LOSS_AVERAGING': 'linear',

    # Log losses etc. every n iterations or 'epoch'
    'LOG_EVERY': 1,

    # Evaluate model every n epochs
    'EVAL_EVERY': 1,

    # Use early stopping
    'EARLY_STOP': False,

    # The metrics used for evaluation, see utils.evaluate.EvalMetrics for
    # options
    'EVAL_METRICS': [
        'JaccardVoxel',
        'JaccardMesh',
        'Chamfer'
    ],

    # Main validation metric according to which the best model is determined.
    # Note: This one must also be part of 'EVAL_METRICS'!
    'MAIN_EVAL_METRIC': 'JaccardMesh',

    # The number of image dimensions (is sometimes set to 3 in the code so may
    # not have an effect for models that exist only for 3D)
    'NDIMS': 3,

    # Voxel2Mesh original parameters
    # (from https://github.com/cvlab-epfl/voxel2mesh).
    # Note that not for all models/architectures all of
    # those parameters are relevant.
    'MODEL_CONFIG': {
        'FIRST_LAYER_CHANNELS': 16,
        'ENCODER_CHANNELS': [16, 32, 64, 128, 256],
        'DECODER_CHANNELS': [128, 64, 32, 16], # Voxel decoder
        'GRAPH_CHANNELS': [32, 32, 32, 32, 32], # Graph decoder
        'NUM_INPUT_CHANNELS': 1,
        'STEPS': 4,
        'DEEP_SUPERVISION': False, # For voxel net
        'BATCH_NORM': False, # Only for graph convs, always True in voxel layers
        # Number of hidden layers in the graph conv blocks
        'GRAPH_CONV_LAYER_COUNT': 4,
        'MESH_TEMPLATE': '../supplementary_material/spheres/icosahedron_162.obj',
        'UNPOOL_INDICES': [0,1,0,1,0],
        'USE_ADOPTIVE_UNPOOL': False,
        # Weighted feature aggregation in graph convs (only possible with
        # pytorch-geometric graph convs)
        'WEIGHTED_EDGES': False,
        # Whether to use a voxel decoder
        'VOXEL_DECODER': True,
        # The graph conv implementation to use
        'GC': GraphConvNorm,
        # Whether to propagate coordinates in the graph decoder in addition to
        # voxel features
        'PROPAGATE_COORDS': False
    },

    # Decay the learning rate by multiplication with 'LR_DECAY_RATE' if no
    # improvement for 'LR_DECAY_AFTER' epochs
    'LR_DECAY_RATE': 0.5,
    'LR_DECAY_AFTER': -1, # -1 = no decay

    # input should be cubic. Otherwise, input should be padded accordingly.
    'PATCH_SIZE': [64, 64, 64],

    # Seed for dataset splitting
    'DATASET_SEED': 1234,

    # Proportions of dataset splits
    'DATASET_SPLIT_PROPORTIONS': [80, 10, 10],

    # The directory where experiments are stored
    'EXPERIMENT_BASE_DIR': "../experiments/",

    # Directory of raw data
    'RAW_DATA_DIR': "/raw/data/dir", # <<<< Needs to set (e.g. in main.py)

    # Directory of preprocessed data
    'PREPROCESSED_DATA_DIR': "/preprocessed/data/dir", # <<<< Needs to set (e.g. in main.py)
}

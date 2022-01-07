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

DATASET_SPLIT_PARAMS = (
    'DATASET_SEED',
    'DATASET_SPLIT_PROPORTIONS',
    'FIXED_SPLIT',
    'OVERFIT'
)

DATASET_PARAMS = (
    'DATASET',
    'RAW_DATA_DIR',
    'PREPROCESSED_DATA_DIR'
)

# Directory for output to check
CHECK_DIR = "../to_check"

# Miscellanous output
MISC_DIR = "../misc"

hyper_ps_default={

    # >>> Note: Using tuples (...) instead of lists [...] may lead to problems
    # when resuming broken trainings (json converts tuples to lists when dumping).
    # Therefore, it is recommended to use lists for parameters here.

    # The name of an experiment (=base folder for all data stored throughout
    # training and testing)
    'EXPERIMENT_NAME': None,

    # Project name used for wandb
    'PROJ_NAME': 'vox2cortex',

    # The architecture to use
    'ARCHITECTURE': 'voxel2meshplusplusgeneric',

    # The dataset to use
    'DATASET': 'TO BE SPECIFIED',

    # The loglevel for output logs
    'LOGLEVEL': 'INFO',

    # Wandb logging group
    'GROUP_NAME': 'uncategorized',

    # The device to train on
    'DEVICE': 'cuda:0',

    # Whether to do overfitting
    'OVERFIT': False,

    # If execution times should be measured for some functions
    'TIME_LOGGING': False,

    # A list of parameters to tune
    'PARAMS_TO_TUNE': None,

    # A list of parameters to fine-tune
    'PARAMS_TO_FINE_TUNE': None,

    # For testing the model from a certain training epoch; if None, the final
    # model is used
    'TEST_MODEL_EPOCH': None,

    # Specification of an ablation study, see utils.ablation_study
    'ABLATION_STUDY': False,

    # The path where templates are stored
    'TEMPLATE_PATH': "../supplementary_material/white_pial/",

    # The template name in dependence of the number of vertices N,
    # 'SELECT_PATH_SIZE' (sps) and 'PATCH_SIZE' (ps)
    'TEMPLATE_NAME': (
        lambda M, N, sps, ps: f"cortex_{M}_1000_3_smoothed_{N}_sps{sps}_ps{ps}.obj"
    ),

    # The number of vertex classes to distinguish (including background)
    'N_V_CLASSES': 2,

    # The number of mesh classes. This is usually the number of non-connected
    # components/structures
    'N_M_CLASSES': 2,

    # The number of vertices in a single template structure
    'N_TEMPLATE_VERTICES': 162,

    # The number of vertices in a single template structure used during testing
    # (may be different than 'N_TEMPLATE_VERTICES'; -1 means that
    # 'N_TEMPLATE_VERTICES' is used)
    'N_TEMPLATE_VERTICES_TEST': -1,

    # The number of reference points in a cortex structure
    'N_REF_POINTS_PER_STRUCTURE': 40962,

    # Either use a mesh or a pointcloud as ground truth. Basically, if one
    # wants to compute only point losses like the Chamfer loss, a pointcloud is
    # sufficient while other losses like cosine distance between vertex normals
    # require a mesh (pointcloud + faces)
    'MESH_TARGET_TYPE': "pointcloud",

    # The type of meshes used, either 'freesurfer' or 'marching cubes'
    'MESH_TYPE': 'marching cubes',

    # The mode for reduction of mesh regularization losses, either 'linear' or
    # 'none'
    'REDUCE_REG_LOSS_MODE': 'none',

    # The structure type for cortex data, either 'cerebral_cortex' or
    # 'white_matter'
    'STRUCTURE_TYPE': "white_matter",

    # Check if data has been transformed correctly. This leads potentially to a
    # larger memory consumption since meshes are voxelized and voxel labels are
    # loaded (even though only one of them is typically used)
    'SANITY_CHECK_DATA': True,

    # The batch size used during training
    'BATCH_SIZE': 1,

    # Optionally provide a norm for gradient clipping
    'CLIP_GRADIENT': False,

    # Activate/deactivate patch mode for the cortex dataset. Possible values
    # are "no", "single-patch", "multi-patch"
    'PATCH_MODE': "no",

    # Accumulate n gradients before doing a backward pass
    'ACCUMULATE_N_GRADIENTS': 1,

    # The number of training epochs
    'N_EPOCHS': 5,

    # Freesurfer ground truth meshes with reduced resolution. 1.0 = original
    # resolution (in terms of number of vertices)
    'REDUCED_FREESURFER': 1.0,

    # Choose either 'voxelized_meshes' or 'aseg' segmentation ground truth
    # labels
    'SEG_GROUND_TRUTH': 'voxelized_meshes',

    # Whether to use curvatures of the meshes. If set to True, the ground truth
    # points are vertices and not sampled surface points
    'PROVIDE_CURVATURES': False,

    # The optimizer used for training
    'OPTIMIZER_CLASS': torch.optim.Adam,

    # Parameters for the optimizer. A separate learning rate for the graph
    # network can be specified
    'OPTIM_PARAMS': {
        'lr': 1e-4, # voxel lr
        'graph_lr': 5e-5,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'weight_decay': 0.0
    },

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
                       EdgeLoss(0.0)],

    # The weights for the mesh loss functions, given are the values from
    # Wickramasinghe et al. Kong et al. used a geometric averaging and weights
    # [0.3, 0.05, 0.46, 0.16]
    'MESH_LOSS_FUNC_WEIGHTS': [1.0, 0.1, 0.1, 1.0],

    # Penalize large vertex displacements, can be seen as a regularization loss
    # function weight
    'PENALIZE_DISPLACEMENT': 0.0,

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
        'Wasserstein',
        'SymmetricHausdorff',
        'JaccardVoxel',
        'JaccardMesh',
        'Chamfer'
    ],

    # Main validation metric according to which the best model is determined.
    # Note: This one must also be part of 'EVAL_METRICS'!
    'MAIN_EVAL_METRIC': 'JaccardMesh',

    # The number of image dimensions. This parameter is deprecated since
    # dimensionality is now inferred from the patch size.
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
        'NORM': 'none', # Only for graph convs, batch norm always used in voxel layers
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
        'PROPAGATE_COORDS': False,
        # Dropout probability of UNet blocks
        'P_DROPOUT_UNET': None,
        # Dropout probability of graph conv blocks
        'P_DROPOUT_GRAPH': None,
        # The used patch size, should be equal to global patch size
        'PATCH_SIZE': [64, 64, 64],
        # The ids of structures that should be grouped in the graph net.
        # Example: if lh_white and rh_white have ids 0 and 1 and lh_pial and
        # rh_pial have ids 2 and 3, then the groups should be specified as
        # ((0,1),(2,3))
        'GROUP_STRUCTS': None,
        # Whether to exchange coordinates between groups
        'EXCHANGE_COORDS': True,
        # The number of neighbors considered for feature aggregation from
        # vertices of different structures in the graph net
        'K_STRUCT_NEIGHBORS': 5,
        # The mechanism for voxel feature aggregations, can be 'trilinear',
        # 'bilinear', or 'lns'
        'AGGREGATE': 'trilinear',
        # Where to take the features from the UNet
        'AGGREGATE_INDICES': [[5,6],[6,7],[7,8]],
    },

    # Decay the learning rate by multiplication with 'LR_DECAY_RATE' if no
    # improvement for 'LR_DECAY_AFTER' epochs
    'LR_DECAY_RATE': 0.5,
    'LR_DECAY_AFTER': -1, # -1 = no decay

    # input should be cubic. Otherwise, input should be padded accordingly.
    'PATCH_SIZE': [64, 64, 64],

    # For selecting a patch from cortex dataset.
    'SELECT_PATCH_SIZE': [192, 224, 192],

    # Seed for dataset splitting
    'DATASET_SEED': 1234,

    # Proportions of dataset splits
    'DATASET_SPLIT_PROPORTIONS': [80, 10, 10],

    # Dict or bool value that allows for specifying fixed ids for dataset
    # splitting.
    # If specified, 'DATASET_SEED' and 'DATASET_SPLIT_PROPORTIONS' will be
    # ignored. The dict should contain values for keys 'train', 'validation',
    # and 'test'. Alternatively, a list of files can be specified containing
    # IDs for 'train', 'validation', and 'test'
    'FIXED_SPLIT': False,

    # The directory where experiments are stored
    'EXPERIMENT_BASE_DIR': "../experiments/",

    # Directory of raw data
    'RAW_DATA_DIR': "/raw/data/dir", # <<<< Needs to set (e.g. in main.py)

    # Directory of preprocessed data, e.g., containing thickness values from
    # FreeSurfer
    'PREPROCESSED_DATA_DIR': "/preprocessed/data/dir", # <<<< Needs to set (e.g. in main.py)

    # Define the measure of uncertainty, possible values: 'mc', None
    'UNCERTAINTY': None,
}

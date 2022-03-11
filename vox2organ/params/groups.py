
""" Experiment-specific parameters. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from params.default import hyper_ps_default
from utils.utils import update_dict
from utils.losses import (
    ChamferAndNormalsLoss,
    ClassAgnosticChamferAndNormalsLoss,
    LaplacianLoss,
    NormalConsistencyLoss,
    EdgeLoss
)
from utils.utils.graph_conv import (
    GraphConvNorm,
)

# This dict contains groups of parameters that kind of belong together in order
# to conduct certain experiments
hyper_ps_groups = {
    # No patch mode, vox2cortex
    'Vox2Cortex no-patch': {
        'BASE_GROUP': None,
        'N_EPOCHS': 100,
        'N_REF_POINTS_PER_STRUCTURE': 50000, # 50K
        'MESH_LOSS_FUNC': [
           ChamferAndNormalsLoss(curv_weight_max=5.0),
           LaplacianLoss(),
           NormalConsistencyLoss(),
           EdgeLoss(0.0)
        ],
        'PATCH_MODE': 'no',
        # Order of structures: lh_white, rh_white, lh_pial, rh_pial; mesh loss
        # weights should respect this order!
        'MESH_LOSS_FUNC_WEIGHTS': [
            [1.0] * 4, # Chamfer
            [0.01] * 2 + [0.0125] * 2, # Cosine,
            [0.1] * 2 + [0.25] * 2, # Laplace,
            [0.001] * 2 + [0.00225] * 2, # NormalConsistency
            [5.0] * 4 # Edge
        ],
        'N_M_CLASSES': 4,
        'PATCH_SIZE': [128, 144, 128],
        'SELECT_PATCH_SIZE': [192, 208, 192],
        'MODEL_CONFIG': {
            'GRAPH_CHANNELS': [256, 64, 64, 64, 64],
            'UNPOOL_INDICES': [0,0,0,0],
            'AGGREGATE_INDICES': [
                [3,4,5,6],
                [2,3,6,7],
                [1,2,7,8],
                [0,1,7,8] # 8 = last decoder skip
            ],
            'NORM': 'batch', # Only for graph convs
            'DECODER_CHANNELS': [64, 32, 16, 8],
            'DEEP_SUPERVISION': True,
            'WEIGHTED_EDGES': False,
            'PROPAGATE_COORDS': True,
            'VOXEL_DECODER': True,
            'GC': GraphConvNorm,
            'GROUP_STRUCTS': [[0, 1], [2, 3]],
        }
    },

    'Vox2Cortex-Parc no-patch': {
       'BASE_GROUP': 'Vox2Cortex no-patch',
        'MESH_LOSS_FUNC': [
           ClassAgnosticChamferAndNormalsLoss(curv_weight_max=5.0),
           LaplacianLoss(),
           NormalConsistencyLoss(),
           EdgeLoss(0.0)
        ],
    },

    # One hemisphere
    'Vox2Cortex-Parc one hemisphere': {
        'BASE_GROUP': 'Vox2Cortex-Parc no-patch',
        'PATCH_MODE': 'single-patch',
        'N_M_CLASSES': 2,
        'PATCH_SIZE': [64, 144, 128],
        'SELECT_PATCH_SIZE': [96, 208, 192],
        'MODEL_CONFIG': {
            'GROUP_STRUCTS': [[0], [1]],
        }
    },

    'CorticalFlow no-patch': {
        'BASE_GROUP': None,
        'N_EPOCHS': 50,
        'N_REF_POINTS_PER_STRUCTURE': 50000, # 50K
        'ARCHITECTURE': 'corticalflow',
        'FREEZE_PRE_TRAINED': True,
        'N_M_CLASSES': 4,
        'PATCH_SIZE': [192, 208, 192],
        'SELECT_PATCH_SIZE': [192, 208, 192],
        'MESH_LOSS_FUNC': [
           ChamferAndNormalsLoss(),
           EdgeLoss(0.0)
        ],
        'PATCH_MODE': 'no',
        # Order of structures: rh_white, rh_pial
        'MESH_LOSS_FUNC_WEIGHTS': [
            [1.0] * 4, # Chamfer
            [0.0] * 4, # Normals
            [1.0] * 4 # Edge
        ],
        # No voxel decoder --> set voxel loss weights to 0
        'VOXEL_LOSS_FUNC_WEIGHTS': [],
        'VOXEL_LOSS_FUNC': [],
        'EVAL_METRICS': [
            'SymmetricHausdorff',
            'JaccardMesh',
            'Chamfer',
            'AverageDistance'
        ],
        'OPTIM_PARAMS': {
            'graph_lr': None,
        },
    },

    'CorticalFlow-Parc no-patch': {
        'BASE_GROUP': 'CorticalFlow no-patch',
        'MESH_LOSS_FUNC': [
           ClassAgnosticChamferAndNormalsLoss(),
           EdgeLoss(0.0)
        ],
    },

    'CorticalFlow no-patch step 1': {
        'BASE_GROUP': 'CorticalFlow no-patch',
        'PRE_TRAINED_MODEL_PATH': None,
        'REDUCED_TEMPLATE': True,
        'MODEL_CONFIG': {
            'ENCODER_CHANNELS': [
                [16, 32, 64, 128, 256],
            ],
            'DECODER_CHANNELS': [
                [128, 64, 32, 16],
            ],
        },
    },
    'CorticalFlow no-patch step 2': {
        'BASE_GROUP': 'CorticalFlow no-patch',
        'PRE_TRAINED_MODEL_PATH': 'previous',
        'REDUCED_TEMPLATE': False,
        'MODEL_CONFIG': {
            'ENCODER_CHANNELS': [
                [16, 32, 64, 128, 256],
                [16, 32, 64],
            ],
            'DECODER_CHANNELS': [
                [128, 64, 32, 16],
                [32, 16],
            ],
        },
    },
    'CorticalFlow no-patch step 3': {
        'BASE_GROUP': 'CorticalFlow no-patch',
        'PRE_TRAINED_MODEL_PATH': 'previous',
        'REDUCED_TEMPLATE': False,
        'MODEL_CONFIG': {
            'ENCODER_CHANNELS': [
                [16, 32, 64, 128, 256],
                [16, 32, 64],
                [16, 32, 64]
            ],
            'DECODER_CHANNELS': [
                [128, 64, 32, 16],
                [32, 16],
                [32, 16]
            ],
        },
    },

    'CorticalFlow-Parc no-patch step 1': {
        'BASE_GROUP': 'CorticalFlow-Parc no-patch',
        'PRE_TRAINED_MODEL_PATH': None,
        'REDUCED_TEMPLATE': True,
        'MODEL_CONFIG': {
            'ENCODER_CHANNELS': [
                [16, 32, 64, 128, 256],
            ],
            'DECODER_CHANNELS': [
                [128, 64, 32, 16],
            ],
        },
    },
    'CorticalFlow-Parc no-patch step 2': {
        'BASE_GROUP': 'CorticalFlow-Parc no-patch',
        'PRE_TRAINED_MODEL_PATH': 'previous',
        'REDUCED_TEMPLATE': False,
        'MODEL_CONFIG': {
            'ENCODER_CHANNELS': [
                [16, 32, 64, 128, 256],
                [16, 32, 64],
            ],
            'DECODER_CHANNELS': [
                [128, 64, 32, 16],
                [32, 16],
            ],
        },
    },
    'CorticalFlow-Parc no-patch step 3': {
        'BASE_GROUP': 'CorticalFlow-Parc no-patch',
        'PRE_TRAINED_MODEL_PATH': 'previous',
        'REDUCED_TEMPLATE': False,
        'MODEL_CONFIG': {
            'ENCODER_CHANNELS': [
                [16, 32, 64, 128, 256],
                [16, 32, 64],
                [16, 32, 64]
            ],
            'DECODER_CHANNELS': [
                [128, 64, 32, 16],
                [32, 16],
                [32, 16]
            ],
        },
    },

    'Vox2Cortex-Parc dropout': {
        'BASE_GROUP': 'Vox2Cortex-Parc no-patch',
        'MODEL_CONFIG': {
            'P_DROPOUT_UNET': 0.2,
            'P_DROPOUT_GRAPH': 0.2,
        }
    },

}

def assemble_group_params(group_name: str):
    """ Combine group params for a certain group name and potential base
    groups.
    """
    group_params = hyper_ps_groups[group_name]
    if group_params['BASE_GROUP'] is not None:
        base_params = assemble_group_params(group_params['BASE_GROUP'])
    else:
        base_params = hyper_ps_default

    return update_dict(base_params, group_params)

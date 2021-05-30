#!/usr/bin/env python3

""" Main file """

import torch

from argparse import ArgumentParser, RawTextHelpFormatter

from utils.params import hyper_ps_default
from utils.modes import ExecModes
from utils.utils import update_dict
from utils.train import training_routine
from utils.tune_params import tuning_routine
from utils.test import test_routine
from utils.train_test import train_test_routine
from utils.losses import (
    ChamferLoss,
    ChamferAndNormalsLoss,
    LaplacianLoss,
    NormalConsistencyLoss,
    EdgeLoss
)
from utils.utils_voxel2meshplusplus.graph_conv import (
    GraphConvNorm,
    GCNConvWrapped,
    GINConvWrapped
)

# Overwrite default parameters for a common training procedure
hyper_ps = {
    # Overwriting std values from utils.params
    #######################
    'EXPERIMENT_NAME': None,  # Attention: "debug" overwrites previous dir"
                              # should be set with console argument
    #######################
    # Learning
    'N_EPOCHS': 2000,
    'EVAL_EVERY': 50,
    'LOG_EVERY': 'epoch',
    'ACCUMULATE_N_GRADIENTS': 1,
    'AUGMENT_TRAIN': True,
    'DATASET_SPLIT_PROPORTIONS': [50, 25, 25],
    'MIXED_PRECISION': True,
    'EVAL_METRICS': [
        'JaccardVoxel',
        'JaccardMesh',
        'Chamfer'
    ],
    'OPTIM_PARAMS': {#
        'lr': 1e-4,
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'weight_decay': 0.0},
    'LR_DECAY_AFTER': 300,
    'DATASET_SEED': 1532,
    'LOSS_AVERAGING': 'linear',
    # CE
    'VOXEL_LOSS_FUNC_WEIGHTS': [1.0],
    'MESH_LOSS_FUNC': [ChamferAndNormalsLoss(),
                       LaplacianLoss(),
                       NormalConsistencyLoss(),
                       EdgeLoss()],
    # 'MESH_LOSS_FUNC_WEIGHTS': [0.3, 0.05, 0.46, 0.16],
    'MESH_LOSS_FUNC_WEIGHTS': [1.0, 0.1, 0.1, 1.0],
    # Model
    'MODEL_CONFIG': {
        'BATCH_NORM': True, # Only for graph convs, always True in voxel layers
        # Decoder channels from Kong, should be multiples of 2
        'DECODER_CHANNELS': [64, 32, 16, 8],
        # Graph decoder channels should be multiples of 2
        'GRAPH_CHANNELS': [128, 64, 32, 16],
        'DEEP_SUPERVISION': True,
        'WEIGHTED_EDGES': False,
        'VOXEL_DECODER': True,
        'GC': GraphConvNorm
    },
}

# Dataset specific parameters
hyper_ps_hippocampus = {
    'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/Task04_Hippocampus/",
    'PATCH_SIZE': (64, 64, 64),
    'BATCH_SIZE': 15,
    'N_M_CLASSES': 1,
    'N_REF_POINTS_PER_STRUCTURE': 3000,
    'N_TEMPLATE_VERTICES': 162,
    'MODEL_CONFIG': {
        'UNPOOL_INDICES': [0,1,1],
    },
    'PROJ_NAME': "hippocampus",
    'MESH_TARGET_TYPE': "mesh"
}
hyper_ps_hippocampus['MODEL_CONFIG']['MESH_TEMPLATE'] =\
    f"../supplementary_material/spheres/icosahedron_{hyper_ps_hippocampus['N_TEMPLATE_VERTICES']}.obj"

hyper_ps_cortex = {
    'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/MALC_CSR/",
    'PATCH_SIZE': (192, 224, 192),
    'BATCH_SIZE': 1,
    'N_M_CLASSES': 2,
    'N_REF_POINTS_PER_STRUCTURE': 40962,
    'N_TEMPLATE_VERTICES': 40962,
    'MODEL_CONFIG': {
        'UNPOOL_INDICES': [0,0,0],
    },
    'PROJ_NAME': "cortex"
}
hyper_ps_cortex['MODEL_CONFIG']['MESH_TEMPLATE'] =\
    f"../supplementary_material/spheres/cortex_white_matter_spheres_{hyper_ps_cortex['N_TEMPLATE_VERTICES']}.obj"

# Overwrite params for overfitting (fewer epochs, no augmentation, smaller
# dataset)
hyper_ps_overfit = {
    # Learning
    'N_EPOCHS': 1000,
    'EVAL_EVERY': 50,
    'BATCH_SIZE': 5,
    'AUGMENT_TRAIN': False,
    'MIXED_PRECISION': False,
}


mode_handler = {
    ExecModes.TRAIN.value: training_routine,
    ExecModes.TEST.value: test_routine,
    ExecModes.TRAIN_TEST.value: train_test_routine,
    ExecModes.TUNE.value: tuning_routine
}


def main(hps):
    """
    Main function for training, validation, test
    """
    argparser = ArgumentParser(description="cortex-parcellation-using-meshes",
                               formatter_class=RawTextHelpFormatter)
    argparser.add_argument('--architecture',
                           type=str,
                           default="voxel2meshplusplusgeneric",
                           help="The name of the algorithm. Supported:\n"
                           "- voxel2mesh\n"
                           "- voxel2meshplusplus\n"
                           "- voxel2meshplusplusgeneric")
    argparser.add_argument('--dataset',
                           type=str,
                           default="Cortex",
                           help="The name of the dataset. Supported:\n"
                           "- Hippocampus\n"
                           "- Cortex")
    argparser.add_argument('--train',
                           action='store_true',
                           help="Train a model.")
    argparser.add_argument('--test',
                           action='store_true',
                           help="Test a model.")
    argparser.add_argument('--tune',
                           default=None,
                           type=str,
                           dest='params_to_tune',
                           nargs='+',
                           help="Specify the name of a parameter to tune.")
    argparser.add_argument('--resume',
                           action='store_true',
                           help="Resume an existing, potentially unfinished"\
                           " experiment.")
    argparser.add_argument('--log',
                           type=str,
                           dest='loglevel',
                           default='INFO',
                           help="Specify log level.")
    argparser.add_argument('--proj',
                           type=str,
                           dest='proj_name',
                           default=None,
                           help="Specify the name of the wandb project.")
    argparser.add_argument('--group',
                           type=str,
                           dest='group_name',
                           default='uncategorized',
                           help="Specify the name of the wandb group.")
    argparser.add_argument('--device',
                           type=str,
                           dest='device',
                           default='cuda:0',
                           help="Specify the device for execution.")
    argparser.add_argument('--overfit',
                           action='store_true',
                           help="Overfit on a few training samples.")
    argparser.add_argument('--time',
                           action='store_true',
                           help="Measure time of some functions.")
    argparser.add_argument('-n', '--exp_name',
                           dest='exp_name',
                           type=str,
                           default=None,
                           help="Name of experiment:\n"
                           "- 'debug' means that the results are  written "
                           "into a directory \nthat might be overwritten "
                           "later. This may be useful for debugging \n"
                           "where the experiment result does not matter.\n"
                           "- Any other name cannot overwrite an existing"
                           " directory.\n"
                           "- If not specified, experiments are automatically"
                           " enumerated with exp_i.")
    args = argparser.parse_args()
    hps['EXPERIMENT_NAME'] = args.exp_name
    hps['ARCHITECTURE'] = args.architecture
    hps['DATASET'] = args.dataset
    hps['LOGLEVEL'] = args.loglevel
    hps['PROJ_NAME'] = args.proj_name
    hps['GROUP_NAME'] = args.group_name
    hps['DEVICE'] = args.device
    hps['OVERFIT'] = args.overfit
    hps['TIME_LOGGING'] = args.time
    hps['PARAMS_TO_TUNE'] = args.params_to_tune

    if args.exp_name == "debug":
        # Overfit when debugging
        hps['OVERFIT'] = True

    torch.cuda.set_device(args.device)

    # Fill hyperparameters with defaults
    hps = update_dict(hyper_ps_default, hps)

    # Dataset specific params
    if args.dataset == 'Hippocampus':
        hps = update_dict(hps, hyper_ps_hippocampus)
    if args.dataset == 'Cortex':
        hps = update_dict(hps, hyper_ps_cortex)

    # Update again for overfitting
    if hps['OVERFIT']:
        hps = update_dict(hps, hyper_ps_overfit)

    if args.params_to_tune is not None:
        mode = ExecModes.TUNE
    else:
        if args.train and not args.test:
            mode = ExecModes.TRAIN.value
        if args.test and not args.train:
            mode = ExecModes.TEST.value
        if args.train and args.test:
            mode = ExecModes.TRAIN_TEST.value
        if not args.test and not args.train:
            print("Please use either --train or --test or both.")
            return

    if hps['ARCHITECTURE'] == "voxel2mesh" and hps['MIXED_PRECISION']:
        raise ValueError("Mixed precision is not supported for original"\
                         " voxel2mesh.")
    if args.architecture == 'voxel2mesh' and hps['BATCH_SIZE'] != 1:
        raise ValueError("Original voxel2mesh only allows for batch size 1."\
                         " Try voxel2meshplusplus for larger batch size.")
    # No voxel decoder --> set voxel loss weights to 0
    if not hps['MODEL_CONFIG']['VOXEL_DECODER']:
        hps['VOXEL_LOSS_FUNC_WEIGHTS'] = []
        hps['VOXEL_LOSS_FUNC'] = []
        if 'JaccardVoxel' in hps['EVAL_METRICS']:
            hps['EVAL_METRICS'].remove('JaccardVoxel')

    # Run
    routine = mode_handler[mode]
    routine(hps, experiment_name=hps['EXPERIMENT_NAME'],
            loglevel=hps['LOGLEVEL'], resume=args.resume)


if __name__ == '__main__':
    main(hyper_ps)

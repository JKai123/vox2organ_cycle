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
    ChamferAndNormalsAndCurvatureLoss,
    LaplacianLoss,
    NormalConsistencyLoss,
    EdgeLoss,
    WassersteinLoss
)
from utils.utils_voxel2meshplusplus.graph_conv import (
    GraphConvNorm,
    GCNConvWrapped,
    GINConvWrapped,
    GeoGraphConvNorm
)

# Overwrite default parameters for a common training procedure
hyper_ps = {
    # Overwriting std values from utils.params
    #######################
    'EXPERIMENT_NAME': None,  # Attention: "debug" overwrites previous dir"
                              # should be set with console argument
    #######################
    # Dataset
    'DATASET_SEED': 1532,
    'DATASET_SPLIT_PROPORTIONS': [50, 25, 25],
    # Learning
    'EVAL_EVERY': 100,
    'LOG_EVERY': 'epoch',
    'ACCUMULATE_N_GRADIENTS': 8,
    'MIXED_PRECISION': True,
    'OPTIMIZER_CLASS': torch.optim.Adam,
    'OPTIM_PARAMS': {
        'lr': 1e-4, # voxel lr
        'graph_lr': 5e-5,
        # SGD
        # 'momentum': 0.9,
        # Adam
        'betas': [0.9, 0.999],
        'eps': 1e-8,
        'weight_decay': 0.0
    },
    'LR_DECAY_AFTER': 300,
    # Loss function
    'LOSS_AVERAGING': 'linear',
    # CE
    'VOXEL_LOSS_FUNC_WEIGHTS': [1.0],
    'MESH_LOSS_FUNC': [
                       # WassersteinLoss(),
                       # ChamferLoss(),
                       ChamferAndNormalsLoss(curv_weight_max=5.0),
                       LaplacianLoss(),
                       NormalConsistencyLoss(),
                       EdgeLoss(0.0)
                      ],
    # 'MESH_LOSS_FUNC': [WassersteinLoss()],
    # 'MESH_LOSS_FUNC_WEIGHTS': [0.3, 0.05, 0.46, 0.16], # Kong
    # 'MESH_LOSS_FUNC_WEIGHTS': [1.0, 0.1, 0.1, 0.1, 1.0], # Wickramasinghe (adapted)
    # 'MESH_LOSS_FUNC_WEIGHTS': [0.1, 0.01, 0.01, 0.01], # Tuned for geometric averaging
    # 'MESH_LOSS_FUNC_WEIGHTS': [0.5, 0.01, 0.1, 0.01], # Tuned on patch
    # 'MESH_LOSS_FUNC_WEIGHTS': [0.1, 0.01, 0.01, 0.01], # Tuned with smaller lr
    # 'MESH_LOSS_FUNC_WEIGHTS': [1.0, 0.5, 0.001, 10.0], # Reverse tuned
    # Model
    'MODEL_CONFIG': {
        'NORM': 'batch', # Only for graph convs
        # Decoder channels from Kong, should be multiples of 2
        'DECODER_CHANNELS': [64, 32, 16, 8],
        'DEEP_SUPERVISION': True,
        'WEIGHTED_EDGES': False,
        'PROPAGATE_COORDS': True,
        'VOXEL_DECODER': True,
        'GC': GraphConvNorm
    },
}

# Dataset specific parameters
hyper_ps_hippocampus = {
    'N_EPOCHS': 2000,
    'AUGMENT_TRAIN': True,
    'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/Task04_Hippocampus/",
    'PATCH_SIZE': [64, 64, 64],
    'BATCH_SIZE': 15,
    'N_M_CLASSES': 1,
    'N_REF_POINTS_PER_STRUCTURE': 1400,
    'N_TEMPLATE_VERTICES': 162,
    'MODEL_CONFIG': {
        'GRAPH_CHANNELS': [128, 64, 32, 16],
        'UNPOOL_INDICES': [0,1,1],
        'AGGREGATE_INDICES': [[0,1],
                              [1,2],
                              [3,4]], # 8 = last decoder skip
    },
    'PROJ_NAME': "hippocampus",
    'MESH_TARGET_TYPE': "mesh"
}
hyper_ps_hippocampus['MODEL_CONFIG']['MESH_TEMPLATE'] =\
    f"../supplementary_material/spheres/icosahedron_{hyper_ps_hippocampus['N_TEMPLATE_VERTICES']}.obj"

hyper_ps_cortex = {
    'FIXED_SPLIT': {
        'train': ['1010_3', '1007_3', '1003_3', '1104_3', '1015_3', '1001_3',
                  '1018_3', '1014_3', '1122_3', '1000_3', '1008_3', '1128_3',
                  '1017_3', '1113_3', '1011_3', '1125_3', '1005_3', '1107_3',
                  '1019_3', '1013_3', '1006_3', '1012_3'],
        'validation': ['1036_3', '1110_3'],
        'test': ['1004_3', '1119_3', '1116_3', '1009_3', '1101_3', '1002_3']
    },
    'NDIMS': 3,
    'N_EPOCHS': 2000,
    'AUGMENT_TRAIN': False,
    'RAW_DATA_DIR': "/mnt/nas/Data_Neuro/MALC_CSR/",
    'PREPROCESSED_DATA_DIR': "/home/fabianb/data/preprocessed/MALC_CSR/",
    'BATCH_SIZE': 3,
    'P_DROPOUT': 0.3,
    'MODEL_CONFIG': {
        'GROUP_STRUCTS': False, # False for single-surface reconstruction
        'GRAPH_CHANNELS': [256, 128, 64, 32, 16],
        'UNPOOL_INDICES': [0,0,0,0],
        'AGGREGATE_INDICES': [[3,4,5,6],
                              [2,3,6,7],
                              [1,2,7,8],
                              [0,1,7,8]], # 8 = last decoder skip
    },
    'PROJ_NAME': "cortex",
    'MESH_TARGET_TYPE': "mesh",
    'STRUCTURE_TYPE': 'cerebral_cortex',
    'REDUCE_REG_LOSS_MODE': 'none',
    'PROVIDE_CURVATURES': True,
    'PATCH_MODE': "single-patch"
}
# Automatically set parameters

###### White matter ######
if hyper_ps_cortex['STRUCTURE_TYPE'] == 'white_matter':
    hyper_ps_cortex['MESH_LOSS_FUNC_WEIGHTS'] = [1.0, 0.01, 0.1, 0.001, 5.0] # Tuned on hemisphere (exp_443/exp_451)
    if hyper_ps_cortex['NDIMS'] == 3:
        if hyper_ps_cortex['PATCH_MODE'] == "single-patch":
            ## Large
            hyper_ps_cortex['MESH_TYPE'] = 'freesurfer'
            hyper_ps_cortex['REDUCED_FREESURFER'] = 0.3
            hyper_ps_cortex['PATCH_ORIGIN'] = [0, 0, 0]
            hyper_ps_cortex['PATCH_SIZE'] = [64, 144, 128]
            hyper_ps_cortex['SELECT_PATCH_SIZE'] = [96, 208, 176]
            hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 40962
            hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 28000
            ## Small
            # hyper_ps_cortex['PATCH_ORIGIN'] = [30, 128, 60]
            # hyper_ps_cortex['PATCH_SIZE'] = [64, 64, 64]
            # hyper_ps_cortex['SELECT_PATCH_SIZE'] = [64, 64, 64]
            # hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 10242
            # hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 15000
            # hyper_ps_cortex['PATCH_ORIGIN'] = [40, 120, 60]
            # hyper_ps_cortex['PATCH_SIZE'] = [64, 80, 48]
            # hyper_ps_cortex['SELECT_PATCH_SIZE'] = [64, 96, 48]
            # hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 10242
            # hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 15000
            ## General
            hyper_ps_cortex['N_M_CLASSES'] = 1
            hyper_ps_cortex['MODEL_CONFIG']['MESH_TEMPLATE'] =\
                f"../supplementary_material/spheres/icosahedron_{hyper_ps_cortex['N_TEMPLATE_VERTICES']}.obj"
        elif hyper_ps_cortex['PATCH_MODE'] == "multi-patch":
            hyper_ps_cortex['PATCH_SIZE'] = [48, 48, 48]
            hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 10242
            hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 11000
            hyper_ps_cortex['N_M_CLASSES'] = 1
            hyper_ps_cortex['MODEL_CONFIG']['MESH_TEMPLATE'] =\
                f"../supplementary_material/spheres/icosahedron_{hyper_ps_cortex['N_TEMPLATE_VERTICES']}.obj"
        else: # no patch mode
            hyper_ps_cortex['N_M_CLASSES'] = 2
            hyper_ps_cortex['PATCH_SIZE'] = [128, 144, 128]
            hyper_ps_cortex['SELECT_PATCH_SIZE'] = [192, 208, 192]
            hyper_ps_cortex['MESH_TYPE'] = 'freesurfer'
            hyper_ps_cortex['REDUCED_FREESURFER'] = 0.3
            hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 40962
            hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 28000
            hyper_ps_cortex['MODEL_CONFIG']['MESH_TEMPLATE'] =\
                f"../supplementary_material/white_matter/cortex_white_matter_icosahedron_{hyper_ps_cortex['N_TEMPLATE_VERTICES']}.obj"
    else: # 2D
        hyper_ps_cortex['N_M_CLASSES'] = 1
        hyper_ps_cortex['PATCH_SIZE'] = [128, 128]
        hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 712
        hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 712
        hyper_ps_cortex['MODEL_CONFIG']['MESH_TEMPLATE'] =\
            f"../supplementary_material/circles/icocircle_{hyper_ps_cortex['N_TEMPLATE_VERTICES']}.obj"

####### Cerebral cortex ######
if hyper_ps_cortex['STRUCTURE_TYPE'] == 'cerebral_cortex':
    hyper_ps_cortex['MESH_LOSS_FUNC_WEIGHTS'] = [1.0, 0.0125, 0.375, 0.0015, 5.0] # Tuned on hemisphere (exp_496)
    if hyper_ps_cortex['NDIMS'] == 3:
        if hyper_ps_cortex['PATCH_MODE'] == "single-patch":
            hyper_ps_cortex['MESH_TYPE'] = 'freesurfer'
            hyper_ps_cortex['REDUCED_FREESURFER'] = 0.3
            hyper_ps_cortex['PATCH_ORIGIN'] = [0, 0, 0]
            hyper_ps_cortex['PATCH_SIZE'] = [64, 144, 128]
            hyper_ps_cortex['SELECT_PATCH_SIZE'] = [96, 208, 176]
            hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 40962
            hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 28000
            hyper_ps_cortex['N_M_CLASSES'] = 1
            hyper_ps_cortex['MODEL_CONFIG']['MESH_TEMPLATE'] =\
                f"../supplementary_material/spheres/icosahedron_{hyper_ps_cortex['N_TEMPLATE_VERTICES']}.obj"
        elif hyper_ps_cortex['PATCH_MODE'] == "multi-patch":
            hyper_ps_cortex['PATCH_SIZE'] = [48, 48, 48]
            hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 10242
            hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 11000
            hyper_ps_cortex['N_M_CLASSES'] = 1
            hyper_ps_cortex['MODEL_CONFIG']['MESH_TEMPLATE'] =\
                f"../supplementary_material/spheres/icosahedron_{hyper_ps_cortex['N_TEMPLATE_VERTICES']}.obj"
        else: # no patch mode
            hyper_ps_cortex['N_M_CLASSES'] = 2
            hyper_ps_cortex['PATCH_SIZE'] = [128, 144, 128]
            hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 40962
            hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 28000
            hyper_ps_cortex['MODEL_CONFIG']['MESH_TEMPLATE'] =\
                f"../supplementary_material/white_matter/cortex_white_matter_convex_both_{hyper_ps_cortex['N_TEMPLATE_VERTICES']}.obj"
    else: # 2D
        hyper_ps_cortex['N_M_CLASSES'] = 1
        hyper_ps_cortex['PATCH_SIZE'] = [128, 128]
        hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 712
        hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 712
        hyper_ps_cortex['MODEL_CONFIG']['MESH_TEMPLATE'] =\
            f"../supplementary_material/circles/icocircle_{hyper_ps_cortex['N_TEMPLATE_VERTICES']}.obj"

####### White matter & cerebral cortex ######
if ('cerebral_cortex' in hyper_ps_cortex['STRUCTURE_TYPE']
    and 'white_matter' in hyper_ps_cortex['STRUCTURE_TYPE']):
    if hyper_ps_cortex['PATCH_MODE'] == "no":
        # Order of structures: lh_white, rh_white, lh_pial, rh_pial; mesh loss
        # weights should respect this order!
        hyper_ps_cortex['MESH_LOSS_FUNC_WEIGHTS'] = [
            [1.0] * 4, # Chamfer
            [0.01] * 2 + [0.025] * 2, # Cosine,
            [0.1] * 2 + [0.25] * 2, # Laplace,
            [0.001] * 2 + [0.0015] * 2, # NormalConsistency
            [5.0] * 4 # Edge
        ]
        hyper_ps_cortex['EVAL_METRICS'] = [
            'Wasserstein',
            'SymmetricHausdorff',
            'JaccardVoxel',
            'JaccardMesh',
            'Chamfer',
            'CorticalThicknessError',
            'AverageDistance'
        ]
        # No patch mode
        hyper_ps_cortex['N_M_CLASSES'] = 4
        hyper_ps_cortex['PATCH_SIZE'] = [128, 144, 128]
        hyper_ps_cortex['SELECT_PATCH_SIZE'] = [192, 208, 192]
        hyper_ps_cortex['MESH_TYPE'] = 'freesurfer'
        hyper_ps_cortex['REDUCED_FREESURFER'] = 0.3
        hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 40962
        hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 44000 # max. number of gt points in this case (batch size 1)
        hyper_ps_cortex['MODEL_CONFIG']['MESH_TEMPLATE'] =\
            f"../supplementary_material/white_pial/cortex_4_ellipsoid_{hyper_ps_cortex['N_TEMPLATE_VERTICES']}_sps{hyper_ps_cortex['SELECT_PATCH_SIZE']}_ps{hyper_ps_cortex['PATCH_SIZE']}.obj"
    elif hyper_ps_cortex['PATCH_MODE'] == "single-patch":
        hyper_ps_cortex['MESH_LOSS_FUNC_WEIGHTS'] = [
            [1.0] * 2, # Chamfer
            [0.01] + [0.0125] , # Cosine,
            [0.1] + [0.25], # Laplace,
            [0.001] + [0.00225], # NormalConsistency
            [5.0] * 2 # Edge
        ]
        hyper_ps_cortex['EVAL_METRICS'] = [
            'Wasserstein',
            'SymmetricHausdorff',
            'JaccardVoxel',
            'JaccardMesh',
            'Chamfer',
            'CorticalThicknessError',
            'AverageDistance'
        ]
        hyper_ps_cortex['N_M_CLASSES'] = 2
        hyper_ps_cortex['PATCH_SIZE'] = [64, 144, 128]
        hyper_ps_cortex['PATCH_ORIGIN'] = [0, 0, 0]
        hyper_ps_cortex['SELECT_PATCH_SIZE'] = [96, 208, 176]
        hyper_ps_cortex['MESH_TYPE'] = 'freesurfer'
        hyper_ps_cortex['REDUCED_FREESURFER'] = 0.3
        hyper_ps_cortex['N_TEMPLATE_VERTICES'] = 40962
        hyper_ps_cortex['N_REF_POINTS_PER_STRUCTURE'] = 28000
        hyper_ps_cortex['MODEL_CONFIG']['MESH_TEMPLATE'] =\
            f"../supplementary_material/rh_white_pial/cortex_2_ellipsoid_{hyper_ps_cortex['N_TEMPLATE_VERTICES']}_sps{hyper_ps_cortex['SELECT_PATCH_SIZE']}_ps{hyper_ps_cortex['PATCH_SIZE']}_po{hyper_ps_cortex['PATCH_ORIGIN']}.obj"
    else:
        raise NotImplementedError()

# Overwrite params for overfitting (fewer epochs, no augmentation, smaller
# dataset)
hyper_ps_overfit = {
    # Learning
    'BATCH_SIZE': 1,
    'AUGMENT_TRAIN': False,
    'FIXED_SPLIT': {'train': [], 'validation': [], 'test': []},
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
                           type=int,
                           default=None,
                           nargs='?',
                           const=-1,
                           help="Test a model, optionally specified by epoch."
                           " If no epoch is specified, the best and the last"
                           " model are evaluated.")
    argparser.add_argument('--tune',
                           default=None,
                           type=str,
                           dest='params_to_tune',
                           nargs='+',
                           help="Specify the name of a parameter to tune.")
    argparser.add_argument('--fine-tune',
                           default=None,
                           type=str,
                           dest='params_to_fine_tune',
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
                           type=int,
                           nargs='?',
                           const=1,
                           default=False,
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
    hps['PARAMS_TO_FINE_TUNE'] = args.params_to_fine_tune
    hps['TEST_MODEL_EPOCH'] = args.test

    if args.params_to_tune and args.params_to_fine_tune:
        raise RuntimeError(
            "Cannot tune and fine-tune parameters at the same time."
        )

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

    if args.params_to_tune or args.params_to_fine_tune:
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

    # Add patch size to model config
    hps['MODEL_CONFIG']['PATCH_SIZE'] = hps['PATCH_SIZE']

    # Set project name automatically
    if hps['PROJ_NAME'] == 'cortex' and hps['NDIMS'] == 2:
        hps['PROJ_NAME'] = 'cortex_2D'

    # Run
    routine = mode_handler[mode]
    routine(hps, experiment_name=hps['EXPERIMENT_NAME'],
            loglevel=hps['LOGLEVEL'], resume=args.resume)


if __name__ == '__main__':
    main(hyper_ps)

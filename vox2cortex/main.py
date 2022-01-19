#!/usr/bin/env python3

""" Main file """

import os
from argparse import ArgumentParser, RawTextHelpFormatter

import torch
from torch.optim import AdamW

from data.supported_datasets import (
    dataset_paths,
)
from params.default import hyper_ps_default
from params.groups import hyper_ps_groups
from utils.modes import ExecModes
from utils.utils import update_dict
from utils.train import training_routine
from utils.tune_params import tuning_routine
from utils.test import test_routine
from utils.train_test import train_test_routine
from utils.ablation_study import (
    AVAILABLE_ABLATIONS,
    set_ablation_params_
)


# Overwrite params for overfitting (often useful for debugging and development)
hyper_ps_overfit = {
    # Learning
    'BATCH_SIZE': 1,
    'SANITY_CHECK_DATA': True
}


# Parameters that are overwritten frequently. For groups of parameters that are
# fixed together, see params.experiments
hyper_ps = {
    #######################
    'EXPERIMENT_NAME': None,  # Attention: "debug" overwrites previous dir"
                              # should be set with console argument
    #######################

    # Parameter group
    'GROUP_NAME': 'Bayesian Vox2Cortex no-patch',

    # Data
    'PROVIDE_CURVATURES': True,
    'SANITY_CHECK_DATA': False, # Save some memory

    # Learning
    'N_EPOCHS': 100,
    'BATCH_SIZE': 2,
    'EVAL_EVERY': 5,
    'CLIP_GRADIENT': 200000,
    'OPTIMIZER_CLASS': torch.optim.AdamW,
    'OPTIM_PARAMS': {
        'weight_decay': 1e-4
    },

    # Inference
    'UNCERTAINTY': 'mc',

    # Evaluation
    'TEST_SPLIT': 'validation',
}

mode_handler = {
    ExecModes.TRAIN.value: training_routine,
    ExecModes.TEST.value: test_routine,
    ExecModes.TRAIN_TEST.value: train_test_routine,
    ExecModes.TUNE.value: tuning_routine
}

def main(hyper_ps):
    """
    Main function for training, validation, test
    """
    argparser = ArgumentParser(description="Vox2Cortex",
                               formatter_class=RawTextHelpFormatter)
    argparser.add_argument('--architecture',
                           type=str,
                           default=hyper_ps_default['ARCHITECTURE'],
                           help="The name of the algorithm. Supported:\n"
                           "- vox2cortex")
    argparser.add_argument('--dataset',
                           type=str,
                           default=hyper_ps_default['DATASET'],
                           help="The name of the dataset.")
    argparser.add_argument('--train',
                           action='store_true',
                           help="Train a model.")
    argparser.add_argument('--test',
                           type=int,
                           default=hyper_ps_default['TEST_MODEL_EPOCH'],
                           nargs='?',
                           const=-1,
                           help="Test a model, optionally specified by epoch."
                           " If no epoch is specified, the best (w.r.t. IoU)"
                           " and the last model are evaluated.")
    argparser.add_argument('--tune',
                           default=hyper_ps_default['PARAMS_TO_TUNE'],
                           type=str,
                           dest='params_to_tune',
                           nargs='+',
                           help="Specify the name of a parameter to tune.")
    argparser.add_argument('--fine-tune',
                           default=hyper_ps_default['PARAMS_TO_FINE_TUNE'],
                           type=str,
                           dest='params_to_fine_tune',
                           nargs='+',
                           help="Specify the name of a parameter to tune.")
    argparser.add_argument('--resume',
                           action='store_true',
                           help="Resume an existing, potentially unfinished"
                           " experiment.")
    argparser.add_argument('--log',
                           type=str,
                           dest='loglevel',
                           default=hyper_ps_default['LOGLEVEL'],
                           help="Specify log level.")
    argparser.add_argument('--proj',
                           type=str,
                           dest='proj_name',
                           default=hyper_ps_default['PROJ_NAME'],
                           help="Specify the name of the wandb project.")
    argparser.add_argument('--group',
                           type=str,
                           dest='group_name',
                           default=hyper_ps_default['GROUP_NAME'],
                           help="Specify the name of the experiment group."
                           " Corresponding parameters are chosen from"
                           " params/groups.py")
    argparser.add_argument('--device',
                           type=str,
                           dest='device',
                           default=hyper_ps_default['DEVICE'],
                           help="Specify the device for execution.")
    argparser.add_argument('--overfit',
                           type=int,
                           nargs='?',
                           const=1, # Assume 1 sample without further spec.
                           default=hyper_ps_default['OVERFIT'],
                           help="Overfit on a few training samples.")
    argparser.add_argument('--time',
                           action='store_true',
                           help="Measure time of some functions.")
    argparser.add_argument('--n_test_vertices',
                           type=int,
                           default=hyper_ps_default['N_TEMPLATE_VERTICES_TEST'],
                           help="Set the number of template vertices during"
                           " testing.")
    argparser.add_argument('--ablation_study',
                           type=str,
                           nargs=1,
                           help="Perform an ablation study."
                           f"Available options are: {AVAILABLE_ABLATIONS}")
    argparser.add_argument('-n', '--exp_name',
                           dest='exp_name',
                           type=str,
                           default=hyper_ps_default['EXPERIMENT_NAME'],
                           help="Name of experiment:\n"
                           "- 'debug' means that the results are  written "
                           "into a directory \nthat might be overwritten "
                           "later. This may be useful for debugging \n"
                           "where the experiment result does not matter.\n"
                           "- Any other name cannot overwrite an existing"
                           " directory.\n"
                           "- If not specified, experiments are automatically"
                           " enumerated with exp_i and stored in"
                           " ../experiments.")
    args = argparser.parse_args()

    # Default params
    hps = hyper_ps_default.copy()

    # Overwrite with group-specific params
    group_name = args.group_name if (
        args.group_name != hyper_ps_default['GROUP_NAME']
    ) else hyper_ps.get('GROUP_NAME', args.group_name)
    hps = update_dict(hps, hyper_ps_groups[group_name])

    # Overwrite with 'often-to-change' or 'under-investigation' params
    hps = update_dict(hps, hyper_ps)

    # Set dataset paths
    hps = update_dict(hps, dataset_paths[args.dataset])

    # Set command line params if they are different from the defaults; if this
    # is true, they overwrite previously set parameters
    ovwr= lambda key, value: (
        value if hyper_ps_default[key] != value else hps[key]
    )
    hps['EXPERIMENT_NAME'] = ovwr('EXPERIMENT_NAME', args.exp_name)
    hps['ARCHITECTURE'] = ovwr('ARCHITECTURE', args.architecture)
    hps['DATASET'] = ovwr('DATASET', args.dataset)
    hps['LOGLEVEL'] = ovwr('LOGLEVEL', args.loglevel)
    hps['PROJ_NAME'] = ovwr('PROJ_NAME', args.proj_name)
    hps['GROUP_NAME'] = ovwr('GROUP_NAME', args.group_name)
    hps['DEVICE'] = ovwr('DEVICE', args.device)
    hps['OVERFIT'] = ovwr('OVERFIT', args.overfit)
    hps['TIME_LOGGING'] = ovwr('TIME_LOGGING', args.time)
    hps['PARAMS_TO_TUNE'] = ovwr('PARAMS_TO_TUNE', args.params_to_tune)
    hps['TEST_MODEL_EPOCH'] = ovwr('TEST_MODEL_EPOCH', args.test)
    hps['PARAMS_TO_FINE_TUNE'] = ovwr(
        'PARAMS_TO_FINE_TUNE', args.params_to_fine_tune
    )
    hps['N_TEMPLATE_VERTICES_TEST'] = ovwr(
        'N_TEMPLATE_VERTICES_TEST', args.n_test_vertices
    )

    if args.ablation_study:
        hps['ABLATION_STUDY'] = args.ablation_study[0]
    else:
        hps['ABLATION_STUDY'] = hyper_ps_default['ABLATION_STUDY']

    if args.params_to_tune and args.params_to_fine_tune:
        raise RuntimeError(
            "Cannot tune and fine-tune parameters at the same time."
        )

    # Training device
    torch.cuda.set_device(args.device)

    # Potentially set params for ablation study
    if args.ablation_study:
        set_ablation_params_(hps, args.ablation_study[0])

    # Set the number of test vertices
    if hps['N_TEMPLATE_VERTICES_TEST'] == -1:
        hps['N_TEMPLATE_VERTICES_TEST'] = hps['N_TEMPLATE_VERTICES']

    # Set execution mode
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

    # Add patch size to model config
    hps['MODEL_CONFIG']['PATCH_SIZE'] = hps['PATCH_SIZE']

    # Automatically choose template
    hps['MODEL_CONFIG']['MESH_TEMPLATE'] = os.path.join(
        hps['TEMPLATE_PATH'],
        hps['TEMPLATE_NAME'](
            hps['N_M_CLASSES'],
            hps['N_TEMPLATE_VERTICES'],
            hps['SELECT_PATCH_SIZE'],
            hps['PATCH_SIZE']
        )
    )

    # Potentially overfit
    if hps['OVERFIT']:
        hps = update_dict(hps, hyper_ps_overfit)

    # Create output dirs
    if not os.path.isdir(hps['MISC_DIR']):
        os.mkdir(hps['MISC_DIR'])
    if not os.path.isdir(hps['CHECK_DIR']):
        os.mkdir(hps['CHECK_DIR'])

    # Run
    routine = mode_handler[mode]
    routine(hps, experiment_name=hps['EXPERIMENT_NAME'],
            loglevel=hps['LOGLEVEL'], resume=args.resume)


if __name__ == '__main__':
    main(hyper_ps)

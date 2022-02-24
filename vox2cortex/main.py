#!/usr/bin/env python3

""" Main file """

import os
from argparse import ArgumentParser, RawTextHelpFormatter

import torch

from data.supported_datasets import (
    dataset_paths,
)
from params.default import hyper_ps_default
from params.groups import assemble_group_params, hyper_ps_groups
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
    # Sanity checks not possible on lrz
    'SANITY_CHECK_DATA': True
}

# Parameters that are overwritten frequently. For groups of parameters that are
# fixed together, see params.groups
hyper_ps_master = {
    # Learning
    'BATCH_SIZE': 2,
    # 'FIXED_SPLIT': [
        # "fold2_train.txt",
        # "fold2_val.txt",
        # "fold2_test.txt"
    # ]

    # LRZ
    # 'MESH_TEMPLATE_PATH': '/mnt/data/fsaverage70/v2c_template/',
    # 'RAW_DATA_DIR': '/mnt/data/OASIS_FS72/',
}

# Overwrite master params if the value is different from the
# default value
def ovwr(hyper_ps, key, value):
    if value != hyper_ps_default[key]:
        hyper_ps[key] = value

mode_handler = {
    ExecModes.TRAIN.value: training_routine,
    ExecModes.TEST.value: test_routine,
    ExecModes.TRAIN_TEST.value: train_test_routine,
    ExecModes.TUNE.value: tuning_routine
}


def single_experiment(hyper_ps, mode, resume):
    """ Run a single experiment.
    """
    # Assemble params from default and group-specific params
    hps = assemble_group_params(hyper_ps['GROUP_NAME'])

    # Set dataset paths
    hps = update_dict(
        hps,
        dataset_paths[hyper_ps.get('DATASET', hps['DATASET'])]
    )

    # Overwrite with master params
    hps = update_dict(hps, hyper_ps)

    # Training device
    torch.cuda.set_device(hps['DEVICE'])

    # Potentially set params for ablation study
    if hps['ABLATION_STUDY']:
        set_ablation_params_(hps, hps['ABLATION_STUDY'])

    # Add patch size to model config
    hps['MODEL_CONFIG']['PATCH_SIZE'] = hps['PATCH_SIZE']

    # Potentially overfit
    if hps['OVERFIT']:
        hps = update_dict(hps, hyper_ps_overfit)

    # Create output dirs
    if not os.path.isdir(hps['MISC_DIR']):
        os.mkdir(hps['MISC_DIR'])
    if not os.path.isdir(hps['CHECK_DIR']):
        os.mkdir(hps['CHECK_DIR'])

    # Run
    # Attention: the routine can change the experiment name
    routine = mode_handler[mode]
    return routine(
        hps,
        experiment_name=hps['EXPERIMENT_NAME'],
        loglevel=hps['LOGLEVEL'],
        resume=resume
    )

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
                           "- vox2cortex\n"
                           "- corticalflow\n"
                           "- v2cflow")
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
    argparser.add_argument('--no-wandb',
                           dest='use_wandb',
                           action='store_false',
                           help="Don't use wandb logging.")
    argparser.add_argument('--proj',
                           type=str,
                           dest='proj_name',
                           default=hyper_ps_default['PROJ_NAME'],
                           help="Specify the name of the wandb project.")
    argparser.add_argument('--group',
                           type=str,
                           dest='group_name',
                           nargs='+',
                           help="Specify the name(s) of the experiment"
                           " group(s)."
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
    argparser.add_argument('--test_on_large',
                           dest='reduced_template',
                           action='store_false',
                           default=hyper_ps_default['REDUCED_TEMPLATE'],
                           help="Test on the large fsaverage template.")
    argparser.add_argument('--ablation_study',
                           type=str,
                           nargs=1,
                           help="Perform an ablation study."
                           f"Available options are: {AVAILABLE_ABLATIONS}")
    argparser.add_argument('--exp_prefix',
                           type=str,
                           default=hyper_ps_default['EXP_PREFIX'],
                           help="A folder prefix for automatically enumerated"
                           " experiments.")
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

    ovwr(hyper_ps, 'EXPERIMENT_NAME', args.exp_name)
    ovwr(hyper_ps, 'ARCHITECTURE', args.architecture)
    ovwr(hyper_ps, 'DATASET', args.dataset)
    ovwr(hyper_ps, 'LOGLEVEL', args.loglevel)
    ovwr(hyper_ps, 'PROJ_NAME', args.proj_name)
    ovwr(hyper_ps, 'GROUP_NAME', args.group_name)
    ovwr(hyper_ps, 'DEVICE', args.device)
    ovwr(hyper_ps, 'OVERFIT', args.overfit)
    ovwr(hyper_ps, 'TIME_LOGGING', args.time)
    ovwr(hyper_ps, 'PARAMS_TO_TUNE', args.params_to_tune)
    ovwr(hyper_ps, 'TEST_MODEL_EPOCH', args.test)
    ovwr(hyper_ps, 'PARAMS_TO_FINE_TUNE', args.params_to_fine_tune)
    ovwr(hyper_ps, 'REDUCED_TEMPLATE', args.reduced_template)
    ovwr(hyper_ps, 'EXP_PREFIX', args.exp_prefix)
    ovwr(hyper_ps, 'USE_WANDB', args.use_wandb)

    if args.ablation_study:
        hyper_ps['ABLATION_STUDY'] = args.ablation_study[0]
    else:
        hyper_ps['ABLATION_STUDY'] = hyper_ps_default['ABLATION_STUDY']

    if args.params_to_tune and args.params_to_fine_tune:
        raise RuntimeError(
            "Cannot tune and fine-tune parameters at the same time."
        )

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

    # Define parameter group name(s)
    param_group_names = args.group_name if (
        args.group_name != hyper_ps_default['GROUP_NAME']
    ) else hyper_ps.get('GROUP_NAME', args.group_name)
    if isinstance(param_group_names, str):
        param_group_names = [param_group_names]
    if not all(n in hyper_ps_groups for n in param_group_names):
        raise RuntimeError("Not all parameter groups exist.")

    previous_exp_name = None

    # Iterate over parameter groups
    for param_group_name in param_group_names:
        # Potentially reference previous experiment
        hyper_ps['PREVIOUS_EXPERIMENT_NAME'] = previous_exp_name
        hyper_ps['GROUP_NAME'] = param_group_name

        previous_exp_name = single_experiment(hyper_ps, mode, args.resume)


if __name__ == '__main__':
    main(hyper_ps_master)

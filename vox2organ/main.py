#!/usr/bin/env python3

""" Main file """

import os
from argparse import ArgumentParser, RawTextHelpFormatter

import torch
import wandb
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
from utils.logging import (
    init_logging,
    init_wandb_logging)
from utils.ablation_study import (
    AVAILABLE_ABLATIONS,
    set_ablation_params_
)
from params.ablation import update_hps_sweep


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
    'BATCH_SIZE': 1,

    # LRZ
    # 'MESH_TEMPLATE_PATH': '/mnt/data/fsaverage70/v2c_template/',
    # 'RAW_DATA_DIR': '/mnt/data/ADNI_FS72/',
    'RAW_DATA_DIR': '/mnt/nas/Data_WholeBody/AbdomenCT-1K/Processed/',
    #'FIXED_SPLIT': None
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


def single_experiment(hyper_ps, ps, mode, resume):
    """ Run a single experiment.
    """
    # Assemble params from default and group-specific params
    hps = assemble_group_params(hyper_ps['GROUP_NAME'])
    init_wandb_logging(
        exp_name=hps['PROJ_NAME'],
        wandb_proj_name=hps['PROJ_NAME'],
        wandb_group_name=hps['GROUP_NAME'],
        wandb_job_type='train',
        params=ps
    )
    ps = wandb.config


    # Set dataset paths
    hps = update_dict(
        hps,
        dataset_paths[hyper_ps.get('DATASET', hps['DATASET'])]
    )

    # Overwrite with master params
    hps = update_dict(hps, hyper_ps_master)

    # Overwrite with abl config
    hps = update_dict(hps, hyper_ps)

    # Training device
    torch.cuda.set_device(hps['DEVICE'])

    hps = update_hps_sweep(hps, ps)

    # Add patch size to model config
    hps['MODEL_CONFIG']['PATCH_SIZE'] = hps['PATCH_SIZE']


    # Create output dirs
    if not os.path.isdir(hps['MISC_DIR']):
        os.mkdir(hps['MISC_DIR'])

    # Run
    # Attention: the routine can change the experiment name
    routine = mode_handler[mode]
    return routine(
        hps,
        experiment_name=hps['EXPERIMENT_NAME'],
        loglevel=hps['LOGLEVEL'],
        resume=resume
    )

def main(ps = None):
    """
    Main function for training, validation, test
    """
    hyper_ps = {}
    dataset = "KiTS"
    group_name = 'Vox2Cortex Abdomen Patient'
    exp_name = "ablation_study_patient_template_4"
    resume = False


    ovwr(hyper_ps, 'EXPERIMENT_NAME', exp_name)
    ovwr(hyper_ps, 'DATASET', dataset)
    ovwr(hyper_ps, 'GROUP_NAME', group_name)
    mode = ExecModes.TRAIN.value
    
    # Define parameter group name(s)
    param_group_names = group_name if (
        group_name != hyper_ps_default['GROUP_NAME']
    ) else hyper_ps.get('GROUP_NAME', group_name)
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

        previous_exp_name = single_experiment(hyper_ps, ps, mode, resume)


if __name__ == '__main__':
    main(hyper_ps_master)

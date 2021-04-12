#!/usr/bin/env python3

""" Main file """

from argparse import ArgumentParser, RawTextHelpFormatter

from utils.params import HyperPs
from utils.modes import Modes
from utils.utils import update_dict

from utils.train import training_routine
from utils.test import test_routine
from utils.train_test import train_test_routine

# Define parameters
hyper_ps = {
    #######################
    'EXPERIMENT_NAME': None,  # Attention: "debug" overwrites previous dir"
                              # should be set with console argument
    #######################
    # Learning
    HyperPs.OPTIM_PARAMS.name: {'lr': 0.0003},
    HyperPs.BATCH_SIZE.name: 64,
}

mode_handler = {
    Modes.TRAIN.value: training_routine,
    Modes.TEST.value: test_routine,
    Modes.TRAIN_TEST.value: train_test_routine
}


def main(hps):
    """
    Main function for training, validation, test
    """
    argparser = ArgumentParser(description="cortex-parcellation-using-meshes",
                               formatter_class=RawTextHelpFormatter)
    argparser.add_argument('algorithm',
                           type=str,
                           help="voxel2mesh")
    argparser.add_argument('dataset',
                           type=str,
                           help="The name of the dataset.")
    argparser.add_argument('--train',
                           action='store_true',
                           help="Train a model.")
    argparser.add_argument('--test',
                           action='store_true',
                           help="Test a model.")
    argparser.add_argument('-v', '--verbose',
                           dest = 'verbose',
                           action='store_true',
                           help="Debug output.")
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

    # Fill hyperparameters with defaults
    default_params = HyperPs.dict()
    hps = update_dict(default_params, hps)

    if args.train and not args.test:
        mode = Modes.TRAIN.value
    if args.test and not args.train:
        mode = Modes.TEST.value
    if args.train and args.test:
        mode = Modes.TRAIN_TEST.value
    if not args.test and not args.train:
        print("Please use either --train or --test or both.")
        return

    # Run
    routine = mode_handler[mode]
    routine(hps, experiment_name=hps['EXPERIMENT_NAME'], verbose=args.verbose)




if __name__ == '__main__':
    main(hyper_ps)

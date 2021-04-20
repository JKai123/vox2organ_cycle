""" Testing the preprocess operations """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from data.dataset import dataset_split_handler
from utils.visualization import show_slices

def run_preprocess_check():

    print("Loading data...")

    hps = {'RAW_DATA_DIR': '/mnt/nas/Data_Neuro/Task04_Hippocampus',
           'PREPROCESSED_DATA_DIR': None,
           'DATASET_SEED': 1234,
           'DATASET_SPLIT_PROPORTIONS': (80, 10, 10),
           'PATCH_SIZE': (64, 64, 64)
          }

    hps_lower = dict((k.lower(), v) for k, v in hps.items())

    # No augmentation
    training_set,\
            _,\
            _ = dataset_split_handler['Hippocampus'](augment_train=False,
                                                             **hps_lower)
    # Augmentation
    training_set_augment,\
            _,\
            _ = dataset_split_handler['Hippocampus'](augment_train=True,
                                                             **hps_lower)

    for iter_in_epoch, (data, data_augment) in tqdm(enumerate(zip(training_set[:5],
                                                  training_set_augment[:5])),
                                                  desc="Testing...",
                                                  position=0,
                                                  leave=True):
        img_slices = [data[0][32, :, :], data[0][:, 32, :], data[0][:, :, 32]]
        label_slices = [data[1][32, :, :], data[1][:, 32, :], data[1][:, :, 32]]
        img_slices_augment = [data_augment[0][32, :, :], data_augment[0][:, 32, :], data_augment[0][:, :, 32]]
        label_slices_augment = [data_augment[1][32, :, :], data_augment[1][:, 32, :], data_augment[1][:, :, 32]]
        show_slices(img_slices, label_slices, "../misc/img" +\
                    str(iter_in_epoch) + ".png")
        show_slices(img_slices, None, "../misc/img" +\
                    str(iter_in_epoch) + "_nolabel.png")
        show_slices(img_slices_augment, label_slices_augment, "../misc/img" +\
                    str(iter_in_epoch) + "_augment.png")
        show_slices(img_slices_augment, None, "../misc/img" +\
                    str(iter_in_epoch) + "_augment_nolabel.png")

if __name__ == '__main__':
    run_preprocess_check()

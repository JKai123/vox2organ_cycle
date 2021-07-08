""" Testing the preprocess operations """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from tqdm import tqdm
import numpy as np

from data.supported_datasets import dataset_split_handler
from utils.visualization import show_slices
from utils.utils import create_mesh_from_voxels
from utils.mesh import Mesh

def run_preprocess_check(dataset):
    """ Check preprocessing for cortex data """

    if dataset == 'Cortex':
        hps = {'RAW_DATA_DIR': '/mnt/nas/Data_Neuro/MALC_CSR/',
               'DATASET_SEED': 1532,
               'DATASET_SPLIT_PROPORTIONS': (100, 0, 0),
               # 'PATCH_SIZE': (192, 224, 192),
               # 'PATCH_SIZE': (64, 144, 128),
               'PATCH_SIZE': (48, 48, 48),
               'N_REF_POINTS_PER_STRUCTURE': 10000, # irrelevant for check
               'MESH_TARGET_TYPE': 'mesh',
               'STRUCTURE_TYPE': 'white_matter',
               # 'PATCH_ORIGIN': (0, 5, 0),
               # 'PATCH_ORIGIN': (30, 128, 60),
               # 'SELECT_PATCH_SIZE': (96, 208, 176),
               # 'SELECT_PATCH_SIZE': (64, 64, 64),
               'PATCH_MODE': "multi-patch",
               'OVERFIT': False,
               'MC_STEP_SIZE': 1
              }
    elif dataset == 'Hippocampus':
        hps = {'RAW_DATA_DIR': '/mnt/nas/Data_Neuro/Task04_Hippocampus',
               'PREPROCESSED_DATA_DIR': None,
               'DATASET_SEED': 1532,
               'DATASET_SPLIT_PROPORTIONS': (100, 0, 0),
               'PATCH_SIZE': (64, 64, 64),
               'N_REF_POINTS_PER_STRUCTURE': 1400, # irrelevant for check
               'OVERFIT': False,
               'MC_STEP_SIZE': 1
              }
    else:
        raise ValueError("Unknown dataset.")

    hps_lower = dict((k.lower(), v) for k, v in hps.items())

    # No augmentation
    print("Loading data...")
    training_set,\
            _,\
            _ = dataset_split_handler[dataset](augment_train=False,
                                                save_dir="../misc",
                                                **hps_lower)
    if dataset == 'Cortex':
        mel = training_set.mean_edge_length()
        print(f"Mean edge length in dataset: {mel:.7f}")

    training_set.check_data()

    # Augmentation
    print("Loading data...")
    training_set_augment,\
            _,\
            _ = dataset_split_handler[dataset](augment_train=True,
                                                save_dir="../misc",
                                                **hps_lower)
    training_set_augment.check_data()

    n_samples = np.min((6, len(training_set)))
    for iter_in_epoch in tqdm(range(n_samples), desc="Testing...", position=0, leave=True):
        # w/o augmentation
        img, label, mesh = training_set.get_item_and_mesh_from_index(iter_in_epoch)
        img, label = img.squeeze(), label.squeeze()
        shape = img.shape
        assert shape == label.shape, "Shapes should be identical."
        img_slices = [img[shape[0]//2, :, :],
                      img[:, shape[1]//2, :],
                      img[:, :, shape[2]//2]]
        label_slices = [label[shape[0]//2, :, :],
                      label[:, shape[1]//2, :],
                      label[:, :, shape[2]//2]]
        mesh.store("../misc/mesh" + str(iter_in_epoch) + ".ply")
        mc_mesh = create_mesh_from_voxels(label)
        mc_mesh.store("../misc/mesh" + str(iter_in_epoch) + "mc.ply")
        show_slices(img_slices, label_slices, "../misc/img" +\
                    str(iter_in_epoch) + ".png")
        show_slices(img_slices, None, "../misc/img" +\
                    str(iter_in_epoch) + "_nolabel.png")

        # /w augmentation
        img, label, mesh = training_set_augment.get_item_and_mesh_from_index(iter_in_epoch)
        img, label = img.squeeze(), label.squeeze()
        shape = img.shape
        assert shape == label.shape, "Shapes should be identical."
        img_slices = [img[shape[0]//2, :, :],
                      img[:, shape[1]//2, :],
                      img[:, :, shape[2]//2]]
        label_slices = [label[shape[0]//2, :, :],
                      label[:, shape[1]//2, :],
                      label[:, :, shape[2]//2]]
        mesh.store("../misc/mesh" + str(iter_in_epoch) + "_augment.ply")
        mc_mesh = create_mesh_from_voxels(label)
        mc_mesh.store("../misc/mesh" + str(iter_in_epoch) + "mc_augment.ply")
        show_slices(img_slices, label_slices, "../misc/img" +\
                    str(iter_in_epoch) + "_augment.png")
        show_slices(img_slices, None, "../misc/img" +\
                    str(iter_in_epoch) + "_augment_nolabel.png")

    print("Results written to ../misc/")

if __name__ == '__main__':
    run_preprocess_check('Cortex')

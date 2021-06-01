""" Testing the preprocess operations """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from tqdm import tqdm

from data.supported_datasets import dataset_split_handler
from utils.visualization import show_slices
from utils.utils import create_mesh_from_voxels
from utils.mesh import Mesh

def run_preprocess_check_hippocampus():
    """ Check preprocessing for Hippocampus data"""

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
                                                     save_dir="../misc",
                                                             **hps_lower)
    training_set.check_data()

    # Augmentation
    training_set_augment,\
            _,\
            _ = dataset_split_handler['Hippocampus'](augment_train=True,
                                                     save_dir="../misc",
                                                             **hps_lower)
    training_set_augment.check_data()

    for iter_in_epoch in tqdm(range(5), desc="Testing...", position=0, leave=True):
        data = training_set.get_item_and_mesh_from_index(iter_in_epoch)
        data_augment = training_set_augment.get_item_and_mesh_from_index(iter_in_epoch)
        img_slices = [data[0].squeeze()[32, :, :],
                      data[0].squeeze()[:, 32, :],
                      data[0].squeeze()[:, :, 32]]
        label_slices = [data[1].squeeze()[32, :, :],
                        data[1].squeeze()[:, 32, :],
                        data[1].squeeze()[:, :, 32]]
        mesh = data[2]
        mesh.store("../misc/mesh" + str(iter_in_epoch) + ".ply")
        img_slices_augment = [data_augment[0].squeeze()[32, :, :],
                              data_augment[0].squeeze()[:, 32, :],
                              data_augment[0].squeeze()[:, :, 32]]
        label_slices_augment = [data_augment[1].squeeze()[32, :, :],
                                data_augment[1].squeeze()[:, 32, :],
                                data_augment[1].squeeze()[:, :, 32]]
        mesh_augment = data_augment[2]
        mesh_augment.store("../misc/mesh" + str(iter_in_epoch) + "_augment.ply")
        show_slices(img_slices, label_slices, "../misc/img" +\
                    str(iter_in_epoch) + ".png")
        show_slices(img_slices, None, "../misc/img" +\
                    str(iter_in_epoch) + "_nolabel.png")
        show_slices(img_slices_augment, label_slices_augment, "../misc/img" +\
                    str(iter_in_epoch) + "_augment.png")
        show_slices(img_slices_augment, None, "../misc/img" +\
                    str(iter_in_epoch) + "_augment_nolabel.png")

def run_preprocess_check_cortex():
    """ Check preprocessing for cortex data """

    print("Loading data...")

    hps = {'RAW_DATA_DIR': '/mnt/nas/Data_Neuro/MALC_CSR/',
           'DATASET_SEED': 1234,
           'DATASET_SPLIT_PROPORTIONS': (100, 0, 0),
           'PATCH_SIZE': (192, 224, 192),
           'N_REF_POINTS_PER_STRUCTURE': 1400
          }

    hps_lower = dict((k.lower(), v) for k, v in hps.items())

    # No augmentation
    training_set,\
            _,\
            _ = dataset_split_handler['Cortex'](augment_train=False,
                                                     save_dir="../misc",
                                                             **hps_lower)
    training_set.check_data()
    # Augmentation (currently not implemented for cortex)
    # training_set_augment,\
            # _,\
            # _ = dataset_split_handler['Cortex'](augment_train=True,
                                                     # save_dir="../misc",
                                                             # **hps_lower)

    for iter_in_epoch in tqdm(range(5), desc="Testing...", position=0, leave=True):
        data = training_set.get_item_and_mesh_from_index(iter_in_epoch)
        # data_augment = training_set_augment.get_item_and_mesh_from_index(iter_in_epoch)
        img_slices = [data[0].squeeze()[32, :, :],
                      data[0].squeeze()[:, 32, :],
                      data[0].squeeze()[:, :, 32]]
        label_slices = [data[1].squeeze()[32, :, :],
                        data[1].squeeze()[:, 32, :],
                        data[1].squeeze()[:, :, 32]]
        mesh = data[2]
        mesh.store("../misc/mesh" + str(iter_in_epoch) + ".ply")
        mc_mesh = create_mesh_from_voxels(data[1])
        mc_mesh.store("../misc/mesh" + str(iter_in_epoch) + "mc.ply")
        # img_slices_augment = [data_augment[0][32, :, :], data_augment[0][:, 32, :], data_augment[0][:, :, 32]]
        # label_slices_augment = [data_augment[1][32, :, :], data_augment[1][:, 32, :], data_augment[1][:, :, 32]]
        # mesh_augment = data_augment[2]
        # mesh_augment.store("../misc/mesh" + str(iter_in_epoch) + "_augment.ply")
        show_slices(img_slices, label_slices, "../misc/img" +\
                    str(iter_in_epoch) + ".png")
        show_slices(img_slices, None, "../misc/img" +\
                    str(iter_in_epoch) + "_nolabel.png")
        # show_slices(img_slices_augment, label_slices_augment, "../misc/img" +\
                    # str(iter_in_epoch) + "_augment.png")
        # show_slices(img_slices_augment, None, "../misc/img" +\
                    # str(iter_in_epoch) + "_augment_nolabel.png")

if __name__ == '__main__':
    run_preprocess_check_cortex()

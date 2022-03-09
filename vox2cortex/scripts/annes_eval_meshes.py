#!/usr/bin/env python3

""" Evaluation script that can be applied directly to predicted meshes (no need
to load model etc.) """

import os
from argparse import ArgumentParser
from sklearn.metrics import jaccard_score, f1_score
import numpy as np
import torch
import trimesh
import nibabel as nib
import matplotlib.pyplot as plt
from trimesh.proximity import longest_ray
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import (
    sample_points_from_meshes,
    knn_points,
    knn_gather
)
import subprocess
from utils.file_handle import read_dataset_ids
from utils.cortical_thickness import _point_mesh_face_distance_unidirectional
from utils.mesh import Mesh
from utils.utils import choose_n_random_points
from data.supported_datasets import valid_ids

PARCEL_NAMES = ['not_defined_unknown', 'caudalanteriorcingulate', 'caudalmiddlefrontal', 'cuneus',
                'entorhinal', 'fusiform', 'inferiorparietal', 'inferiortemporal', 'isthmuscingulate',
                'lateraloccipital', 'lateralorbitofrontal', 'lingual', 'medialorbitofrontal', 'middletemporal',
                'parahippocampal', 'paracentral', 'parsopercularis', 'parsorbitalis', 'parstriangularis',
                'pericalcarine', 'postcentral', 'posteriorcingulate', 'precentral', 'precuneus',
                'rostralanteriorcingulate', 'rostralmiddlefrontal', 'superiorfrontal', 'superiorparietal',
                'superiortemporal', 'supramarginal', 'transversetemporal', 'insula']

EXTRA_18_LIST = ['Afterthought-1', 'Colin27-1', 'HLN-12-1', 'HLN-12-10', 'HLN-12-11', 'HLN-12-12', 'HLN-12-2',
                 'HLN-12-3',
                 'HLN-12-4', 'HLN-12-5', 'HLN-12-6', 'HLN-12-7', 'HLN-12-8', 'HLN-12-9', 'MMRR-3T7T-2-1',
                 'MMRR-3T7T-2-2',
                 'Twins-2-1', 'Twins-2-2']
# for vox2cortex
RAW_DATA_DIR = "/mnt/nas/Data_Neuro/"
# EXPERIMENT_DIR = "/mnt/nas/Users/Anne/Vox2Cortex_Experiments/Parcellation/"
EXPERIMENT_DIR = '../experiments/'
#FS_DIR = "/mnt/nas/Data_Neuro/OASIS/FS_full_72"
# mindboggle:
FS_DIR = '/mnt/nas/Data_Neuro/Mindboggle/FreeSurfer/Users/arno.klein/Data/Mindboggle101/subjects'
# EXPERIMENT_DIR = "/mnt/nas/Users/Anne/Vox2Cortex_Experiments/OASIS72/"


SURF_NAMES = ("lh_white", "rh_white", "lh_pial", "rh_pial")
# SURF_NAMES = ("rh_white", "rh_pial")
PARTNER = {"rh_white": "rh_pial",
           "rh_pial": "rh_white",
           "lh_white": "lh_pial",
           "lh_pial": "lh_white"}

MODES = ('ad_hd', 'thickness', 'trt')


def eval_trt(mri_id, surf_name, eval_params, epoch, device="cuda:1",
             subfolder="meshes"):
    pred_folder = os.path.join(eval_params['log_path'])
    if "TRT" not in pred_folder:
        raise ValueError("Test-Retest evaluation is meant for TRT dataset.")

    # Skip every second scan (was already compared to its predecessor)
    subject_id, scan_id = mri_id.split("/")
    scan_id_int = int(scan_id.split("_")[1])
    if scan_id_int % 2 == 0:
        return None
    scan_id_next = f"T1_{str(scan_id_int + 1).zfill(2)}"
    mri_id_next = "/".join([subject_id, scan_id_next])

    # Load predicted meshes
    s_index = SURF_NAMES.index(surf_name)
    pred_mesh_path = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.ply"
    )
    pred_mesh = trimesh.load(pred_mesh_path)
    pred_mesh.remove_duplicate_faces();
    pred_mesh.remove_unreferenced_vertices();
    #
    pred_mesh_partner_path = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id_next}_epoch{epoch}_struc{s_index}_meshpred.ply"
    )
    pred_mesh_partner = trimesh.load(pred_mesh_partner_path)
    pred_mesh_partner.remove_duplicate_faces();
    pred_mesh_partner.remove_unreferenced_vertices();

    # Register to each other with ICP as done by DeepCSR
    trans_affine, cost = pred_mesh.register(pred_mesh_partner)
    print("Average distance before registration for mesh"
          f" {mri_id}, {surf_name}: {cost}")
    v_pred_mesh_new = trimesh.transform_points(pred_mesh.vertices,
                                               trans_affine)
    pred_mesh.vertices = v_pred_mesh_new
    # _, cost = pred_mesh.register(pred_mesh_partner)
    # print("Average distance after registration for mesh"
    # f" {mri_id}, {surf_name}: {cost}")

    # Compute ad, hd, percentage > 1mm, percentage > 2mm with pytorch3d
    pred_mesh = Meshes(
        [torch.from_numpy(pred_mesh.vertices).float().to(device)],
        [torch.from_numpy(pred_mesh.faces).long().to(device)]
    )
    pred_mesh_partner = Meshes(
        [torch.from_numpy(pred_mesh_partner.vertices).float().to(device)],
        [torch.from_numpy(pred_mesh_partner.faces).long().to(device)]
    )

    # compute with pytorch3d:
    pred_pcl = sample_points_from_meshes(pred_mesh, 100000)
    pred_pcl_partner = sample_points_from_meshes(pred_mesh_partner, 100000)

    # compute point to mesh distances and metrics; not exactly the same as
    # trimesh, it's always a bit larger than the trimesh distances, but a
    # lot faster.
    print(f"Computing point to mesh distances for files {pred_mesh_path} and"
          f" {pred_mesh_partner_path}...")
    P2G_dist = _point_mesh_face_distance_unidirectional(
        Pointclouds(pred_pcl_partner), pred_mesh
    ).cpu().numpy()
    G2P_dist = _point_mesh_face_distance_unidirectional(
        Pointclouds(pred_pcl), pred_mesh_partner
    ).cpu().numpy()

    assd2 = (P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])
    hd2 = max(np.percentile(P2G_dist, 90),
              np.percentile(G2P_dist, 90))
    greater_1 = ((P2G_dist > 1).sum() + (G2P_dist > 1).sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])
    greater_2 = ((P2G_dist > 2).sum() + (G2P_dist > 2).sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])

    print("\t > Average symmetric surface distance {:.4f}".format(assd2))
    print("\t > Hausdorff surface distance {:.4f}".format(hd2))
    print("\t > Greater 1 {:.4f}%".format(greater_1 * 100))
    print("\t > Greater 2 {:.4f}%".format(greater_2 * 100))

    return assd2, hd2, greater_1, greater_2


def trt_output(results, summary_file):
    ad = results[:, 0]
    hd = results[:, 1]
    greater_1 = results[:, 2]
    greater_2 = results[:, 3]

    cols_str = ';'.join(
        ['MEAN_OF_AD', 'STD_OF_AD', 'MEAN_OF_HD', 'STD_OF_HD',
         'MEAN_OF_>1', 'STD_OF_>1', 'MEAN_OF_>2', 'STD_OF_>2']
    )
    mets_str = ';'.join(
        [str(np.mean(ad)), str(np.std(ad)),
         str(np.mean(hd)), str(np.std(hd)),
         str(np.mean(greater_1)), str(np.std(greater_1)),
         str(np.mean(greater_2)), str(np.std(greater_2))]
    )

    with open(summary_file, 'w') as output_csv_file:
        output_csv_file.write(cols_str + '\n')
        output_csv_file.write(mets_str + '\n')


def eval_thickness(mri_id, surf_name, eval_params, epoch, device="cuda:0",
                   method="nearest", subfolder="meshes"):
    """ Cortical thickness biomarker.
    :param method: 'nearest' or 'ray'.
    """
    print("Evaluate thickness using " + method + " correspondences.")

    pred_folder = os.path.join(eval_params['log_path'])
    thickness_folder = os.path.join(
        eval_params['log_path'], 'thickness'
    )
    if not os.path.isdir(thickness_folder):
        os.mkdir(thickness_folder)

    # load ground-truth meshes
    try:
        gt_mesh_path = os.path.join(
            eval_params['gt_mesh_path'], mri_id, '{}.stl'.format(surf_name)
        )
        gt_mesh = trimesh.load(gt_mesh_path)
    except ValueError:
        gt_mesh_path = os.path.join(
            eval_params['gt_mesh_path'], mri_id, '{}.ply'.format(surf_name)
        )
        gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.remove_duplicate_faces()
    gt_mesh.remove_unreferenced_vertices()
    gt_pntcloud = gt_mesh.vertices
    gt_normals = gt_mesh.vertex_normals
    if "pial" in surf_name:  # point to inside
        gt_normals = - gt_normals
    try:
        gt_mesh_path_partner = os.path.join(
            eval_params['gt_mesh_path'],
            mri_id,
            '{}.stl'.format(PARTNER[surf_name])
        )
        gt_mesh_partner = trimesh.load(gt_mesh_path_partner)
    except ValueError:
        gt_mesh_path_partner = os.path.join(
            eval_params['gt_mesh_path'],
            mri_id,
            '{}.ply'.format(PARTNER[surf_name])
        )
        gt_mesh_partner = trimesh.load(gt_mesh_path_partner)
    gt_mesh_partner.remove_duplicate_faces()
    gt_mesh_partner.remove_unreferenced_vertices()

    # Load predicted meshes
    s_index = SURF_NAMES.index(surf_name)
    pred_mesh_path = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.ply"
    )
    pred_mesh = trimesh.load(pred_mesh_path)
    pred_mesh.remove_duplicate_faces()
    pred_mesh.remove_unreferenced_vertices()
    pred_pntcloud = pred_mesh.vertices
    pred_normals = pred_mesh.vertex_normals
    if "pial" in surf_name:  # point to inside
        pred_normals = - pred_normals
    s_index_partner = SURF_NAMES.index(PARTNER[surf_name])
    pred_mesh_path_partner = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id}_epoch{epoch}_struc{s_index_partner}_meshpred.ply"
    )
    pred_mesh_partner = trimesh.load(pred_mesh_path_partner)
    pred_mesh_partner.remove_duplicate_faces()
    pred_mesh_partner.remove_unreferenced_vertices()

    if method == "ray":
        # Choose subset of predicted vertices and closest vertices from gt
        pred_pntcloud_sub, pred_idx = choose_n_random_points(
            pred_pntcloud, 10000, return_idx=True
        )
        pred_normals_sub = pred_normals[pred_idx]
        _, gt_idx, gt_pntcloud_sub = knn_points(
            torch.from_numpy(pred_pntcloud_sub)[None].float().to(device),
            torch.from_numpy(gt_pntcloud)[None].float().to(device),
            K=1,
            return_nn=True
        )
        gt_idx = gt_idx.squeeze().cpu().numpy()
        gt_pntcloud_sub = gt_pntcloud_sub.squeeze().cpu().numpy()
        gt_normals_sub = gt_normals[gt_idx]

        # Compute thickness measure using
        # trimesh.proximity.longest_ray
        print(f"Computing ray distances for file {pred_mesh_path}...")
        gt_thickness = longest_ray(gt_mesh_partner, gt_pntcloud_sub, gt_normals_sub)
        pred_thickness = longest_ray(pred_mesh_partner, pred_pntcloud_sub, pred_normals_sub)

        # Set inf values to nan
        gt_thickness[~np.isfinite(gt_thickness)] = np.nan
        pred_thickness[~np.isfinite(pred_thickness)] = np.nan

        error = np.abs(pred_thickness - gt_thickness)

    elif method == "nearest":
        # Use all vertices
        pred_idx = np.array(range(pred_pntcloud.shape[0]))
        gt_idx = np.array(range(gt_pntcloud.shape[0]))

        # Move to gpu
        gt_pntcloud = Pointclouds(
            [torch.from_numpy(gt_pntcloud).float().to(device)]
        )
        gt_mesh_partner = Meshes(
            [torch.from_numpy(gt_mesh_partner.vertices).float().to(device)],
            [torch.from_numpy(gt_mesh_partner.faces).long().to(device)],
        )
        pred_pntcloud = Pointclouds(
            [torch.from_numpy(pred_pntcloud).float().to(device)]
        )
        pred_mesh_partner = Meshes(
            [torch.from_numpy(pred_mesh_partner.vertices).float().to(device)],
            [torch.from_numpy(pred_mesh_partner.faces).long().to(device)],
        )

        # Compute thickness measure using nearest face distance
        print(f"Computing nearest distances for file {pred_mesh_path}...")
        gt_thickness = _point_mesh_face_distance_unidirectional(
            gt_pntcloud, gt_mesh_partner
        ).squeeze().cpu().numpy()
        pred_thickness = _point_mesh_face_distance_unidirectional(
            pred_pntcloud, pred_mesh_partner
        ).squeeze().cpu().numpy()

        # Compute error w.r.t. to nearest gt vertex
        _, nearest_idx, _ = knn_points(
            pred_pntcloud.points_padded(),
            gt_pntcloud.points_padded(),
            K=1,
            return_nn=True
        )
        nearest_idx = nearest_idx.squeeze().cpu().numpy()
        error = np.abs(pred_thickness - gt_thickness[nearest_idx])

        pred_pntcloud = pred_pntcloud.points_packed().cpu().numpy()
        gt_pntcloud = gt_pntcloud.points_packed().cpu().numpy()

    else:
        raise ValueError("Unknown method {}.".format(method))

    error_mean = np.nanmean(error)
    error_median = np.nanmedian(error)

    print("\t > Thickness error mean {:.4f}".format(error_mean))
    print("\t > Thickness error median {:.4f}".format(error_median))

    # Store
    th_gt_file = os.path.join(
        thickness_folder, f"{mri_id}_struc{s_index}_gt.thickness"
    )
    np.save(th_gt_file, gt_thickness)
    th_pred_file = os.path.join(
        thickness_folder, f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.thickness"
    )
    np.save(th_pred_file, pred_thickness)
    err_file = os.path.join(
        thickness_folder, f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.thicknesserror"
    )
    error_features = np.zeros(pred_pntcloud.shape[0])
    error_features[pred_idx] = error
    np.nan_to_num(error_features, copy=False, nan=0.0)
    np.save(err_file, error_features)

    return error_mean, error_median


def thickness_output(results, summary_file):
    means_all = results[:, 0]
    medians_all = results[:, 1]
    mean_of_means = np.mean(means_all)
    std_of_means = np.std(means_all)
    mean_of_medians = np.mean(medians_all)
    std_of_medians = np.std(medians_all)

    cols_str = ';'.join(
        ['MEAN_OF_MEANS', 'STD_OF_MEANS', 'MEAN_OF_MEDIANS', 'STD_OF_MEDIANS']
    )
    mets_str = ';'.join(
        [str(mean_of_means), str(std_of_means),
         str(mean_of_medians), str(std_of_medians)]
    )

    with open(summary_file, 'w') as output_csv_file:
        output_csv_file.write(cols_str + '\n')
        output_csv_file.write(mets_str + '\n')


def remap_parcellation_labels(labels, remap_config):
    """
    Function to remap the label values into the desired range of algorithm
    :param remap_config FS_DKT refers to freesurfer DKT-40 atlas, for which we remove labels 0,1, 32, 33 ending
    up with a total of 32 labels
    """

    if remap_config == 'FS_DKT':
        label_list = [2, 3, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
                      22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35]  # 32 total labels
    else:
        raise ValueError("Invalid argument value for remap config, only valid options are FS_DKT")

    new_labels = np.zeros_like(labels)
    for i, label in enumerate(label_list):

        label_present = np.zeros_like(labels)
        if isinstance(label, list):
            label_present[labels == label[0]] = 1
            label_present[labels == label[1]] = 1
        else:
            label_present[labels == label] = 1
        new_labels = new_labels + (i + 1) * label_present
    # set labels that now have label value 1 to 0 (meaning unknown label)
    # new_labels[new_labels==1] = 0
    return new_labels


def read_vtk(filename: str):
    points = []
    polys = []
    point_data = []
    with open(filename, 'r') as f:
        line = f.readline()
        while line:
            if 'POINTS' in line:
                n_points = int(line.split()[1])
                for _ in range(n_points):
                    points.append(np.array(
                        list(map(float, list(f.readline().split())))
                    ))
                line = f.readline()
            elif 'POLYGONS' in line:
                n_polys = int(line.split()[1])
                for _ in range(n_polys):
                    polys.append(np.array(
                        list(map(int, f.readline().split(" ")))[1:]
                    ))
                line = f.readline()
            elif 'POINT_DATA' in line:
                n_pd = int(line.split()[1])
                while not line.replace("\n", "").replace(".", "", 1).isdigit():
                    line = f.readline()
                # First line already read
                point_data.append(np.array(int(line.split()[0])))
                for _ in range(n_pd - 1):
                    point_data.append(np.array(int(f.readline().split()[0])))
                line = f.readline()
            else:
                line = f.readline()

        return np.stack(points), np.stack(polys), np.stack(point_data)


def eval_parcellation_dice(mri_id, surf_name, eval_params, epoch, device='cuda:0', subfolder='meshes'):
    """ Evaluate parcellations in terms of Jaccard and Dice.
        """
    print(f"Evaluate parcellation of pat {mri_id} {surf_name}")

    pred_folder = os.path.join(eval_params['log_path'])
    parc_folder = os.path.join(
        eval_params['log_path'], 'parc'
    )
    hemi = surf_name.split("_")[0]

    if not os.path.isdir(parc_folder):
        os.mkdir(parc_folder)

    # Load ground-truth mesh
    try:
        gt_mesh_path = os.path.join(
            eval_params['gt_mesh_path'], mri_id, '{}.stl'.format(surf_name)
        )
        gt_mesh = trimesh.load(gt_mesh_path, process=False)
    except ValueError:
        gt_mesh_path = os.path.join(
            eval_params['gt_mesh_path'], mri_id, '{}.ply'.format(surf_name)
        )
        gt_mesh = trimesh.load(gt_mesh_path, process=False)

    # Load gt parcellation from original FS output folder
    # for Mindboggle it is a bit different:
    if 'Mindboggle' in eval_params['dataset']:
        if os.path.isfile(os.path.join(FS_DIR, mri_id, 'label', hemi + '.labels.DKT31.manual.annot')):
            aparc_annot = nib.freesurfer.io.read_annot(
                os.path.join(FS_DIR, mri_id, 'label', hemi + '.labels.DKT31.manual.annot'))
            gt_parc = aparc_annot[0]
            # colors = aparc_annot[1]
            # label_names = aparc_annot[2]
        else:  # if annotation file doesn't exist in freesurfer folder, load .vtk file
            if mri_id in EXTRA_18_LIST:
                dataset_name = 'Extra-18'
            else:
                dataset_name = '-'.join(mri_id.split('-')[0:-1])
            surface_path = os.path.join(RAW_DATA_DIR,'Mindboggle', dataset_name + '_surfaces', mri_id,
                                        hemi + '.labels.DKT31.manual.vtk')
            _, _, gt_parc = read_vtk(surface_path)
    else:
        gt_parc = nib.freesurfer.io.read_annot(
            os.path.join(
                eval_params['fs_path'],
                mri_id,
                'label',
                '{}.aparc.DKTatlas40.annot'.format(hemi)
            )
        )[0]

    include_labels = np.arange(0, 32)

    #num_parcel_classes = len(include_labels)
    #gt_parc = torch.from_numpy(gt_parc.astype(np.int32))[None].to(device)

    # Load predicted mesh
    s_index = SURF_NAMES.index(surf_name)
    pred_mesh_path = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.ply"
    )
    pred_mesh = trimesh.load(pred_mesh_path, process=False)

    # load predicted parcellation:
    parcel_path = os.path.join(
        '/mnt/nas/Data_Neuro/Parcellation_atlas/fsaverage/v2c_template/',
        surf_name + ".aparc.DKTatlas40.annot"
    )

    # Load predicted vertex classes from template
    pred_parc = nib.freesurfer.io.read_annot(parcel_path)[0]
    #print(f'debug dice, pred parc min ', pred_parc.min())

    # remap labels
    gt_parc = remap_parcellation_labels(gt_parc, 'FS_DKT')
    pred_parc = remap_parcellation_labels(pred_parc, 'FS_DKT')


    pred_parc = torch.from_numpy(pred_parc.astype(np.int32))[None].to(device)
    gt_parc = torch.from_numpy(gt_parc.astype(np.int32))[None].to(device)

    pred_pntcloud = torch.from_numpy(
        pred_mesh.vertices
    )[None].float().to(device)
    gt_pntcloud = torch.from_numpy(
        gt_mesh.vertices
    )[None].float().to(device)

    # Jaccard on predicted mesh
    x_nn = knn_points(pred_pntcloud, gt_pntcloud, K=1)
    surf_parc = pred_parc.squeeze()
    neighbor_parc = knn_gather(
        gt_parc.unsqueeze(-1), x_nn.idx
    ).long().squeeze()
    jaccard_p = jaccard_score(
        neighbor_parc.cpu(),
        surf_parc.cpu(),
        labels=include_labels,
        average=None
    )
    dice_p = f1_score(
        neighbor_parc.cpu(),
        surf_parc.cpu(),
        labels=include_labels,
        average=None
    )

    # Jaccard on gt mesh
    x_nn = knn_points(gt_pntcloud, pred_pntcloud, K=1)
    surf_parc = gt_parc.squeeze()
    neighbor_parc = knn_gather(
        pred_parc.unsqueeze(-1), x_nn.idx
    ).long().squeeze()
    jaccard_g = jaccard_score(
        neighbor_parc.cpu(),
        surf_parc.cpu(),
        labels=include_labels,
        average=None
    )
    dice_g = f1_score(
        neighbor_parc.cpu(),
        surf_parc.cpu(),
        labels=include_labels,
        average=None
    )

    # Average of both directions
    jaccard = (jaccard_p + jaccard_g) / 2
    dice = (dice_p + dice_g) / 2

    print(f"Mean Jaccard: {jaccard[1:].mean()}")
    print(f"Mean Dice: {dice[1:].mean()}")
    #print(f'debug Dice scores ', dice[1:])

    jacc_file = os.path.join(
        parc_folder, f"{mri_id}_struc{s_index}_jacc.npy"
    )
    np.save(jacc_file, jaccard[1:])
    dice_file = os.path.join(
        parc_folder, f"{mri_id}_struc{s_index}_dice.npy"
    )
    np.save(dice_file, dice[1:])

    return jaccard[1:], dice[1:]



def eval_ad_hd_pytorch3d(mri_id, surf_name, eval_params, epoch,
                         device="cuda:0", subfolder="meshes"):
    """ AD and HD computed with pytorch3d. """
    pred_folder = os.path.join(eval_params['log_path'])
    ad_hd_folder = os.path.join(
        eval_params['log_path'], 'ad_hd'
    )
    if not os.path.isdir(ad_hd_folder):
        os.mkdir(ad_hd_folder)

    # load ground-truth mesh
    try:
        gt_mesh_path = os.path.join(eval_params['gt_mesh_path'], mri_id, '{}.stl'.format(surf_name))
        gt_mesh = trimesh.load(gt_mesh_path)
    except ValueError:
        gt_mesh_path = os.path.join(eval_params['gt_mesh_path'], mri_id, '{}.ply'.format(surf_name))
        gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.remove_duplicate_faces();
    gt_mesh.remove_unreferenced_vertices();
    gt_mesh = Meshes(
        [torch.from_numpy(gt_mesh.vertices).float().to(device)],
        [torch.from_numpy(gt_mesh.faces).long().to(device)]
    )

    # load predicted mesh
    # file endings depending on the post-processing:
    # orig, pp, top_fix
    s_index = SURF_NAMES.index(surf_name)
    pred_mesh_path = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.ply"
    )
    pred_mesh = trimesh.load(pred_mesh_path)
    pred_mesh.remove_duplicate_faces();
    pred_mesh.remove_unreferenced_vertices();
    pred_mesh = Meshes(
        [torch.from_numpy(pred_mesh.vertices).float().to(device)],
        [torch.from_numpy(pred_mesh.faces).long().to(device)]
    )

    # compute with pytorch3d:
    gt_pcl = sample_points_from_meshes(gt_mesh, 100000)
    pred_pcl = sample_points_from_meshes(pred_mesh, 100000)

    # compute point to mesh distances and metrics; not exactly the same as
    # trimesh, it's always a bit larger than the trimesh distances, but a
    # lot faster.
    print(f"Computing point to mesh distances for file {pred_mesh_path}...")
    P2G_dist = _point_mesh_face_distance_unidirectional(
        Pointclouds(gt_pcl), pred_mesh
    ).cpu().numpy()
    G2P_dist = _point_mesh_face_distance_unidirectional(
        Pointclouds(pred_pcl), gt_mesh
    ).cpu().numpy()

    assd2 = (P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.shape[0] + G2P_dist.shape[0])
    hd2 = max(np.percentile(P2G_dist, 90),
              np.percentile(G2P_dist, 90))

    print("\t > Average symmetric surface distance {:.4f}".format(assd2))
    print("\t > Hausdorff surface distance {:.4f}".format(hd2))

    ad_pred_file = os.path.join(
        ad_hd_folder, f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.ad.npy"
    )
    np.save(ad_pred_file, assd2)
    hd_pred_file = os.path.join(
        ad_hd_folder, f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.hd.npy"
    )
    np.save(hd_pred_file, hd2)

    return assd2, hd2


def eval_ad_hd_trimesh(mri_id, surf_name, eval_params, epoch, subfolder="meshes"):
    print('>>' * 5 + " Evaluating mri {} and surface {}".format(mri_id, surf_name))
    pred_folder = os.path.join(eval_params['log_path'])

    # load ground truth
    gt_pcl, gt_pcl_path, gt_mesh_path = None, None, None

    # load ground-truth mesh
    try:
        gt_mesh_path = os.path.join(eval_params['gt_mesh_path'], mri_id, '{}.stl'.format(surf_name))
        gt_mesh = trimesh.load(gt_mesh_path)
    except ValueError:
        gt_mesh_path = os.path.join(eval_params['gt_mesh_path'], mri_id, '{}.ply'.format(surf_name))
        gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.remove_duplicate_faces();
    gt_mesh.remove_unreferenced_vertices();
    print("GT mesh loaded from {} with {} vertices and {} faces".format(
        gt_mesh_path, gt_mesh.vertices.shape, gt_mesh.faces.shape))
    # sample point cloud for ground-truth mesh
    gt_pcl = gt_mesh.sample(100000)
    print("Point cloud with {} dimensions sampled from ground-truth mesh".format(gt_pcl.shape))

    # load predicted mesh
    # file endings depending on the post-processing:
    # orig, pp, top_fix
    s_index = SURF_NAMES.index(surf_name)
    pred_mesh_path = os.path.join(
        pred_folder,
        subfolder,
        f"{mri_id}_epoch{epoch}_struc{s_index}_meshpred.ply"
    )
    pred_mesh = trimesh.load(pred_mesh_path)
    pred_mesh.remove_duplicate_faces();
    pred_mesh.remove_unreferenced_vertices();
    print("Predicted mesh loaded from {} with {} vertices and {} faces".format(
        pred_mesh_path, pred_mesh.vertices.shape, pred_mesh.faces.shape))
    # sampling point cloud in predicted mesh
    pred_pcl = pred_mesh.sample(100000)
    print("Point cloud with {} dimensions sampled from predicted mesh".format(pred_pcl.shape))

    # compute point to mesh distances and metrics
    print(f"Computing point to mesh distances for file {pred_mesh_path}...")
    _, P2G_dist, _ = trimesh.proximity.closest_point(pred_mesh, gt_pcl)
    _, G2P_dist, _ = trimesh.proximity.closest_point(gt_mesh, pred_pcl)
    print("point to mesh distances computed")
    # Average symmetric surface distance
    print("computing metrics...")
    assd = ((P2G_dist.sum() + G2P_dist.sum()) / float(P2G_dist.size + G2P_dist.size))
    print("\t > Average symmetric surface distance {:.4f}".format(assd))
    # Hausdorff distance
    hd = max(np.percentile(P2G_dist, 90), np.percentile(G2P_dist, 90))
    print("\t > Hausdorff surface distance {:.4f}".format(hd))

    # log and metrics write csv
    cols_str = ';'.join(['MRI_ID', 'SURF_NAME', 'ASSD', 'HD'])
    mets_str = ';'.join([mri_id, surf_name, str(assd), str(hd)])
    print('REPORT_COLS;{}'.format(cols_str))
    print('REPORT_VALS;{}'.format(mets_str))
    met_csv_file_path = os.path.join(eval_params['log_path'],
                                     "{}_{}_{}.csv".format(eval_params['metrics_csv_prefix'], mri_id, surf_name))
    with open(met_csv_file_path, 'w') as output_csv_file:
        output_csv_file.write(mets_str + '\n')
    print('>>' * 5 + " Evaluation for {} and {}".format(mri_id, surf_name))

    return assd, hd


def dice_output(results, summary_file):
    jacc_all = results[:, 0,:]
    dice_all = results[:, 1,:]
    #dice_all = results

    dice_mean = np.mean(dice_all)
    dice_std = np.std(dice_all)
    jacc_mean = np.mean(jacc_all)
    jacc_std = np.std(jacc_all)

    dice_cols = ['DICE_ALL_MEAN', 'DICE_ALL_STD']
    for parc_str in PARCEL_NAMES[1:]:
        dice_cols.append(parc_str + '_MEAN')
        dice_cols.append(parc_str + '_STD')
    dice_cols_str = ';'.join(dice_cols)

    dice_mets = [str(dice_mean), str(dice_std)]
    dice_parc_mean = np.mean(dice_all, axis=0)
    dice_parc_std = np.std(dice_all, axis=0)
    for m, s in zip(dice_parc_mean, dice_parc_std):
        dice_mets.append(str(m))
        dice_mets.append(str(s))

    dice_mets_str = ';'.join(dice_mets)

    jacc_cols = ['JACCARD_ALL_MEAN', 'JACCARD_ALL_STD']
    for parc_str in PARCEL_NAMES[1:]:
        jacc_cols.append(parc_str + '_MEAN')
        jacc_cols.append(parc_str + '_STD')
    jacc_cols_str = ';'.join(jacc_cols)

    jacc_mets = [str(jacc_mean), str(jacc_std)]
    jacc_parc_mean = np.mean(jacc_all, axis=0)
    jacc_parc_std = np.std(jacc_all, axis=0)
    for m, s in zip(jacc_parc_mean, jacc_parc_std):
        jacc_mets.append(str(m))
        jacc_mets.append(str(s))

    jacc_mets_str = ';'.join(jacc_mets)

    with open(summary_file, 'w') as output_csv_file:
        output_csv_file.write(dice_cols_str + '\n')
        output_csv_file.write(dice_mets_str + '\n')
        output_csv_file.write('\n')
        output_csv_file.write(jacc_cols_str + '\n')
        output_csv_file.write(jacc_mets_str + '\n')


def ad_hd_output(results, summary_file):
    assd_all = results[:, 0]
    hd_all = results[:, 1]
    assd_mean = np.mean(assd_all)
    assd_std = np.std(assd_all)
    hd_mean = np.mean(hd_all)
    hd_std = np.std(hd_all)

    cols_str = ';'.join(['AD_MEAN', 'AD_STD', 'HD_MEAN', 'HD_STD'])
    mets_str = ';'.join([str(assd_mean), str(assd_std), str(hd_mean), str(hd_std)])

    with open(summary_file, 'w') as output_csv_file:
        output_csv_file.write(cols_str + '\n')
        output_csv_file.write(mets_str + '\n')


mode_to_function = {"ad_hd": eval_ad_hd_pytorch3d,
                    "thickness": eval_thickness,
                    "trt": eval_trt,
                    "parc": eval_parcellation_dice}
mode_to_output_file = {"ad_hd": ad_hd_output,
                       "thickness": thickness_output,
                       "trt": trt_output,
                       "parc": dice_output}

if __name__ == '__main__':
    argparser = ArgumentParser(description="Mesh evaluation procedure")
    argparser.add_argument('exp_name',
                           type=str,
                           help="Name of experiment under evaluation.")
    argparser.add_argument('epoch',
                           type=int,
                           help="The epoch to evaluate.")
    argparser.add_argument('n_test_vertices',
                           help="The number of template vertices for each"
                                " structure that was used during testing.")
    argparser.add_argument('dataset',
                           type=str,
                           help="The dataset.")
    argparser.add_argument('mode',
                           type=str,
                           help="The evaluation to perform, possible values"
                                " are " + str(MODES))
    argparser.add_argument('--meshfixed',
                           action='store_true',
                           help="Use MeshFix'ed meshes for evaluation.")

    args = argparser.parse_args()
    exp_name = args.exp_name
    epoch = args.epoch
    mode = args.mode
    dataset = args.dataset
    meshfixed = args.meshfixed

    # Provide params
    eval_params = {}
    if "OASIS" in dataset or "Mindboggle" in dataset:
        subdir = "CSR_data"
    else:
        subdir = ""
    eval_params['gt_mesh_path'] = os.path.join(
        RAW_DATA_DIR,
        dataset.replace("_small", "").replace("_large", "").replace("_orig", ""),
        subdir
    )
    eval_params['exp_path'] = os.path.join(EXPERIMENT_DIR, exp_name)
    if 'Mindboggle' in dataset:
        dataset = 'Mindboggle'
    eval_params['log_path'] = os.path.join(
        EXPERIMENT_DIR, exp_name,
        "test_template_"
        + str(args.n_test_vertices)
        + f"_{dataset}"
    )
    eval_params['metrics_csv_prefix'] = "eval_" + mode
    eval_params['fs_path'] = FS_DIR
    eval_params['dataset'] = dataset

    if meshfixed:
        eval_params['metrics_csv_prefix'] += "_meshfixed"
        subfolder = "meshfix"
    else:
        subfolder = "meshes"

    # Read dataset split
    dataset_file = os.path.join(eval_params['log_path'], 'dataset_ids.txt')

    # Use all valid ids in the case of test-retest (everything is 'test') and
    # test split otherwise
    if mode == 'trt':
        ids = valid_ids(eval_params['gt_mesh_path'])
    else:
        ids = read_dataset_ids(dataset_file)

    res_all = []
    res_surfaces = {s: [] for s in SURF_NAMES}
    for mri_id in ids:
        for surf_name in SURF_NAMES:
            result = mode_to_function[mode](
                mri_id, surf_name, eval_params, epoch, subfolder=subfolder
            )
            if result is not None:
                res_all.append(result)
                res_surfaces[surf_name].append(result)

    # Averaged over surfaces
    summary_file = os.path.join(
        eval_params['log_path'],
        f"{eval_params['metrics_csv_prefix']}_summary.csv"
    )
    mode_to_output_file[mode](np.array(res_all), summary_file)

    # Per-surface results
    for surf_name in SURF_NAMES:
        summary_file = os.path.join(
            eval_params['log_path'],
            f"{surf_name}_{eval_params['metrics_csv_prefix']}_summary.csv"
        )
        mode_to_output_file[mode](np.array(res_surfaces[surf_name]), summary_file)

    print("Done.")

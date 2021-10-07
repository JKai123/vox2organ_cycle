#!/usr/bin/env python3

""" Evaluation script that can be applied directly to predicted meshes (no need
to load model etc.) """

import os
from argparse import ArgumentParser

import numpy as np
import torch
import trimesh
import matplotlib.pyplot as plt
from trimesh.proximity import longest_ray, closest_point
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import (
    sample_points_from_meshes,
    knn_gather,
    knn_points
)

from utils.file_handle import read_dataset_ids
from utils.cortical_thickness import _point_mesh_face_distance_unidirectional
from utils.mesh import Mesh
from utils.utils import choose_n_random_points

RAW_DATA_DIR = "/mnt/nas/Data_Neuro/MALC_CSR/"
EXPERIMENT_DIR = "/home/fabianb/work/cortex-parcellation-using-meshes/experiments/"
# SURF_NAMES = ("lh_white", "rh_white", "lh_pial", "rh_pial")
SURF_NAMES = ("rh_white", "rh_pial")
PARTNER = {"rh_white": "rh_pial",
           "rh_pial": "rh_white",
           "lh_white": "lh_pial",
           "lh_pial": "lh_white"}

MODES = ('ad_hd', 'thickness')

def eval_thickness_ray(mri_id, surf_name, eval_params, epoch, device="cuda:1",
                       method="ray", subfolder="meshes"):
    """ Cortical thickness biomarker.
    :param method: 'nearest' or 'ray'.
    """
    pred_folder = os.path.join(eval_params['log_path'])
    thickness_folder = os.path.join(
        eval_params['log_path'], 'thickness_evaluation_' + subfolder
    )
    if not os.path.isdir(thickness_folder):
        os.mkdir(thickness_folder)

    # load ground-truth meshes
    gt_mesh_path = os.path.join(
        eval_params['gt_mesh_path'], mri_id, '{}.stl'.format(surf_name)
    )
    gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.remove_duplicate_faces()
    gt_mesh.remove_unreferenced_vertices()
    gt_pntcloud = gt_mesh.vertices
    gt_normals = gt_mesh.vertex_normals
    if "pial" in surf_name: # point to inside
        gt_normals = - gt_normals
    gt_mesh_path_partner = os.path.join(
        eval_params['gt_mesh_path'],
        mri_id,
        '{}.stl'.format(PARTNER[surf_name])
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
    if "pial" in surf_name: # point to inside
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
        print("Computing ray distances...")
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
        print("Computing nearest distances...")
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

    # Store mesh with error as color
    mesh_path = os.path.join(
        thickness_folder, f"{mri_id}_{surf_name}_thicknesserror.ply"
    )
    error_features = np.zeros(pred_pntcloud.shape[0])
    error_features[pred_idx] = error
    np.nan_to_num(error_features, copy=False, nan=0.0)
    Mesh(
        pred_mesh.vertices, pred_mesh.faces, features=error_features
    ).store_with_features(
        mesh_path, vmin=0.0, vmax=4.0
    )
    mesh_path = os.path.join(
        thickness_folder, f"{mri_id}_{surf_name}_thickness_pred.ply"
    )
    error_features = np.zeros(pred_pntcloud.shape[0])
    error_features[pred_idx] = pred_thickness
    np.nan_to_num(error_features, copy=False, nan=0.0)
    Mesh(
        pred_mesh.vertices, pred_mesh.faces, features=error_features
    ).store_with_features(
        mesh_path, vmin=0.0, vmax=4.0
    )
    mesh_path = os.path.join(
        thickness_folder, f"{mri_id}_{surf_name}_thickness_gt.ply"
    )
    error_features = np.zeros(gt_pntcloud.shape[0])
    error_features[gt_idx] = gt_thickness
    np.nan_to_num(error_features, copy=False, nan=0.0)
    Mesh(
        gt_mesh.vertices, gt_mesh.faces, features=error_features
    ).store_with_features(
        mesh_path, vmin=0.0, vmax=4.0
    )

    hist_file = os.path.join(
        thickness_folder, f"{mri_id}_{surf_name}_errorhisto.png"
    )
    plt.hist(error[np.isfinite(error)], bins=100)
    plt.savefig(hist_file)
    plt.clf()

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
        output_csv_file.write(cols_str+'\n')
        output_csv_file.write(mets_str+'\n')

def eval_ad_hd_pytorch3d(mri_id, surf_name, eval_params, epoch,
                         device="cuda:1", subfolder="meshes"):
    """ AD and HD computed with pytorch3d. """
    pred_folder = os.path.join(eval_params['log_path'])

    # load ground-truth mesh
    gt_mesh_path = os.path.join(eval_params['gt_mesh_path'], mri_id, '{}.stl'.format(surf_name))
    gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.remove_duplicate_faces(); gt_mesh.remove_unreferenced_vertices();
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
    pred_mesh.remove_duplicate_faces(); pred_mesh.remove_unreferenced_vertices();
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
    print("computing point to mesh distances...")
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

    return assd2, hd2

def eval_ad_hd_trimesh(mri_id, surf_name, eval_params, epoch, subfolder="meshes"):

    print('>>' * 5 + " Evaluating mri {} and surface {}".format(mri_id, surf_name))
    pred_folder = os.path.join(eval_params['log_path'])

    # load ground truth
    gt_pcl, gt_pcl_path, gt_mesh_path = None, None, None

    # load ground-truth mesh
    gt_mesh_path = os.path.join(eval_params['gt_mesh_path'], mri_id, '{}.stl'.format(surf_name))
    gt_mesh = trimesh.load(gt_mesh_path)
    gt_mesh.remove_duplicate_faces(); gt_mesh.remove_unreferenced_vertices();
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
    pred_mesh.remove_duplicate_faces(); pred_mesh.remove_unreferenced_vertices();
    print("Predicted mesh loaded from {} with {} vertices and {} faces".format(
        pred_mesh_path, pred_mesh.vertices.shape, pred_mesh.faces.shape))
    # sampling point cloud in predicted mesh
    pred_pcl = pred_mesh.sample(100000)
    print("Point cloud with {} dimensions sampled from predicted mesh".format(pred_pcl.shape))

    # compute point to mesh distances and metrics
    print("computing point to mesh distances...")
    _, P2G_dist, _ = trimesh.proximity.closest_point(pred_mesh, gt_pcl)
    _, G2P_dist, _ = trimesh.proximity.closest_point(gt_mesh, pred_pcl)
    print("point to mesh distances computed")
    #Average symmetric surface distance
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
    met_csv_file_path = os.path.join(eval_params['log_path'], "{}_{}_{}.csv".format(eval_params['metrics_csv_prefix'], mri_id, surf_name))
    with open(met_csv_file_path, 'w') as output_csv_file:
        output_csv_file.write(mets_str+'\n')
    print('>>' * 5 + " Evaluation for {} and {}".format(mri_id, surf_name))

    return assd, hd

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
        output_csv_file.write(cols_str+'\n')
        output_csv_file.write(mets_str+'\n')

mode_to_function = {"ad_hd": eval_ad_hd_pytorch3d,
                    "thickness": eval_thickness_ray}
mode_to_output_file = {"ad_hd": ad_hd_output,
                       "thickness": thickness_output}
if __name__ == '__main__':
    argparser = ArgumentParser(description="Mesh evaluation procedure")
    argparser.add_argument('exp_name',
                           type=str,
                           help="Name of experiment under evaluation.")
    argparser.add_argument('epoch',
                           type=int,
                           help="The epoch to evaluate.")
    argparser.add_argument('n_test_vertices',
                           type=int,
                           help="The number of template vertices for each"
                           " structure that was used during testing.")
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
    meshfixed = args.meshfixed

    # Provide params
    eval_params = {}
    eval_params['gt_mesh_path'] = RAW_DATA_DIR
    eval_params['exp_path'] = os.path.join(EXPERIMENT_DIR, exp_name)
    eval_params['log_path'] = os.path.join(
        EXPERIMENT_DIR, exp_name, "test_template_" + str(args.n_test_vertices)
    )
    eval_params['metrics_csv_prefix'] = "eval_" + mode

    if meshfixed:
        eval_params['metrics_csv_prefix'] += "_meshfixed"
        subfolder = "meshfix"
    else:
        subfolder = "meshes"

    dataset_file = os.path.join(eval_params['exp_path'], 'dataset_ids.txt')
    ids = read_dataset_ids(dataset_file)

    res_all = []
    for mri_id in ids:
        for surf_name in SURF_NAMES:
            result = mode_to_function[mode](
                mri_id, surf_name, eval_params, epoch, subfolder=subfolder
            )
            res_all.append(result)

    summary_file = os.path.join(
        eval_params['log_path'],
        f"{eval_params['metrics_csv_prefix']}_summary.csv"
    )

    # Write output
    mode_to_output_file[mode](np.array(res_all), summary_file)

    print("Done.")

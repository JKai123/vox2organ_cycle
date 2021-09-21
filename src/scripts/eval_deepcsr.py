#!/usr/bin/env python3

""" Evaluation script from DeepCSR. """

import os
from argparse import ArgumentParser

import numpy as np
import torch
import trimesh
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.ops import sample_points_from_meshes

from utils.file_handle import read_dataset_ids
from utils.cortical_thickness import _point_mesh_face_distance_unidirectional

RAW_DATA_DIR = "/mnt/nas/Data_Neuro/MALC_CSR/"
EXPERIMENT_DIR = "/home/fabianb/work/cortex-parcellation-using-meshes/experiments/"
SURF_NAMES = ("lh_white", "rh_white", "lh_pial", "rh_pial")

def eval_ad_hd_pytorch3d(mri_id, surf_name, eval_params, epoch, device="cuda:1"):
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
        "meshes/",
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

def eval_ad_hd_trimesh(mri_id, surf_name, eval_params, epoch):

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
        "meshes/",
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

if __name__ == '__main__':
    argparser = ArgumentParser(description="DeepCSR evaluation procedure")
    argparser.add_argument('-n', '--exp_name',
                           dest='exp_name',
                           type=str,
                           default=None,
                           help="Name of experiment under evaluation.")
    argparser.add_argument('--epoch',
                           type=int,
                           default=3000,
                           help="The epoch to evaluate.")
    args = argparser.parse_args()
    exp_name = args.exp_name
    epoch = args.epoch

    # Provide params
    eval_params = {}
    eval_params['gt_mesh_path'] = RAW_DATA_DIR
    eval_params['exp_path'] = os.path.join(EXPERIMENT_DIR, exp_name)
    eval_params['log_path'] = os.path.join(EXPERIMENT_DIR, exp_name, "test")
    eval_params['metrics_csv_prefix'] = "eval_deepcsr"

    dataset_file = os.path.join(eval_params['exp_path'], 'dataset_ids.txt')
    ids = read_dataset_ids(dataset_file)

    assd_all = []
    hd_all = []
    for mri_id in ids:
        for surf_name in SURF_NAMES:
            assd, hd = eval_ad_hd_pytorch3d(
                mri_id, surf_name, eval_params, epoch
            )
            assd_all.append(assd)
            hd_all.append(hd)

    assd_mean = np.mean(assd_all)
    assd_std = np.std(assd_all)
    hd_mean = np.mean(hd_all)
    hd_std = np.std(hd_all)

    summary_file = os.path.join(eval_params['log_path'], 'eval_deepcsr_summary.csv')
    cols_str = ';'.join(['AD_MEAN', 'AD_STD', 'HD_MEAN', 'HD_STD'])
    mets_str = ';'.join([str(assd_mean), str(assd_std), str(hd_mean), str(hd_std)])

    with open(summary_file, 'w') as output_csv_file:
        output_csv_file.write(cols_str+'\n')
        output_csv_file.write(mets_str+'\n')


    print("Done.")

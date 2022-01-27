
""" Single inference on a registered nifty brain scan. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
import json
from argparse import ArgumentParser, RawTextHelpFormatter

import torch
import nibabel
import numpy as np

from utils.mesh import Mesh
from models.model_handler import ModelHandler
from data.dataset import img_with_patch_size
from utils.utils import dict_to_lower_dict
from utils.utils_voxel2meshplusplus.graph_conv import GraphConvNorm
from utils.coordinate_transform import (
    transform_mesh_affine,
    normalize_vertices_per_max_dim,
)


surf_names = ("lh_white", "rh_white", "lh_pial", "rh_pial")
default_dir = "./inference-output/"


def main():
    argparser = ArgumentParser(
        description="Model inference CLI",
        formatter_class=RawTextHelpFormatter
    )
    argparser.add_argument(
        'MODEL',
        help="The stored model."
    )
    argparser.add_argument(
        'PARAMS',
        help="The stored parameter configuration."
    )
    argparser.add_argument(
        'IMAGE',
        help="A registerd nifty image."
    )
    argparser.add_argument(
        'TEMPLATE',
        help="The template to apply."
    )
    argparser.add_argument(
        '--output-dir',
        dest='output_dir',
        default=default_dir,
        help="An optional output directory. If not specified, the output is"
        f" written to {default_dir}"
    )
    argparser.add_argument(
        '--device',
        default="cuda:0",
        help="The device on which to perform inference."
    )

    args = argparser.parse_args()
    torch.cuda.set_device(args.device)

    # Load params
    print("Loading params...")
    with open(args.PARAMS, 'r') as f:
        hps = json.load(f)
    assert str(hps['PATCH_SIZE']) in args.TEMPLATE
    assert str(hps['SELECT_PATCH_SIZE']) in args.TEMPLATE
    hps['MODEL_CONFIG']['MESH_TEMPLATE'] = args.TEMPLATE
    hps['MODEL_CONFIG']['GC'] = GraphConvNorm

    # Load model
    print("Loading_model...")
    model_config = dict_to_lower_dict(hps['MODEL_CONFIG'])
    model = ModelHandler[hps['ARCHITECTURE']].value(
        ndims=hps['NDIMS'],
        n_v_classes=hps['N_V_CLASSES'],
        n_m_classes=hps['N_M_CLASSES'],
        patch_shape=hps['PATCH_SIZE'],
        **model_config
    ).float()
    model.load_state_dict(torch.load(args.MODEL, map_location='cpu'))
    model.cuda()
    model.eval()

    # Load image
    print("Loading image...")
    lower_limit = np.array((0, 0, 0) , dtype=int)
    upper_limit = np.array(hps['SELECT_PATCH_SIZE'], dtype=int)
    img = nibabel.load(args.IMAGE).get_fdata()
    img, trans_affine_1 = img_with_patch_size(
        img,
        hps['SELECT_PATCH_SIZE'],
        is_label=False,
        mode='crop',
        crop_at=(lower_limit + upper_limit) // 2
    )
    if hps['PATCH_SIZE'] != hps['SELECT_PATCH_SIZE']:
        img, trans_affine_2 = img_with_patch_size(
            img, hps['PATCH_SIZE'], is_label=False, mode='interpolate'
        )
    else:
        trans_affine_2 = np.eye(hps['NDIMS'] + 1) # Identity
    img_affine = trans_affine_2 @ trans_affine_1

    # Affine transformation
    _, norm_affine = normalize_vertices_per_max_dim(
        torch.zeros(hps['NDIMS']).view(-1, hps['NDIMS']),
        hps['PATCH_SIZE'],
        return_affine=True
    )
    trans_affine = norm_affine @ img_affine

    # Prediction
    print("Prediction...")
    with torch.no_grad():
        pred = model(img.cuda()[None, None])

    # Output in image coordinates
    print(f"Writing output to {args.output_dir}...")
    vertices, faces = model.__class__.pred_to_verts_and_faces(pred)
    vertices = vertices[-1].squeeze().cpu().numpy()
    faces = faces[-1].squeeze().cpu().numpy()
    vertices, faces = transform_mesh_affine(
        vertices, faces, np.linalg.inv(trans_affine)
    )
    try:
        os.mkdir(args.output_dir)
    except:
        answer = input(f"{args.output_dir} not empty, overwrite? (y/n)")
        while answer not in ("y", "n"):
            answer = input(f"{args.output_dir} not empty, overwrite? (y/n)")
        if answer != "y":
            print("Aborting.")
            return
    for i, (v, f) in enumerate(zip(vertices, faces)): # Each surface separately
        pred_mesh_filename = os.path.join(
            args.output_dir,
            surf_names[i] + ".ply"
        )
        Mesh(v, f).store(pred_mesh_filename)

    print("Finished.")


if __name__ == '__main__':
    main()


""" Convenience script to visualize freesurfer thickness """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from argparse import ArgumentParser

import numpy as np
import trimesh
from nibabel.freesurfer.io import read_morph_data
from matplotlib import cm

from utils.utils import normalize_min_max

argparser = ArgumentParser(description="Visualize FreeSurfer thickness.")
argparser.add_argument("FILE_NAME",
                       type=str,
                       help="The file id, e.g., 1000_3.")
argparser.add_argument("OUT_NAME",
                       type=str,
                       help="The output file name.")
args = argparser.parse_args()
in_file_name = args.FILE_NAME
out_file_name = args.OUT_NAME

mesh_fn = f"/mnt/nas/Data_Neuro/MALC_CSR/{in_file_name}/lh_white.stl"
thickness_fn = f"/mnt/nas/Data_Neuro/MALC_CSR/FS/FS/{in_file_name}/surf/lh.thickness"

# Problem: Merged vertices do not correspond to order of thickness
mesh = trimesh.load(mesh_fn, process=False)
thickness = read_morph_data(thickness_fn)

viridis = cm.get_cmap('viridis')
colors = viridis(normalize_min_max(thickness))
mesh.visual.vertex_colors = colors
mesh.export(out_file_name)

print("Average freesurfer thickness: ", str(np.mean(thickness)))

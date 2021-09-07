
""" Convenience script to visualize freesurfer thickness """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from argparse import ArgumentParser

import numpy as np
from trimesh.base import Trimesh
from nibabel.freesurfer.io import read_morph_data, read_geometry
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

mesh_fn = f"/mnt/nas/Data_Neuro/MALC_CSR/FS/FS/{in_file_name}/surf/lh.white"
thickness_fn = f"/mnt/nas/Data_Neuro/MALC_CSR/FS/FS/{in_file_name}/surf/lh.thickness"

# Use nibabel.freesurfer.io.read_geometry to load mesh since lh_pial.stl etc.
# contain duplicate vertices
vertices, faces = read_geometry(mesh_fn)
mesh = Trimesh(vertices, faces)
thickness = read_morph_data(thickness_fn)

viridis = cm.get_cmap('viridis')
colors = viridis(normalize_min_max(thickness))
mesh.visual.vertex_colors = colors
mesh.export(out_file_name)

print("Average freesurfer thickness: ", str(np.mean(thickness)))

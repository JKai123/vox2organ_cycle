
""" Convenience script for the computation of cortical thickness biomarkers. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from argparse import ArgumentParser

import torch
import trimesh
from matplotlib import cm
from matplotlib.colors import Normalize

from utils.cortical_thickness import _point_mesh_face_distance_unidirectional

if __name__ == '__main__':
    argparser = ArgumentParser(description="Measure cortical thickness.")
    argparser.add_argument('WHITE_SURFACE',
                           type=str,
                           help="The file name of a white surface mesh.")
    argparser.add_argument('PIAL_SURFACE',
                           type=str,
                           help="The file name of a pial surface mesh.")
    argparser.add_argument('OUTPUT',
                           type=str,
                           default="thickness_mesh.ply",
                           help="Name of output file.")

    args = argparser.parse_args()
    outpath = args.OUTPUT
    white_mesh_fn = args.WHITE_SURFACE
    pial_mesh_fn = args.PIAL_SURFACE

    white_mesh = trimesh.load(white_mesh_fn)
    pial_mesh = trimesh.load(pial_mesh_fn)

    # To pytorch3d
    white_vertices = torch.from_numpy(white_mesh.vertices).float().cuda()
    pial_vertices = torch.from_numpy(pial_mesh.vertices).float().cuda()
    pial_faces = torch.from_numpy(pial_mesh.faces).long().cuda()

    white_pntcloud = Pointclouds([white_vertices])
    pial_mesh = Meshes([pial_vertices], [pial_faces])

    point_to_face = _point_mesh_face_distance_unidirectional(
        white_pntcloud, pial_mesh
    )

    # Generate output mesh (color corresponds to vertex thickness)
    out_mesh = white_mesh
    autumn = cm.get_cmap('autumn')
    color_norm = Normalize(vmin=0.1, vmax=3.0, clip=True)
    colors = autumn(color_norm(point_to_face.cpu().numpy()))
    out_mesh.visual.vertex_colors = colors
    out_mesh.export(outpath)

    avg_thickness = point_to_face.mean()

    print("Average squared thickness value: ", str(avg_thickness))
    print("Output mesh as been written to ", outpath)

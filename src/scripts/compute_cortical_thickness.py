
""" Convenience script for the computation of cortical thickness biomarkers. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from argparse import ArgumentParser

import torch
import trimesh
from pytorch3d.structures import Meshes, Pointclouds
from pytorch3d.loss.point_mesh_distance import _PointFaceDistance
from matplotlib import cm

from utils.utils import normalize_min_max

point_face_distance = _PointFaceDistance.apply

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

    # The following is taken from pytorch3d.loss.point_to_mesh_distance
    # Packed representation for white matter pointclouds
    points = white_pntcloud.points_packed()  # (P, 3)
    points_first_idx = white_pntcloud.cloud_to_packed_first_idx()
    max_points = white_pntcloud.num_points_per_cloud().max().item()

    # Packed representation for faces
    verts_packed = pial_mesh.verts_packed()
    faces_packed = pial_mesh.faces_packed()
    tris = verts_packed[faces_packed]  # (T, 3, 3)
    tris_first_idx = pial_mesh.mesh_to_faces_packed_first_idx()
    max_tris = pial_mesh.num_faces_per_mesh().max().item()

    # Point to face distance: shape # (P,)
    point_to_face = point_face_distance(
        points, points_first_idx, tris, tris_first_idx, max_points
    )
    # Take root as point_face_distance returns squared distances
    point_to_face = torch.sqrt(point_to_face)

    # Generate output mesh (color corresponds to vertex thickness)
    out_mesh = white_mesh
    viridis = cm.get_cmap('viridis')
    colors = viridis(normalize_min_max(point_to_face.cpu().numpy()))
    out_mesh.visual.vertex_colors = colors
    out_mesh.export(outpath)

    avg_thickness = point_to_face.mean()

    print("Average squared thickness value: ", str(avg_thickness))
    print("Output mesh as been written to ", outpath)


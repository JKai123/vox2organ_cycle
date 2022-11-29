from pycpd import AffineRegistration
from pycpd import DeformableRegistration
import numpy as np
from enum import Enum
import open3d as o3d
from tqdm import tqdm


def compute_avg_mesh(meshes):
    meshes_verts = [mesh.vertices for mesh in meshes]
    meshes_verts = np.stack(meshes_verts)
    mean_verts = np.mean(meshes_verts, axis=0)
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mean_verts), o3d.utility.Vector3iVector(meshes[0].triangles))


def cpd_single(source, target, method="deform"):
    if method == "affine":
        reg = AffineRegistration(X=target, Y=source, w=0.9999)
    if method == "deform":
        reg = DeformableRegistration(X=target, Y=source, alpha=0.1)
    ty = reg.register()
    return ty

def cpd(meshes, path):
    target = compute_avg_mesh(meshes)
    tmeshes = []
    for i, mesh in tqdm(enumerate(meshes)):
        tverts = cpd_single(np.asarray(mesh.vertices), np.asarray(target.vertices))
        tmeshes.append(o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(tverts), o3d.utility.Vector3iVector(mesh.triangles)))
        o3d.io.write_triangle_mesh(path + "/error_" + str(i) + ".ply", mesh)
    return tmeshes

""" Funcitons that perform Generalized Procrustes analysis 
https://medium.com/@olga_kravchenko/generalized-procrustes-analysis-with-python-numpy-c571e8e8a421
"""
__author__ = "Johannes Kaiser"
__email__ = "johannes.kaiser@tum.de"

import numpy as np
import torch
import scipy
import torch
import open3d as o3d
from pytorch3d.loss import (
    chamfer_distance)
from pytorch3d.structures import (
    Pointclouds
)
from tqdm import tqdm
def center_origin(shape_trimesh):
    center = torch.mean(shape_trimesh.verts_packed(), dim=0)
    shape_trimesh.offset_verts_(-center)


def unit_scale(shape_trimesh, typ="2norm"):
    # scale by frobenius,
    # scale by vertices 2 norm
    # scale by bounding box
    if typ == "frobenius":
        norm = np.sqrt(np.sum(np.square(shape_trimesh.vertices_packed())))
    if typ == "2norm":
        norm = np.sqrt(np.sum(np.square(shape_trimesh.vertices_packed()), axis = 1))
        norm = np.max(norm)
        mesh_verts = (np.asarray(shape_trimesh.vertices_packed())) / norm
    if typ == "bounding box":
        # oriented_bb = shape_trimesh.get_oriented_bounding_box
        raise NotImplementedError

def rotate(shape_target, shape_trimesh):
    # Kabsch algorithm @Wikipedia
    Q = np.asarray(shape_target.vertices)
    P = np.asarray(shape_trimesh.vertices)

    H = np.dot(P.T, Q)
    # compute SVD
    u, _, vt = scipy.linalg.svd(H)
    # construct S: an identity matrix with the smallest singular value replaced by sgn(|U*V^t|)
    s = np.eye(Q.shape[1])
    s[-1, -1] = np.sign(np.linalg.det(np.dot(u, vt)))
    # compute optimal rotational transformation
    r_opt = np.dot(np.dot(u, s), vt)
    # Rotate shape
    shape_trimesh.rotate(r_opt) # TODO transpose might be wrong


def rotate_ICP(shape_target, shape_trimesh):
    source = o3d.geometry.PointCloud(shape_trimesh.vertices)
    target = o3d.geometry.PointCloud(shape_target.vertices)
    threshold = 0.2
    # print("Initial allignment")
    # evaluation = o3d.pipelines.registration.evaluate_registration(source, target, threshold)
    # print(evaluation)
    # print("Apply Point to points ICP")
    reg_p2p = o3d.pipelines.registration.registration_icp(
        source, target, threshold, estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint()
    )
    # print(reg_p2p)
    # print("Transformation is:")
    # print(reg_p2p.transformation)
    shape_trimesh.transform(reg_p2p.transformation)


def compute_avg_mesh(meshes):
    meshes_verts = [mesh.vertices for mesh in meshes]
    meshes_verts = np.stack(meshes_verts)
    mean_verts = np.mean(meshes_verts, axis=0)
    return o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(mean_verts), o3d.utility.Vector3iVector(meshes[0].triangles))


def compute_error(meshes, target):
    target_verts = np.asarray(target.vertices)
    diff = 0
    for mesh in meshes:
        mesh_verts = (np.asarray(mesh.vertices))
        test= mesh_verts - target_verts
        test= np.square(mesh_verts - target_verts)
        test= np.sum(np.square(mesh_verts - target_verts), axis=1)
        test= np.sqrt(np.sum(np.square(mesh_verts - target_verts), axis=1))
        diff += np.sum(np.sqrt(np.sum(np.square(mesh_verts - target_verts), axis=1)))/mesh_verts.shape[0]
    diff = diff/len(meshes)
    return diff


def compute_error_closest_points(meshes, target):
    diff = 0
    mesh_list = []
    target_list = []
    for mesh in tqdm(meshes):
        mesh_list.append(np.asarray(mesh.vertices))
        target_list.append(np.asarray(target.vertices))
    mverts = torch.tensor(np.float32(np.stack(mesh_list, axis=0)))
    tverts = torch.tensor(np.float32(np.stack(target_list, axis=0)))
    source_pc = Pointclouds(mverts)
    target_pc = Pointclouds(tverts)
    diff += chamfer_distance(source_pc, target_pc)[0]
    return diff


def gpa(meshes, path):
    #for i, mesh in enumerate(meshes):
        #o3d.io.write_triangle_mesh(path + "/" + "_original_" + str(i) + ".ply", mesh)
    for mesh in meshes:
        center_origin(mesh)
        # unit_scale(mesh)
    return meshes
    
    # target = meshes[0]
    # error = np.inf
    # thresh = 10
    # loop = 0
    # while error > thresh:
    #     loop +=1
    #     for i, mesh in enumerate(meshes):
    #         # print("indiv_error_pre_" + str(i) + "_" + str(compute_error([mesh], target)))
    #         rotate_ICP(target, mesh)
    #         # print("indiv_error_aft_" + str(i) + "_" +str(compute_error([mesh], target)))
    #         if loop == 100:
    #             o3d.io.write_triangle_mesh(path + "/error_" + str(loop) + "_" + str(i) + ".ply", mesh)
            
    #     error = compute_error_closest_points(meshes, target)
    #     print("error: "  + str(error))
    #     error = compute_error(meshes, target)
    #     print("error: "  + str(error))
    #     #target = compute_avg_mesh(meshes)
    #     #o3d.io.write_triangle_mesh(path + "/" + str(loop) + "_" + "target" + ".ply", target)

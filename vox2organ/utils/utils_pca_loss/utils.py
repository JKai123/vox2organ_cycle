import os
import numpy as np
import open3d as o3d
from tqdm import tqdm


def get_meshes_from_dir(data_path, typ='o3d'):
    files = []
    files_number = np.array([])
    files_order = np.array([])

    for ply_file in os.listdir(data_path):
        files.append(ply_file)
        num = ''
        for c in ply_file:
            if c.isdigit():
                num = num + c
        num = num[5:-1]
        files_number = np.append(files_number, int(num))
    files_order = np.argsort(files_number)
    print(files_order)
    print(type(files_order))

    meshes = []

    if typ == 'o3d':
        for _, ordered_Index in enumerate(files_order): 
            ply_file = files[ordered_Index]   
            print("Load File ", ply_file)
            meshes.append(o3d.io.read_triangle_mesh(os.path.join(data_path, ply_file)))
    orderer_list = [files[i] for i in files_order]
    return meshes, orderer_list


def get_multi_meshes_from_dir(data_path):
    files = {}
    cases = []
    organs = []
    print("Load Meshes")
    for ply_file in tqdm(os.listdir(data_path)):
        num = ''
        for c in ply_file:
            if c.isdigit():
                num = num + c
        case = num[0:5]
        organ = num[-1]
        key = case + organ
        files[key] = ply_file
        if case not in cases:
            cases.append(case)
        if organ not in organs:
            organs.append(organ)
    
    meshes = []
    print("Combine Meshes")
    for case in  tqdm(cases):
        case_meshes = []
        for organ in organs:
            ply_file = files[case + organ]
            case_meshes.append(o3d.io.read_triangle_mesh(os.path.join(data_path, ply_file)))
        total_mesh = None
        for mesh in case_meshes:
            if total_mesh == None:
                total_mesh = mesh
            else:
                total_mesh += mesh
        meshes.append(total_mesh)
    return meshes


def save_meshes(meshes, file_names,  path):
    if not os.path.exists(path):
        os.makedirs(path)
    for i, mesh in enumerate(meshes):
        print("Saving " + file_names[i])
        o3d.io.write_triangle_mesh(os.path.join(path, file_names[i]), mesh)

import sys, os
import numpy as np
import open3d as o3d
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_ply

# Data structures and functions for rendering
from pytorch3d.structures import Meshes

# Set paths
DATA_DIR = "experiments/viz2"
CWD_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(CWD_PATH, DATA_DIR)


def load_data():
    files = []
    files_number = np.array([])
    files_order = np.array([])

    for ply_file in os.listdir(DATA_PATH):
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

    for _, ordered_Index in enumerate(files_order): 
        ply_file = files[ordered_Index]   
        print("Load File ", ply_file)
        meshes.append(o3d.io.read_triangle_mesh(os.path.join(DATA_PATH, ply_file)))
    
    return meshes


def visualize_data(data):
    print("Computing normal and rendering it.")
    data.compute_vertex_normals()
    print(np.asarray(data.triangle_normals))
    o3d.visualization.draw_geometries([data])


def plotting_routine():
    meshes =  load_data()
    visualize_data(meshes[0])



if __name__ == '__main__':
    plotting_routine()
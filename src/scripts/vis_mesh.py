#!/usr/bin/env python3

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

""" Visualization of 3D data """

import numpy as np

from utils.utils import read_vertices_from_ply
from utils.visualization import show_pointcloud_pynt

def main():
    filename = "../supplementary_material/example_meshes/lh_pial.ply"
    data = read_vertices_from_ply(filename) # Read the point cloud
    show_pointcloud_pynt(filename)


if __name__ == "__main__":
    main()

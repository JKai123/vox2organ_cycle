""" Visualization of data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from typing import Union

import open3d as o3d
import pyvista as pv

from pyntcloud import PyntCloud

def show_pointcloud(filenames: Union[str, list], backend='pyvista'):
    """
    Show a point cloud stored in a file (e.g. .ply) using open3d or pyvista.
    An alternative is based on PyntCloud and pyvista, see
    'show_pointcloud_pynt'

    :param str filename: A list of files or a directory name.
    :param str backend: 'open3d' or 'pyvista' (default)
    """
    if isinstance(filenames, str):
        if os.path.isdir(filenames):
            path = filenames
            filenames = os.listdir(path)
            filenames = [os.path.join(path, fn) for fn in filenames]
        else:
            filenames = [filenames]

    for fn in filenames:
        print(f"File: {fn}")
        if backend == 'open3d':
            show_pointcloud_open3d(fn)
        else:
            show_pointcloud_pynt(fn)

def show_pointcloud_open3d(filename: str):
    """
    Show a point cloud stored in a file (e.g. .ply) using open3d.
    An alternative is based on PyntCloud and pyvista, see
    'show_pointcloud_pynt'

    :param str filename: The file that should be visualized.
    """
    cloud = o3d.io.read_point_cloud(filename)
    print(cloud)
    o3d.visualization.draw_geometries([cloud])

def show_pointcloud_pynt(filename: str):
    """
    Show a point cloud stored in a file (e.g. .ply) using pyntcloud/pyvista.

    :param str filename: The file that should be visualized.
    """
    cloud = PyntCloud.from_file(filename)
    print(cloud)

    plotter = pv.Plotter()
    plotter.add_mesh(cloud.to_instance("pyvista", mesh=True), color='red', point_size=5)
    plotter.add_mesh(cloud.to_instance("pyvista", mesh=True), color='white')
    plotter.show()

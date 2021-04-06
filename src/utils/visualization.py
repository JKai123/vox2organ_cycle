""" Visualization of data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import open3d as o3d

from mpl_toolkits.mplot3d import axes3d
from pyntcloud import PyntCloud

def show_pointcloud(filename: str):
    """
    Show a point cloud stored in a file (e.g. .ply) using open3d.
    !!! Leads sometimes to a GLFW error, see 'show_pointcloud_pynt for an
    alternative.

    :param str filename: The file that should be visualized.
    """
    cloud = o3d.io.read_point_cloud(filename)
    # Leads to error "GLFW Error: GLX: Failed to create context: GLXBadFBConfig"
    o3d.visualization.draw_geometries([cloud])

def show_pointcloud_pynt(filename: str):
    """
    Show a point cloud stored in a file (e.g. .ply) using pyntcloud.

    :param str filename: The file that should be visualized.
    """
    cloud = PyntCloud.from_file(filename)
    cloud.plot()


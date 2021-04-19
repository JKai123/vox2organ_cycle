""" Visualization of data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from typing import Union

import open3d as o3d
import nibabel as nib
import matplotlib.pyplot as plt

from pyntcloud import PyntCloud
from utils.utils import find_label_to_img

def show_pointcloud(filenames: Union[str, list], backend='open3d'):
    """
    Show a point cloud stored in a file (e.g. .ply) using open3d or pyvista.
    An alternative is based on PyntCloud and pyvista, see
    'show_pointcloud_pynt'

    :param str filenames: A list of files or a directory name.
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
        elif backend == 'pyvista':
            show_pointcloud_pynt(fn)
        else:
            raise ValueError("Unknown backend {}".format(backend))

def show_pointcloud_open3d(filename: str):
    """
    Show a point cloud stored in a file (e.g. .ply) using open3d.
    An alternative is based on PyntCloud and pyvista, see
    'show_pointcloud_pynt'

    :param str filename: The file that should be visualized.
    """
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    print(mesh)
    o3d.visualization.draw_geometries([mesh])

def show_pointcloud_pynt(filename: str):
    """
    Show a point cloud stored in a file (e.g. .ply) using pyntcloud/pyvista.

    :param str filename: The file that should be visualized.
    """
    import pyvista as pv
    cloud = PyntCloud.from_file(filename)
    print(cloud)

    plotter = pv.Plotter()
    plotter.add_mesh(cloud.to_instance("pyvista", mesh=True), color='red', point_size=5)
    plotter.add_mesh(cloud.to_instance("pyvista", mesh=True), color='white')
    plotter.show()

def show_img_slices_3D(filenames: str, show_label=True):
    """
    Show three centered slices of a 3D image

    :param str filenames: A list of files or a directory name.
    :param bool show_label: Try to find label corresponding to image and show
    image and label together if possible.
    """

    if isinstance(filenames, str):
        if os.path.isdir(filenames):
            path = filenames
            filenames = os.listdir(path)
            filenames = [os.path.join(path, fn) for fn in filenames]
        else:
            filenames = [filenames]

    for fn in filenames:
        img3D = nib.load(fn)
        assert img3D.ndim == 3, "Image dimension not equal to 3."

        # Try to find ground truth
        base_dir = '/'.join(fn.split('/')[:-2]) # Dataset base directory
        img_id = fn.split('/')[-1].split('.')[0] # Name of image without type
        label_name = find_label_to_img(base_dir, img_id)

        img1 = img3D.get_fdata() # get np.ndarray
        img1 = img1[int(img3D.shape[0]/2), :, :]
        img2 = img3D.get_fdata() # get np.ndarray
        img2 = img2[:, int(img3D.shape[1]/2), :]
        img3 = img3D.get_fdata() # get np.ndarray
        img3 = img3[:, :, int(img3D.shape[2]/2)]

        if label_name is not None and show_label:
            # Read and show ground truth
            label3D = nib.load(label_name)

            label1 = label3D.get_fdata() # get np.ndarray
            label1 = label1[int(label3D.shape[0]/2), :, :]
            label2 = label3D.get_fdata() # get np.ndarray
            label2 = label2[:, int(label3D.shape[1]/2), :]
            label3 = label3D.get_fdata() # get np.ndarray
            label3 = label3[:, :, int(label3D.shape[2]/2)]

            show_slices([img1, img2, img3], labels=[label1, label2, label3])

        else:
            show_slices([img1, img2, img3])

def show_slices(slices, labels=None, save_path=None):
    """
    Visualize image slices in a row.

    :param array-like slices: The image slices to visualize.
    :param array-like labels (optional): The image segmentation label slices.
    """

    _, axs = plt.subplots(1, len(slices))

    for i, s in enumerate(slices):
        axs[i].imshow(s, cmap="gray")

    if labels is not None:
        for i, l in enumerate(labels):
            axs[i].imshow(l, cmap="OrRd", alpha=0.3)

    plt.suptitle("Image Slices")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)

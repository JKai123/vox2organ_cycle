""" Visualization of data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from typing import Union

import open3d as o3d
import nibabel as nib
import matplotlib.pyplot as plt

from pyntcloud import PyntCloud

def find_label_to_img(base_dir: str, img_id: str, label_dir_id="label"):
    """
    Get the label file corresponding to an image file.
    Note: This is only needed in the case where the dataset is not represented
    by a data.dataset.

    :param str base_dir: The base directory containing the label directory.
    :param str img_id: The id of the image that is also cotained in the label
    file.
    :param str label_dir_id: The string that identifies the label directory.
    :return The label file name.
    """
    label_dir = None
    label_name = None
    for d in os.listdir(base_dir):
        d_full = os.path.join(base_dir, d)
        if (os.path.isdir(d_full) and (label_dir_id in d)):
            label_dir = d_full
            print(f"Found label directory '{label_dir}'.")
    if label_dir is None:
        print(f"No label directory found in {base_dir}, maybe adapt path"\
              " specification or search string.")
        return None
    # Label dir found
    for ln in os.listdir(label_dir):
        if img_id == ln.split('.')[0]:
            label_name = ln

    if label_name is None:
        print(f"No file with id '{img_id}' found in directory"\
              " '{label_dir}'.")
        return None
    return os.path.join(label_dir, label_name)


def show_pointcloud(filenames: Union[str, list], backend='open3d'):
    """
    Show a point cloud stored in a file (e.g. .ply) using open3d or pyvista.

    :param str filenames: A list of files or a directory name.
    :param str backend: 'open3d' or 'pyvista' (default)
    """
    if isinstance(filenames, str):
        if os.path.isdir(filenames):
            path = filenames
            filenames = os.listdir(path)
            filenames.sort()
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
            filenames.sort()
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

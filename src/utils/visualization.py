""" Visualization of data """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import os
from typing import Union

import numpy as np
# import open3d as o3d # Leads to double logging, uncomment if needed
import nibabel as nib
import matplotlib.pyplot as plt
import trimesh
import torch

from pyntcloud import PyntCloud
from skimage.measure import find_contours

from data.cortex_labels import combine_labels
from utils.coordinate_transform import normalize_vertices_per_max_dim

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
            show_pointcloud_pyvista(fn)
        else:
            raise ValueError("Unknown backend {}".format(backend))

def show_pointcloud_open3d(filename: str):
    """
    Show a point cloud stored in a file (e.g. .ply) using open3d.
    An alternative is based on pyvista, see
    'show_pointcloud_pyvista'

    :param str filename: The file that should be visualized.
    """
    mesh = o3d.io.read_triangle_mesh(filename)
    mesh.compute_vertex_normals()
    print(mesh)
    o3d.visualization.draw_geometries([mesh])

def show_pointcloud_pyvista(filename: str):
    """
    Show a point cloud stored in a file (e.g. .ply) using pyvista.

    :param str filename: The file that should be visualized.
    """
    import pyvista as pv
    cloud = pv.read(filename)
    print(cloud)

    plotter = pv.Plotter()
    plotter.add_mesh(cloud)
    plotter.show()

def show_img_slices_3D(filenames: str, show_label=True, dataset="Cortex",
                       label_mode='contour', labels_from_mesh: str=None, 
                       output_file=None):
    """
    Show three centered slices of a 3D image

    :param str filenames: A list of files or a directory name.
    :param bool show_label: Try to find label corresponding to image and show
    image and label together if possible.
    :param dataset: Either 'Hippocampus' or 'Cortex'
    :param label_mode: Either 'contour' or 'fill'
    :param labels_from_mesh: Path to a mesh that is used as mesh label then.
    :param output_dir: Optionally specify an output file.
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
        print(f"Loading image {fn}...")
        assert img3D.ndim == 3, "Image dimension not equal to 3."

        img1 = img3D.get_fdata() # get np.ndarray
        img1 = img1[int(img3D.shape[0]/2), :, :]
        img2 = img3D.get_fdata() # get np.ndarray
        img2 = img2[:, int(img3D.shape[1]/2), :]
        img3 = img3D.get_fdata() # get np.ndarray
        img3 = img3[:, :, int(img3D.shape[2]/2)]

        # Try to find ground truth
        if labels_from_mesh is None:
            if dataset == "Hippocampus":
                labels = _get_labels_hippocampus(fn)
            elif dataset == "Cortex":
                labels = _get_labels_cortex(fn)
            else:
                raise ValueError(f"Unknown dataset {dataset}")
        else:
            labels = _get_labels_from_mesh(
                labels_from_mesh, patch_size=img3D.get_fdata().shape
            )

        if labels is not None and show_label:
            # Read and show ground truth
            show_slices([img1, img2, img3], labels=labels,
                        label_mode=label_mode, save_path=output_file)

        else:
            show_slices([img1, img2, img3], save_path=output_file)

def _get_labels_from_mesh(mesh_labels, patch_size):
    """ Generate voxel labels from mesh prediction(s)."""

    # Mesh processing requires pytorch3d
    from utils.utils import voxelize_mesh

    if not isinstance(mesh_labels, list):
        mesh_labels = [mesh_labels]

    label1, label2, label3 = [], [], []
    for ml in mesh_labels:
        mesh = trimesh.load(ml)
        vertices = torch.from_numpy(mesh.vertices)
        faces = torch.from_numpy(mesh.faces)
        # Potentially normalize: if the mean of all vertex coordinates is > 2,
        # it is assumed that the coordinates are not normalized
        if vertices.mean() > 2:
            vertices = normalize_vertices_per_max_dim(vertices, patch_size)

        voxelized = voxelize_mesh(vertices, faces, patch_size, 1).cpu().numpy()
        label1.append(voxelized[int(patch_size[0]/2), :, :])
        label2.append(voxelized[:, int(patch_size[1]/2), :])
        label3.append(voxelized[:, :, int(patch_size[2]/2)])

    return [label1, label2, label3]

def _get_labels_cortex(filename):
    """ Get label slices for all three axes. """
    sample_dir = "/".join(filename.split("/")[:-1])
    label_name = os.path.join(sample_dir, "aseg.nii.gz")
    try:
        label3D = nib.load(label_name)
    except FileNotFoundError:
        print("[Warning] Label " + label_name + " could not be found.")
        return None

    label1_pial = label3D.get_fdata() # get np.ndarray
    label1_pial = combine_labels(
        label1_pial, ('right_cerebral_cortex', 'left_cerebral_cortex')
    )
    label1_pial = label1_pial[int(label3D.shape[0]/2), :, :]

    label1_white = label3D.get_fdata() # get np.ndarray
    label1_white = combine_labels(
        label1_white, ('right_white_matter', 'left_white_matter')
    )
    label1_white = label1_white[int(label3D.shape[0]/2), :, :]

    label2_pial = label3D.get_fdata() # get np.ndarray
    label2_pial = combine_labels(
        label2_pial, ('right_cerebral_cortex', 'left_cerebral_cortex')
    )
    label2_pial = label2_pial[:, int(label3D.shape[1]/2), :]

    label2_white = label3D.get_fdata() # get np.ndarray
    label2_white = combine_labels(
        label2_white, ('right_white_matter', 'left_white_matter')
    )
    label2_white = label2_white[:, int(label3D.shape[1]/2), :]

    label3_pial = label3D.get_fdata() # get np.ndarray
    label3_pial = combine_labels(
        label3_pial, ('right_cerebral_cortex', 'left_cerebral_cortex')
    )
    label3_pial = label3_pial[:, :, int(label3D.shape[2]/2)]

    label3_white = label3D.get_fdata() # get np.ndarray
    label3_white = combine_labels(
        label3_white, ('right_white_matter', 'left_white_matter')
    )
    label3_white = label3_white[:, :, int(label3D.shape[2]/2)]

    return [[label1_pial, label1_white],
            [label2_pial, label2_white],
            [label3_pial, label3_white]]

def _get_labels_hippocampus(filename):
    """ Get label slices for all three axes. """
    base_dir = '/'.join(filename.split('/')[:-2]) # Dataset base directory
    img_id = filename.split('/')[-1].split('.')[0] # Name of image without type
    label_name = find_label_to_img(base_dir, img_id)

    try:
        label3D = nib.load(label_name)
    except FileNotFoundError:
        print("[Warning] Label " + label_name + " could not be found.")
        return None

    label1 = label3D.get_fdata() # get np.ndarray
    label1 = label1[int(label3D.shape[0]/2), :, :]
    label2 = label3D.get_fdata() # get np.ndarray
    label2 = label2[:, int(label3D.shape[1]/2), :]
    label3 = label3D.get_fdata() # get np.ndarray
    label3 = label3[:, :, int(label3D.shape[2]/2)]

    return [[label1], [label2], [label3]]


def show_slices(slices, labels=None, save_path=None, label_mode='contour'):
    """
    Visualize image slices in a row.

    :param array-like slices: The image slices to visualize.
    :param array-like labels (optional): The image segmentation label slices.
    """

    assert label_mode in ('contour', 'fill')
    colors = ('blue', 'green')

    _, axs = plt.subplots(1, len(slices))
    if len(slices) == 1:
        axs = [axs]

    for i, s in enumerate(slices):
        axs[i].imshow(s, cmap="gray")

    if labels is not None:
        for i, l in enumerate(labels):
            if not isinstance(l, list):
                l_ = [l]
            else:
                l_ = l

            for ll, col in zip(l_, colors):
                if label_mode == 'fill':
                    axs[i].imshow(ll, cmap="OrRd", alpha=0.3)
                else:
                    contours = find_contours(ll, np.max(ll)/2)
                    for c in contours:
                        axs[i].plot(c[:, 1], c[:, 0], linewidth=0.5,
                                    color=col)

    plt.suptitle("Image Slices")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

def show_img_with_contour(img, vertices, edges, save_path=None):
    if vertices.ndim != 2 or edges.ndim != 2:
        raise ValueError("Vertices and edges should be in packed"
                         " representation.")
    plt.imshow(img, cmap="gray")
    vertices_edges = vertices[edges]

    plt.plot(vertices_edges[:,0,1], vertices_edges[:,0,0], color="red",
             marker='x', markeredgecolor="gray", markersize=1, linewidth=1)

    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

def show_difference(img_1, img_2, save_path=None):
    """
    Visualize the difference of two 3D images in the center axes.

    :param array-like img_1: The first image
    :param array-like img_2: The image that should be compared to the first one
    :param save_path: Where the image is exported to
    """
    shape_1 = img_1.shape
    img_1_slices = [img_1[shape_1[0]//2, :, :],
                    img_1[:, shape_1[1]//2, :],
                    img_1[:, :, shape_1[2]//2]]
    shape_2 = img_2.shape
    assert shape_1 == shape_2, "Compared images should be of same shape."
    img_2_slices = [img_2[shape_2[0]//2, :, :],
                    img_2[:, shape_2[1]//2, :],
                    img_2[:, :, shape_2[2]//2]]
    diff = [(i1 != i2).long() for i1, i2 in zip(img_1_slices, img_2_slices)]

    _, axs = plt.subplots(1, len(img_1_slices))
    if len(img_1_slices) == 1:
        axs = [axs]

    for i, s in enumerate(img_1_slices):
        axs[i].imshow(s, cmap="gray")

    for i, l in enumerate(diff):
        axs[i].imshow(l, cmap="OrRd", alpha=0.6)

    plt.suptitle("Difference")
    if save_path is None:
        plt.show()
    else:
        plt.savefig(save_path)
        plt.close()

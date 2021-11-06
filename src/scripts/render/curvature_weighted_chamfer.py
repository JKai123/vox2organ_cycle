
""" Create a mesh-patch that is color-coded based on curvature weights. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import torch
import numpy as np
import pyvista as pv

from utils.mesh import curv_from_cotcurv_laplacian

# Load mesh
mesh = pv.read("../supplementary_material/example_meshes/lh_pial.ply")
assert mesh.is_all_triangles()

# Normalize
verts = mesh.points / 144.
# Faces in pyvista are stored in the format
# [num. vertices in f1, v1, v2, ..., num. vertices in f2, ...]
faces_mask = np.ones_like(mesh.faces, dtype=bool)
faces_mask[::4] = False
faces = mesh.faces[faces_mask].reshape((-1, 3))

# Compute curvature per vertex
curv = curv_from_cotcurv_laplacian(
    torch.from_numpy(verts),
    torch.from_numpy(faces)
)
weights = torch.minimum(1 + curv, torch.tensor(5.0))
np.save("../misc/weights.npz", weights)

# Extract patch

# Display
pv.set_plot_theme('doc')
plotter = pv.Plotter()
plotter.add_mesh(mesh, scalars=weights, cmap='hot')

fn = "../misc/curv_weighted_chamfer.png"
plotter.show(screenshot=fn)
print("Stored screenshot at ", fn)


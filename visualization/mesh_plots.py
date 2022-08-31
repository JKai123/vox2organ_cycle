import sys, os
import numpy as np
import plotly.graph_objects as go
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_obj

# Data structures and functions for rendering
from pytorch3d.structures import Meshes


# Set paths
DATA_DIR = "../experiments/exp_15/meshes"
obj_filename = os.path.join(DATA_DIR, "Case_00039_epoch15_struc0_meshpred")

# Load obj file
mesh = load_objs_as_meshes([obj_filename])

verts = mesh.verts_padded

fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
                   alphahull=5,
                   opacity=0.4,
                   color='cyan')])
fig.show()

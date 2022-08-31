import sys, os
import numpy as np
import plotly.graph_objects as go
# Util function for loading meshes
from pytorch3d.io import load_objs_as_meshes, load_ply

# Data structures and functions for rendering
from pytorch3d.structures import Meshes

def verts_to_pts(verts):
    x = verts[:,0]
    y = verts[:,1]
    z = verts[:,2]
    return x,y,z

# Set paths
DATA_DIR = "experiments/viz"
CWD_PATH = os.path.abspath(os.getcwd())
DATA_PATH = os.path.join(CWD_PATH, DATA_DIR)

x = []
y = []
z = []

for ply_file in os.listdir(DATA_PATH):
    verts, faces = load_ply(os.path.join(DATA_PATH, ply_file))
    x_temp, y_temp, z_temp = verts_to_pts(verts)
    x.append(x_temp)
    y.append(y_temp)
    z.append(z_temp)


fig = go.Figure(
    layout=go.Layout(
        updatemenus = [
            dict(
                type="buttons",
                buttons=[
                    dict(
                        label="Play",
                        method="animate",
                        args=[None])])]),
    frames=[go.Frame(data=[go.Mesh3d(x=x[k], y=y[k], z=z[k],
                   alphahull=5,
                   opacity=0.4,
                   color='cyan')]) for k, _ in enumerate(x)]
)

# fig = go.Figure(data=[go.Mesh3d(x=x, y=y, z=z,
#                   alphahull=5,
#                   opacity=0.4,
#                   color='cyan')])
fig.show()


meshes = Meshes(verts=[verts], faces=[faces])


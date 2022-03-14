
""" Convenience script to assemble a template from individual files. This file
can be copied into a folder where the files "lh_white.ply", "rh_white.ply",
"lh_pial.ply", and "rh_pial.ply" exist. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import trimesh
from trimesh.scene.scene import Scene

lh_white = trimesh.load("lh_white.smoothed.ply")
rh_white = trimesh.load("rh_white.smoothed.ply")
lh_pial = trimesh.load("lh_pial.smoothed.ply")
rh_pial = trimesh.load("rh_pial.smoothed.ply")

scene = Scene()

scene.add_geometry(lh_white, geom_name="lh_white")
scene.add_geometry(rh_white, geom_name="rh_white")
scene.add_geometry(lh_pial, geom_name="lh_pial")
scene.add_geometry(rh_pial, geom_name="rh_pial")

scene.export("fsaverage_smoothed_163842.obj")

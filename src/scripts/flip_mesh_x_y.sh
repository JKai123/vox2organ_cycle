#!/bin/bash

# Learn how to embed python in bash script :)

# Flip a mesh in x- and -y direction by multiplication of -1 and write the result to Desktop

orig_mesh=$1
export orig_mesh

python - << EOF
import os
import sys
import trimesh

fn = os.environ['orig_mesh']
print("Transforming", fn)
mesh = trimesh.load(fn)
mesh.vertices[:, [0,1]] = mesh.vertices[:, [0,1]] * (-1)
mesh.export(os.path.join("/mnt/c/Users/Fabian/Desktop/", fn.replace("/", "_")))

sys.exit(0)
EOF

echo "Wrote output to /mnt/c/Users/Fabian/Desktop/."

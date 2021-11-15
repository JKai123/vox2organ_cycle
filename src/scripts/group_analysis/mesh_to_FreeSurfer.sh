#!/bin/bash

# Transform any input mesh (typically .ply) to FreeSurfer space equal to
# lh.white etc.

IN_MESH=$1
ORIG_MGZ=$2
TRANS_AFFINE_FILE=$3
TRANS_FOLDER=$4
OUT_MESH=$5

export IN_MESH
export TRANS_AFFINE_FILE
export OUT_MESH

# Get trk --> scanner matrices as described in
# https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems,
# Example 3

# norig
norig_file="$TRANS_FOLDER/norig.txt"
if test -f "$norig_file"; then
    rm $norig_file
fi
mri_info --vox2ras "$ORIG_MGZ" >> $norig_file
# norig inverse
norig_inv_file="$TRANS_FOLDER/norig_inv.txt"
if test -f "$norig_inv_file"; then
    rm $norig_inv_file
fi
python -c "import numpy as np;\
    mat = np.loadtxt('$norig_file');\
    mat_inv = np.linalg.inv(mat);\
    np.savetxt('$norig_inv_file', mat_inv)"
export norig_inv_file

# torig
torig_file="$TRANS_FOLDER/torig.txt"
if test -f "$torig_file"; then
    rm $torig_file
fi
mri_info --vox2ras-tkr "$ORIG_MGZ" >> $torig_file
export torig_file

# Transform mesh
echo "Processing $IN_MESH"

python - << EOF
import os
import sys
import trimesh
import numpy as np
import nibabel as nib

trans_affine = np.loadtxt(os.environ['TRANS_AFFINE_FILE'])
norig_inv = np.loadtxt(os.environ['norig_inv_file'])
torig = np.loadtxt(os.environ['torig_file'])

mesh = trimesh.load(os.environ['IN_MESH'], process=False)
verts = mesh.vertices
verts_affine = np.concatenate(
    (verts.T, np.ones((1, verts.shape[0]))),
    axis=0
)

# Transform
verts_affine= torig @ norig_inv @ trans_affine @ verts_affine

new_verts = verts_affine.T[:, :-1]

nib.freesurfer.io.write_geometry(os.environ['OUT_MESH'], new_verts, mesh.faces)

sys.exit(0)
EOF

if [ $? -eq 0 ]; then
    echo "Finished transformation."
else
    echo "Errors occured during transformation."
fi
exit 0

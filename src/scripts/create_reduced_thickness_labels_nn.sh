#!/bin/bash

set -e

# Create thickness files for reduced resolution FS meshes. This is done by
# first applying the FreeSurfer command 'mris_convert' and then transforming
# vertex coordinates with 'transform_affine'

MALC_DIR="/mnt/nas/Data_Neuro/MALC_CSR"
FS_DIR="$MALC_DIR/FS/FS"
PREPROCESSED_DIR="/home/fabianb/data/preprocessed/MALC_CSR"
REDUCED="_reduced_0.3"

for d in $FS_DIR/*; do
    ids=$(basename $d)
    for id in $ids; do
        d_surf="$d/surf"
        d_mri="$d/mri"
        d_preprocessed="$PREPROCESSED_DIR/$id"
        d_malc="$MALC_DIR/$id"
        if [ "${id: -1}" = "3" ] && [ -d $d_malc ]; then
            for hemisphere in lh rh; do
                for surface in white pial; do
                    # Get trk --> scanner matrices as described in
                    # https://surfer.nmr.mgh.harvard.edu/fswiki/CoordinateSystems,
                    # Example 3
                    norig_file="$d_preprocessed/norig.txt"
                    if test -f "$norig_file"; then
                        rm $norig_file
                    fi
                    mri_info --vox2ras "$d_mri/orig.mgz" >> $norig_file
                    torig_file="$d_preprocessed/torig.txt"
                    if test -f "$torig_file"; then
                        rm $torig_file
                    fi
                    mri_info --vox2ras-tkr "$d_mri/orig.mgz" >> $torig_file
                    # Inverse
                    torig_inv_file="$d_preprocessed/torig_inv.txt"
                    if test -f "$torig_inv_file"; then
                        rm $torig_inv_file
                    fi
                    python -c "import numpy as np;\
                        mat = np.loadtxt('$torig_file');\
                        mat_inv = np.linalg.inv(mat);\
                        np.savetxt('$torig_inv_file', mat_inv)"

                    # inv(transform_affine.txt)
                    trans_affine="$d_malc/transform_affine.txt"
                    trans_affine_inv_file="$d_preprocessed/trans_affine_inv.txt"
                    python -c "import numpy as np;\
                        mat = np.loadtxt('$trans_affine');\
                        mat_inv = np.linalg.inv(mat);\
                        np.savetxt('$trans_affine_inv_file', mat_inv)"

                    # Transform mesh
                    full_mesh="$d_surf/$hemisphere.$surface"
                    full_thickness="$d_surf/$hemisphere.thickness"
                    reduced_mesh="$d_malc/$hemisphere"_"$surface$REDUCED.stl"
                    reduced_thickness="$d_preprocessed/$hemisphere"_"$surface$REDUCED.thickness"
                    echo "Processing $full_mesh"
                    python3 create_reduced_thickness_labels.py\
                        $full_mesh\
                        $full_thickness\
                        $reduced_mesh\
                        $reduced_thickness\
                        --transform $torig_inv_file $norig_file $trans_affine_inv_file

                    # Remove intermediate files
                    #rm $norig_file
                    #rm $torig_file
                    #rm $torig_inv_file
                done
            done
        fi
    done
done

#!/bin/bash

# Input: Predicted mesh and per-vertex labels
# Output: Per-vertex labels mapped to fsaverage template from FreeSurfer

# This script performs the following steps in order to create registered per-vertex thickness labels that can be used for group analysis:
# - Map the input mesh to the space before image registration (space of orig.mgz)
# - Align the mesh to the corresponding FreeSurfer mesh with ICP
# - Map per-vertex values to the FreeSurfer mesh with nearest neighbors
# - Map per-vertex values to template sphere (fsaverage/surf/?h.

# Arguments
EXP_NAME=$1
EPOCH=$2
N_TEST_VERTICES=$3
DATASET=$4

# Surfaces
STRUCTURES=(lh_white rh_white lh_pial rh_pial)
# Paths
EXPERIMENT_BASE_DIR=/home/fabianb/work/cortex-parcellation-using-meshes/experiments
EXPERIMENT_DIR=$EXPERIMENT_BASE_DIR/$EXP_NAME
# UPDATE FOR NEWER EXPERIMENTS
TEST_DIR=$EXPERIMENT_DIR/test_template_${N_TEST_VERTICES}_${DATASET}
#TEST_DIR=$EXPERIMENT_DIR/test_template_${N_TEST_VERTICES}
TMP_DIR=$TEST_DIR/reg_tmp
PRED_MESH_DIR=$TEST_DIR/meshes
DATASET_SHORT=${DATASET%_large}
DATASET_SHORT=${DATASET_SHORT%_small}
DATASET_SHORT=${DATASET_SHORT%_orig}
DATA_BASE_DIR=/mnt/nas/Data_Neuro/$DATASET_SHORT
# Only for ADNI_large:
FS_BASE_DIR=/mnt/nas/Data_Neuro/$DATASET_SHORT/ADNI_large_files/ADNI_large_files
ID_SUFFIX=_orig

# Exit on keyboard interrupt
control_c() {
	exit
}
trap control_c SIGINT

# Create dir for temporary files
if [ ! -d $TMP_DIR ]; then
	mkdir $TMP_DIR
fi

### Iterate over files ###
echo "Start registering thickness of ${TEST_DIR}"
for PRED_MESH_FILE in $PRED_MESH_DIR/*; do
	# Clean tmp dir
	rm $TMP_DIR/*

	PRED_MESH_FN=$(basename $PRED_MESH_FILE)
	if [[ $PRED_MESH_FN != *"meshpred"* ]]; then
		continue
	fi
	ID=${PRED_MESH_FN%_epoch*}
	STRUC_NR=${PRED_MESH_FN%%_meshpred*}
	STRUC_NR=${STRUC#*_epoch*_}
	STRUC=${STRUCTURES[$STRUC_NR]}
	HEMI=${STRUC%_white}
	HEMI=${HEMI%_pial}
	STRUC_GROUP=${STRUC#${HEMI}_}
	OUT_DIR=$TEST_DIR/thickness
    echo ""
    echo ""
	echo "### Process file ${ID} and structure ${STRUC} ###"

	# To FreeSurfer
	ORIG_MGZ=$FS_BASE_DIR/$ID$ID_SUFFIX/mri/orig.mgz
	TRANS_AFFINE_FILE=$DATA_BASE_DIR/$ID/transform_affine.txt
	OUT_MESH=$TMP_DIR/transformed_$HEMI.$STRUC_GROUP
	echo ""
	echo "### Mesh to FreeSurfer ###"
	bash mesh_to_FreeSurfer.sh $PRED_MESH_FILE $ORIG_MGZ $TRANS_AFFINE_FILE $TMP_DIR $OUT_MESH
	if [ $? -eq 0 ]; then
		echo "Generated ${OUT_MESH}"
    else
        continue
	fi

	# Transfer values to FS mesh
	echo ""
	echo "### Values to FS mesh ###"
	PRED_VALUES=$TEST_DIR/thickness/${PRED_MESH_FN%.ply}.thickness.npy
	TRANSFERRED_VALUES=$OUT_DIR/${PRED_MESH_FN%.ply}_transferred.thickness
	FIXED_MESH=$FS_BASE_DIR/$ID$ID_SUFFIX/surf/$HEMI.$STRUC_GROUP
	python3 values_to_FS_mesh.py $OUT_MESH $PRED_VALUES $FIXED_MESH $TRANSFERRED_VALUES
	if [ $? -eq 0 ]; then
		echo "Generated ${TRANSFERRED_VALUES}"
    else
        continue
	fi

	# Transfer values to template sphere
	echo ""
	echo "### Values to template ###"
	SPHERE_REG=$FS_BASE_DIR/$ID$ID_SUFFIX/surf/$HEMI.sphere.reg
	OUT_FILE=$OUT_DIR/${PRED_MESH_FN%.ply}.thickness.reg.npy
	python3 values_to_fsaverage.py $HEMI $SPHERE_REG $TRANSFERRED_VALUES $OUT_FILE
	if [ $? -eq 0 ]; then
		echo "Wrote ${OUT_FILE}"
    else
        continue
	fi
done

echo "Finished."
exit 0



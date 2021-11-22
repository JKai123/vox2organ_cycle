#!/bin/bash

# Apply meshfix postprocessing to a certain experiment

EXPERIMENT_DIR="../experiments"

Help()
{
    # Display help
    echo "Automatic application of MeshFix to all test predictions in an experiment"
    echo
    echo "Syntax: apply_meshfix EXP_NAME EPOCH N_TEST_VERTICES"
    echo
}

while getopts ":h" option; do
    case $option in
        h)
            Help
            exit;;
    esac
done

exp_name=$1
epoch=$2
n_test_vertices=$3

if [ -z "$3" ]; then
    Help
    exit 1
fi

# Find experiment directory
if [ ! -d $EXPERIMENT_DIR ]; then
    EXPERIMENT_DIR="../$EXPERIMENT_DIR"
fi
if [ ! -d $EXPERIMENT_DIR ]; then
    echo "Experiment directory could not be found."
    exit 1
fi

# Generate meshfix directory if necessary
test_dir="$EXPERIMENT_DIR/$exp_name/test_template_$n_test_vertices"
meshfix_dir="$test_dir/meshfix"
if [ ! -d $meshfix_dir ]; then
    mkdir $meshfix_dir
fi

# Fix all predicted meshes
mesh_dir="$test_dir/meshes"
for mesh_file in $mesh_dir/*; do
    fn=$(basename $mesh_file)
    if [[ "$fn" == *"epoch$epoch"*"meshpred.ply" ]]; then
        echo "Fixing $mesh_file..."
        MeshFix $mesh_file $meshfix_dir/$fn
    fi
done

echo "Fixed meshes have been written to $meshfix_dir"
echo "Done."

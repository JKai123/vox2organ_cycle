How to create the a per-vertex group study from predicted surface meshes:

0. First check if in file `scripts/group_analysi/mesht_to_FreeSurfer.sh` you need
to load the image with or without preprocessing (see comment therein).
Currently it seems like you need process=True for predicted and process=False
for FreeSurfer meshes.

1. Per-vertex thickness mapped to fsaverage template.
```
cd scripts/group_analysis
bash create_registered_thickness_labels.sh
```

2. Per-vertex t-test
```
python3 -m scripts.group_analysis.AD_degeneration
```

3. Visualization
```
python3 -m scripts.vis_mesh $FREESURFER_HOME/subjects/fsaverage/surf/<hemisphere>.smoothwm --values /path/to/p-thickness-file
```

4. Smooth with MeshLab
```
Filters -> Color Creation and Processing -> Smooth: Laplacian Vertex Color
```

5. Final visualization
```
python3 -m scripts.render.create_mesh_with_background
```

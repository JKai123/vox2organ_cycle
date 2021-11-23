# Vox2Cortex

This repository implements Vox2Cortex, a fast deep learning-based method for reconstruction of cortical surfaces from MRI.

## Installation
1. Make sure you use python 3.8
2. Clone this (Vox2Cortex) repo
```
    git clone https://gitlab.lrz.de/ga63wus/vox2cortex.git
```
3. Install using pip
```
    cd vox2cortex/
    pip install .
```
4. Clone and install our [pytorch3d fork](https://github.com/fabibo3/pytorch3d) as described therein (basically in analogy to how vox2cortex is installed as described above).

## Usage
You can include new cortex datasets directly in `vox2cortex/data/supported_datasets.py` and `vox2cortex/data/dataset_handler.py`. It is generally assumed that the cortex data (preprocessed with [FreeSurfer](https://surfer.nmr.mgh.harvard.edu/)) is stored in the form `data-raw-directory/sample-ID/sample-data`, where `sample-data` includes MRI scans and ground-truth surfaces.

A training with subsequent model testing can be started with
```
    cd vox2cortex/
    python3 main.py --train --test
```
For further information about command-line options see
```
    python3 main.py --help
```
All model parameters (and also for optimization, testing, tuning, etc.) are set in `vox2cortex/main.py`. For an extensive documentation and default parameters see `vox2cortex/utils/params.py`.

Further scripts facilitating the general workflow can be found in `src/scripts/`, e.g., for template creation. This folder also contains scripts for model evaluation that operate directly on predicted meshes (in contrast to in `utills/evaluation.py` that operates on model predictions and therefore requires the model to be available).

## Coordinate convention
The coordinate convention is the following:
- Mesh coordinates are image coordinates normalized w.r.t. the largest image dimension, see functions `utils.utils.normalize_vertices_per_max_dim`
 and `utils.utils.unnormalize_vertices_per_max_dim`.
- [`torch.nn.functional.grid_sample`](https://pytorch.org/docs/stable/nn.functional.html?highlight=grid_sample#torch.nn.functional.grid_sample) requires each image
 dimension to be normalized separately, such that coordinate -1 is one boundary and coordinate +1 the other boundary of the image in the respective dimension.
Therefore, if one wants to sample features from the image, it is necessary to convert to image coordinates first using `utils.utils.unnormalize_vertices_per_max_dim`
, normalizing w.r.t. each image dimension separately using `utils.utils.normalize_vertices`, and flipping of x- and z-axis. See example below and `check/check_coordsystems.py` for exemplary
usage.
```
import torch
import torch.nn.functional as F
a = torch.tensor([[[0,0,0],[0,0,1],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]],[[0,0,0],[0,0,0],[0,0,0]]]).float()
c = torch.nonzero(a).float() - 1 # coords in [-1,1]
c = torch.flip(c, dims=[1]) # z,y,x --> x,y,z
a = a[None][None]
c = c[None][None][None]
print(F.grid_sample(a, c, align_corners=True))
```
Output:
```
tensor([[[[[1.]]]]])
```

## Normal convention
The normal convention follows the convention used in most libraries like
pytorch3d or trimesh. That is, the face indices are ordered such that the face
normal of a face with vertex indices (i, j, k) calculates as (vj - vi) x (vk - vi).

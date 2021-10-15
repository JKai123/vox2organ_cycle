# Cortex Parcellation using Meshes

Master's thesis at ai-med. The project deals with 3D segmentation/parcellation of cortex structures based on mesh template deformation similar to [Pixel2Mesh](https://arxiv.org/abs/1804.01654) and [Voxel2Mesh](https://arxiv.org/abs/1912.03681).

The long-living branches are structured as follows:

- master: most tested, very stable
- run: ready to run but maybe not tested extensively
- dev: development in progress, potentially not runnable

## Installation
Installation using conda:
```
    conda install requirements.yml --name <conda-env-name>
```
In addition, clone and install our [pytorch3d fork](https://github.com/fabibo3/pytorch3d) as described therein (under 'Install from a local clone' in `INSTALL.md`).

## Usage
A training with subsequent model testing can be started with
```
    cd src/
    python3 main.py --train --test
```
For further information about command-line options see
```
    python3 main.py --help
```
All model parameters (and also for optimization, testing, tuning, etc.) are set in `src/main.py`. For an extensive documentation of parameters see `src/utils/params.py`.

Further scripts facilitating the general workflow can be found in `src/scripts/`, e.g., for visualization of images and meshes. This folder also contains scripts for model evaluation that operate directly on predicted meshes (in contrast to in `utills/evaluation.py` that operates on model predictions and therefore requires the model to be available). The script `scripts/apply_meshfix.sh`can be used to process all predicted test meshes of an experiment with [MeshFix](https://github.com/MarcoAttene/MeshFix-V2.1) (requires MeshFix to be installed and available in PATH).

The folder `src/check/` contains some scripts for checking preprocessing steps (e.g., run `python3 -m check.check_preprocess` to check preprocessing operations) or implementations (e.g., `python3 -m check.check_coordsystems` to check the implementation of coordinate transformations).

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

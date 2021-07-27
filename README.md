# Cortex Parcellation using Meshes

Master's thesis at ai-med. The project deals with an approach for cortex parcellation based on pixel2mesh/voxel2mesh.

The long-living branches are structured as follows:

- master: most tested, very stable  
- run: usually ready to run but maybe not tested extensively
- dev: development in progress, probably not runnable  

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

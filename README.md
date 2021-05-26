# Cortex Parcellation using Meshes

Master's thesis at ai-med. The project deals with an approach for cortex parcellation based on pixel2mesh/voxel2mesh.

The long-living branches are structured as follows:

- master: most tested, very stable  
- run: usually ready to run but maybe not tested extensively
- dev: development in progress, probably not runnable  

## Coordinate convention
[`torch.nn.functional.grid_sample`](https://pytorch.org/docs/stable/nn.functional.html?highlight=grid_sample#torch.nn.functional.grid_sample) requires flipped coordinates, see example below. By convention, we transform all coordinates such that it would be possible to sample from them using `torch.nn.functional.grid_sample`.
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

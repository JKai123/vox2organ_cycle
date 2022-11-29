import numpy as np

def mesh_vert_sqr_dist(meshes1, meshes2):
    dist = 0
    dist_list = []
    for (mesh1, mesh2) in zip(meshes1, meshes2):
        vert1 = np.asarray(mesh1.vertices)
        vert2 = np.asarray(mesh2.vertices)
        dist += np.linalg.norm(vert1-vert2)
        dist_list += [np.linalg.norm(vert1-vert2)]
    return dist/len(meshes1), dist_list

def subspace_ranges(meshes_transf):
    data_stacked = np.stack(meshes_transf)
    dim_maxima = data_stacked.max(axis=0)
    dim_minima = data_stacked.min(axis=0)
    return dim_minima, dim_maxima
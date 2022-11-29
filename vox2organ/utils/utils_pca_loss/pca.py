__author__ = "Johannes Kaiser"
__email__ = "johannes.kaiser@tum.de"

import numpy as np
import open3d as o3d
from sklearn.decomposition import PCA


def pca(meshes, dims):
    """ Functions that perform PCA 
    https://www.askpython.com/python/examples/principal-component-analysis
    """
    # No normalization wrt the variance of the indiv variables, as this is what we are interested in
    flattened_verts_list = []
    for mesh in meshes:
        flattened_verts_list.append(np.asarray(mesh.vertices[0:15000]).flatten())
    data = np.stack(flattened_verts_list)
    mean = np.mean(data, axis = 0)
    data = data - mean
    cov_mat = np.cov(data, rowvar = False)
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)
    #sort the eigenvalues in descending order
    test2 = np.amax(eigen_vectors)
    sorted_index = np.argsort(eigen_values)[::-1]
    
    sorted_eigenvalue = eigen_values[sorted_index]
    #similarly sort the eigenvectors 
    sorted_eigenvectors = eigen_vectors[:,sorted_index]


    mean = mean.reshape(3, -1).T
    ev_list = []
    for i in range(sorted_eigenvectors.shape[1]):
        ev = sorted_eigenvectors[:,i]
        ev = ev.reshape(3, -1).T
        ev_list.append(ev)
    sorted_eigenvectors = np.stack(ev_list)
    return(mean, sorted_eigenvectors, sorted_eigenvalue, None)


def scikit_pca(meshes, dims):
    flattened_verts_list = []
    for mesh in meshes:
        flattened_verts_list.append(np.asarray(mesh.vertices).flatten())

    data = np.stack(flattened_verts_list)
    mean = np.mean(data, axis = 0)
    data = data - mean
    if dims == None:
        pca = PCA()
    else:
        pca = PCA(n_components=dims) # TODO remove variavne normalization
    pca.fit(data)
    exp_var = pca.explained_variance_
    eig_vec = pca.components_
    
    return(mean, eig_vec, exp_var, pca)

    
def project_subspace(meshes, mean, eigenvectors, eigenvalues):
    # Transform in new space
    data, edges_list = transf_meshes(meshes, mean, eigenvectors, eigenvalues)
    
    # Transform back
    data = np.dot(data, eigenvectors)    
    
    # Add mean
    data += mean
    
    # Reformulate meshes
    meshes = []
    vert_list = []
    for i, verts in enumerate(data):
        vert_list.append(verts.reshape(-1, 3))
        meshes.append(o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(verts.reshape(-1, 3)), edges_list[i]))
    return meshes


def transf_meshes(meshes, mean, eigenvectors, eigenvalues):
     # meshes to data
    flattened_verts_list = []
    edges_list = []
    for mesh in meshes:
        flattened_verts_list.append(np.asarray(mesh.vertices).flatten())
        edges_list.append(mesh.triangles)
    data = np.stack(flattened_verts_list)
    # mean = np.mean(data, axis = 0)
    data = data - mean
    
    # Transform data in subspace
    data = np.dot(data, eigenvectors.T)
    return data, edges_list
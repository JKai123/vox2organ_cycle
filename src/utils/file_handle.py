
""" Put module information here """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

import numpy as np

def read_obj(filepath):
    """ Read an .obj file in a way that separate mesh objects/structures
    are not merged
    """
    vertices = []
    faces = []
    normals = []

    vertices_structure = []
    faces_structure = []
    normals_structure = []
    V_prev = 0
    with open(filepath) as fp:
        line = fp.readline()
        cnt = 1
        while line:
            if line[0] != '#' and line[0] != 'o':
                cnt = cnt + 1
                values = [float(x) for x in line.split('\n')[0].split(' ')[1:]]
                if line[:2] == 'vn':
                    normals_structure.append(values)
                elif line[0] == 'v':
                    vertices_structure.append(values)
                elif line[0] == 'f':
                    faces_structure.append(values)
            if line[0] == 'o' and any((
                len(vertices_structure) > 0,
                len(faces_structure) > 0,
                len(normals_structure) > 0
            )):
                vertices.append(vertices_structure)
                faces.append(np.array(faces_structure) - V_prev)
                normals.append(normals_structure)
                V_prev += len(vertices_structure)
                vertices_structure = []
                faces_structure = []
                normals_structure = []

            line = fp.readline()

        vertices.append(vertices_structure)
        faces.append(np.array(faces_structure) - V_prev)
        normals.append(normals_structure)

        vertices = np.array(vertices)
        normals = np.array(normals)
        faces = np.array(faces)
        faces = np.int64(faces) - 1
        if normals.size > 0:
            return vertices, faces, normals
        else:
            return vertices, faces, None

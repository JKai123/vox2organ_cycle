
""" Utility functions for sphere templates. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from pytorch3d.utils import ico_sphere

def generate_sphere_template(centers: dict, radii: dict, level=6):
    """ Generate a template with spheres centered at centers and corresponding
    radii
    - level 6: 40962 vertices
    - level 7: 163842 vertices

    :param centers: A dict containing {structure name: structure center}
    :param radii: A dict containing {structure name: structure radius}
    :param level: The ico level to use

    :returns: A trimesh.scene.scene.Scene
    """
    if len(centers) != len(radii):
        raise ValueError("Number of centroids and radii must be equal.")

    scene = Scene()
    for (k, c), (_, r) in zip(centers.items(), radii.items()):
        # Get unit sphere
        sphere = ico_sphere(level)
        # Scale adequately
        v = sphere.verts_packed() * r + c

        v = v.cpu().numpy()
        f = sphere.faces_packed().cpu().numpy()

        mesh = Trimesh(v, f)

        scene.add_geometry(mesh, geom_name=k)

    return scene

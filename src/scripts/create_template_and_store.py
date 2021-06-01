
""" Create a mesh template and store it """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from data.cortex import Cortex

structure_type = 'white_matter'

template_path = "../supplementary_material/spheres/cortex_" + structure_type + "_convex.obj"

dataset, _, _ = Cortex.split("/mnt/nas/Data_Neuro/MALC_CSR/",
                             1532,
                             (100, 0, 0),
                             False,
                             "../misc",
                             patch_size=(192, 224, 192),
                             structure_type=structure_type,
                             mesh_target_type='mesh',
                             n_ref_points_per_structure = None)

path = dataset.store_convex_cortex_template(template_path, 200000)

print("Template stored at " + path)

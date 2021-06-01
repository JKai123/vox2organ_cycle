
""" Create a mesh template and store it """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from data.cortex import Cortex

template_path = "../supplementary_material/spheres/cortex_white_matter_convex_40770.obj"

dataset, _, _ = Cortex.split("/mnt/nas/Data_Neuro/MALC_CSR/",
                             1532,
                             (100, 0, 0),
                             (192, 224, 192),
                             False,
                             "../misc")

path = dataset.store_convex_cortex_template(template_path)

print("Template stored at " + path)

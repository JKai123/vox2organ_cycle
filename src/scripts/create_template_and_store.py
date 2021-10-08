
""" Create a mesh template and store it """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from data.cortex import Cortex

structure_type = ('white_matter', 'cerebral_cortex')

# Important!!! (defines coordinate normalization in the template)
patch_size = [64, 144, 128]
select_patch_size = [96, 208, 176]
patch_origin=[0, 0, 0]

template_path = f"../supplementary_material/rh_white_pial/cortex_4_ellipsoid_40962_sps{select_patch_size}_ps{patch_size}.obj"

split = {
    'train': ['1000_3', '1001_3', '1002_3', '1006_3', '1007_3', '1008_3',
              '1009_3', '1010_3', '1011_3', '1012_3', '1013_3', '1014_3',
              '1015_3', '1036_3', '1017_3', '1107_3', '1128_3', '1101_3',
              '1003_3', '1004_3', '1125_3', '1005_3', '1122_3', '1110_3',
              '1119_3', '1113_3', '1018_3', '1019_3', '1104_3', '1116_3'],
    'validation': [],
    'test': []
}

print("Creating dataset...")
dataset, _, _ = Cortex.split(raw_data_dir="/mnt/nas/Data_Neuro/MALC_CSR/",
                             augment_train=False,
                             save_dir="../misc",
                             dataset_seed=1532,
                             # dataset_split_proportions=(100, 0, 0),
                             fixed_split=split,
                             patch_origin=patch_origin,
                             select_patch_size=select_patch_size,
                             patch_size=patch_size,
                             structure_type=structure_type,
                             mesh_target_type='mesh',
                             reduced_freesurfer=0.3,
                             n_ref_points_per_structure=10000, # irrelevant
                             mesh_type='freesurfer',
                             preprocessed_data_dir="/home/fabianb/data/preprocessed/MALC_CSR/",
                             patch_mode="no")
print("Dataset created.")
print("Creating template...")

# path = dataset.store_convex_cortex_template(
    # template_path, n_min_points=40000, n_max_points=60000
# )
path = dataset.store_ellipsoid_template(template_path)

if path is not None:
    print("Template stored at " + path)

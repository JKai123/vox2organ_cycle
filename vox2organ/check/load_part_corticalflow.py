
""" Loading only a part of cortical flow model. """

__author__ = "Fabi Bongratz"
__email__ = "fabi.bongratz@gmail.com"

from models.corticalflow import CorticalFlow

part_model = "../experiments/exp_9/intermediate.model"

model = CorticalFlow(
    2,
    1,
    [12, 12, 12],
    [[16, 32, 64, 128, 256], [16, 32, 64]],
    [[128, 64, 32, 16], [32, 16]],
    "../supplementary_material/white_pial/cortex_2_1000_3_smoothed_41602_sps[96, 208, 192]_ps[96, 208, 192].obj",
    None,
    3
)

model.load_part(part_model)
model.train()
model.freeze_pre_trained()

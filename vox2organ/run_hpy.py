import wandb
from utils.hyp_sweep import get_sweep_config
from main import main

sweep_configuration = get_sweep_config
sweep_id = wandb.sweep(sweep=sweep_configuration, project='final_sweeps_mri')
# sweep_id = 'rsgrepwx'
wandb.agent(sweep_id, function=main, count=100, project='final_sweeps_mri')
# main(sweep_configuration)
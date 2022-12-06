import wandb
from utils.hyp_sweep import get_sweep_config
from main import main

sweep_configuration = get_sweep_config
# sweep_id = wandb.sweep(sweep=sweep_configuration, project='final_sweeps')
# sweep_id = 'hvyg2c7l'
wandb.agent("hvyg2c7l", function=main, count=100, project='final_sweeps')
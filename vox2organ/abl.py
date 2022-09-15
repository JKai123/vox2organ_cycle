from main import main
from params.ablation import get_sweep_config
import wandb
wandb.login()
sweep_config = get_sweep_config()
sweep_id = wandb.sweep(sweep_config, project="pytorch-sweeps-demo")
wandb.agent(sweep_id, main, count=5)
wandb.finish()
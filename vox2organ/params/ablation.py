
""" Experiment-specific parameters. """

__author__ = "Johannes Kaiser"
__email__ = "johannes.kaiser@tum.de"


# This dict contains groups of sweep 
# to conduct parameter sweeps
sweep_config = {
    # No patch mode, vox2cortex
    'method': 'random'
}

metric = {
    'name': 'TotalLoss',
    'goal': 'minimize'
}


parameters_dict = {
    'N_EPOCHS': {
        'value': 20
        },
    'chamfer_loss': {
        'distribution': 'uniform',
        'min': 0,
        'max': 1
        },
    'cosine_loss': {
        'distribution': 'uniform',
        'min': 0,
        'max': 1
        },
    'laplace_loss': {
        'distribution': 'uniform',
        'min': 0,
        'max': 1
        },
    'normal_loss': {
        'distribution': 'uniform',
        'min': 0,
        'max': 1
        },
    'edge_loss': {
        'distribution': 'uniform',
        'min': 0,
        'max': 1
        },
    }



def get_sweep_config():
    config = sweep_config
    config['metric'] = metric
    config['parameters'] = parameters_dict
    return config

def update_hps_sweep(hps, config):
    hps['N_EPOCHS'] = config.N_EPOCHS
    hps['MESH_LOSS_FUNC_WEIGHTS'] = [
            [config.chamfer_loss] * 5, # Chamfer
            [config.cosine_loss] * 5, # Cosine,
            [config.laplace_loss] * 5, # Laplace,
            [config.normal_loss] * 5, # NormalConsistency
            [config.edge_loss] * 5 # Edge
        ]
    return hps


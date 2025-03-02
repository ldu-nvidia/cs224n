sweep_config = {
    'method': 'random'
    }
    metric = {
    'name': 'cross_entropy',
    'goal': 'minimize'   
    }

sweep_config['metric'] = metric

parameters_dict = {
    'fc_layer_size': {
        'values': [128, 256, 512]
        },
    'lora_rank' : {
        'values': [2, 4, 8, 32]
        },
    'regularization' : {
        'values': [1e-5, 1e-4, 1e-3]
        }
    }

sweep_config['parameters'] = parameters_dict

parameters_dict.update({
    'epochs': {
        'value': 5}
    })

parameters_dict.update({
    'learning_rate': {
        # a flat distribution between 0 and 0.1
        'distribution': 'uniform',
        'min': 0,
        'max': 1e-3
      },
    'batch_size': {
        'distribution': 'q_log_uniform_values',
        'q': 8,
        'min': 2,
        'max': 32,
      }
    })

sweep_config['parameters'] = parameters_dict
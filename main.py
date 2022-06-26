import json
import os
import numpy as np
from datetime import datetime
from pprint import pprint
from copy import copy
from itertools import product

time_format = "%d-%m-%Y--%H-%M-%S"

# datetime object containing current date and time
now = datetime.now()

start = now.strftime(time_format)

# load the configuration of this run
with open('config.json', 'r') as config_json:
    config = json.load(config_json)

#
# grab the time series from the dataset
#
if config['dataset']['file type'] == 'json':
    with open(config['dataset']['file path']) as dataset:
        ts = json.load(dataset)

used = config['dataset']['series used']

if used == 'all':
    print(f'Runing the experiment with all {len(ts)} time series...')
    datasets = {
        f'T{k}': {
            'raw': v,
            'legnth': len(v),
        } for k, v in ts.items()
    }
else:
    print(f'Using series number {used}...')
    datasets = {
        f'T{i}': {
            'raw': ts[str(i)],
            'legnth': len(ts[str(i)]),
        } for i in used
    }

#
# grab configuration set-ups
#
# transformation
t_config = config['transformation config']
xthresholds = t_config['xthresholds']

# threshold of the transformation
thresholds = list(product(xthresholds, xthresholds))

#
# regression/forecast modelling
modelling_config = config['modelling config']

v_size = modelling_config['validation size']
t_size = modelling_config['test size']
horizon = modelling_config['forecast horizon']
measure = modelling_config['score measure']

if type(v_size) != int and type(v_size) != float:
    raise ValueError('Invalid validation size in config.json')

if type(t_size) != int and type(t_size) != float:
    raise ValueError('Invalid test size in config.json')

if type(horizon) != int:
    raise ValueError('Invalid forecast horizon in config.json')

#
# performance measurement (used in validation, testing)
#
# predefined performance measurement options
measures = {
    # !!! If the horizon > 1, need a new distance measure.
    'SAPE': lambda y, y_hat: np.abs(y_hat-y)/np.mean((np.abs(y_hat), np.abs(y)))
}

if measure == 'SMAPE':
    score = copy(measures['SAPE'])
else:
    raise ValueError('Invalid score measure in config.json')

#
# models
models = config['models']

#
# create policy sets for the agents
#
# predefined model hyperparameters to tune and their search space
models_params = {
    'MLP': {
        'n of lags': [1, 2],
        'strucs': [(100, ), (100, 100, )],
        'max iter': [100]
    },
    'EN': {},
    'AutoARIMA': {},
    'ETS': {}
}

# create policy sets
MLP_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'n lag': h[1],
        'struc': h[2],
        'max iter': h[3]
    } for h in product(thresholds, *models_params['MLP'].values())
]

EN_policies = [{}]
AutoARIMA_policies = [{}]
ETS_policies = [{}]

#
# make directory for this experiment
#
if not os.path.exists(f'experiment_info/{start}/'):
    os.mkdir(f'experiment_info/{start}/')

#
# let the agents play
#
run_info = {}
for model in models:
    if model == "MLP":
        import MLP as mlp
        with open(f'experiment_info/{start}/{model}_raw.json', 'x') as file:
            json.dump(mlp.raw_info, file, indent=4)
        with open(f'experiment_info/{start}/{model}_tran.json', 'x') as file:
            json.dump(mlp.tran_info, file, indent=4)
        """Clear the memory
        del mlp.raw_info
        del mlp.tran_info
        """
    elif model == "EN":
        import EN as en
    elif model == "AutoARIMA":
        import AutoARIMA as autoarima
    elif model == "ETS":
        import ETS as ets

#
# process the experiment run info for a bit
#
for agent in run_info:
    pass

now = datetime.now()
end = now.strftime(time_format)

#
# record information
#
# run
with open('experiment_info/runs.json', 'r') as file:
    ran = json.load(file)

new_run = {
    'run time': [start, end],
    'config input': config,
    'model hyper_params': {
        m: models_params[m] for m in models
    },
    'messages': None
}

ran.append(new_run)

with open('experiment_info/runs.json', 'w') as file:
    json.dump(ran, file, indent=4)

print('End of main')

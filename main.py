import json
import os
import numpy as np
import warnings
from datetime import datetime
from copy import copy
from itertools import product

time_format = "%d-%m-%Y--%H-%M-%S"
messages = {}

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

use = config['dataset']['use series']
min_len = config['dataset']['min length']

# use time series with length >= 80
if use == 'all':
    print(f'Reading all {len(ts)} time series...')
    datasets = {
        f'T{k}': {
            'raw': v,
            'legnth': len(v),
        } for k, v in ts.items() if len(v) >= min_len
    }
else:
    print(f'Reading series number {use}...')
    datasets = {
        f'T{i}': {
            'raw': ts[str(i)],
            'legnth': len(ts[str(i)]),
        } for i in use if len(ts[str(i)]) >= min_len
    }

used = list(datasets.keys())
n_series = len(used)
print(f'Total of {n_series} series are used after length filtering.')

# Prompt the user about unreasonable selection of series used.
if n_series <= 3:
    warnings.warn("Less than 4 series are chosen. Check the specified min_length or use_series.")

    proceed = input("Less than 4 series are chosen. Do you want to proceed? [yes/no]")
    if proceed == "no":
        print("Process terminated.")
        exit()
    elif proceed == "yes":
        pass
    else:
        print("Invalid response. Process terminated.")
        exit()
    messages['Warning'] = [f'Only {n_series} series are chosen for this run.']
if n_series == 0:
    raise RuntimeError('No series is chosen.')
if n_series > 50:
    proceed = input("You are running more than 50 series. Do you want to proceed? [yes/no]")
    if proceed == "no":
        print("Process terminated.")
        exit()
    elif proceed == "yes":
        pass
    else:
        print("Invalid response. Process terminated.")
        exit()
#
# grab configuration set-ups
#
# transformation
t_config = config['transformation config']
xthresholds = [round(x, 3) for x in np.arange(*t_config['xthresholds'])]

# threshold of the transformation
thresholds = list(product(xthresholds, xthresholds))

#
# regression/forecast modelling
v_size = config['modelling config']['validation size']
t_size = config['modelling config']['test size']
horizon = config['modelling config']['forecast horizon']
measure = config['modelling config']['score measure']

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
        'n of lags': [1, 2], # [1, 2, 3, 4, 5],
        'strucs': [(0, ), (3, )], # [(0, ), (1, ), (3, ), (5, ), (7, ), (9, )],
        'max iter': [500]
    },
    'EN': {
        'n of lags': [1, 2], # [1, 2, 3, 4, 5],
        'alpha' : [0.1, 0.3], # [round(x, 3) for x in np.arange(0.1, 1.1, 0.1)],
        'l1 ratio': [0.1, 0.5], # [round(x, 3) for x in np.arange(0.1, 1.1, 0.1)]
    },
    'ETS': {
        'seasonal periods': [12],  # known monthly data, search space should be [1, 4, 12, 52]
        'trend': ['add', 'mul'],
        'seasonal':['add', 'mul'],
        'damped trend': [True, False]
    },
    'AutoARIMA': {}
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

EN_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'n lag': h[1],
        'alpha': h[2],
        'l1 ratio': h[3]
    } for h in product(thresholds, *models_params['EN'].values())
]

ETS_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'seasonal periods': h[1],
        'trend': h[2],
        'seasonal': h[3],
        'damped trend': h[4]
    } for h in product(thresholds, *models_params['ETS'].values())
]

AutoARIMA_policies = [{}]

#
# make directory for this experiment
#
if not os.path.exists(f'experiment_info/{start}/'):
    os.mkdir(f'experiment_info/{start}/')

#
# let the agents play
#
run_info = {}
print(f'Running {models}...')


for model in models:
    if model == "MLP":
        try:
            import MLP as mlp
            with open(f'experiment_info/{start}/{model}_raw.json', 'x') as file:
                json.dump(mlp.raw_info, file, indent=4)
            with open(f'experiment_info/{start}/{model}_tran.json', 'x') as file:
                json.dump(mlp.tran_info, file, indent=4)
            """Clear the memory
            del mlp.raw_info
            del mlp.tran_info
            """
        except Exception as e:
            print(f'Exception {e.__class__} occurred in running {model}.')
            print(f'{model}: no .json info is generated.')
            
        else:
            print(f'{model} agent has completed successfully.')
            print(f'{model}: .json info generated.')

    elif model == "EN":
        try:
            import EN as en
            with open(f'experiment_info/{start}/{model}_raw.json', 'x') as file:
                json.dump(en.raw_info, file, indent=4)
            with open(f'experiment_info/{start}/{model}_tran.json', 'x') as file:
                json.dump(en.tran_info, file, indent=4)
            """Clear the memory
            del en.raw_info
            del en.tran_info
            """
        except Exception as e:
            print(f'Exception {e.__class__} occurred in running {model}.')
            print(f'{model}: no .json info is generated.')
            
        else:
            print(f'{model} agent has completed successfully.')
            print(f'{model}: .json info generated.')

    elif model == "AutoARIMA":
        """
        try:
            import AutoARIMA as autoarima
        except Exception as e:
            print(f'Exception {e.__class__} occurred in running {model}.')
            print(f'{model}: no .json info is generated.')
            
        else:
            print(f'{model} agents ran successfully.')
            print(f'{model}: .json info generated.')"""

    elif model == "ETS":
        try:
            import ETS as ets
            with open(f'experiment_info/{start}/{model}_raw.json', 'x') as file:
                json.dump(ets.raw_info, file, indent=4)
            with open(f'experiment_info/{start}/{model}_tran.json', 'x') as file:
                json.dump(ets.tran_info, file, indent=4)
            """Clear the memory
            del ets.raw_info
            del ets.tran_info
            """
        except Exception as e:
            print(f'Exception {e.__class__} occurred in running {model}.')
            print(f'{model}: no .json info is generated.')
            
        else:
            print(f'{model} agent has comleted successfully.')
            print(f'{model}: .json info generated.')
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
    'number of series used': n_series,
    'model hyper_params': {
        m: models_params[m] for m in models
    },
    'messages': messages
}

ran.append(new_run)

with open('experiment_info/runs.json', 'w') as file:
    json.dump(ran, file, indent=4)

print('End of main')

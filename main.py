import json
import os
import numpy as np
import warnings
from datetime import datetime
from copy import copy
from itertools import product

def get_time(time_format: str = "%d-%m-%Y--%H-%M-%S") -> str:
    now = datetime.now()
    return now.strftime(time_format)

def prompt_time(time_format: str = "%d-%m-%Y--%H-%M-%S") -> None:
    print(f'Timestamp: {get_time(time_format)}')


# -----------------------------------------------------------
# Start
time_format = "%d-%m-%Y--%H-%M-%S"

messages = {}

start = get_time()

print(f'Programme starting at {start}')


# ------------------------------------------------------------
# load the configuration of this run
with open('config.json', 'r') as config_json:
    config = json.load(config_json)

# grab the time series from the dataset
if config['dataset']['file type'] == 'json':
    with open(config['dataset']['file path']) as dataset:
        ts = json.load(dataset)

use = config['dataset']['use series']
min_len = config['dataset']['min length']

datasets = {}
lens = []

# use time series with length >= min_len
if use == 'all':
    print(f'Reading all {len(ts)} time series...')
    for k, v in ts.items():
        l = len(v)
        if l >= min_len:
            datasets[k] = {
                'raw': v,
                'length': l
            }
            lens.append(l)
elif type(use) == list:
    print(f'Reading series number {use}...')
    for i in use:
        l = len(ts[str(i)])
        if l >= min_len:
            datasets[f'T{i}'] = {
                'raw': ts[str(i)],
                'length': l
            }
            lens.append(l)
elif type(use) == int:
    count = 0
    for k, v in ts.items():
        if count == use:
            break
        l = len(v)
        if l >= min_len:
            datasets[k] = {
                'raw': v,
                'length': l
            }
            lens.append(l)
            count += 1

used = list(datasets.keys()) # a list of indices of used time series
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


# ------------------------------------------------------------------------
# grab configuration set-ups
# transformation
t_config = config['transformation config']
xthresholds = [round(x, 3) for x in np.arange(*t_config['xthresholds'])]

# threshold of the transformation
thresholds = list(product(xthresholds, xthresholds))

# regression/forecast modelling configuration
v_size = config['modelling config']['validation size']
t_size = config['modelling config']['test size']
horizon = config['modelling config']['forecast horizon']
measure = config['modelling config']['score measure']
models = config['models']

if type(v_size) != int and type(v_size) != float:
    raise ValueError('Invalid validation size in config.json')

if type(t_size) != int and type(t_size) != float:
    raise ValueError('Invalid test size in config.json')

if type(horizon) != int:
    raise ValueError('Invalid forecast horizon in config.json')


# -----------------------------------------------------------------------
# performance measurement (used in validation, testing)
# predefined performance measurement options
# @TODO: Define a better distance measure adaptable for higher dimension!!!
def SAPE(y, y_hat):
    return np.abs(y_hat-y)/np.mean((np.abs(y_hat), np.abs(y)))

measures = {
    'SAPE': SAPE
}

if measure == 'SMAPE':
    score = copy(measures['SAPE'])
else:
    raise ValueError('Invalid score measure in config.json')


# ----------------------------------------------------------------------------
# Create policy sets for the agents.
# Predefined model hyperparameters to tune and their search space
models_params = {
    'MLP': { # 3x5=15 policies x36 thresholds x10 val = 41 secs
        'n of lags': [1, 3, 5],
        'strucs': [(0, ), (1, ), (3, ), (5, ), (7, )],
        'max iter': [500]
    },
    'EN': {  # 3x4x4=48 policies x36 thresholds x10 val = 1 sec
        'n of lags': [1, 3, 5],
        'alpha' : [10**scale for scale in [-1, 0, 1, 2]],
        'l1 ratio': [round(x, 3) for x in np.arange(0.01, 1.01, 0.3)]
    },
    'ETS': {  # 2x2x2=8 policies x36 thresholds x10 val = 27 sec
        'seasonal periods': [12],  # known monthly data, search space should be [1, 4, 12, 52]
        'trend': ['add', 'mul'],
        'seasonal':['add', 'mul'],
        'damped trend': [True, False]
    },
    'XGB' :{  # 3x3x2x3 54
        'n of lags': [1, 3, 5],
        'max depth': [3, 10, 17], # 3 ~ 20
        'booster': ['gbtree', 'dart'], # gblineaer uses linear functions
        'subsample ratio': [0.1, 0.4, 0.7], # 0 ~ 1
    },
    'LGBM' : {  # 3x3x4x2=72 policies x36 thresholds x10 val = 7 min
        'n of lags': [1, 3, 5],
        'max depth': [-1, 10, 20], # -1 ~ 32
        'min split gain': [0, 3, 5], # 0 ~ 5
        'importance type': ['split', 'gain']
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

XGB_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'n lag': h[1],
        'max depth': h[2],
        'booster': h[3],
        'subsample ratio': h[4]
    } for h in product(thresholds, *models_params['XGB'].values())
]

LGBM_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'n lag': h[1],
        'max depth': h[2],
        'min split gain': h[3],
        'importance type': h[4]
    } for h in product(thresholds, *models_params['LGBM'].values())
]

AutoARIMA_policies = [{}]


# --------------------------------------------------------------------------
# multiprocessing with pool
n_workers = config['execution config']['n of workers']
worker_count = len(os.sched_getaffinity(0))
print(f'You have {worker_count} workers deployable.')
if type(n_workers) != int and n_workers >= worker_count:
    raise ValueError('Invalid number of works.')


# --------------------------------------------------------------------------
# Let the agents play.
proceed = input(f'You are running {models} with {n_workers} workers. Do you want to proceed? [yes/no]')
if proceed == "yes":
    print(f'Running {models} with {n_workers} workers...')
elif proceed == "no":
    print("Process terminated.")
    exit()
else:
    print("Invalid response. Process terminated.")
    exit()

run_info = {}

# make directory for information of this experiment
if not os.path.exists(f'experiment_info/{start}/'):
    os.mkdir(f'experiment_info/{start}/')

for model in models:
    if model == "MLP":
        try:
            from MLP import run_mlp
            mlp_raw_info, mlp_tran_info = run_mlp(
                datasets, v_size, t_size, horizon, score, MLP_policies, n_workers
            )
            # one time series costs about 1 kb in the .json file
            with open(f'experiment_info/{start}/{model}_raw.json', 'x') as file:
                json.dump(mlp_raw_info, file, indent=4)
            with open(f'experiment_info/{start}/{model}_tran.json', 'x') as file:
                json.dump(mlp_tran_info, file, indent=4)
            """Clear the memory
            del mlp.raw_info
            del mlp.tran_info
            """
        except Exception as e:
            print(f'Exception {e.__class__} occurred in running {model}.')
            print(f'{model}: no .json info is generated.')
            prompt_time()
        else:

            print(f'{model} agent has completed successfully.')
            print(f'{model}: .json info generated.')
            prompt_time()

    elif model == "EN":
        try:
            from EN import run_en
            en_raw_info, en_tran_info = run_en(
                datasets, v_size, t_size, horizon, score, EN_policies, n_workers
            )
            with open(f'experiment_info/{start}/{model}_raw.json', 'x') as file:
                json.dump(en_raw_info, file, indent=4)
            with open(f'experiment_info/{start}/{model}_tran.json', 'x') as file:
                json.dump(en_tran_info, file, indent=4)
            """Clear the memory
            del en.raw_info
            del en.tran_info
            """
        except Exception as e:
            print(f'Exception {e.__class__} occurred in running {model}.')
            print(f'{model}: no .json info is generated.')
            prompt_time()
            
        else:
            print(f'{model} agent has completed successfully.')
            print(f'{model}: .json info generated.')
            prompt_time()

    elif model == "XGB":
        # try:
        from XGB import run_xgb
        xgb_raw_info, xgb_tran_info = run_xgb(
            datasets, v_size, t_size, horizon, score, XGB_policies, n_workers
        )
        with open(f'experiment_info/{start}/{model}_raw.json', 'x') as file:
            json.dump(xgb_raw_info, file, indent=4)

        with open(f'experiment_info/{start}/{model}_tran.json', 'x') as file:
            json.dump(xgb_tran_info, file, indent=4)

    elif model == "LGBM":
        try:
            from LGBM import run_lgbm
            lgbm_raw_info, lgbm_tran_info = run_lgbm(
                datasets, v_size, t_size, horizon, score, LGBM_policies
            )
            with open(f'experiment_info/{start}/{model}_raw.json', 'x') as file:
                json.dump(lgbm_raw_info, file, indent=4)

            with open(f'experiment_info/{start}/{model}_tran.json', 'x') as file:
                json.dump(lgbm_tran_info, file, indent=4)
        except Exception as e:
            print(f'Exception {e.__class__} occurred in running {model}.')
            print(f'{model}: no .json info is generated.')
            prompt_time()

        else:
            print(f'{model} agent has completed successfully.')
            print(f'{model}: .json info generated.')
            prompt_time()

    elif model == "ETS":
        try:
            from ETS import run_ets
            ets_raw_info, ets_tran_info = run_ets(datasets, v_size, t_size, horizon, score, ETS_policies, n_workers)

            with open(f'experiment_info/{start}/{model}_raw.json', 'x') as file:
                json.dump(ets_raw_info, file, indent=4)
            with open(f'experiment_info/{start}/{model}_tran.json', 'x') as file:
                json.dump(ets_tran_info, file, indent=4)

                """Clear the memory
                del ets.raw_info
                del ets.tran_info
                """
        except Exception as e:
            print(f'Exception {e.__class__} occurred in running {model}.')
            print(f'{model}: no .json info is generated.')
            prompt_time()
            
        else:
            print(f'{model} agent has comleted successfully.')
            print(f'{model}: .json info generated.')
            prompt_time()

    elif model == "AutoARIMA":
        """
        try:
            import AutoARIMA as autoarima
        except Exception as e:
            print(f'Exception {e.__class__} occurred in running {model}.')
            print(f'{model}: no .json info is generated.')
            prompt_time()
            
        else:
            print(f'{model} agents ran successfully.')
            print(f'{model}: .json info generated.')
            prompt_time()"""


# ---------------------------------------------------------
# Process and log the information regarding the run.
for agent in run_info:
    pass

end = get_time()

with open('experiment_info/runs.json', 'r') as file:
    ran = json.load(file)

new_run = {
    'run time': [start, end],
    'config input': config,
    'number of series used': n_series,
    'Avg. len of series': np.mean(lens),
    'model hyper_params': {
        m: models_params[m] for m in models
    },
    'messages': messages
}

ran.append(new_run)

with open('experiment_info/runs.json', 'w') as file:
    json.dump(ran, file, indent=4)

print('End of main')
prompt_time()

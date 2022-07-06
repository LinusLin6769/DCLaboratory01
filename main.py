import json
import os
import numpy as np
import warnings
import math
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

file_name = config['dataset']['file name']

# grab the time series from the dataset
if config['dataset']['file type'] == 'json':
    with open(config['dataset']['file path']+file_name) as dataset:
        ts = json.load(dataset)
else:
    raise ValueError('The programme currently only deals with .JSON files.')

use = config['dataset']['use series']
min_len = config['dataset']['min length']
max_len = config['dataset']['max length']
cumulate_used = config['dataset']['cumulate used']  # this number normally is 0 (without any previous run)
datasets = {}
lens = []

# use time series with length >= min_len
if use == 'all':
    print(f'Reading all {len(ts)} time series...')
    for k, v in ts.items():
        l = len(v)
        if l >= min_len and l <= max_len:
            datasets[k] = {
                'raw': v,
                'length': l
            }
            lens.append(l)
elif type(use) == list:
    print(f'Reading series number {use}...')
    for i in use:
        l = len(ts[str(i)])
        if l >= min_len and l <= max_len:
            datasets[f'T{i}'] = {
                'raw': ts[str(i)],
                'length': l
            }
            lens.append(l)
elif type(use) == int:
    count = 0
    for k, v in ts.items():
        if count == (cumulate_used + use):
            break
        l = len(v)
        if l >= min_len and l <= max_len:
            count += 1
            if count > cumulate_used:
                datasets[k] = {
                    'raw': v,
                    'length': l
                }
                lens.append(l)

used = list(datasets.keys()) # a list of indices of used time series
n_series = len(used)
print(f'Total of {n_series} series are used after length filtering.')

# Prompt the user about unreasonable selection of series used.
if n_series == 0:
    raise RuntimeError('No series is chosen.')

if n_series <= 3:
    warnings.warn("Less than 4 series are chosen. Check the specified min_length, max_length or use_series.")

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
interp_kind = t_config['interp kind']
use_states = t_config['use states']

# transformation policies
t_policies = list(product(xthresholds, xthresholds, interp_kind, use_states))

# regression/forecast modelling configuration
retrain_window = config['modelling config']['retrain window']
v_size = config['modelling config']['validation size']
t_size = config['modelling config']['test size']
horizon = config['modelling config']['forecast horizon']
measure = config['modelling config']['score measure']
models = config['models']

if type(retrain_window) != int and type(retrain_window) != float:
    raise ValueError('Invalid retrain window in config.json')

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
    if type(y) == float or type(y) == int:
        return np.abs(y_hat-y)/np.mean((np.abs(y_hat), np.abs(y)))

    zero = [0 for _ in range(len(y))]
    return math.dist(y, y_hat)/np.mean(math.dist(y, zero), math.dist(y_hat, zero))

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
import model_policies
from model_policies import models_params, all_policies


# --------------------------------------------------------------------------
# multiprocessing with pool
n_workers = config['execution config']['n of workers']
worker_count = len(os.sched_getaffinity(0))
print(f'You have {worker_count} workers deployable.')
if type(n_workers) != int and n_workers >= worker_count:
    raise ValueError('Invalid number of works.')


# --------------------------------------------------------------------------
# Let the agents play.
if config['type of run'] == 'real run':
    print('This is a real run.')
    dir = 'experiment_info'
elif config['type of run'] == 'test':
    print('This is a test run.')
    dir = 'test_experiment_info'

proceed = input(f'You are running {models} while using {n_workers} parallel workers. Do you want to proceed? [yes/no]')

if proceed == "yes":
    pass
elif proceed == "no":
    print("Process terminated.")
    exit()
else:
    print("Invalid response. Process terminated.")
    exit()

run_info = {}

# make directory for information of this experiment
if not os.path.exists(f'{dir}/{start}/'):
    os.mkdir(f'{dir}/{start}/')

from MLP import run_mlp
from EN import run_en
from ETS import run_ets
from LGBM import run_lgbm
from XGB import run_xgb
from RF import run_rf
from LSVR import run_lsvr

run_funcs = {
    'MLP': run_mlp,
    'EN': run_en,
    'ETS': run_ets,
    'LGBM': run_lgbm,
    'XGB': run_xgb,
     'RF': run_rf,
    'LSVR': run_lsvr
}

took_time = {k: None for k in models}

# start training the models
for model in models:
    # try:
    go = get_time()
    raw_info, tran_info = run_funcs[model](
        datasets, v_size, retrain_window, t_size, horizon, score, all_policies[model], n_workers
    )
    # one time series costs about 1 kb in the .json file
    with open(f'{dir}/{start}/{model}_raw.json', 'x') as file:
        json.dump(raw_info, file, indent=4)
    with open(f'{dir}/{start}/{model}_tran.json', 'x') as file:
        json.dump(tran_info, file, indent=4)

    took_time[model] = [go, get_time()]
    """except Exception as e:
        print(f'Exception {e.__class__} occurred in running {model}.')
        print(f'{model}: NO .json info is generated.')
        prompt_time()
    else:

        print(f'{model} agent has completed successfully.')
        print(f'{model}: .json info generated.')
        prompt_time()"""


# ---------------------------------------------------------
# Process and log the information regarding the run.
end = get_time()

with open(f'{dir}/runs.json', 'r') as file:
    ran = json.load(file)

new_run = {
    'run time': [start, end],
    'config input': config,
    'number of series used': n_series,
    'Avg. len of series': np.mean(lens),
    'took time': {
        m: took_time[m] for m in models
    },
    'model hyper_params': {
        m: models_params[m] for m in models
    },
    'messages': messages
}

# create a copy of the information of this run and put it into the experiment directory
with open(f'{dir}/{start}/run_info.json', 'x') as file:
    json.dump(new_run, file, indent=4)

ran.append(new_run)

with open(f'{dir}/runs.json', 'w') as file:
    json.dump(ran, file, indent=4)

print('End of main')
prompt_time()

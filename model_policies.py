from __main__ import file_name, t_policies
from itertools import product
import numpy as np

models_monthly_params = {
    'MLP': { # 3x5=15 policies x36 thresholds x10 val x20 workers = 41 secs
        'n of lags': [1, 3, 5],
        'strucs': [(0, ), (1, ), (3, ), (5, ), (7, )],
        'max iter': [500]
    },
    'EN': {  # 3x4x4=48 policies x36 thresholds x10 val x20 workers = 1 sec
        'n of lags': [1, 3, 5],
        'alpha' : [10**scale for scale in [-1, 0, 1, 2, 3]],
        'l1 ratio': [0.01, 0.3, 0.5, 0.8, 0.9] # [round(x, 3) for x in np.arange(0.01, 1.01, 0.3)]
    },
    'ETS': {  # 2x2x2=8 policies x36 thresholds x10 val x20 workers = 27 sec
        'auto': [True]
    },
    'XGB' :{  # 3x3x2x3=54 policies x36 thresholds x10 val x20 workers = 60 min!!!!!
        'n of lags': [1, 3, 5],
        'max depth': [3, 10, 17], # 3 ~ 20
        'booster': ['gbtree', 'dart'], # gblineaer uses linear functions
        'subsample ratio': [0.1, 0.4, 0.7], # 0 ~ 1
    },
    'LGBM' : {  # 3x3x4x2=72 policies x36 thresholds x10 val x20 workers = 7 min
        'n of lags': [1, 3, 5],
        'max depth': [-1], # -1 ~ 32
        'min split gain': [0], # 0 ~ 5
        'importance type': ['split']  # 'gain' doesn't not tend to win
    },
    'RF': {  # 3x3x3x1x1=27 policies x36 thresholds x10 val x20 workers = 1 min 30 sec
        'n of lags': [1, 3, 5],
        'max depth': [None], # None, or 1 ~ 32
        'min samples split': [0.005], # defalut=2, int or 0.0001 ~ 0.5 as a fraction of n of samples
        'min impurity decrease': [0], # defalut=0, 0 ~ 1
        'ccp alpha': [0] # default=0 (no pruning), 0 ~ 1
    },
    'AutoARIMA': {}
}

models_weekly_params = {
    'MLP': { # 3x5=15 policies x36 thresholds x10 val x20 workers = 41 secs
        'n of lags': [4, 8, 12],
        'strucs': [(0, ), (5, ), (10, ), (15, ), (20, )],
        'max iter': [2000]
    },
    'EN': {  # 3x4x4=48 policies x36 thresholds x10 val x20 workers = 1 sec
        'n of lags': [4, 8, 12],
        'alpha' : [10**scale for scale in [-1, 0, 1, 2, 3]],
        'l1 ratio': [0.01, 0.3, 0.5, 0.8, 0.9] # [round(x, 3) for x in np.arange(0.01, 1.01, 0.3)]
    },
    'ETS': {  # 2x2x2=8 policies x36 thresholds x10 val x20 workers = 27 sec
        'auto': [True]
    },
    'XGB' :{  # 3x3x2x3=54 policies x36 thresholds x10 val x20 workers = 60 min!!!!!
        'n of lags': [4, 8, 12],
        'max depth': [3, 10, 17], # 3 ~ 20
        'booster': ['gbtree', 'dart'], # gblineaer uses linear functions
        'subsample ratio': [0.1, 0.4, 0.7], # 0 ~ 1
    },
    'LGBM' : {  # 3x3x4x2=72 policies x36 thresholds x10 val x20 workers = 7 min
        'n of lags': [4, 8, 12],
        'max depth': [-1], # -1 ~ 32
        'min split gain': [0], # 0 ~ 5
        'importance type': ['split']  # 'gain' doesn't not tend to win
    },
    'RF': {  # 3x3x3x1x1=27 policies x36 thresholds x10 val x20 workers = 1 min 30 sec
        'n of lags': [4, 8, 12],
        'max depth': [None], # None, or 1 ~ 32
        'min samples split': [0.005], # defalut=2, int or 0.0001 ~ 0.5 as a fraction of n of samples
        'min impurity decrease': [0], # defalut=0, 0 ~ 1
        'ccp alpha': [0] # default=0 (no pruning), 0 ~ 1
    },
    'AutoARIMA': {}
}

models_daily_params = {
    'MLP': { # 3x5=15 policies x36 thresholds x10 val x20 workers = 41 secs
        'n of lags': [7, 14, 21],
        'strucs': [(20, )],
        'max iter': [2000]
    },
    'EN': {  # 3x4x4=48 policies x36 thresholds x10 val x20 workers = 1 sec
        'n of lags': [7, 14, 21],
        'alpha' : [10**scale for scale in [2]],
        'l1 ratio': [0.5] # [round(x, 3) for x in np.arange(0.01, 1.01, 0.3)]
    },
    'ETS': {  # 2x2x2=8 policies x36 thresholds x10 val x20 workers = 27 sec
        'auto': [True]
    },
    'XGB' :{  # 3x3x2x3=54 policies x36 thresholds x10 val x20 workers = 60 min!!!!!
        'n of lags': [7, 14, 21],
        'max depth': [10], # 3 ~ 20
        'booster': ['gbtree'], # gblineaer uses linear functions
        'subsample ratio': [0.5], # 0 ~ 1
    },
    'LGBM' : {  # 3x3x4x2=72 policies x36 thresholds x10 val x20 workers = 7 min
        'n of lags': [7, 14, 21],
        'max depth': [-1], # -1 ~ 32
        'min split gain': [0], # 0 ~ 5
        'importance type': ['split']  # 'gain' doesn't not tend to win
    },
    'RF': {  # 3x3x3x1x1=27 policies x36 thresholds x10 val x20 workers = 1 min 30 sec
        'n of lags': [7, 14, 21],
        'max depth': [None], # None, or 1 ~ 32
        'min samples split': [0.005], # defalut=2, int or 0.0001 ~ 0.5 as a fraction of n of samples
        'min impurity decrease': [0], # defalut=0, 0 ~ 1
        'ccp alpha': [0] # default=0 (no pruning), 0 ~ 1
    },
    'LSVR' : {
        'n of lags': [7, 14, 21],
    },
    'AutoARIMA': {}
}

models_hourly_params = {
    'MLP': { # 3x5=15 policies x36 thresholds x10 val x20 workers = 41 secs
        'n of lags': [12, 24, 36],
        'strucs': [(40, )],
        'max iter': [2000]
    },
    'EN': {  # 3x4x4=48 policies x36 thresholds x10 val x20 workers = 1 sec
        'n of lags': [12, 24, 36],
        'alpha' : [10**scale for scale in [2]],
        'l1 ratio': [0.5] # [round(x, 3) for x in np.arange(0.01, 1.01, 0.3)]
    },
    'ETS': {  # 2x2x2=8 policies x36 thresholds x10 val x20 workers = 27 sec
        'auto': [True]
    },
    'XGB' :{  # 3x3x2x3=54 policies x36 thresholds x10 val x20 workers = 60 min!!!!!
        'n of lags': [12, 24, 36],
        'max depth': [10], # 3 ~ 20
        'booster': ['gbtree'], # gblineaer uses linear functions
        'subsample ratio': [0.5], # 0 ~ 1
    },
    'LGBM' : {  # 3x3x4x2=72 policies x36 thresholds x10 val x20 workers = 7 min
        'n of lags': [12, 24, 36],
        'max depth': [-1], # -1 ~ 32
        'min split gain': [0], # 0 ~ 5
        'importance type': ['split']  # 'gain' doesn't not tend to win
    },
    'RF': {  # 3x3x3x1x1=27 policies x36 thresholds x10 val x20 workers = 1 min 30 sec
        'n of lags': [12, 24, 36],
        'max depth': [None], # None, or 1 ~ 32
        'min samples split': [0.005], # defalut=2, int or 0.0001 ~ 0.5 as a fraction of n of samples
        'min impurity decrease': [0], # defalut=0, 0 ~ 1
        'ccp alpha': [0] # default=0 (no pruning), 0 ~ 1
    },
    'LSVR': {
        'n of lags': [12, 24, 36],
    },
    'AutoARIMA': {}
}

if 'monthly' in file_name.lower():
    models_params = models_monthly_params
elif 'weekly' in file_name.lower():
    models_params = models_weekly_params
elif 'daily' in file_name.lower():
    models_params = models_daily_params
elif 'hourly' in file_name.lower():
    models_params = models_hourly_params

# create policy sets
MLP_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'interp kind': h[0][2],
        'use states': h[0][3],
        'n lag': h[1],
        'struc': h[2],
        'max iter': h[3]
    } for h in product(t_policies, *models_params['MLP'].values())
]

EN_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'interp kind': h[0][2],
        'use states': h[0][3],
        'n lag': h[1],
        'alpha': h[2],
        'l1 ratio': h[3]
    } for h in product(t_policies, *models_params['EN'].values())
]

ETS_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'interp kind': h[0][2],
        'use states': h[0][3],
        'auto': h[1]
    } for h in product(t_policies, *models_params['ETS'].values())
]

XGB_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'interp kind': h[0][2],
        'use states': h[0][3],
        'n lag': h[1],
        'max depth': h[2],
        'booster': h[3],
        'subsample ratio': h[4]
    } for h in product(t_policies, *models_params['XGB'].values())
]

LGBM_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'interp kind': h[0][2],
        'use states': h[0][3],
        'n lag': h[1],
        'max depth': h[2],
        'min split gain': h[3],
        'importance type': h[4]
    } for h in product(t_policies, *models_params['LGBM'].values())
]

RF_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'interp kind': h[0][2],
        'use states': h[0][3],
        'n lag': h[1],
        'max depth': h[2],
        'min samples split': h[3],
        'min impurity decrease': h[4],
        'ccp alpha': h[5]
    } for h in product(t_policies, *models_params['RF'].values())
]

LSVR_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'interp kind': h[0][2],
        'use states': h[0][3],
        'n lag': h[1]
    } for h in product(t_policies, *models_params['LSVR'].values())
]

AutoARIMA_policies = [{}]

all_policies = {
    'MLP': MLP_policies,
    'EN': EN_policies,
    'ETS': ETS_policies,
    'LGBM': LGBM_policies,
    'XGB': XGB_policies,
    'RF': RF_policies,
    'LSVR': LSVR_policies
}

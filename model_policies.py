from __main__ import file_name, t_policies
from itertools import product
import numpy as np

models_monthly_params = {
    'MLP': { 
        'n of lags': [3, 6, 12],
        'strucs': [(6, ), (12, ), (6, 6, ), (12, 6, )],
        'max iter': [500]
    },
    'EN': { 
        'n of lags': [3, 6, 12],
        'alpha' : [10**scale for scale in [-1, 0, 1, 2]], # 0.1, 0, 1, 10, 100, 1000
        'l1 ratio': [0.1, 0.5, 0.9]  # 0 ~ 1
    },
    'ETS': { 
        'auto': [True]
    },
    'LGBM' : {  
        'n of lags': [3, 6, 12],
        'max depth': [-1], # -1 ~ 32
        'min split gain': [0], # 0 ~ 5
        'importance type': ['split']  # 'gain' doesn't not tend to win
    },
    'RF': {  
        'n of lags': [3, 6, 12],
        'max depth': [None], # None, or 1 ~ 32
        'min samples split': [0.005, 0.01], # defalut=2, int or 0.0001 ~ 0.5 as a fraction of n of samples
        'min impurity decrease': [0], # defalut=0, 0 ~ 1
        'ccp alpha': [0] # default=0 (no pruning), 0 ~ 1
    },
    'LSVR' : {
        'n of lags': [3, 6, 12],
        'tol': [0.001, 0.01],
        'c': [1, 0.1],
        # max iter: 500 ~ 2000
    },
    'MA': {
        'q': [1]
    },
    'XGB' :{  
        'n of lags': [3, 6, 12],
        'max depth': [10], # 3 ~ 20
        'booster': ['gbtree'], # gblineaer uses linear functions
        'subsample ratio': [0.5], # 0 ~ 1
    },
    'AutoARIMA': {}
}

models_weekly_params = {
    'MLP': { 
        'n of lags': [3, 7, 14],
        'strucs': [(6, ), (12, ), (6, 6, ), (12, 6, )],
        'max iter': [500]
    },
    'EN': { 
        'n of lags': [3, 7, 14],
        'alpha' : [10**scale for scale in [-1, 0, 1, 2]], # 0.1, 0, 1, 10, 100, 1000
        'l1 ratio': [0.1, 0.5, 0.9]  # 0 ~ 1
    },
    'ETS': { 
        'auto': [True]
    },
    'LGBM' : {  
        'n of lags': [3, 7, 14],
        'max depth': [-1], # -1 ~ 32
        'min split gain': [0], # 0 ~ 5
        'importance type': ['split']  # 'gain' doesn't not tend to win
    },
    'RF': {  
        'n of lags': [3, 7, 14],
        'max depth': [None], # None, or 1 ~ 32
        'min samples split': [0.005, 0.01], # defalut=2, int or 0.0001 ~ 0.5 as a fraction of n of samples
        'min impurity decrease': [0], # defalut=0, 0 ~ 1
        'ccp alpha': [0] # default=0 (no pruning), 0 ~ 1
    },
    'LSVR' : {
        'n of lags': [3, 7, 14],
        'tol': [0.001, 0.01],
        'c': [1, 0.1],
        # max iter: 500 ~ 2000
    },
    'MA': {
        'q': [1]
    },
    'XGB' :{  
        'n of lags': [3, 7, 14],
        'max depth': [10], # 3 ~ 20
        'booster': ['gbtree'], # gblineaer uses linear functions
        'subsample ratio': [0.5], # 0 ~ 1
    },
    'AutoARIMA': {}
}

models_daily_params = {
    'MLP': { 
        'n of lags': [3, 7, 14],
        'strucs': [(6, ), (12, ), (6, 6, ), (12, 6, )],
        'max iter': [500]
    },
    'EN': { 
        'n of lags': [3, 7, 14],
        'alpha' : [10**scale for scale in [-1, 0, 1, 2]], # 0.1, 0, 1, 10, 100, 1000
        'l1 ratio': [0.1, 0.5, 0.9]  # 0 ~ 1
    },
    'ETS': { 
        'auto': [True]
    },
    'LGBM' : {  
        'n of lags': [3, 7, 14],
        'max depth': [-1], # -1 ~ 32
        'min split gain': [0], # 0 ~ 5
        'importance type': ['split']  # 'gain' doesn't not tend to win
    },
    'RF': {  
        'n of lags': [3, 7, 14],
        'max depth': [None], # None, or 1 ~ 32
        'min samples split': [0.005, 0.01], # defalut=2, int or 0.0001 ~ 0.5 as a fraction of n of samples
        'min impurity decrease': [0], # defalut=0, 0 ~ 1
        'ccp alpha': [0] # default=0 (no pruning), 0 ~ 1
    },
    'LSVR' : {
        'n of lags': [3, 7, 14],
        'tol': [0.001, 0.01],
        'c': [1, 0.1],
        # max iter: 500 ~ 2000
    },
    'MA': {
        'q': [1]
    },
    'XGB' :{  
        'n of lags': [3, 7, 14],
        'max depth': [10], # 3 ~ 20
        'booster': ['gbtree'], # gblineaer uses linear functions
        'subsample ratio': [0.5], # 0 ~ 1
    },
    'AutoARIMA': {}
}

models_hourly_params = {
    'MLP': { 
        'n of lags': [3, 7, 14],
        'strucs': [(6, ), (12, ), (6, 6, ), (12, 6, )],
        'max iter': [500]
    },
    'EN': { 
        'n of lags': [3, 7, 14],
        'alpha' : [10**scale for scale in [-1, 0, 1, 2]], # 0.1, 0, 1, 10, 100, 1000
        'l1 ratio': [0.1, 0.5, 0.9]  # 0 ~ 1
    },
    'ETS': { 
        'auto': [True]
    },
    'LGBM' : {  
        'n of lags': [3, 7, 14],
        'max depth': [-1], # -1 ~ 32
        'min split gain': [0], # 0 ~ 5
        'importance type': ['split']  # 'gain' doesn't not tend to win
    },
    'RF': {  
        'n of lags': [3, 7, 14],
        'max depth': [None], # None, or 1 ~ 32
        'min samples split': [0.005, 0.01], # defalut=2, int or 0.0001 ~ 0.5 as a fraction of n of samples
        'min impurity decrease': [0], # defalut=0, 0 ~ 1
        'ccp alpha': [0] # default=0 (no pruning), 0 ~ 1
    },
    'LSVR' : {
        'n of lags': [3, 7, 14],
        'tol': [0.001, 0.01],
        'c': [1, 0.1],
        # max iter: 500 ~ 2000
    },
    'MA': {
        'q': [1]
    },
    'XGB' :{  
        'n of lags': [3, 7, 14],
        'max depth': [10], # 3 ~ 20
        'booster': ['gbtree'], # gblineaer uses linear functions
        'subsample ratio': [0.5], # 0 ~ 1
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
raw_MLP_policies = [
    {
        'n lag': h[0],
        'struc': h[1],
        'max iter': h[2]
    } for h in product(*models_params['MLP'].values())
]

tran_MLP_policies = [
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

raw_EN_policies = [
    {
        'n lag': h[0],
        'alpha': h[1],
        'l1 ratio': h[2]
    } for h in product(*models_params['EN'].values())
]

tran_EN_policies = [
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

raw_ETS_policies = [
    {
        'auto': h[0]
    } for h in product(*models_params['ETS'].values())
]

tran_ETS_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'interp kind': h[0][2],
        'use states': h[0][3],
        'auto': h[1]
    } for h in product(t_policies, *models_params['ETS'].values())
]

raw_XGB_policies = [
    {
        'n lag': h[0],
        'max depth': h[1],
        'booster': h[2],
        'subsample ratio': h[3]
    } for h in product(*models_params['XGB'].values())
]

tran_XGB_policies = [
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

raw_LGBM_policies = [
    {
        'n lag': h[0],
        'max depth': h[1],
        'min split gain': h[2],
        'importance type': h[3]
    } for h in product(*models_params['LGBM'].values())
]

tran_LGBM_policies = [
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

raw_RF_policies = [
    {
        'n lag': h[0],
        'max depth': h[1],
        'min samples split': h[2],
        'min impurity decrease': h[3],
        'ccp alpha': h[4]
    } for h in product(*models_params['RF'].values())
]

tran_RF_policies = [
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

raw_LSVR_policies = [
    {
        'n lag': h[0],
        'tol': h[1],
        'c': h[2]
    } for h in product(*models_params['LSVR'].values())
]

tran_LSVR_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'interp kind': h[0][2],
        'use states': h[0][3],
        'n lag': h[1],
        'tol': h[2],
        'c': h[3]
    } for h in product(t_policies, *models_params['LSVR'].values())
]

raw_MA_policies = [
    {
        'q': h[0]
    } for h in product(*models_params['MA'].values())
]

tran_MA_policies = [
    {
        'thres up': h[0][0],
        'thres down': h[0][1],
        'interp kind': h[0][2],
        'use states': h[0][3],
        'q': h[1]
    } for h in product(t_policies, *models_params['MA'].values())
]

AutoARIMA_policies = [{}]

all_policies = {
    'MLP': {
        'raw': raw_MLP_policies,
        'tran': tran_MLP_policies
    },
    'EN': {
        'raw': raw_EN_policies,
        'tran': tran_EN_policies
    },
    'ETS': {
        'raw': raw_ETS_policies,
        'tran': tran_ETS_policies
    },
    'LGBM': {
        'raw': raw_LGBM_policies,
        'tran': tran_LGBM_policies
    },
    'XGB': {
        'raw': raw_XGB_policies,
        'tran': tran_XGB_policies
    },
    'RF': {
        'raw': raw_RF_policies,
        'tran': tran_RF_policies
    },
    'LSVR': {
        'raw': raw_LSVR_policies,
        'tran': tran_LSVR_policies,
    },
    'MA': {
        'raw': raw_MA_policies,
        'tran': tran_MA_policies
    }
}

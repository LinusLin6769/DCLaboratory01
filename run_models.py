from dc_transformation import DCTransformer
from parallel_validation import ParallelValidation
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor

from sklearn.exceptions import ConvergenceWarning as skConvWarn
from multiprocessing import Pool, Process, Pipe
from typing import List, Tuple, Dict
from itertools import product
from p_tqdm import p_map
from tqdm import trange, tqdm
from copy import copy
import data_prep, ts_analysis
import warnings
import numpy as np
import pandas as pd

class RunModels:
    def __init__(self, model, datasets, v_size, t_size, horizon, score, policies, n_workers) -> None:
        self.model = model
        self.datasets = datasets
        self.v_size = v_size
        self.t_size = t_size
        self.horizon = horizon
        self.score = score
        self.policies = policies
        self.n_workers = n_workers

        # return variables
        self.raw_info = {}
        self.tran_info = {}

        # can deal with these models
        self.val_models = {
            'EN': self.val_en,
            'ETS': self.val_ets,
            'MLP': self.val_mlp,
            'LGBM': self.val_lgbm,
            'XGB': self.val_xgb
        }
    
    def non_positive_value_check(self, series_id, series)-> bool:
        skip = False
        for s in series:
            if s <= 0:
                self.raw_info[series_id] = {
                    'message': 'Skipped: existence of value <= 0.'
                }
                self.tran_info[series_id] = {
                    'message': 'Skipped: existence of value <= 0.'
                }
                skip = True
                break
        return skip
    
    def model_selection(self, series_id, raw_policy_errs, tran_policy_errs):

        best_raw_val_SMAPE_ind = np.argmin(raw_policy_errs)
        best_tran_val_SMAPE_ind = np.argmin(tran_policy_errs)

        self.raw_info[series_id]['val SMAPE'] = raw_policy_errs[best_raw_val_SMAPE_ind]
        self.tran_info[series_id]['test SMAPE'] = tran_policy_errs[best_tran_val_SMAPE_ind]
        
        self.raw_info[series_id]['best model'] = copy(self.policies[best_raw_val_SMAPE_ind])
        self.tran_info[series_id]['best model'] = copy(self.policies[best_tran_val_SMAPE_ind])
        
        del self.raw_info[series_id]['best model']['thres up']
        del self.raw_info[series_id]['best model']['thres down']

    def run_model(self)-> Tuple[List]:
        raw_info = {}
        tran_info = {}

        for i, entry in tqdm(self.datasets.items(), decs=f'Running {self.model}'):
            series = entry['raw']

            if self.non_positive_value_check(i, series): continue

            # ------------------------------------------------------------------
            # validation
            # split
            N = len(series)
            n_test = self.t_size if type(self.t_size) == int else int(self.t_size * N)
            n_val = self.v_size if type(self.v_size) == int else int(self.v_size * N)
            split = n_val + n_test

            # suppress convergence warning during validation
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=skConvWarn)
                warnings.filterwarnings(action='ignore', category=UserWarning)

                raw_policy_errs, tran_policy_errs = self.val_models[self.model](i, series, n_val, split)

            # model selection for both raw and tran with the all their validation errors
            self.model_selection(i, raw_policy_errs, tran_policy_errs)

            # ------------------------------------------------------------------
            # test
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=skConvWarn)
                for j in trange(n_test, desc=f'Testing series {i}'):
                    if j == n_test-1:
                        train_v = series
                    else:
                        train_v = series[:-n_test+j+1]
                    train = train_v[:-self.horizon]
                    val = train_v[-self.horizon:]

                    # raw
                    # make design matrix and target vector
                    lag = self.raw_info[i]['best model']['n lag']
                    rX, ry = data_prep.ts_prep(train, nlag=lag, horizon=self.horizon)
                    train_X, val_X = rX, train[-lag:]
                    train_y, val_y = ry, val

                    self.test_models[self.model]

        return raw_info, tran_info

    def val_en(self, series_id, series, n_val, split) -> Tuple[List]:
        
        with Pool(processes=self.n_workers) as p:
            arg = {
                'series': series,
                'n_val': n_val,
                'split': split,
                'horizon': self.horizon,
                'score': self.score
            }
            par_val = ParallelValidation(arg, model=self.model)
            res = tqdm(
                iterable=p.imap(par_val.run_parallel, self.policies),
                desc=f'Validating series {series_id}',
                total=len(self.policies)
            )
        
        res1, res2 = zip(*res)
        res1 = [np.mean(e) for e in res1]
        res2 = [np.mean(e) for e in res2]

        return res1, res2


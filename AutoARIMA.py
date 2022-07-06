from enum import auto
from tkinter.tix import Tree
from pmdarima.arima import auto_arima
from dc_transformation import DCTransformer
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning as statsConvWarn
from pmdarima.warnings import ModelFitWarning
from multiprocessing import Pool, Process, Pipe
from typing import List, Tuple, Dict
from p_tqdm import p_map
from itertools import product
from tqdm import trange, tqdm
from copy import copy
import data_prep, ts_analysis
from parallel_validation import ParallelValidation
import warnings
import numpy as np
import pandas as pd

def run_autoarima(datasets, v_size, retrain_window, t_size, horizon, score, policies, n_workers) -> Tuple[Dict]:


    raw_info = {}
    tran_info = {}

    for i, entry in tqdm(datasets.items(), desc='Running AutoARIMA'):
        series = entry['raw']

        """If series has non-positive values, then skip.
        The reason being the current diffusion estimation is std(log-return).
        It cannot handle non-positve values."""
        skip = False
        for s in series:
            if s <= 0:
                raw_info[i] = {
                    'message': 'Skipped: existence of value <= 0.'
                }
                tran_info[i] = {
                    'message': 'Skipped: existene of value <= 0.'
                }
                skip = True
                break
        if skip: continue

        #
        # validation
        #
        # split
        N = len(series)
        n_test = t_size if type(t_size) == int else int(t_size * N)
        n_val = v_size if type(v_size) == int else int(v_size * N)
        retrain_window = retrain_window if type(retrain_window) == int else int(retrain_window * N)
        split = n_val + n_test
        raw_policy_errs = []
        tran_policy_errs = []

        # suppress convergence warning during validation
        with warnings.catch_warnings():
            old_settings = np.seterr(all='ignore')  # suppress overflow warnings
            warnings.filterwarnings(action='ignore', category=ModelFitWarning)
            warnings.filterwarnings(action='ignore', category=UserWarning)
            
            with Pool(processes=n_workers) as p:
                arg = {
                    'series': series,
                    'n_val': n_val,
                    'split': split,
                    'retrain_window': retrain_window,
                    'horizon': horizon,
                    'score': score,
                }
                par_val = ParallelValidation(arg, model='AutoARIMA')
                res = tqdm(
                    iterable=p.imap(par_val.run_parallel, policies),
                    desc=f'Validating series {i}',
                    total=len(policies)
                )
                res = list(res)

            np.seterr(**old_settings)  # restore the warning settings
        
        raw_policy_errs, tran_policy_errs = zip(*res)
        raw_policy_errs = [np.mean(e) for e in raw_policy_errs]
        tran_policy_errs = [np.nanmean(e) for e in tran_policy_errs]
        
        # model selection with all the validation errors
        best_raw_val_SMAPE_ind = np.argmin(raw_policy_errs)
        best_tran_val_SMAPE_ind = np.argmin(tran_policy_errs)
        best_raw_val_SMAPE = raw_policy_errs[best_raw_val_SMAPE_ind]
        best_tran_val_SMAPE = tran_policy_errs[best_tran_val_SMAPE_ind]
        best_raw_policy = copy(policies[best_raw_val_SMAPE_ind])
        best_tran_policy = copy(policies[best_tran_val_SMAPE_ind])
        del best_raw_policy['thres up']
        del best_raw_policy['thres down']
        del best_raw_policy['interp kind']
        del best_raw_policy['use states']

        #
        # test
        #
        raw_test_errs = []
        tran_test_errs = []
        raw_y_hats = []
        tran_y_hats = []

        with warnings.catch_warnings():
            old_settings = np.seterr(all='ignore')
            warnings.filterwarnings(action='ignore', category=ModelFitWarning)
            warnings.filterwarnings(action='ignore', category=UserWarning)
            for j in trange(n_test, desc=f'Testing series {i}'):
                if j == n_test-1:
                    train_v = series
                else:
                    train_v = series[:-n_test+j+1]
                train = train_v[:-horizon]
                val = train_v[-horizon:]

                # raw
                if j % retrain_window == 0:
                    rmodel = auto_arima(train, suppress_warnings=True, n_jobs=-1)
                else:
                    rmodel.update([train[-1]])

                y, y_hat = val[0], rmodel.predict(n_periods=horizon, return_conf_int=False)[0]

                raw_test_errs.append(score(y, y_hat))
                raw_y_hats.append(y_hat)

                # with transformation
                # @NOTE: Estimation of sigma can be improved!!!
                sigma = np.std(np.diff(np.log(train)))
                thres = (sigma*best_tran_policy['thres up'], -sigma*best_tran_policy['thres down'])
                t = DCTransformer()
                t.transform(train, threshold=thres, kind=best_tran_policy['interp kind'])
                ttrain = t.tdata1

                if best_tran_policy['use states']:
                    tstates = t.status
                    tstates_onehot = data_prep.one_hot(tstates, list(t.STATUS_CODE.keys()))
                    tX_states = tstates_onehot[:-horizon] # 2d
                    tval_X_states = tstates_onehot[-horizon] # 1d

                    if j % retrain_window == 0:
                        tmodel = auto_arima(ttrain, X=tX_states, suppress_warnings=True, n_jobs=-1)
                    else:
                        tmodel.update([ttrain[-1]], X=[tX_states[-1]])
                    # predict
                    y, ty_hat = val[0], tmodel.predict(n_periods=horizon, X=[tval_X_states])[0]
                else:
                    if j % retrain_window == 0:
                        tmodel = auto_arima(ttrain, suppress_warnings=True, n_jobs=-1)
                    else:
                        tmodel.update([ttrain[-1]])
                    # predict
                    y, ty_hat = val[0], tmodel.predict(n_periods=horizon)[0]

                tran_test_errs.append(score(y, ty_hat))
                tran_y_hats.append(ty_hat)
            np.seterr(**old_settings)  # restore the warning settings

        raw_info[i] = {
            'message': None,  # placeholder for other information
            'test SMAPE': round(np.mean(raw_test_errs), 6),
            'val SMAPE': round(best_raw_val_SMAPE, 6),
            'best model': best_raw_policy,
            'y hats': raw_y_hats  # probably should put elsewhere
        }

        tran_info[i] = {
            'message': None,  # placeholder for other information
            'test SMAPE': round(np.mean(tran_test_errs), 6),
            'val SMAPE': round(best_tran_val_SMAPE, 6),
            'best model': best_tran_policy,
            'y hats': tran_y_hats  # probably should put elsewhere
        }
    
    return raw_info, tran_info

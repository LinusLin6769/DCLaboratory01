# from __main__ import datasets, v_size, t_size, horizon, score
# from __main__ import ETS_policies as policies
from dc_transformation import DCTransformer
from target_transformation import TargetTransformation
from sktime.forecasting.ets import AutoETS
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning as statsConvWarn
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.exceptions import ConvergenceWarning as skConvWarn
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

def run_ets_tran(pool, arg, policies, ind):

    par_val = ParallelValidation(arg, model='ETS', type='tran')
    res = tqdm(
        iterable=pool.imap(par_val.run_parallel, policies),
        desc=f'Tran: validating series {ind}',
        total=len(policies))

    return list(res)

def run_ets_raw(pool, arg, policies, ind):

    par_val = ParallelValidation(arg, model='ETS', type='raw')
    res = tqdm(
        iterable=pool.imap(par_val.run_parallel, policies),
        desc=f'Raw: validating series {ind}',
        total=len(policies))

    return list(res)


def run_ets(datasets, ttype, v_size, retrain_window, t_size, horizon, gap, score, policies, n_workers) -> Tuple[Dict]:

    raw_info = {}
    tran_info = {}

    for i, entry in tqdm(datasets.items(), desc='Running ETS'):
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
        raw_policy_errs = None
        tran_policy_errs = None

        # suppress convergence warning during validation
        with warnings.catch_warnings():
            old_settings = np.seterr(all='ignore')  # suppress overflow warnings
            warnings.filterwarnings(action='ignore', category=statsConvWarn)
            warnings.filterwarnings(action='ignore', category=UserWarning)
            
            raw_policies= policies['raw']
            tran_policies = policies['tran']

            with Pool(processes=n_workers) as p:
                arg = {
                    'series': series,
                    'ttype': ttype,
                    'n_val': n_val,
                    'split': split,
                    'retrain_window': retrain_window,
                    'horizon': horizon,
                    'gap': gap,
                    'score': score,
                }

                raw_policy_errs = run_ets_raw(pool=p, arg=arg, policies=raw_policies, ind=i)
                tran_policy_errs = run_ets_tran(pool=p, arg=arg, policies=tran_policies, ind=i)

            np.seterr(**old_settings)  # restore the warning settings

        raw_policy_errs = [np.mean(e) for e in raw_policy_errs]
        tran_policy_errs = [np.nanmean(e) for e in tran_policy_errs]
        
        # model selection with all the validation errors
        best_raw_val_SMAPE_ind = np.argmin(raw_policy_errs)
        best_tran_val_SMAPE_ind = np.argmin(tran_policy_errs)
        best_raw_val_SMAPE = raw_policy_errs[best_raw_val_SMAPE_ind]
        best_tran_val_SMAPE = tran_policy_errs[best_tran_val_SMAPE_ind]
        best_raw_policy = copy(raw_policies[best_raw_val_SMAPE_ind])
        best_tran_policy = copy(tran_policies[best_tran_val_SMAPE_ind])

        #
        # test
        #
        raw_test_errs = []
        tran_test_errs = []
        raw_y_hats = []
        tran_y_hats = []

        with warnings.catch_warnings():
            old_settings = np.seterr(all='ignore')
            warnings.filterwarnings(action='ignore', category=statsConvWarn)
            warnings.filterwarnings(action='ignore', category=UserWarning)
            for j in trange(n_test, desc=f'Testing series {i}'):
                if j == n_test-1:
                    train_v = series
                else:
                    train_v = series[:-n_test+j+1]
                train = train_v[:-horizon-gap]
                val = train_v[-horizon:]
                
                # target transformation
                tt = TargetTransformation(type=ttype)
                train = tt.transform(train)                

                # raw
                if j % retrain_window == 0:
                    rmodel = AutoETS(auto=best_raw_policy['auto'])
                    rmodel.fit(pd.Series(train))
                else:
                    rmodel.update(pd.Series(train[-1], index=[len(train)-1]))

                # prediction
                y_hat_temp = float(rmodel.predict(horizon+gap))

                # back transformation
                y_hat = tt.back_transform(train.tolist() + [y_hat_temp])[-horizon]
                
                # validation
                y = val[0]

                raw_test_errs.append(score(y, y_hat))
                raw_y_hats.append(y_hat)

                train = train_v[:-horizon-gap]

                # with transformation
                # @NOTE: Estimation of sigma can be improved!!!
                sigma = np.std(np.diff(np.log(train)))
                thres = (sigma*best_tran_policy['thres up'], -sigma*best_tran_policy['thres down'])
                t = DCTransformer()
                t.transform(train, threshold=thres, kind=best_tran_policy['interp kind'])
                ttrain = t.tdata1

                # target transformation
                tt = TargetTransformation(type=ttype)
                ttrain = tt.transform(ttrain)

                if j % retrain_window == 0:
                    tmodel = AutoETS(auto=best_tran_policy['auto'])
                    tmodel.fit(pd.Series(ttrain))
                else:
                    tmodel.update(pd.Series(ttrain[-1], index=[len(ttrain)-1]))

                ty_hat_temp = float(tmodel.predict(horizon+gap))
                
                ty_hat = tt.back_transform(ttrain.tolist() + [ty_hat_temp])[-horizon]

                y = val[0]

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

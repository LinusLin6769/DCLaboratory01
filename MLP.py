# from __main__ import datasets, v_size, t_size, horizon, score
# from __main__ import MLP_policies as policies
from dc_transformation import DCTransformer
from parallel_validation import ParallelValidation
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import ConvergenceWarning as skConvWarn
from multiprocessing import Pool, Process, Pipe
from typing import List, Tuple, Dict
from itertools import product
from tqdm import trange, tqdm
from p_tqdm import p_map
from copy import copy
import data_prep, ts_analysis
import warnings
import numpy as np
import pandas as pd



def run_mlp(datasets, v_size, t_size, horizon, score, policies, n_workers) -> Tuple[Dict]:
    raw_info = {}
    tran_info = {}

    for i, entry in tqdm(datasets.items(), desc='Running MLP'):
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
        split = n_val + n_test

        # suppress convergence warning during validation
        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=skConvWarn)
            warnings.filterwarnings(action='ignore', category=UserWarning)
            
            with Pool(processes=n_workers) as p:
                arg = {
                    'series': series,
                    'n_val': n_val,
                    'split': split,
                    'horizon': horizon,
                    'score': score
                }
                par_val = ParallelValidation(arg, model='MLP')
                res = tqdm(
                    iterable=p.imap(par_val.run_parallel, policies),
                    desc=f'Validating series {i}',
                    total=len(policies))
                res = list(res)
        
        raw_policy_errs, tran_policy_errs = zip(*res)
        raw_policy_errs = [np.mean(e) for e in raw_policy_errs]
        tran_policy_errs = [np.mean(e) for e in tran_policy_errs]
        
        # model selection with all the validation errors
        best_raw_val_SMAPE_ind = np.argmin(raw_policy_errs)
        best_tran_val_SMAPE_ind = np.argmin(tran_policy_errs)
        best_raw_val_SMAPE = raw_policy_errs[best_raw_val_SMAPE_ind]
        best_tran_val_SMAPE = tran_policy_errs[best_tran_val_SMAPE_ind]
        best_raw_policy = copy(policies[best_raw_val_SMAPE_ind])
        best_tran_policy = copy(policies[best_tran_val_SMAPE_ind])
        del best_raw_policy['thres up']
        del best_raw_policy['thres down']

        #
        # test
        #
        raw_test_errs = []
        tran_test_errs = []
        raw_y_hats = []
        tran_y_hats = []

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=skConvWarn)
            for j in trange(n_test, desc=f'Series {i}, testing'):
                if j == n_test-1:
                    train_v = series
                else:
                    train_v = series[:-n_test+j+1]
                train = train_v[:-horizon]
                val = train_v[-horizon:]

                # raw
                rX, ry = data_prep.ts_prep(train, nlag=best_raw_policy['n lag'], horizon=horizon)
                train_X, val_X = rX, train[-best_raw_policy['n lag']:]
                train_y, val_y = ry, val
                
                # converge to linear regression if no hidden layer
                if best_raw_policy['struc'] == (0, ):
                    rmodel = LinearRegression()
                else:
                    rmodel = MLPRegressor(hidden_layer_sizes=best_raw_policy['struc'], max_iter=best_raw_policy['max iter'], random_state=1)
                
                rmodel.fit(train_X, train_y.ravel())
                y, y_hat = val_y[0], rmodel.predict([val_X])[0]
                raw_test_errs.append(score(y, y_hat))
                raw_y_hats.append(y_hat)

                # with transformation
                # @NOTE: Estimation of sigma can be improved!!!
                sigma = np.std(np.diff(np.log(train)))
                thres = (sigma*best_tran_policy['thres up'], -sigma*best_tran_policy['thres down'])
                t = DCTransformer()
                t.transform(train, threshold=thres)
                ttrain = t.tdata1

                tX, ty = data_prep.ts_prep(ttrain, nlag=best_tran_policy['n lag'], horizon=horizon)
                ttrain_X, tval_X = tX, ttrain[-best_tran_policy['n lag']:]
                ttrain_y, val_y = ty, val

                # converge to linear regression if no hidden layer
                if best_tran_policy['struc'] == (0, ):
                    tmodel = LinearRegression()
                else:
                    tmodel = MLPRegressor(hidden_layer_sizes=best_tran_policy['struc'], max_iter=best_tran_policy['max iter'], random_state=1)
                
                tmodel.fit(ttrain_X, ttrain_y.ravel())
                y, ty_hat = val_y[0], tmodel.predict([tval_X])[0]
                tran_test_errs.append(score(y, ty_hat))
                tran_y_hats.append(ty_hat)

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

if __name__ == '__main__':
    pass

from dc_transformation import DCTransformer
from target_transformation import TargetTransformation
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

def run_rf_tran(pool, arg, policies, ind):

    par_val = ParallelValidation(arg, model='RF', type='tran')
    res = tqdm(
        iterable=pool.imap(par_val.run_parallel, policies),
        desc=f'Tran: validating series {ind}',
        total=len(policies))

    return list(res)

def run_rf_raw(pool, arg, policies, ind):
    par_val = ParallelValidation(arg, model='RF', type='raw')
    res = tqdm(
        iterable=pool.imap(par_val.run_parallel, policies),
        desc=f'Raw: validating series {ind}',
        total=len(policies))

    return list(res)


def run_rf(datasets, ttype, v_size, retrain_window, t_size, horizon, gap, score, policies, n_workers) -> Tuple[Dict]:

    raw_info = {}
    tran_info = {}

    for i, entry in tqdm(datasets.items(), desc='Running RF'):
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
            warnings.filterwarnings(action='ignore', category=skConvWarn)
            warnings.filterwarnings(action='ignore', category=UserWarning)

            raw_policies = policies['raw']
            tran_policies = policies['tran']

            with Pool(processes=n_workers) as p:
                arg = {
                    'series': series,
                    'ttype': ttype,
                    'n_val': n_val,
                    'retrain_window': retrain_window,
                    'split': split,
                    'horizon': horizon,
                    'gap': gap,
                    'score': score
                }

                raw_policy_errs = run_rf_raw(pool=p, arg=arg, policies=raw_policies, ind=i)

                tran_policy_errs = run_rf_tran(pool=p, arg=arg, policies=tran_policies, ind=i)

        raw_policy_errs = [np.mean(e) for e in raw_policy_errs]
        tran_policy_errs = [np.mean(e) for e in tran_policy_errs]
        
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
        raw_feature_importances = []
        tran_feature_importances = []

        with warnings.catch_warnings():
            warnings.filterwarnings(action='ignore', category=skConvWarn)
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
                rX, ry = data_prep.ts_prep(train, nlag=best_raw_policy['n lag'], horizon=horizon, gap=gap)
                train_X, val_X = rX, train[-best_raw_policy['n lag']:]
                train_y, val_y = ry, val

                if j % retrain_window == 0:
                    rmodel = RandomForestRegressor(
                        max_depth=best_raw_policy['max depth'],
                        min_samples_split=best_raw_policy['min samples split'],
                        min_impurity_decrease=best_raw_policy['min impurity decrease'],
                        ccp_alpha=best_raw_policy['ccp alpha']
                    )
                    rmodel.fit(train_X, train_y.ravel())

                # prediction
                y_hat_temp = rmodel.predict([val_X])[0]

                # back transformation
                y_hat = tt.back_transform(train.tolist() + [y_hat_temp])[-horizon]
                
                # validation
                y = val_y[0]

                raw_test_errs.append(score(y, y_hat))
                raw_y_hats.append(y_hat)
                raw_feature_importances.append(rmodel.feature_importances_.tolist())

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

                tX, ty = data_prep.ts_prep(ttrain, nlag=best_tran_policy['n lag'], horizon=horizon, gap=gap)

                if best_tran_policy['use states']:
                    tstates = t.status[best_tran_policy['n lag']-1:]
                    tstates_onehot = data_prep.one_hot(tstates, list(t.STATUS_CODE.keys()))
                    tX_states = tstates_onehot[:-horizon]
                    tval_X_states = tstates_onehot[-horizon]

                    ttrain_X, tval_X = np.append(tX, tX_states, axis=1), np.append(ttrain[-best_tran_policy['n lag']:], tval_X_states, axis=0)
                else:
                    ttrain_X, tval_X = tX, ttrain[-best_tran_policy['n lag']:]

                ttrain_y, val_y = ty, val

                if j % retrain_window == 0:
                    tmodel = RandomForestRegressor(
                        max_depth=best_tran_policy['max depth'],
                        min_samples_split=best_tran_policy['min samples split'],
                        min_impurity_decrease=best_tran_policy['min impurity decrease'],
                        ccp_alpha=best_tran_policy['ccp alpha']
                    )
                    tmodel.fit(ttrain_X, ttrain_y.ravel())

                ty_hat_temp = tmodel.predict([tval_X])[0]
                
                ty_hat = tt.back_transform(ttrain.tolist() + [ty_hat_temp])[-horizon]

                y = val_y[0]

                tran_test_errs.append(score(y, ty_hat))
                tran_y_hats.append(ty_hat)
                tran_feature_importances.append(tmodel.feature_importances_.tolist())

        raw_info[i] = {
            'message': None,  # placeholder for other information
            'test SMAPE': round(np.mean(raw_test_errs), 6),
            'val SMAPE': round(best_raw_val_SMAPE, 6),
            'best model': best_raw_policy,
            'y hats': raw_y_hats,  # probably should put elsewhere
            'feature importances': raw_feature_importances
        }

        tran_info[i] = {
            'message': None,  # placeholder for other information
            'test SMAPE': round(np.mean(tran_test_errs), 6),
            'val SMAPE': round(best_tran_val_SMAPE, 6),
            'best model': best_tran_policy,
            'y hats': tran_y_hats,  # probably should put elsewhere
            'feature importances': tran_feature_importances
        }
    
    return raw_info, tran_info

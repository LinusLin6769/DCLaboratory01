from dc_transformation import DCTransformer
from parallel_validation import ParallelValidation
from typing import List, Tuple, Dict
from itertools import product
import lightgbm as lgbm
from tqdm import trange, tqdm
from p_tqdm import p_map
from copy import copy
import data_prep, ts_analysis
import warnings
import numpy as np
import pandas as pd

def run_lgbm(datasets, v_size, retrain_window, t_size, horizon, score, policies, n_workers) -> Tuple[Dict]:
    raw_info = {}
    tran_info = {}

    for i, entry in tqdm(datasets.items(), desc='Running LGBM'):
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
            old_settings = np.seterr(all='ignore')
            warnings.filterwarnings(action='ignore', category=UserWarning)

            raw_policies = policies['raw']
            tran_policies = policies['tran']

            # raw
            for policy in tqdm(raw_policies, desc=f'Raw: validating series {i}'):
                raw_val_errs = []

                # n_val folds rolling validation
                for v in range(n_val):
                    train_v = series[:-split+v+1]
                    train = train_v[:-horizon]
                    val = train_v[-horizon:]

                    # raw
                    rX, ry = data_prep.ts_prep(train, nlag=policy['n lag'], horizon=horizon)
                    train_X, val_X = rX, train[-policy['n lag']:]
                    train_y, val_y = ry, val
                    
                    if v % retrain_window == 0:
                        rmodel = lgbm.LGBMRegressor(
                            max_depth=policy['max depth'],
                            min_split_gain=policy['min split gain'],
                            importance_type=policy['importance type'],
                            n_jobs=n_workers,
                            random_state=0
                        )
                        rmodel.fit(train_X, train_y.ravel())

                    y, y_hat = val_y[0], rmodel.predict([val_X])[0]
                    raw_val_errs.append(score(y, y_hat))

                raw_policy_errs.append(np.nanmean(raw_val_errs))

            # tran
            for policy in tqdm(tran_policies, desc=f'Tran: validating series {i}'):
                tran_val_errs = []

                # n_val folds rolling validation
                for v in range(n_val):
                    train_v = series[:-split+v+1]
                    train = train_v[:-horizon]
                    val = train_v[-horizon:]

                    # with transformation
                    # @NOTE: Estimation of sigma can be improved!!!
                    sigma = np.std(np.diff(np.log(train)))
                    thres = (sigma*policy['thres up'], -sigma*policy['thres down'])
                    t = DCTransformer()
                    t.transform(train, threshold=thres, kind=policy['interp kind'])
                    ttrain = t.tdata1

                    if len(ttrain) > 1:
                        tX, ty = data_prep.ts_prep(ttrain, nlag=policy['n lag'], horizon=horizon)
                        if policy['use states']:
                            tstates = t.status[policy['n lag']-1:]
                            tstates_onehot = data_prep.one_hot(tstates, list(t.STATUS_CODE.keys()))
                            tX_states = tstates_onehot[:-horizon]
                            tval_X_states = tstates_onehot[-horizon]
                            
                            ttrain_X, tval_X = np.append(tX, tX_states, axis=1), np.append(ttrain[-policy['n lag']:], tval_X_states, axis=0)
                        else:
                            ttrain_X, tval_X = tX, ttrain[-policy['n lag']:]

                        ttrain_y, val_y = ty, val

                        if v % retrain_window == 0:
                            tmodel = lgbm.LGBMRegressor(
                            max_depth=policy['max depth'],
                            min_split_gain=policy['min split gain'],
                            importance_type=policy['importance type'],
                            n_jobs=n_workers,
                            random_state=0
                            )
                            tmodel.fit(ttrain_X, ttrain_y.ravel())
                        y, ty_hat = val_y[0], tmodel.predict([tval_X])[0]
                        tran_val_errs.append(score(y, ty_hat))
                        
                    else:
                        tran_val_errs.append(0.999)

                tran_policy_errs.append(np.nanmean(tran_val_errs))

            np.seterr(**old_settings)

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

        # with warnings.catch_warnings():
        for j in trange(n_test, desc=f'Testing series {i}'):
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
            
            if j % retrain_window == 0:
                rmodel = lgbm.LGBMRegressor(
                    max_depth=best_raw_policy['max depth'],
                    min_split_gain=best_raw_policy['min split gain'],
                    importance_type=best_raw_policy['importance type'],
                    random_state=0
                )
                rmodel.fit(train_X, train_y.ravel())

            y, y_hat = val_y[0], rmodel.predict([val_X])[0]
            raw_test_errs.append(score(y, y_hat))
            raw_y_hats.append(y_hat)

            # with transformation
            """Transformation has to be improved!!!"""
            sigma = np.std(np.diff(np.log(train)))
            thres = (sigma*best_tran_policy['thres up'], -sigma*best_tran_policy['thres down'])
            t = DCTransformer()
            t.transform(train, threshold=thres, kind=best_tran_policy['interp kind'])
            ttrain = t.tdata1

            tX, ty = data_prep.ts_prep(ttrain, nlag=best_tran_policy['n lag'], horizon=horizon)

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
                tmodel = lgbm.LGBMRegressor(
                    max_depth=best_tran_policy['max depth'],
                    min_split_gain=best_tran_policy['min split gain'],
                    importance_type=best_tran_policy['importance type'],
                    random_state=0
                )
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

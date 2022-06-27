from __main__ import datasets, v_size, t_size, horizon, score
from __main__ import ETS_policies as policies
from dc_transformation import DCTransformer
from statsmodels.tools.sm_exceptions import InterpolationWarning
from statsmodels.tools.sm_exceptions import ConvergenceWarning as statsConvWarn
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.exceptions import ConvergenceWarning as skConvWarn
from itertools import product
from tqdm import trange, tqdm
from copy import copy
import data_prep, ts_analysis
import warnings
import numpy as np
import pandas as pd

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
                'message': 'Existence of value <= 0, skipped.'
            }
            tran_info[i] = {
                'message': 'Existene of value <= 0, skipped.'
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
    raw_policy_errs = []
    tran_policy_errs = []

    # suppress convergence warning during validation
    with warnings.catch_warnings():
        old_settings = np.seterr(over='ignore')  # suppress overflow warnings
        warnings.filterwarnings(action='ignore', category=statsConvWarn)

        for policy in tqdm(policies, desc=f'Series {i}, validating'):
            raw_val_errs = []
            tran_val_errs = []

            # n_val folds rolling validation
            for v in range(n_val):
                train_v = series[:-split+v+1]
                train = train_v[:-horizon]
                val = train_v[-horizon:]
                    
                # raw
                rmodel = ExponentialSmoothing(
                    train,
                    seasonal_periods=policy['seasonal periods'],
                    trend=policy['trend'],
                    seasonal=policy['seasonal'],
                    damped_trend=policy['damped trend']
                ).fit()
                y, y_hat = val[0], rmodel.forecast(horizon).tolist()[0] # the forecast function returns np.array, which is not acceptable for json
                raw_val_errs.append(score(y, y_hat))

                # with transformation
                """Transformation has to be improved!!!"""
                sigma = np.std(np.diff(np.log(train)))
                thres = (sigma*policy['thres up'], -sigma*policy['thres down'])
                t = DCTransformer()
                t.transform(train, threshold=thres)
                ttrain = t.tdata1

                tmodel = ExponentialSmoothing(
                    ttrain,
                    seasonal_periods=policy['seasonal periods'],
                    trend=policy['trend'],
                    seasonal=policy['seasonal'],
                    damped_trend=policy['damped trend']
                ).fit()
                y, ty_hat = val[0], tmodel.forecast(horizon).tolist()[0]
                tran_val_errs.append(score(y, ty_hat))

            raw_policy_errs.append(np.mean(raw_val_errs))
            tran_policy_errs.append(np.mean(tran_val_errs))
        np.seterr(**old_settings)  # restore the warning settings
    
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
        old_settings = np.seterr(over='ignore')
        warnings.filterwarnings(action='ignore', category=statsConvWarn)
        for j in trange(n_test, desc=f'Series {i}, testing'):
            if j == n_test-1:
                train_v = series
            else:
                train_v = series[:-n_test+j+1]
            train = train_v[:-horizon]
            val = train_v[-horizon:]

            # raw
            rmodel = ExponentialSmoothing(
                train,
                seasonal_periods=policy['seasonal periods'],
                trend=policy['trend'],
                seasonal=policy['seasonal'],
                damped_trend=policy['damped trend']
            ).fit()
            y, y_hat = val[0], rmodel.forecast(horizon).tolist()[0]
            raw_test_errs.append(score(y, y_hat))
            raw_y_hats.append(y_hat)

            # with transformation
            """Transformation has to be improved!!!"""
            sigma = np.std(np.diff(np.log(train)))
            thres = (sigma*best_tran_policy['thres up'], -sigma*best_tran_policy['thres down'])
            t = DCTransformer()
            t.transform(train, threshold=thres)
            ttrain = t.tdata1

            tmodel = ExponentialSmoothing(
                ttrain,
                seasonal_periods=policy['seasonal periods'],
                trend=policy['trend'],
                seasonal=policy['seasonal'],
                damped_trend=policy['damped trend']
            ).fit()
            y, ty_hat = val[0], tmodel.forecast(horizon).tolist()[0]
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

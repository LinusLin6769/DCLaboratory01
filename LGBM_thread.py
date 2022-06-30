from dc_transformation import DCTransformer
from parallel_validation import ParallelValidation
from threading import Thread
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

# generator function that creates equal chunk size of indices for the tasks
def split_tasks(tasks, n_chunk):
    p, q = divmod(len(tasks), n_chunk)
    return (tasks[i*p + min(i, q):(i+1)*p + min(i+1, q)] for i in range(n_chunk))

def run_chunk_tasks(indices, all_tasks, func, res):
    print('run chunk tasks is called')
    print(indices, all_tasks, func, res)
    for i in indices:
        res.append(func(all_tasks[i]))
    print(res)

def run_lgbm(datasets, v_size, t_size, horizon, score, policies, n_workers) -> Tuple[Dict]:

    # @NOTE: Do we have other better choices of number of threads???
    n_threads = n_workers*2
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
        split = n_val + n_test

        raw_policy_errs = []
        tran_policy_errs = []
        res = []
        # suppress convergence warning during validation
        with warnings.catch_warnings():
            old_settings = np.seterr(all='ignore')
            warnings.filterwarnings(action='ignore', category=UserWarning)
            
            # split the tasks (validation of the policies) into chunks
            n_tasks = len(policies)
            chunks = list(split_tasks(range(n_tasks), n_threads))

            arg = {
                'series': series,
                'n_val': n_val,
                'split': split,
                'horizon': horizon,
                'score': score
            }
            par_val = ParallelValidation(arg, model='LGBM')
            thread_tasks = []

            for chunk in chunks:
                # create the thread and its respective tasks designating in a chunk
                thread_tasks.append(Thread(target=run_chunk_tasks, args=(chunk, policies, par_val.run_parallel, res)))

            for t in tqdm(thread_tasks, desc=f'Validating series {i}'):
                t.start()
            
            print('tasks are finished')
            print(res)
            for t in thread_tasks:
                t.join()
            
            print('tasks are closed')
            print(res)
            np.seterr(**old_settings)

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
            t.transform(train, threshold=thres)
            ttrain = t.tdata1

            tX, ty = data_prep.ts_prep(ttrain, nlag=best_tran_policy['n lag'], horizon=horizon)
            ttrain_X, tval_X = tX, ttrain[-best_tran_policy['n lag']:]
            ttrain_y, val_y = ty, val

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

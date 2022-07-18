# Responsible for running sklearn regressors: EN, MLP, LSVR, RF

from dc_transformation import DCTransformer
from target_transformation import TargetTransformation
from parallel_validation import ParallelValidation
from sklearn.linear_model import ElasticNet
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from sklearn.ensemble import RandomForestRegressor

from sklearn.exceptions import ConvergenceWarning as skConvWarn
from multiprocessing import Pool, Process, Pipe
from typing import List, Tuple, Dict, Callable, Sequence, Any, Union
from itertools import product
from p_tqdm import p_map
from tqdm import trange, tqdm
from copy import copy
import data_prep, ts_analysis
import warnings
import numpy as np
import pandas as pd

class RunModel:
    def __init__(self, model, datasets, ttype, v_size, retrain_window, t_size, horizon, gap, score, policies, n_workers) -> None:

        self.model: str                 = model
        self.datasets: Sequence         = datasets
        self.ttype: str                 = ttype
        self.v_size: Union[float, int]  = v_size
        self.retrain_window: int        = retrain_window
        self.t_size: Union[float, int]  = t_size
        self.horizon: int               = horizon
        self.gap: int                   = gap
        self.score: Callable            = score
        self.policies: List[Dict]       = policies
        self.n_workers: int             = n_workers

    def run_tran(self,
            model: str,
            pool: Callable,
            arg: Dict[str, Any],
            policies:List[Dict],
            ind: int
        ) -> List[List[float]]:

        par_val = ParallelValidation(arg, model=model, type='tran')
        res = tqdm(
            iterable=pool.imap(par_val.run_parallel, policies),
            desc=f'Tran: validating series {ind}',
            total=len(policies))

        return list(res)

    def run_raw(self,
            model: str,
            pool: Callable,
            arg: Dict[str, Any],
            policies: List[Dict],
            ind: int
        ) -> List[List[float]]:

        par_val = ParallelValidation(arg, model=model, type='raw')
        res = tqdm(
            iterable=pool.imap(par_val.run_parallel, policies),
            desc=f'Raw: validating series {ind}',
            total=len(policies))

        return list(res)


    def run_model(self) -> Tuple[Dict]:

        raw_info = {}
        tran_info = {}

        for i, entry in tqdm(self.datasets.items(), desc='Running EN'):
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
            n_test = self.t_size if type(self.t_size) == int else int(self.t_size * N)
            n_val = self.v_size if type(self.v_size) == int else int(self.v_size * N)
            retrain_window = self.retrain_window if type(self.retrain_window) == int else int(self.retrain_window * N)
            split = n_val + n_test
            raw_policy_errs = []
            tran_policy_errs = []

            # suppress convergence warning during validation
            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=skConvWarn)
                warnings.filterwarnings(action='ignore', category=UserWarning)

                raw_policies = self.policies['raw']
                tran_policies = self.policies['tran']

                with Pool(processes=self.n_workers) as p:
                    arg = {
                        'series': series,
                        'ttype': self.ttype,
                        'n_val': n_val,
                        'retrain_window': retrain_window,
                        'split': split,
                        'horizon': self.horizon,
                        'gap': self.gap,
                        'score': self.score
                    }

                    raw_policy_errs = self.run_raw(model=self.model, pool=p, arg=arg, policies=raw_policies, ind=i)
                    tran_policy_errs = self.run_tran(model=self.model, pool=p, arg=arg, policies=tran_policies, ind=i)

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

            raw_coeffs = []
            tran_coeffs = []

            raw_feature_importances = []
            tran_feature_importances = []

            with warnings.catch_warnings():
                warnings.filterwarnings(action='ignore', category=skConvWarn)
                for j in trange(n_test, desc=f'Testing series {i}'):
                    if j == n_test-self.horizon:
                        train_v = series
                    else:
                        train_v = series[:-n_test+j+self.horizon]
                    
                    train = train_v[:-self.horizon-self.gap]
                    val = train_v[-self.horizon:]

                    # target transformation
                    tt = TargetTransformation(type=self.ttype)
                    train = tt.transform(train)

                    # raw
                    rX, ry = data_prep.ts_prep(train, nlag=best_raw_policy['n lag'], horizon=self.horizon, gap=self.gap)
                    train_X, val_X = rX, train[-best_raw_policy['n lag']:]
                    train_y, val_y = ry, val

                    if j % retrain_window == 0:
                        # model switch
                        if self.model == 'EN':
                            rmodel = ElasticNet(alpha=best_raw_policy['alpha'], l1_ratio=best_raw_policy['l1 ratio'], random_state=0)
                            rmodel.fit(train_X, train_y)

                        elif self.model == 'MLP':
                            # converge to linear regression if no hidden layer
                            if best_raw_policy['struc'] == (0, ):
                                rmodel = LinearRegression()
                            else:
                                rmodel = MLPRegressor(hidden_layer_sizes=best_raw_policy['struc'], max_iter=best_raw_policy['max iter'], random_state=1)
                            rmodel.fit(train_X, train_y.ravel())
                        
                        elif self.model == 'LSVR':
                            rmodel = LinearSVR(
                                tol=best_raw_policy['tol'],
                                C=best_raw_policy['c']
                            )
                            rmodel.fit(train_X, train_y.ravel())

                        elif self.model == 'RF':
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
                    y_hat = tt.back_transform(train.tolist() + [y_hat_temp])[-self.horizon]
                    
                    # validation
                    y = val_y[0]

                    raw_test_errs.append(self.score(y, y_hat))
                    raw_y_hats.append(y_hat)

                    if self.model == 'EN':
                        c = list(rmodel.intercept_) + rmodel.coef_.tolist()
                        raw_coeffs.append(c)
                    elif self.model == 'RF':
                        raw_feature_importances.append(rmodel.feature_importances_.tolist())

                    train = train_v[:-self.horizon-self.gap]

                    # with transformation
                    # @NOTE: Estimation of sigma can be improved!!!
                    sigma = np.std(np.diff(np.log(train)))
                    thres = (sigma*best_tran_policy['thres up'], -sigma*best_tran_policy['thres down'])
                    t = DCTransformer()
                    t.transform(train, threshold=thres, kind=best_tran_policy['interp kind'])
                    ttrain = t.tdata1

                    # target transformation
                    tt = TargetTransformation(type=self.ttype)
                    ttrain = tt.transform(ttrain)

                    tX, ty = data_prep.ts_prep(ttrain, nlag=best_tran_policy['n lag'], horizon=self.horizon, gap=self.gap)

                    if best_tran_policy['use states']:
                        tstates = t.status[best_tran_policy['n lag']-1:]
                        tstates_onehot = data_prep.one_hot(tstates, list(t.STATUS_CODE.keys()))
                        tX_states = tstates_onehot[:-self.horizon]
                        tval_X_states = tstates_onehot[-self.horizon]

                        ttrain_X, tval_X = np.append(tX, tX_states, axis=1), np.append(ttrain[-best_tran_policy['n lag']:], tval_X_states, axis=0)
                    else:
                        ttrain_X, tval_X = tX, ttrain[-best_tran_policy['n lag']:]

                    ttrain_y, val_y = ty, val

                    if j % retrain_window == 0:
                        if self.model == 'EN':
                            tmodel = ElasticNet(alpha=best_tran_policy['alpha'],l1_ratio=best_tran_policy['l1 ratio'], random_state=0)
                            tmodel.fit(ttrain_X, ttrain_y)
                        
                        elif self.model == 'MLP':
                            # converge to linear regression if no hidden layer
                            if best_tran_policy['struc'] == (0, ):
                                tmodel = LinearRegression()
                            else:
                                tmodel = MLPRegressor(hidden_layer_sizes=best_tran_policy['struc'], max_iter=best_tran_policy['max iter'], random_state=1)
                            tmodel.fit(ttrain_X, ttrain_y.ravel())

                        elif self.model == 'LSVR':
                            tmodel = LinearSVR(
                                tol=best_tran_policy['tol'],
                                C=best_tran_policy['c']
                            )
                            tmodel.fit(ttrain_X, ttrain_y.ravel())

                        elif self.model == 'RF':
                            tmodel = RandomForestRegressor(
                                max_depth=best_tran_policy['max depth'],
                                min_samples_split=best_tran_policy['min samples split'],
                                min_impurity_decrease=best_tran_policy['min impurity decrease'],
                                ccp_alpha=best_tran_policy['ccp alpha']
                            )
                            tmodel.fit(ttrain_X, ttrain_y.ravel())

                    ty_hat_temp = tmodel.predict([tval_X])[0]
                    
                    ty_hat = tt.back_transform(ttrain.tolist() + [ty_hat_temp])[-self.horizon]

                    y = val_y[0]

                    tran_test_errs.append(self.score(y, ty_hat))
                    tran_y_hats.append(ty_hat)

                    if self.model == 'EN':
                        c = list(tmodel.intercept_) + tmodel.coef_.tolist()
                        tran_coeffs.append(c)
                    elif self.model == 'RF':
                        tran_feature_importances.append(tmodel.feature_importances_.tolist())

            raw_info[i] = {
                'message': None,  # placeholder for other information
                'test SMAPE': round(np.mean(raw_test_errs), 6),
                'val SMAPE': round(best_raw_val_SMAPE, 6),
                'best model': best_raw_policy,
                'y hats': raw_y_hats,  # probably should put elsewhere
            }

            tran_info[i] = {
                'message': None,  # placeholder for other information
                'test SMAPE': round(np.mean(tran_test_errs), 6),
                'val SMAPE': round(best_tran_val_SMAPE, 6),
                'best model': best_tran_policy,
                'y hats': tran_y_hats,  # probably should put elsewhere
            }

            if self.model == 'EN':
                raw_info[i]['coeffs'] = raw_coeffs
                tran_info[i]['coeffs'] = tran_coeffs
            elif self.model == 'RF':
                raw_info[i]['feature importances'] = raw_feature_importances
                tran_info[i]['feature importances'] = tran_feature_importances
        
        return raw_info, tran_info

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

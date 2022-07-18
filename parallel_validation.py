from pmdarima import auto_arima
from dc_transformation import DCTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import LinearSVR
from statsmodels.tsa.api import ExponentialSmoothing
from sktime.forecasting.ets import AutoETS
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from target_transformation import TargetTransformation
import xgboost as xgb
import lightgbm as lgbm
from typing import Any, Callable, List, Sequence, Tuple, Dict
import data_prep, ts_analysis
import numpy as np
import pandas as pd

class ParallelValidation:
    def __init__(self, args: Dict, model: str, type: str) -> None:
        self.args = args
        self.model = model
        self.type = type

        self.classes = {
            'skl regressors': ['EN', 'MLP', 'LSVR', 'RF'],
            'sktime forecasters': ['ETS'],
            'manual forecasters': ['MA']}

    def run_parallel(self, p: List[Dict]) -> List[float]:
        if self.model in self.classes['skl regressors']:
            if self.type == 'raw':
                return self.skl_val_raw(policy=p, **self.args)

            elif self.type == 'tran':
                return self.skl_val_tran(policy=p, **self.args)
            
            else:
                raise ValueError('Invalid model type. Should be either raw or tran.')
        
        elif self.model == 'ETS':
            if self.type == 'raw':
                return self.val_raw_ets(policy=p, **self.args)
        
            elif self.type == 'tran':
                return self.val_tran_ets(policy=p, **self.args)
        
            else:
                raise ValueError('Invalid model type. Should be either raw or tran')

        elif self.model == 'MA':
            if self.type == 'raw':
                return self.val_raw_ma(policy=p, **self.args)
        
            elif self.type == 'tran':
                return self.val_tran_ma(policy=p, **self.args)
        
            else:
                raise ValueError('Invalid model type. Should be either raw or tran')

    def skl_val_raw(self,
            policy: Dict[str, Any],
            series: Sequence,
            ttype: str,
            n_val: int,
            retrain_window: int,
            split: int,
            horizon: int,
            gap: int,
            score: Callable
        ) -> Tuple[List]:
        
        raw_val_errs = []

        # n_val folds rolling validation
        for v in range(n_val):
            train_v = series[:-split+v+1]  #  includes the validation point that should be excluded during transformation
            train = train_v[:-horizon-gap]
            val = train_v[-horizon:]
        
            # target transformation
            tt = TargetTransformation(type=ttype)
            train = tt.transform(train)

            # raw
            rX, ry = data_prep.ts_prep(train, nlag=policy['n lag'], horizon=horizon, gap=gap)
            train_X, val_X = rX, train[-policy['n lag']:]
            train_y, val_y = ry, val

            if v % retrain_window == 0:
                
                # model switcher
                if self.model == 'EN':
                    rmodel = ElasticNet(
                        alpha=policy['alpha'],
                        l1_ratio=policy['l1 ratio'],
                        random_state=0)
                    rmodel.fit(train_X, train_y)

                elif self.model == 'MLP':
                    # converge to linear regression if no hidden layer
                    if policy['struc'] == (0, ):
                        rmodel = LinearRegression()
                    else:
                        rmodel = MLPRegressor(hidden_layer_sizes=policy['struc'], max_iter=policy['max iter'], random_state=1)
                    rmodel.fit(train_X, train_y.ravel())

                elif self.model == 'LSVR':
                    rmodel = LinearSVR(
                        tol=policy['tol'],
                        C=policy['c']
                    )
                    rmodel.fit(train_X, train_y.ravel())

                elif self.model == 'RF':
                    rmodel = RandomForestRegressor(
                        max_depth=policy['max depth'],
                        min_samples_split=policy['min samples split'],
                        min_impurity_decrease=policy['min impurity decrease'],
                        ccp_alpha=policy['ccp alpha']
                    )
                    rmodel.fit(train_X, train_y.ravel())

            # prediction, the models here share the same syntax in predicting
            y_hat_temp = rmodel.predict([val_X])[0]

            # back transformation
            y_hat = tt.back_transform(train.tolist() + [y_hat_temp])[-horizon]
            
            # validation
            y = val_y[0]

            raw_val_errs.append(score(y, y_hat))

        return raw_val_errs

    def skl_val_tran(self,
            policy: Dict[str, Any],
            series: Sequence,
            ttype: str,
            n_val: int,
            retrain_window: int,
            split: int,
            horizon: int,
            gap: int,
            score: Callable
        ) -> Tuple[List]:
        
        tran_val_errs = []

        # n_val folds rolling validation
        for v in range(n_val):
            train_v = series[:-split+v+1]  #  includes the validation point that should be excluded during transformation
            train = train_v[:-horizon-gap]
            val = train_v[-horizon:]

            # with transformation
            # @NOTE: Estimation of sigma can be improved!!!
            sigma = np.std(np.diff(np.log(train)))
            thres = (sigma*policy['thres up'], -sigma*policy['thres down'])
            t = DCTransformer()
            t.transform(train, threshold=thres, kind=policy['interp kind'])
            ttrain = t.tdata1

            # target transformation
            tt = TargetTransformation(type=ttype)
            ttrain = tt.transform(ttrain)

            if len(ttrain) > 1:
                tX, ty = data_prep.ts_prep(ttrain, nlag=policy['n lag'], horizon=horizon, gap=gap)

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
                    # model switch
                    if self.model == 'EN':
                        tmodel = ElasticNet(
                            alpha=policy['alpha'],
                            l1_ratio=policy['l1 ratio'],
                            random_state=0
                        )
                        tmodel.fit(ttrain_X, ttrain_y)

                    elif self.model == 'MLP':
                        # coverge to linear regression if no hidden layer
                        if policy['struc'] == (0, ):
                            tmodel = LinearRegression()
                        else:
                            tmodel = MLPRegressor(hidden_layer_sizes=policy['struc'], max_iter=policy['max iter'], random_state=1)
                        tmodel.fit(ttrain_X, ttrain_y.ravel())

                    elif self.model == 'LSVR':
                        tmodel = LinearSVR(
                            tol=policy['tol'],
                            C=policy['c']
                        )
                        tmodel.fit(ttrain_X, ttrain_y.ravel())

                    elif self.model == 'RF':
                        tmodel = RandomForestRegressor(
                            max_depth=policy['max depth'],
                            min_samples_split=policy['min samples split'],
                            min_impurity_decrease=policy['min impurity decrease'],
                            ccp_alpha=policy['ccp alpha']
                        )
                        tmodel.fit(ttrain_X, ttrain_y.ravel())

                # prediction
                ty_hat_temp = tmodel.predict([tval_X])[0]
                
                ty_hat = tt.back_transform(ttrain.tolist() + [ty_hat_temp])[-horizon]

                y = val_y[0]

                tran_val_errs.append(score(y, ty_hat))
            else:
                tran_val_errs.append(0.999)
        
        return tran_val_errs
    
    def val_raw_ets(self, policy, series, ttype, n_val, retrain_window, split, horizon, gap, score) -> Tuple[List]:

        raw_val_errs = []

        # n_val folds rolling validation
        for v in range(n_val):
            train_v = series[:-split+v+1]
            train = train_v[:-horizon-gap]
            val = train_v[-horizon:]

            # target transformation
            tt = TargetTransformation(type=ttype)
            train = tt.transform(train)

            # raw
            if v % retrain_window == 0:
                rmodel = AutoETS(auto=policy['auto'])
                rmodel.fit(pd.Series(train))
            else:
                rmodel.update(pd.Series(train[-1], index=[len(train)-1]))

            # prediction
            y_hat_temp = float(rmodel.predict(horizon+gap))

            # back transformation
            y_hat = tt.back_transform(train.tolist() + [y_hat_temp])[-horizon]
            
            # validation
            y = val[0]

            raw_val_errs.append(score(y, y_hat))
        
        return raw_val_errs


    def val_tran_ets(self, policy, series, ttype, n_val, retrain_window, split, horizon, gap, score) -> Tuple[List]:

        tran_val_errs = []

        # n_val folds rolling validation
        for v in range(n_val):
            train_v = series[:-split+v+1]
            train = train_v[:-horizon-gap]
            val = train_v[-horizon:]

            # with transformation
            # @NOTE: Estimation of sigma can be improved!!!
            sigma = np.std(np.diff(np.log(train)))
            thres = (sigma*policy['thres up'], -sigma*policy['thres down'])
            t = DCTransformer()
            t.transform(train, threshold=thres, kind=policy['interp kind'])
            ttrain = t.tdata1

            # target transformation
            tt = TargetTransformation(type=ttype)
            ttrain = tt.transform(ttrain)

            if len(ttrain) > 1:

                if v % retrain_window == 0:
                    tmodel = AutoETS(auto=policy['auto'])
                    tmodel.fit(pd.Series(ttrain))
                else:
                    tmodel.update(pd.Series(ttrain[-1], index=[len(ttrain)-1]))

                ty_hat_temp = float(tmodel.predict(horizon+gap))
                
                ty_hat = tt.back_transform(ttrain.tolist() + [ty_hat_temp])[-horizon]

                y = val[0]

                tran_val_errs.append(score(y, ty_hat))
            else:
                tran_val_errs.append(0.999)
        
        return tran_val_errs
    
    def val_raw_ma(self, policy, series, ttype, n_val, retrain_window, split, horizon, gap, score) -> Tuple[List]:

        raw_val_errs = []

        # n_val folds rolling validation
        for v in range(n_val):
            train_v = series[:-split+v+1]
            train = train_v[:-horizon-gap]
            val = train_v[-horizon:]

            # target transformation
            tt = TargetTransformation(type=ttype)
            train = tt.transform(train)

            # raw

            # prediction
            y_hat_temp = np.mean(train[-policy['q']:])

            # back transformation
            y_hat = tt.back_transform(train.tolist() + [y_hat_temp])[-horizon]
            
            # validation
            y = val[0]

            raw_val_errs.append(score(y, y_hat))
        
        return raw_val_errs


    def val_tran_ma(self, policy, series, ttype, n_val, retrain_window, split, horizon, gap, score) -> Tuple[List]:

        tran_val_errs = []

        # n_val folds rolling validation
        for v in range(n_val):
            train_v = series[:-split+v+1]
            train = train_v[:-horizon-gap]
            val = train_v[-horizon:]

            # with transformation
            # @NOTE: Estimation of sigma can be improved!!!
            sigma = np.std(np.diff(np.log(train)))
            thres = (sigma*policy['thres up'], -sigma*policy['thres down'])
            t = DCTransformer()
            t.transform(train, threshold=thres, kind=policy['interp kind'])
            ttrain = t.tdata1

            # target transformation
            tt = TargetTransformation(type=ttype)
            ttrain = tt.transform(ttrain)

            if len(ttrain) > 1:
                
                # prediction
                ty_hat_temp = np.mean(ttrain[-policy['q']:])
                
                ty_hat = tt.back_transform(ttrain.tolist() + [ty_hat_temp])[-horizon]

                y = val[0]

                tran_val_errs.append(score(y, ty_hat))
            else:
                tran_val_errs.append(0.999)
        
        return tran_val_errs

    # @NOTE: the build for AutoARIMA is not complete due to its possibility of failing to fit. Exception handling is needed.
    def val_autoarima(self, policy, series, n_val, retrain_window, split, horizon, score) -> Tuple[List]:

        raw_val_errs = []
        tran_val_errs = []

        # n_val folds rolling validation
        for v in range(n_val):
            train_v = series[:-split+v+1]  #  includes the validation point that should be excluded during transformation
            train = train_v[:-horizon]
            val = train_v[-horizon:]
        
            # raw
            if v % retrain_window == 0:
                rmodel = auto_arima(train, suppress_warnings=True)
            else:
                rmodel.update([train[-1]])

            y, y_hat = val[0], rmodel.predict(n_periods=horizon, return_conf_int=False)[0]
            raw_val_errs.append(score(y, y_hat))

            # with transformation
            # @NOTE: Estimation of sigma can be improved!!!
            sigma = np.std(np.diff(np.log(train)))
            thres = (sigma*policy['thres up'], -sigma*policy['thres down'])
            t = DCTransformer()
            t.transform(train, threshold=thres, kind=policy['interp kind'])
            ttrain = t.tdata1

            if len(ttrain) > 1:

                if policy['use states']:
                    tstates = t.status
                    tstates_onehot = data_prep.one_hot(tstates, list(t.STATUS_CODE.keys()))
                    tX_states = tstates_onehot[:-horizon] # 2d
                    tval_X_states = tstates_onehot[-horizon] # 1d

                    if v % retrain_window == 0:
                        tmodel = auto_arima(ttrain, X=tX_states, suppress_warnings=True)
                    else:
                        tmodel.update([ttrain[-1]], X=[tX_states[-1]])
                    
                    # predict
                    y, ty_hat = val[0], tmodel.predict(n_periods=horizon, X=[tval_X_states])[0]
                else:
                    # without states
                    if v % retrain_window == 0:
                        tmodel = auto_arima(ttrain, suppress_warnings=True)
                    else:
                        tmodel.update([ttrain[-1]])
                    # predict
                    y, ty_hat = val[0], tmodel.predict(n_periods=horizon)[0]
                
                tran_val_errs.append(score(y, ty_hat))
            else:
                tran_val_errs.append(0.999)
        
        return raw_val_errs, tran_val_errs

    # @NOTE: Not used because the XGboost library has its own parallelism that conflicts with other customised parallel settings.
    def val_xgb(self, policy, series, n_val, retrain_window, split, horizon, score) -> Tuple[List]:
        raw_val_errs = []
        tran_val_errs = []

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
                rmodel = xgb.XGBRegressor(
                    max_depth=policy['max depth'],
                    booster=policy['booster'],
                    subsample=policy['subsample ratio'],
                    random_state=0
                )
                rmodel.fit(train_X, train_y)

            y, y_hat = val_y[0], rmodel.predict([val_X])[0]
            raw_val_errs.append(score(y, y_hat))

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
                    tmodel = xgb.XGBRegressor(
                    max_depth=policy['max depth'],
                    booster=policy['booster'],
                    subsample=policy['subsample ratio'],
                    random_state=0
                    )
                    tmodel.fit(ttrain_X, ttrain_y)
                y, ty_hat = val_y[0], tmodel.predict([tval_X])[0]
                tran_val_errs.append(score(y, ty_hat))
            else:
                tran_val_errs.append(0.999)
        
        return raw_val_errs, tran_val_errs

    # @NOTE: LightGBM doesn't work with multiprocessing. This function is not used. The reason is the same with the XGboost library.
    def val_lgbm(self, policy, series, n_val, retrain_window, split, horizon, score) -> Tuple[List]:
        raw_val_errs = []
        tran_val_errs = []

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
                    random_state=0
                )
                rmodel.fit(train_X, train_y.ravel())
            y, y_hat = val_y[0], rmodel.predict([val_X])[0]
            raw_val_errs.append(score(y, y_hat))

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
                    random_state=0
                    )
                    tmodel.fit(ttrain_X, ttrain_y.ravel())
                y, ty_hat = val_y[0], tmodel.predict([tval_X])[0]
                tran_val_errs.append(score(y, ty_hat))
            else:
                tran_val_errs.append(0.999)
        
        return raw_val_errs, tran_val_errs

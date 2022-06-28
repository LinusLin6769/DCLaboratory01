from dc_transformation import DCTransformer
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.linear_model import ElasticNet
import xgboost as xgb
import lightgbm as lgbm
from typing import List, Tuple, Dict
import data_prep, ts_analysis
import numpy as np
import pandas as pd

class ParallelValidation:
    def __init__(self, args: Dict, model: str) -> None:
        self.args = args
        self.model = model
        
        self.models = {
            'MLP': self.val_mlp,
            'ETS': self.val_ets,
            'EN': self.val_en,
            'XGB': self.val_xgb,
            'LGBM': self.val_lgbm
        }

    def run_parallel(self, p):
        return self.models[self.model](policy=p, **self.args)

    def val_mlp(self, policy, series, n_val, split, horizon, score) -> Tuple[List]:

        raw_val_errs = []
        tran_val_errs = []

        # n_val folds rolling validation
        for v in range(n_val):
            train_v = series[:-split+v+1]  #  includes the validation point that should be excluded during transformation
            train = train_v[:-horizon]
            val = train_v[-horizon:]
        
            # raw
            rX, ry = data_prep.ts_prep(train, nlag=policy['n lag'], horizon=horizon)
            train_X, val_X = rX, train[-policy['n lag']:]
            train_y, val_y = ry, val

            # converge to linear regression if no hidden layer
            if policy['struc'] == (0, ):
                rmodel = LinearRegression()
            else:
                rmodel = MLPRegressor(hidden_layer_sizes=policy['struc'], max_iter=policy['max iter'], random_state=1)

            rmodel.fit(train_X, train_y.ravel())
            y, y_hat = val_y[0], rmodel.predict([val_X])[0]
            raw_val_errs.append(score(y, y_hat))

            # with transformation
            """Transformation has to be improved!!!"""
            sigma = np.std(np.diff(np.log(train)))
            thres = (sigma*policy['thres up'], -sigma*policy['thres down'])
            t = DCTransformer()
            t.transform(train, threshold=thres)
            ttrain = t.tdata1

            if len(ttrain) > 1:
                tX, ty = data_prep.ts_prep(ttrain, nlag=policy['n lag'], horizon=horizon)
                ttrain_X, tval_X = tX, ttrain[-policy['n lag']:]
                ttrain_y, val_y = ty, val

                # coverge to linear regression if no hidden layer
                if policy['struc'] == (0, ):
                    tmodel = LinearRegression()
                else:
                    tmodel = MLPRegressor(hidden_layer_sizes=policy['struc'], max_iter=policy['max iter'], random_state=1)
                
                tmodel.fit(ttrain_X, ttrain_y.ravel())
                y, ty_hat = val_y[0], tmodel.predict([tval_X])[0]
                tran_val_errs.append(score(y, ty_hat))
            else:
                tran_val_errs.append(0.999)
        
        return raw_val_errs, tran_val_errs

    def val_ets(self, policy, series, n_val, split, horizon, score) -> Tuple[List]:

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

            if len(ttrain) > 1:
                tmodel = ExponentialSmoothing(
                    ttrain,
                    seasonal_periods=policy['seasonal periods'],
                    trend=policy['trend'],
                    seasonal=policy['seasonal'],
                    damped_trend=policy['damped trend']
                ).fit()
                y, ty_hat = val[0], tmodel.forecast(horizon).tolist()[0]
                tran_val_errs.append(score(y, ty_hat))
            else:
                tran_val_errs.append(0.999)
        
        return raw_val_errs, tran_val_errs

    def val_en(self, policy, series, n_val, split, horizon, score) -> Tuple[List]:

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
            
            rmodel = ElasticNet(
                alpha=policy['alpha'],
                l1_ratio=policy['l1 ratio'],
                random_state=0)
            rmodel.fit(train_X, train_y)
            y, y_hat = val_y[0], rmodel.predict([val_X])[0]
            raw_val_errs.append(score(y, y_hat))

            # with transformation
            """Transformation has to be improved!!!"""
            sigma = np.std(np.diff(np.log(train)))
            thres = (sigma*policy['thres up'], -sigma*policy['thres down'])
            t = DCTransformer()
            t.transform(train, threshold=thres)
            ttrain = t.tdata1

            if len(ttrain) > 1:
                tX, ty = data_prep.ts_prep(ttrain, nlag=policy['n lag'], horizon=horizon)
                ttrain_X, tval_X = tX, ttrain[-policy['n lag']:]
                ttrain_y, val_y = ty, val

                tmodel = ElasticNet(alpha=policy['alpha'],l1_ratio=policy['l1 ratio'], random_state=0)
                tmodel.fit(ttrain_X, ttrain_y)
                y, ty_hat = val_y[0], tmodel.predict([tval_X])[0]
                tran_val_errs.append(score(y, ty_hat))
            else:
                tran_val_errs.append(0.999)
        
        return raw_val_errs, tran_val_errs
    
    def val_xgb(self, policy, series, n_val, split, horizon, score) -> Tuple[List]:
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
            """Transformation has to be improved!!!"""
            sigma = np.std(np.diff(np.log(train)))
            thres = (sigma*policy['thres up'], -sigma*policy['thres down'])
            t = DCTransformer()
            t.transform(train, threshold=thres)
            ttrain = t.tdata1

            if len(ttrain) > 1:
                tX, ty = data_prep.ts_prep(ttrain, nlag=policy['n lag'], horizon=horizon)
                ttrain_X, tval_X = tX, ttrain[-policy['n lag']:]
                ttrain_y, val_y = ty, val

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

    # @NOTE: LightGBM doesn't work with multiprocessing. This function is not used.
    def val_lgbm(self, policy, series, n_val, split, horizon, score) -> Tuple[List]:
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
            """Transformation has to be improved!!!"""
            sigma = np.std(np.diff(np.log(train)))
            thres = (sigma*policy['thres up'], -sigma*policy['thres down'])
            t = DCTransformer()
            t.transform(train, threshold=thres)
            ttrain = t.tdata1

            if len(ttrain) > 1:
                tX, ty = data_prep.ts_prep(ttrain, nlag=policy['n lag'], horizon=horizon)
                ttrain_X, tval_X = tX, ttrain[-policy['n lag']:]
                ttrain_y, val_y = ty, val

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

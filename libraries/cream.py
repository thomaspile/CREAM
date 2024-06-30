import pandas as pd
import numpy as np
import os
import sys
import talib
from binance.client import Client
from geneticalgorithm import geneticalgorithm as ga
from datetime import datetime
import sklearn
from sklearn import metrics
import lightgbm
from hyperopt import hp, STATUS_OK, tpe, atpe, fmin, Trials, SparkTrials, pyll
import plotly
import plotly.graph_objects as go
import plotly.offline as py
from matplotlib import pyplot as plt
import copy

# sys.path.insert(0, 'libraries')
from model_builder import OptimalModel, ModelBuilder, ModelPlots


class TimeSeriesFeatures:
        
    def build_rolling_features(df, var, roll_lengths=[1, 2, 3, 5, 7]):

        for n in roll_lengths:
            df[f'{var}_rolling_mean_{n}_delta'] = (df[var] / df[var].rolling(window=n, closed='left').mean()) - 1
            df[f'{var}_rolling_max_{n}_delta'] = (df[var] / df[var].rolling(window=n, closed='left').max()) - 1
            df[f'{var}_rolling_min_{n}_delta'] = (df[var] / df[var].rolling(window=n, closed='left').min()) - 1
            if n > 1:
                df[f'{var}_rolling_std_{n}_delta'] = (df[var] / df[var].rolling(window=n, closed='left').std()) - 1

        return df

    def build_rsi_features(df, var, rsi_timeperiods=[10, 25, 50, 100, 200]):

        for rsi_timeperiod in rsi_timeperiods:
            df[f'{var}_rsi_{rsi_timeperiod}'] = talib.RSI(df[var], timeperiod=rsi_timeperiod)

        return df

    def build_macd_features(df_in, var, macd_fastp_values=[10, 25, 50, 100, 200], 
                            macd_slowp_values=[10, 25, 50, 100, 200], 
                            macd_sigp_values=[10, 25, 50, 100, 200]):

        df = df_in.copy()

        for macd_fastp in macd_fastp_values:
            for macd_slowp in macd_slowp_values:
                for macd_sigp in macd_sigp_values:
                    df[f'{var}_macd_{macd_fastp}_{macd_slowp}_{macd_sigp}'], _, _ = talib.MACD(df[var], 
                                                                              fastperiod=macd_fastp, 
                                                                              slowperiod=macd_slowp, 
                                                                              signalperiod=macd_sigp)    

        return df

    def build_bollinger_features(df_in, var, bol_periods=[5, 10, 20, 30, 50], stdNbrs=[2, 4, 6, 8]):

        df = df_in.copy()

        for bol_period in bol_periods:
            for bol_stdNbr in stdNbrs:    
                upper, middle, lower = talib.BBANDS(df[var]*100000, timeperiod=bol_period, nbdevup=bol_stdNbr, nbdevdn=bol_stdNbr, matype=0)
                df[f'bol_upper_{bol_period}_{bol_stdNbr}'], df[f'bol_middle_{bol_period}_{bol_stdNbr}'], df[f'bol_lower_{bol_period}_{bol_stdNbr}'] = upper/100000, middle/100000, lower/100000

        return df

    def build_outcomes(df_in, var, prop_to_rise, time_var, n_forward_looks = [1, 2, 3, 5, 10]):

        df = df_in.copy()
        for n in n_forward_looks:
            df[f'max_{n}_forward'] = (df.iloc[::-1]
                    .rolling(n, on=time_var, min_periods=0)[var]
                    .max()
                    .iloc[::-1])

            df[f'delta_{n}_forward'] = (df[f'max_{n}_forward'] / df[var]) - 1

            df[f'outcome_{n}_forward_up_{prop_to_rise}'] = np.where(df[f'delta_{n}_forward'] >= prop_to_rise, 1, 0)

        return df
    

class AppliedStrategy:
    
    def __init__(self, df_train_in, df_test_in, cols, params, outcome_var=None, time_var='date', price_var='price', cutoff=None):
        
        self.model = None
        
        self.df_train = df_train_in.copy()
        self.df_test = df_test_in.copy()
        self.cols = cols
        self.params = params
        self.outcome_var = outcome_var
        self.time_var = time_var
        self.cutoff = cutoff
        self.price_var = price_var
        
        print(f'params={params}')
        
        self.model_params = copy.deepcopy(self.params)
        if 'outcome_var' in self.params.keys():
            self.outcome_var = self.params['outcome_var']
            del self.model_params['outcome_var']
        elif self.outcome_var is None:
            raise ValueError('Must provide outcome_var in either params or function parameter')
        if 'cutoff' in self.model_params.keys():
            self.cutoff = self.params['cutoff']
            del self.model_params['cutoff']
            
    def build_and_predict(self):

        builder = ModelBuilder()

        self.model, self.importance, self.errors = builder.build_lgb(self.df_train[self.cols], self.df_train[self.outcome_var], 
                                                      self.df_test[self.cols], self.df_test[self.outcome_var],
                                                      None, None, 
                                                      self.model_params, outcome_type='classification', 
                                                      n_jobs=-1, seed=123)

        self.df_train['pred'] = self.model.predict_proba(self.df_train[self.cols])[:, 1]  
        self.df_test['pred'] = self.model.predict_proba(self.df_test[self.cols])[:, 1] 
        
    def define_buys(self, df, cutoff, time_var='date', price_var='price'):
        
        df = df.sort_values(time_var)
        df['buy'] = np.where(df['pred'] >= cutoff, df[price_var], np.nan)
        df['buy_loss'] = np.where((df['pred'] >= cutoff) & (df[self.outcome_var] == 0), df[price_var], np.nan)
        df['buy_win'] = np.where((df['pred'] >= cutoff) & (df[self.outcome_var] == 1), df[price_var], np.nan)  
        df['should_buy'] = np.where(df[self.outcome_var]==1, df[price_var], np.nan)
        
        return df
    
    def plot_validation(self):
        
        ModelPlots.feature_importance(self.importance, title='Feature Importance', n_features=10)
        
    def plot_buys(self, label_wins=True, cutoff=None, figsize=(15, 5), plot_actuals=False):
        
        if self.model is None:
            self.build_and_predict()
            
        if cutoff is not None:
            self.df_train = self.define_buys(self.df_train, cutoff, self.time_var, self.price_var)
            self.df_test = self.define_buys(self.df_test, cutoff, self.time_var, self.price_var)            
        elif self.cutoff is not None:
            self.df_train = self.define_buys(self.df_train, self.cutoff, self.time_var, self.price_var)
            self.df_test = self.define_buys(self.df_test, self.cutoff, self.time_var, self.price_var)
        else:
            raise ValueError('Cutoff must be supplied to plot buys')
            
        _, axs = plt.subplots(1, 2, figsize=figsize)
        
        ax = axs[0]
        df = self.df_train
        ax.plot(df[self.time_var], df[self.price_var])
        if label_wins:
            ax.scatter(df[self.time_var], df['buy_loss'], color='r', s=50)
            ax.scatter(df[self.time_var], df['buy_win'], color='g', s=50) 
            ax.legend(['price', 'Losing Buy', 'Winning Buy'])
        else:
            ax.scatter(df[self.time_var], df['buy'], color='y', s=50)
            ax.legend(['price', 'buy'])
        if plot_actuals:
            ax.scatter(df[self.time_var], df['should_buy'], color='y', s=50)
        ax.set_title('Train Set- Buy Strategy')
        
        ax = axs[1]
        df = self.df_test
        ax.plot(df[self.time_var], df[self.price_var])
        if label_wins:
            ax.scatter(df[self.time_var], df['buy_loss'], color='r', s=50)
            ax.scatter(df[self.time_var], df['buy_win'], color='g', s=50) 
            ax.legend(['price', 'Losing Buy', 'Winning Buy'])
        else:
            ax.scatter(df[self.time_var], df['buy'], color='y', s=50)
            ax.legend(['price', 'buy'])
        if plot_actuals:
            ax.scatter(df[self.time_var], df['should_buy'], color='y', s=50)            
        ax.set_title('Test Set- Buy Strategy')
        
        
# def apply_parameters():
    
#     df_train = df_train_in.copy()
#     df_test = df_test_in.copy()
    
#     model_params = copy.deepcopy(params)

#     if 'outcome_var' in model_params.keys():
#         outcome_var = params['outcome_var']
#         del model_params['outcome_var']
#     elif outcome_var is None:
#         raise ValueError('Must provide outcome_var in either params or function parameter')
#     if 'cutoff' in model_params.keys():
#         cutoff = params['cutoff']
#         del model_params['cutoff']
            
#     builder = ModelBuilder()
    
#     model, importance, errors = builder.build_lgb(df_train[cols], df_train[outcome_var], 
#                                                   df_test[cols], df_test[outcome_var],
#                                                   None, None, 
#                                                   model_params, outcome_type='classification', 
#                                                   n_jobs=-1, seed=123)
    
#     df_train['pred'] = model.predict_proba(df_train[cols])[:, 1]  
#     df_test['pred'] = model.predict_proba(df_test[cols])[:, 1] 
    
#     if cutoff is not None
#         df = df.sort_values('open_time')
#         df['buy'] = np.where(df['pred'] >= cutoff, df['open'], np.nan)
#         df['buy_loss'] = np.where(df['buy'].notnull() & df[outcome_var] == 0, df['open'], np.nan)
#         df['buy_win'] = np.where(df['buy'].notnull() & df[outcome_var] == 1, df['open'], np.nan)  
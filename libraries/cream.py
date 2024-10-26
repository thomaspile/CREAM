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


class MACD:
    
    def __init__(self, series, periods=None, fastp=12, slowp=26, sigp=9):
        
        self.series = series.copy().reset_index(drop=True)
        try:
            self.series_name = series.__dict__['_cacher'][0]
        except:
            self.series_name = None
            
        self.periods = periods

        self.macd, self.macd_signal, self.macdHist = talib.MACD(self.series, fastperiod=fastp, 
                                                                slowperiod=slowp, signalperiod=sigp)
        
        self.scale_to_series()
        self.get_crossovers()
        self.get_divergence()
        self.get_divergence_features()
    
    def scale_to_series(self):
        
        '''Only used if you want to plot the MACD line and signal on the series plot'''
    
        scaler = MinMaxScaler(feature_range=(min(self.series), max(self.series)))
        
        scaler.fit(np.array(self.macd).reshape(-1, 1))
        self.macd_scaled = scaler.transform(np.array(self.macd).reshape(-1, 1))
        self.macd_signal_scaled = scaler.transform(np.array(self.macd_signal).reshape(-1, 1))
    
    def get_crossovers(self):
        
        self.crossovers = ['None']
        self.idxs_up = []
        self.idxs_down = []

        for i in range(1, len(self.macd_signal)):
            if self.macd[i] > self.macd_signal[i] and self.macd[i - 1] <= self.macd_signal[i - 1]:
                self.crossovers.append("Up")
                self.idxs_up.append(i)
            elif self.macd[i] < self.macd_signal[i] and self.macd[i - 1] >= self.macd_signal[i - 1]:
                self.crossovers.append("Down")
                self.idxs_down.append(i)
            else:
                self.crossovers.append('None')
                
        df_up_signals = pd.DataFrame({'crossover_idx':self.idxs_up, 'crossover_type':['up']*len(self.idxs_up)})
        df_down_signals = pd.DataFrame({'crossover_idx':self.idxs_down, 'crossover_type':['down']*len(self.idxs_down)})
        
        self.df_macd_crossovers = pd.concat([df_up_signals, df_down_signals]).sort_values('crossover_idx', ascending=True)
 
    def get_divergence(self):
        
        self.df_macd = pd.DataFrame({'macd':self.macd, 'macd_signal':self.macd_signal})
        
        if self.periods is not None:
            self.df_macd['idx'] = self.periods
        else:
            self.df_macd = self.df_macd.reset_index().rename(columns={'index':'idx'})         
        
        self.df_macd['macd_signal_divergence'] = self.df_macd['macd'] - self.df_macd['macd_signal']        
    
    def get_divergence_features(self, n_back_checks=10, n_periods_gradient=4):
        
        for i in range(1, n_back_checks + 1):
            self.df_macd[f'macd_signal_divergence_minus_{i}'] = self.df_macd['macd_signal_divergence'].shift(i) 
        
        self.df_macd[f'macd_signal_divergence_gradient_{n_periods_gradient}'] = (self.df_macd[f'macd_signal_divergence'] - self.df_macd[f'macd_signal_divergence_minus_{n_periods_gradient}']) / n_periods_gradient
        self.df_macd[f'macd_signal_divergence_rolling_std_{n_back_checks}_delta'] = (self.df_macd['macd_signal_divergence'] / self.df_macd['macd_signal_divergence'].rolling(window=n_back_checks, closed='left').std()) - 1
        
        # self.df_macd[self.df_macd['idx'].isin(self.df_macd_crossovers['idx'].values
        
    def plot(self, figsize=(24, 20)):

        _, axs = plt.subplots(2, 1, figsize=figsize)

        ax = axs[0]
        ax.plot(self.series, 'black')
        
        for idx in macd.idxs_up:
            ax.axvline(idx, c='g')
            ax.text(x=idx - 5, y=min(self.series) + 15, s='Upturn', fontsize=18)
        for idx in macd.idxs_down:
            ax.axvline(idx, c='r')
            ax.text(x=idx - 5, y=min(self.series), s='Downturn', fontsize=18)

        # ax.legend(['Price', 'Ascended', 'Double Ascended', 'Descended', 'Double Descended', 'Triple Descended'], prop={'size': 20})

        ax.set_title('Macd', fontsize=26)
        ax.set_xlabel('Index', fontsize=22)
        ax.set_ylabel(self.series_name, fontsize=22);

        ax.set_xticklabels([int(x) for x in ax.get_xticks()], fontsize=18)
        ax.set_yticklabels([int(x) for x in ax.get_yticks()], fontsize=18);

        ax = axs[1]
        ax.plot(macd.df_macd['idx'], macd.df_macd['macd_signal_divergence'])
        ax.plot(macd.df_macd['idx'], macd.df_macd['macd_signal_divergence_gradient_4'])
        ax.axhline(0, c='r')            
            

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
    
    def __init__(self, df_train_in, df_test_in, cols, params, model=None, outcome_var=None, time_var='date', price_var='price', cutoff=None):
        
        self.model = model
        
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
    
    def predict(self):
    
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
            self.build()
            
        self.predict()
            
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
        
        
def get_turning_points(array):
    ''' turning_points(array) -> min_indices, max_indices
    Finds the turning points within an 1D array and returns the indices of the minimum and 
    maximum turning points in two separate lists.
    '''
    idx_max, idx_min = [], []
    if (len(array) < 3): 
        return idx_min, idx_max

    NEUTRAL, RISING, FALLING = range(3)
    def get_state(a, b):
        if a < b: return RISING
        if a > b: return FALLING
        return NEUTRAL

    ps = get_state(array[0], array[1])
    begin = 1
    for i in range(2, len(array)):
        s = get_state(array[i - 1], array[i])
        if s != NEUTRAL:
            if ps != NEUTRAL and ps != s:
                idx = (begin + i - 1) // 2
                left_scale = array[i-1] - array[i-2]
                right_scale = array[i-1] - array[i]
                left_prop = (array[i-1] / array[i-2]) - 1         
                right_prop = (array[i-1] / array[i]) - 1  
                if s == FALLING: 
                    idx_max.append((idx, scale, prop)) 
                else:
                    idx_min.append((idx, scale, prop))
            begin = i
            ps = s
    return idx_min, idx_max


# # Define signal

# t = df_2022.index.values
# s = df_2022.a_index.values

# # Execute EMD on signal
# IMF = EMD().emd(s,t)
# N = IMF.shape[0]+1

# # Plot results
# plt.subplot(N,1,1)
# plt.plot(t, s, 'r')
# plt.title("EMF for Cotton Prices")
# plt.xlabel("Time [Days]")

# for n, imf in enumerate(IMF):
#     plt.subplot(N,1,n+2)
#     plt.plot(t, imf, 'g')
#     plt.title("IMF "+str(n+1))
#     plt.xlabel("Time [Days]")

# plt.tight_layout()
# plt.savefig('simple_example')
# plt.show()

        
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
import pandas as pd
import numpy as np
import warnings
from hyperopt import hp, STATUS_OK, tpe, atpe, fmin, Trials, SparkTrials, pyll


class LongContract:
    
    def __init__(self, size, value, entry_price):
        
        self.size = size
        self.value = value
        self.entry_price = entry_price
        
    def update(self):
        
        pass
        

def long_short_strategy(df_model, df_prices, buy_threshold, sell_threshold, 
                        sell_point_execute, flexible_quantities,
                        base_exchange_prop, base_leverage_ratio, price_var, start_balance=100):

    # weight model as 100% currently since just 1
    w_model_buy = 1

    if price_var in df_model.columns:
        df_model = df_model.drop(columns=price_var).copy()
    if 'price' in df_model.columns:
        df_model = df_model.drop(columns='price')

    df = df_model.merge(df_prices, on='date', how='left')
    df['price'] = df[price_var]

    df['balance'] = np.nan
    df.loc[0, 'balance'] = start_balance

    df['model_buy_ind'] = np.where(df['model_prob'] >= buy_threshold, 1, 0)
    df['model_sell_ind'] = np.where(df['model_prob'] <= sell_threshold, 1, 0)

    bought = False
    stock_held = 0
    balance = start_balance
    sell_point_count = 0
    total_quantity_long = 0
    total_quantity_short = 0
    
    contract_tracking = []

    for i, pit in df.iterrows():
        
        margin_to_post = start_balance * base_exchange_prop
        
        margin_to_post = min(margin_to_post, balance)
        
        if margin_to_post > 0 and pit.model_buy_ind == 1:
            if flexible_quantities:
                leverage_ratio = base_leverage_ratio * ((1 - buy_threshold) + pit.model_prob)
            else:
                leverage_ratio = base_leverage_ratio

            contract_value = margin_to_post * leverage_ratio
            quantity_long = contract_value / pit.price

            df.loc[i, 'long'] = 1
            df.loc[i, 'margin_to_post'] = margin_to_post
            df.loc[i, 'contract_value'] = contract_value
            df.loc[i, 'contract_quantity'] = quantity_long
            df.loc[i, 'contract_id'] = 'cont_' + str(pit.date)[0:10] 

            balance = balance - margin_to_post
            total_quantity_long += quantity_long
            
            # LongContract(size=quantity_long, value=contract_value, entry_price=price)
            
            # contract = {'type':'long', 'margin':margin_to_post, 'entry_price':price, 'liquidation_price':None}
            # contracts.update(contract)

        else:
            df.loc[i, 'long'] = 0
            df.loc[i, 'margin_to_post'] = 0
            df.loc[i, 'contract_value'] = 0
            df.loc[i, 'quantity_long'] = 0
            
        df.loc[i, 'total_quantity_long'] = total_quantity_long
        df.loc[i, 'balance'] = balance
        
        df_contracts = df[df.long == 1][['contract_id', 'price', 'margin_to_post', 'contract_value', 'contract_quantity']].copy()
        df_contracts = df_contracts.rename(columns={'price':'entry_price', 'margin_to_post':'margin_posted', 'contract_value':'original_value'}) 
        df_contracts['date'] = pit.date
        df_contracts['current_price'] = pit.price
        df_contracts['price_difference'] = df_contracts['current_price'] - df_contracts['entry_price']
        df_contracts['current_value'] = pit.price * df_contracts['contract_quantity']
        df_contracts['value_difference'] = df_contracts['current_value'] - df_contracts['original_value']
        df_contracts['margin_gain'] = df_contracts['value_difference'] / df_contracts['margin_posted']
        
        contract_tracking.append(df_contracts)
    
    df_contract_tracking = pd.concat(contract_tracking).reset_index(drop=True)
    df_contract_tracking = df_contract_tracking.sort_values(['contract_id', 'date']).reset_index(drop=True)
    
    return df, df_contract_tracking

#         df.loc[i, 'balance'] = balance
#         df.loc[i, 'stock_held'] = stock_held
#         df.loc[i, 'portfolio_value'] = balance + pit.price * stock_held

#     df_final_balance = df.balance.iat[-1]
#     df_final_stock_held = df.stock_held.iat[-1]
#     df_final_price = df.price.iat[-1]

#     portfolio_value = df_final_balance + df_final_price*df_final_stock_held

#     if balance != df_final_balance:
#         raise ValueError('Balance mismatch between algorithm ('+str(balance)+') and df record ('+str(df_final_balance)+')')
#     if math.isnan(portfolio_value):
#         print('df_final_balance: ' + str(df_final_balance))
#         print('df_final_price: ' + str(df_final_price))
#         print('df_final_stock_held: ' + str(df_final_stock_held))
#         raise ValueError('portfolio_value is nan')

    # return portfolio_value, df

def buy_sell_strategy(df_model, df_prices, buy_threshold, sell_threshold, sell_point_execute, flexible_quantities,
                      base_exchange_prop, price_var, start_balance=100):

    # buy_sigma_threshold = 0.7
    # sell_sigma_threshold = 0.7

    # weight model as 100% currently since just 1
    w_model_buy = 1

    if price_var in df_model.columns:
        df_model = df_model.drop(columns=price_var).copy()
    if 'price' in df_model.columns:
        df_model = df_model.drop(columns='price')

    df = df_model.merge(df_prices, on='date', how='left')
    df['price'] = df[price_var]

    #         df = df_in[['date', 'price', 'model_prob']].reset_index().copy()

    df['balance'] = np.nan
    df.loc[0, 'balance'] = start_balance

    df['model_buy_ind'] = np.where(df['model_prob'] >= buy_threshold, 1, 0)
    df['model_sell_ind'] = np.where(df['model_prob'] <= sell_threshold, 1, 0)

    bought = False
    stock_held = 0
    balance = start_balance
    sell_point_count = 0

    #base_exchange = 100

    for i, pit in df.iterrows():
        base_exchange = (balance/pit.price) * base_exchange_prop
        # buy_sigma = w_model_buy * pit.model_buy_ind + w_macd_buy * pit.macd_buy_ind + w_bol_buy * pit.bol_buy_ind
        # buy_sigma = w_model_buy * pit.model_buy_ind
        # df.loc[i, 'buy_sigma'] = buy_sigma
        if pit.model_buy_ind == 1:
            if flexible_quantities:
                buy_quantity = base_exchange * ((1 - buy_threshold) + pit.model_prob)
            else:
                buy_quantity = base_exchange
            buy_quantity = min(buy_quantity, balance/pit.price)
            df.loc[i, 'buy'] = 1
            df.loc[i, 'buy_quantity'] = buy_quantity
            balance = balance - buy_quantity*pit.price
            bought = True
            stock_held = stock_held + buy_quantity
        else:
            df.loc[i, 'buy'] = 0
            df.loc[i, 'buy_quantity'] = 0

        sell_exchange = stock_held * base_exchange_prop
        # sell_sigma = w_rsi_sell * pit.rsi_sell_ind + w_macd_sell * pit.macd_sell_ind + w_bol_sell * pit.bol_sell_ind
        # df.loc[i, 'sell_sigma'] = sell_sigma
        if stock_held > 0:
            if pit.model_sell_ind == 1:
                sell_point_count = sell_point_count + 1
                if flexible_quantities:
                    sell_quantity = sell_exchange * (sell_threshold + (1 - pit.model_prob))
                else:
                    sell_quantity = sell_exchange                    
                sell_quantity = min(sell_quantity, stock_held)
                if sell_point_count >= sell_point_execute:
                    df.loc[i, 'sell'] = 1
                    df.loc[i, 'sell_quantity'] = sell_quantity
                    balance = balance + sell_quantity*pit.price
                    stock_held = stock_held - sell_quantity
                    bought = False
                    sell_point_count = 0
                else:
                    df.loc[i, 'sell'] = 0
                    df.loc[i, 'sell_quantity'] = 0
            else:
                df.loc[i, 'sell'] = 0
                df.loc[i, 'sell_quantity'] = 0
        else:
            df.loc[i, 'sell'] = 0
            df.loc[i, 'sell_quantity'] = 0

        df.loc[i, 'balance'] = balance
        df.loc[i, 'stock_held'] = stock_held
        df.loc[i, 'portfolio_value'] = balance + pit.price * stock_held

    df_final_balance = df.balance.iat[-1]
    df_final_stock_held = df.stock_held.iat[-1]
    df_final_price = df.price.iat[-1]

    portfolio_value = df_final_balance + df_final_price*df_final_stock_held

    if balance != df_final_balance:
        raise ValueError('Balance mismatch between algorithm ('+str(balance)+') and df record ('+str(df_final_balance)+')')
    if math.isnan(portfolio_value):
        print('df_final_balance: ' + str(df_final_balance))
        print('df_final_price: ' + str(df_final_price))
        print('df_final_stock_held: ' + str(df_final_stock_held))
        raise ValueError('portfolio_value is nan')

    return portfolio_value, df
    
    
class StrategyOptimiser:
    
    def __init__(self, df, df_prices, search_space, optimiser, evals, hp_algo=tpe.suggest, n_cv_folds=1, price_var='close'):
        
        self.df = df
        self.df_prices = df_prices
        self.search_space = search_space
        self.optimiser = optimiser
        self.evals = evals
        self.hp_algo = hp_algo
        self.n_cv_folds = n_cv_folds
        self.price_var = price_var
        
        if self.n_cv_folds > 1:
            with warnings.catch_warnings():
                warnings.simplefilter(action='ignore', category=FutureWarning)
                self.folds = np.array_split(self.df, self.n_cv_folds)

    def buy_sell_strategy(self, df_model, df_prices, buy_threshold, sell_threshold, sell_point_execute, flexible_quantities, base_exchange_prop, start_balance=100):

        # buy_sigma_threshold = 0.7
        # sell_sigma_threshold = 0.7

        # weight model as 100% currently since just 1
        w_model_buy = 1
        
        if self.price_var in df_model.columns:
            df_model = df_model.drop(columns=self.price_var).copy()
        if 'price' in df_model.columns:
            df_model = df_model.drop(columns='price')
            
        df = df_model.merge(df_prices, on='date', how='left')
        df['price'] = df[self.price_var]
        
#         df = df_in[['date', 'price', 'model_prob']].reset_index().copy()

        df['balance'] = np.nan
        df.loc[0, 'balance'] = start_balance

        df['model_buy_ind'] = np.where(df['model_prob'] >= buy_threshold, 1, 0)
        df['model_sell_ind'] = np.where(df['model_prob'] <= sell_threshold, 1, 0)

        bought = False
        stock_held = 0
        balance = start_balance
        sell_point_count = 0

        #base_exchange = 100

        for i, pit in df.iterrows():
            base_exchange = (balance/pit.price) * base_exchange_prop
            # buy_sigma = w_model_buy * pit.model_buy_ind + w_macd_buy * pit.macd_buy_ind + w_bol_buy * pit.bol_buy_ind
            # buy_sigma = w_model_buy * pit.model_buy_ind
            # df.loc[i, 'buy_sigma'] = buy_sigma
            if pit.model_buy_ind == 1:
                if flexible_quantities:
                    buy_quantity = base_exchange * ((1 - buy_threshold) + pit.model_prob)
                else:
                    buy_quantity = base_exchange
                buy_quantity = min(buy_quantity, balance/pit.price)
                df.loc[i, 'buy'] = 1
                df.loc[i, 'buy_quantity'] = buy_quantity
                balance = balance - buy_quantity*pit.price
                bought = True
                stock_held = stock_held + buy_quantity
            else:
                df.loc[i, 'buy'] = 0
                df.loc[i, 'buy_quantity'] = 0

            sell_exchange = stock_held * base_exchange_prop
            # sell_sigma = w_rsi_sell * pit.rsi_sell_ind + w_macd_sell * pit.macd_sell_ind + w_bol_sell * pit.bol_sell_ind
            # df.loc[i, 'sell_sigma'] = sell_sigma
            if stock_held > 0:
                if pit.model_sell_ind == 1:
                    sell_point_count = sell_point_count + 1
                    if flexible_quantities:
                        sell_quantity = sell_exchange * (sell_threshold + (1 - pit.model_prob))
                    else:
                        sell_quantity = sell_exchange                    
                    sell_quantity = min(sell_quantity, stock_held)
                    if sell_point_count >= sell_point_execute:
                        df.loc[i, 'sell'] = 1
                        df.loc[i, 'sell_quantity'] = sell_quantity
                        balance = balance + sell_quantity*pit.price
                        stock_held = stock_held - sell_quantity
                        bought = False
                        sell_point_count = 0
                    else:
                        df.loc[i, 'sell'] = 0
                        df.loc[i, 'sell_quantity'] = 0
                else:
                    df.loc[i, 'sell'] = 0
                    df.loc[i, 'sell_quantity'] = 0
            else:
                df.loc[i, 'sell'] = 0
                df.loc[i, 'sell_quantity'] = 0

            df.loc[i, 'balance'] = balance
            df.loc[i, 'stock_held'] = stock_held
            df.loc[i, 'portfolio_value'] = balance + pit.price * stock_held

        df_final_balance = df.balance.iat[-1]
        df_final_stock_held = df.stock_held.iat[-1]
        df_final_price = df.price.iat[-1]

        portfolio_value = df_final_balance + df_final_price*df_final_stock_held

        if balance != df_final_balance:
            raise ValueError('Balance mismatch between algorithm ('+str(balance)+') and df record ('+str(df_final_balance)+')')
        if math.isnan(portfolio_value):
            print('df_final_balance: ' + str(df_final_balance))
            print('df_final_price: ' + str(df_final_price))
            print('df_final_stock_held: ' + str(df_final_stock_held))
            raise ValueError('portfolio_value is nan')

        return portfolio_value, df

    def objective_optimise_strategy(self, params):

        if self.n_cv_folds > 1:
 
            portfolio_values = []
            for df_fold in self.folds:
                portfolio_value, prices_strategy = self.buy_sell_strategy(df_fold, 
                                                                          self.df_prices,
                                                                          buy_threshold=params['buy_threshold'],
                                                                          sell_threshold=params['sell_threshold'],
                                                                          sell_point_execute=params['sell_point_execute'], 
                                                                          flexible_quantities=params['flexible_quantities'], 
                                                                          base_exchange_prop=params['base_exchange_prop'], 
                                                                          start_balance=params['start_balance'])
                portfolio_values.append(portfolio_value)

            objective = np.mean(portfolio_values)
            cv_scores = portfolio_values

        else:

            balance, prices_strategy = self.buy_sell_strategy(df, 
                                                              self.df_prices,
                                                              buy_threshold=params['buy_threshold'],
                                                              sell_threshold=params['sell_threshold'],
                                                              sell_point_execute=params['sell_point_execute'], 
                                                              flexible_quantities=params['flexible_quantities'], 
                                                              base_exchange_prop=params['base_exchange_prop'], 
                                                              start_balance=params['start_balance'])
            objective = balance
            cv_scores = []

        return {'loss': - objective, 'params': params, 'status': STATUS_OK, 'cv_scores':cv_scores}
    
    def optimise(self, pop_size=100, max_iteration_without_improv=None):
            
        bayes_trials = Trials()

        best = fmin(fn=self.objective_optimise_strategy, 
                    space=self.search_space, 
                    algo=self.hp_algo, 
                    max_evals=self.evals, 
                    trials=bayes_trials)

        strategy = pd.DataFrame(sorted(bayes_trials.results, key = lambda x: x['loss'])).params[0]    
        cv_scores = pd.DataFrame(sorted(bayes_trials.results, key = lambda x: x['loss'])).cv_scores[0]
        trials = bayes_trials     

        print('Best params:', strategy)
        print('Cross Validation Scores:', cv_scores)

        return strategy
    
    def apply_strategy(self, df, strategy_params, plot=True):

        df = df.copy()

        portfolio_value, applied_strategy = self.buy_sell_strategy(df, 
                                                                   self.df_prices,
                                                      buy_threshold=strategy_params['buy_threshold'],
                                                      sell_threshold=strategy_params['sell_threshold'],
                                                      sell_point_execute=strategy_params['sell_point_execute'], 
                                                      flexible_quantities=strategy_params['flexible_quantities'], 
                                                      base_exchange_prop=strategy_params['base_exchange_prop'], 
                                                      start_balance=100)

        print('Final portfolio value: ' + str(portfolio_value))

        if plot:
            buys = applied_strategy[applied_strategy.buy_quantity > 0]
            sells = applied_strategy[applied_strategy.sell_quantity > 0]

            Candle = go.Candlestick(x=applied_strategy.date,
                                open=applied_strategy.price,
                                high=applied_strategy.price,
                                low=applied_strategy.price,
                                close=applied_strategy.price 
                                )

            Trace = go.Scatter(x=buys.date,
                               y=buys.price,
                               mode='markers',
                               name ='buy',
                               marker=go.scatter.Marker(size=10, color='blue', symbol='cross')
                               )

            Trace2 = go.Scatter(x=sells.date,
                               y=sells.price,
                               mode='markers',
                               name ='sell',
                               marker=go.scatter.Marker(size=10, color='black', symbol='cross')
                               )

            py.plot([Candle, Trace, Trace2])    

        return applied_strategy

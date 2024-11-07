import pandas as pd
import numpy as np
import warnings
from hyperopt import hp, STATUS_OK, tpe, atpe, fmin, Trials, SparkTrials, pyll
import plotly
import plotly.graph_objects as go
import plotly.offline as py
from scipy import stats
import matplotlib.pyplot as plt


class LongContract:
    
    def __init__(self, size, value, entry_price):
        
        self.size = size
        self.value = value
        self.entry_price = entry_price
        self.status = 'live'
        
    def update(self):
        
        pass
    

class Portfolio:
    
    def __init__(self, buy_threshold, sell_threshold, flexible_quantities, adjusting_margins,
                        base_exchange_prop, base_leverage_ratio, stop_loss_prop, price_var, start_balance=100):
        
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold
        self.flexible_quantities = flexible_quantities
        self.adjusting_margins = adjusting_margins
        self.base_exchange_prop = base_exchange_prop
        self.base_leverage_ratio = base_leverage_ratio
        self.stop_loss_prop = stop_loss_prop
        self.balance = start_balance
        
    
    

def long_short_strategy(df_model, df_prices, buy_threshold, sell_threshold, 
                        flexible_quantities, adjusting_margins,
                        base_exchange_prop, base_leverage_ratio, stop_loss_prop, price_var, start_balance=100):

    # weight model as 100% currently since just 1
    w_model_buy = 1

    if price_var in df_model.columns:
        df_model = df_model.drop(columns=price_var).copy()
    if 'price' in df_model.columns:
        df_model = df_model.drop(columns='price')
        
    df = pd.merge_asof(df_prices[['date', 'open','high','low','close']],
                      df_model[['date', 'model_prob']].rename(columns={'date':'model_date'}), 
                      left_on='date', 
                      right_on='model_date', 
                      direction='backward')
    df = df[df.model_date.notnull()].reset_index(drop=True)
    df['date'] = pd.to_datetime(df['date'])
    df['model_date'] = pd.to_datetime(df['model_date'])
    df['days_since_model'] = (df['date'] - df['model_date']).dt.days
    df = df[df['days_since_model'] <= 6].reset_index(drop=True)

    # df = df_model.merge(df_prices, on='date', how='left')
    df['price'] = df[price_var]

    df['balance'] = np.nan
    df['portfolio_value'] = np.nan
    df.loc[0, 'balance'] = start_balance

    df['model_buy_ind'] = np.where((df['model_prob'] >= buy_threshold) & (df['days_since_model'] == 0), 1, 0)
    df['model_sell_ind'] = np.where((df['model_prob'] <= sell_threshold) & (df['days_since_model'] == 0), 1, 0)
    
    df['open_long'] = 0
    df['open_short'] = 0
    df['close_long'] = 0
    df['close_short'] = 0
    
    df['contract_id'] = ''
    df['margin_to_post'] = np.nan
    df['contract_value'] = np.nan
    df['contract_quantity'] = np.nan
    
    bought = False
    stock_held = 0
    balance = start_balance
    sell_point_count = 0
    total_quantity_long = 0
    total_quantity_short = 0
    n_active_long_contracts = 0
    
    contract_tracking = []
    liquidated = []
    closed = []
    stop_losses = []
    
    
    def close_long_contract(df, i, contract, balance, total_quantity_long, n_active_long_contracts, summed_value_differences, summed_margins, closed):
        
        df.loc[i, 'close_long'] = df.loc[i, 'close_long'] + 1
        # df.loc[i, 'contract_value'] = contract.current_value
        # df.loc[i, 'contract_quantity'] = contract.contract_quantity
        df.loc[i, 'contract_id'] = df.loc[i, 'contract_id'] + ', ' + contract.contract_id            
        
        # print(balance, contract.margin_posted, contract.value_difference, contract.contract_quantity)
        balance = balance + contract.margin_posted + contract.value_difference
        total_quantity_long = total_quantity_long - contract.contract_quantity

        n_active_long_contracts = n_active_long_contracts - 1
        summed_value_differences = summed_value_differences - contract.value_difference
        summed_margins = summed_margins - contract.margin_posted

        closed.append(contract.contract_id)
        
        return df, balance, total_quantity_long, n_active_long_contracts, summed_value_differences, summed_margins, closed

    def close_short_contract(df, i, contract, balance, total_quantity_short, n_active_short_contracts, summed_value_differences, summed_margins, closed):
        
        df.loc[i, 'close_short'] = df.loc[i, 'close_short'] + 1
        # df.loc[i, 'contract_value'] = contract.current_value
        # df.loc[i, 'contract_quantity'] = contract.contract_quantity
        df.loc[i, 'contract_id'] = df.loc[i, 'contract_id'] + ', ' + contract.contract_id           

        balance = balance + contract.margin_posted + contract.value_difference
        total_quantity_short = total_quantity_short - contract.contract_quantity

        n_active_short_contracts = n_active_short_contracts - 1
        summed_value_differences = summed_value_differences - contract.value_difference
        summed_margins = summed_margins - contract.margin_posted

        closed.append(contract.contract_id)

        return df, balance, total_quantity_short, n_active_short_contracts, summed_value_differences, summed_margins, closed
    
    for i, pit in df.iterrows():
        
        df_contracts = df[((df.open_long == 1) | (df.open_short == 1))\
                          & (~df['contract_id'].isin(liquidated)) & (~df['contract_id'].isin(closed))]\
        [['contract_id', 'open_long', 'open_short', 'price', 'margin_to_post', 'contract_value', 'contract_quantity']].copy()
        
        if df_contracts.shape[0] > 0:
            df_contracts = df_contracts.rename(columns={'open_long':'long',
                                                        'open_short':'short',
                                                        'price':'entry_price', 
                                                        'margin_to_post':'margin_posted', 
                                                        'contract_value':'original_value'}) 
            df_contracts['date'] = pit.date
            df_contracts['current_price'] = pit.price
            df_contracts['price_difference'] = df_contracts['current_price'] - df_contracts['entry_price']
            df_contracts['current_value'] = pit.price * df_contracts['contract_quantity']
            df_contracts['value_difference'] = df_contracts['current_value'] - df_contracts['original_value']
            df_contracts['margin_gain'] = df_contracts['value_difference'] / df_contracts['margin_posted']
            
            
            summed_value_differences = df_contracts['value_difference'].sum()
            summed_margins = df_contracts['margin_posted'].sum()
            
            
            # Stop losses
            df_contracts['stop_loss'] = np.where(df_contracts['margin_gain'] <= - stop_loss_prop, 1, 0)
            
            for idx, contract in df_contracts[df_contracts['stop_loss'] == 1].iterrows():
                stop_losses.append(contract.contract_id)
                
                if contract.contract_id.startswith('long'):
                    df, balance, total_quantity_long, n_active_long_contracts, summed_value_differences, summed_margins, closed = close_long_contract(df, i, contract, balance, total_quantity_long, n_active_long_contracts, summed_value_differences, summed_margins, closed)
                else:   
                    df, balance, total_quantity_short, n_active_short_contracts, summed_value_differences, summed_margins, closed = close_short_contract(df, i, contract, balance, total_quantity_short, n_active_short_contracts, summed_value_differences, summed_margins, closed)                    
                
            stop_losses = list(set(stop_losses))

            
            # Liquidations
            df_contracts['liquidated'] = np.where(df_contracts['margin_gain'] <= -1, 1, 0)

            for contract in df_contracts[df_contracts['liquidated'] == 1].contract_id.values:
                liquidated.append(contract)
            liquidated = list(set(liquidated))
            

            contract_tracking.append(df_contracts)
            
            df_active_long = df_contracts[(df_contracts.long == 1) & (~df_contracts['contract_id'].isin(liquidated)) & (~df_contracts['contract_id'].isin(stop_losses))]
            df_active_short = df_contracts[(df_contracts.short == 1) & (~df_contracts['contract_id'].isin(liquidated)) & (~df_contracts['contract_id'].isin(stop_losses))]
            
            n_active_long_contracts = df_active_long.shape[0]
            n_active_short_contracts = df_active_short.shape[0]
            
        else:
            n_active_long_contracts = 0
            n_active_short_contracts = 0
            summed_value_differences = 0
            summed_margins = 0
        
        if adjusting_margins:
            margin_to_post = balance * base_exchange_prop
            margin_to_post = min(margin_to_post, balance)
        else:
            margin_to_post = start_balance * base_exchange_prop
            margin_to_post = min(margin_to_post, balance)            
        
        if pit.model_buy_ind == 1:
            
            if n_active_short_contracts == 0 and margin_to_post > 0: 
                if flexible_quantities:
                    leverage_ratio = base_leverage_ratio * ((1 - buy_threshold) + pit.model_prob)
                else:
                    leverage_ratio = base_leverage_ratio

                contract_value = margin_to_post * leverage_ratio
                quantity_long = contract_value / pit.price

                df.loc[i, 'open_long'] = 1
                df.loc[i, 'margin_to_post'] = margin_to_post
                df.loc[i, 'contract_value'] = contract_value
                df.loc[i, 'contract_quantity'] = quantity_long
                df.loc[i, 'contract_id'] = 'long_' + str(pit.date)[0:10]

                balance = balance - margin_to_post
                total_quantity_long = total_quantity_long + quantity_long
                
                n_active_long_contracts = n_active_long_contracts + 1
                summed_margins = summed_margins + margin_to_post
                
            elif n_active_short_contracts > 0:
                
                df_active_short = df_active_short.sort_values('margin_gain', ascending=False).reset_index(drop=True)
                df_max_gain = df_active_short.loc[0]
                
                df.loc[i, 'close_short'] = 1
                df.loc[i, 'contract_value'] = df_max_gain.current_value
                df.loc[i, 'contract_quantity'] = df_max_gain.contract_quantity
                df.loc[i, 'contract_id'] = df_max_gain.contract_id            
            
                balance = balance + df_max_gain.margin_posted + df_max_gain.value_difference
                total_quantity_short = total_quantity_short - df_max_gain.contract_quantity
                
                n_active_short_contracts = n_active_short_contracts - 1
                summed_value_differences = summed_value_differences - df_max_gain.value_difference
                summed_margins = summed_margins - df_max_gain.margin_posted
                
                closed.append(df_max_gain.contract_id)
                
        elif pit.model_sell_ind == 1:
            
            if n_active_long_contracts == 0 and margin_to_post > 0:
                if flexible_quantities:
                    leverage_ratio = base_leverage_ratio * (sell_threshold + (1 - pit.model_prob))
                else:
                    leverage_ratio = base_leverage_ratio  

                contract_value = - margin_to_post * leverage_ratio
                quantity_short = contract_value / pit.price

                df.loc[i, 'open_short'] = 1
                df.loc[i, 'margin_to_post'] = margin_to_post
                df.loc[i, 'contract_value'] = contract_value
                df.loc[i, 'contract_quantity'] = quantity_short
                df.loc[i, 'contract_id'] = 'short_' + str(pit.date)[0:10]
                
                balance = balance - margin_to_post
                total_quantity_short = total_quantity_short + quantity_short
                
                n_active_short_contracts = n_active_short_contracts + 1
                summed_margins = summed_margins + margin_to_post
            
            elif n_active_long_contracts > 0:
                
                df_active_long = df_active_long.sort_values('margin_gain', ascending=False).reset_index(drop=True)
                df_max_gain = df_active_long.loc[0]
                
                df.loc[i, 'close_long'] = 1
                df.loc[i, 'contract_value'] = df_max_gain.current_value
                df.loc[i, 'contract_quantity'] = df_max_gain.contract_quantity
                df.loc[i, 'contract_id'] = df_max_gain.contract_id            
            
                balance = balance + df_max_gain.margin_posted + df_max_gain.value_difference
                total_quantity_long = total_quantity_long - df_max_gain.contract_quantity
                
                n_active_long_contracts = n_active_long_contracts - 1
                summed_value_differences = summed_value_differences - df_max_gain.value_difference
                summed_margins = summed_margins - df_max_gain.margin_posted
                
                closed.append(df_max_gain.contract_id)
                
        total_value_long = total_quantity_long * pit.price
        df.loc[i, 'total_quantity_long'] = total_quantity_long
        df.loc[i, 'total_value_long'] = total_value_long

        total_value_short = total_quantity_short * pit.price
        df.loc[i, 'total_quantity_short'] = total_quantity_short
        df.loc[i, 'total_value_short'] = total_value_short
        
        df.loc[i, 'n_active_long_contracts'] = n_active_long_contracts
        df.loc[i, 'n_active_short_contracts'] = n_active_short_contracts
        
        df.loc[i, 'balance'] = balance
        
        portfolio_value = balance + summed_margins + summed_value_differences
        df.loc[i, 'portfolio_value'] = portfolio_value
    
    if len(contract_tracking) > 0:
        df_contract_tracking = pd.concat(contract_tracking).reset_index(drop=True)
        df_contract_tracking = df_contract_tracking.sort_values(['contract_id', 'date']).reset_index(drop=True)
    else:
        df_contract_tracking = pd.DataFrame()

    df['balance'] = np.round(df['balance'], 2)
    df['portfolio_value'] = np.round(df['portfolio_value'], 2)
    
    df['total_quantity_long'] = np.round(df['total_quantity_long'], 2)
    df['total_quantity_short'] = np.round(df['total_quantity_short'], 2)
    
    df['total_value_long'] = np.round(df['total_value_long'], 2)
    df['total_value_short'] = np.round(df['total_value_short'], 2)
    
    df['n_active_long_contracts'] = df['n_active_long_contracts'].astype(int)
    df['n_active_short_contracts'] = df['n_active_short_contracts'].astype(int)
    
    cumulative_max_value = df['portfolio_value'].cummax()
    df['drawdown'] = (df['portfolio_value']/cumulative_max_value) - 1.0
    greatest_drawdown = df['drawdown'].min()
    
    return df, df_contract_tracking, portfolio_value, greatest_drawdown


# def long_short_strategy_old(df_model, df_prices, buy_threshold, sell_threshold, 
#                         sell_point_execute, flexible_quantities, adjusting_margins,
#                         base_exchange_prop, base_leverage_ratio, price_var, start_balance=100):

#     # weight model as 100% currently since just 1
#     w_model_buy = 1

#     if price_var in df_model.columns:
#         df_model = df_model.drop(columns=price_var).copy()
#     if 'price' in df_model.columns:
#         df_model = df_model.drop(columns='price')

#     df = df_model.merge(df_prices, on='date', how='left')
#     df['price'] = df[price_var]

#     df['balance'] = np.nan
#     df.loc[0, 'balance'] = start_balance

#     df['model_buy_ind'] = np.where(df['model_prob'] >= buy_threshold, 1, 0)
#     df['model_sell_ind'] = np.where(df['model_prob'] <= sell_threshold, 1, 0)
    
#     df['open_long'] = np.nan
#     df['open_short'] = np.nan
#     df['close_long'] = np.nan
#     df['close_short'] = np.nan
    
#     df['contract_id'] = ''
#     df['margin_to_post'] = np.nan
#     df['contract_value'] = np.nan
#     df['contract_quantity'] = np.nan
    
#     bought = False
#     stock_held = 0
#     balance = start_balance
#     sell_point_count = 0
#     total_quantity_long = 0
#     total_quantity_short = 0
#     n_active_long_contracts = 0
    
#     contract_tracking = []
#     liquidated = []
#     closed = []

#     for i, pit in df.iterrows():
        
#         df_contracts = df[((df.open_long == 1) | (df.open_short == 1))\
#                           & (~df['contract_id'].isin(liquidated)) & (~df['contract_id'].isin(closed))]\
#         [['contract_id', 'open_long', 'open_short', 'price', 'margin_to_post', 'contract_value', 'contract_quantity']].copy()
        
#         if df_contracts.shape[0] > 0:
#             df_contracts = df_contracts.rename(columns={'open_long':'long',
#                                                         'open_short':'short',
#                                                         'price':'entry_price', 
#                                                         'margin_to_post':'margin_posted', 
#                                                         'contract_value':'original_value'}) 
#             df_contracts['date'] = pit.date
#             df_contracts['current_price'] = pit.price
#             df_contracts['price_difference'] = df_contracts['current_price'] - df_contracts['entry_price']
#             df_contracts['current_value'] = pit.price * df_contracts['contract_quantity']
#             df_contracts['value_difference'] = df_contracts['current_value'] - df_contracts['original_value']
#             df_contracts['margin_gain'] = df_contracts['value_difference'] / df_contracts['margin_posted']

            
            
                   
#             df_contracts['liquidated'] = np.where(df_contracts['margin_gain'] <= -1, 1, 0)

#             for contract in df_contracts[df_contracts['liquidated'] == 1].contract_id.values:
#                 liquidated.append(contract)
#             liquidated = list(set(liquidated))

#             contract_tracking.append(df_contracts)
            
#             df_active_long = df_contracts[(df_contracts.long == 1) & (~df_contracts['contract_id'].isin(liquidated))]
#             df_active_short = df_contracts[(df_contracts.short == 1) & (~df_contracts['contract_id'].isin(liquidated))]
            
#             n_active_long_contracts = df_active_long.shape[0]
#             n_active_short_contracts = df_active_short.shape[0]
            
#             summed_value_differences = df_contracts['value_difference'].sum()
#             summed_margins = df_contracts['margin_posted'].sum()
#         else:
#             n_active_long_contracts = 0
#             n_active_short_contracts = 0
#             summed_value_differences = 0
#             summed_margins = 0
        
#         if adjusting_margins:
#             margin_to_post = balance * base_exchange_prop
#             margin_to_post = min(margin_to_post, balance)
#         else:
#             margin_to_post = start_balance * base_exchange_prop
#             margin_to_post = min(margin_to_post, balance)            
        
#         if pit.model_buy_ind == 1:
            
#             if n_active_short_contracts == 0 and margin_to_post > 0.01: 
#                 if flexible_quantities:
#                     leverage_ratio = base_leverage_ratio * ((1 - buy_threshold) + pit.model_prob)
#                 else:
#                     leverage_ratio = base_leverage_ratio

#                 contract_value = margin_to_post * leverage_ratio
#                 quantity_long = contract_value / pit.price

#                 df.loc[i, 'open_long'] = 1
#                 df.loc[i, 'margin_to_post'] = margin_to_post
#                 df.loc[i, 'contract_value'] = contract_value
#                 df.loc[i, 'contract_quantity'] = quantity_long
#                 df.loc[i, 'contract_id'] = 'long_' + str(pit.date)[0:10]

#                 balance = balance - margin_to_post
#                 total_quantity_long = total_quantity_long + quantity_long
                
#                 n_active_long_contracts = n_active_long_contracts + 1
#                 summed_margins = summed_margins + margin_to_post
                
#             elif n_active_short_contracts > 0:
                
#                 df_active_short = df_active_short.sort_values('margin_gain', ascending=False).reset_index(drop=True)
#                 df_max_gain = df_active_short.loc[0]
                
#                 df.loc[i, 'close_short'] = 1
#                 df.loc[i, 'contract_value'] = df_max_gain.current_value
#                 df.loc[i, 'contract_quantity'] = df_max_gain.contract_quantity
#                 df.loc[i, 'contract_id'] = df_max_gain.contract_id            
            
#                 balance = balance + df_max_gain.margin_posted + df_max_gain.value_difference
#                 total_quantity_short = total_quantity_short - df_max_gain.contract_quantity
                
#                 n_active_short_contracts = n_active_short_contracts - 1
#                 summed_value_differences = summed_value_differences - df_max_gain.value_difference
#                 summed_margins = summed_margins - df_max_gain.margin_posted
                
#                 closed.append(df_max_gain.contract_id)
                
#         elif pit.model_sell_ind == 1:
            
#             if n_active_long_contracts == 0 and margin_to_post > 0:
#                 if flexible_quantities:
#                     leverage_ratio = base_leverage_ratio * (sell_threshold + (1 - pit.model_prob))
#                 else:
#                     leverage_ratio = base_leverage_ratio  

#                 contract_value = - margin_to_post * leverage_ratio
#                 quantity_short = contract_value / pit.price

#                 df.loc[i, 'open_short'] = 1
#                 df.loc[i, 'margin_to_post'] = margin_to_post
#                 df.loc[i, 'contract_value'] = contract_value
#                 df.loc[i, 'contract_quantity'] = quantity_short
#                 df.loc[i, 'contract_id'] = 'short_' + str(pit.date)[0:10]
                
#                 balance = balance - margin_to_post
#                 total_quantity_short = total_quantity_short + quantity_short
                
#                 n_active_short_contracts = n_active_short_contracts + 1
#                 summed_margins = summed_margins + margin_to_post
            
#             elif n_active_long_contracts:
                
#                 df_active_long = df_active_long.sort_values('margin_gain', ascending=False).reset_index(drop=True)
#                 df_max_gain = df_active_long.loc[0]
                
#                 df.loc[i, 'close_long'] = 1
#                 df.loc[i, 'contract_value'] = df_max_gain.current_value
#                 df.loc[i, 'contract_quantity'] = df_max_gain.contract_quantity
#                 df.loc[i, 'contract_id'] = df_max_gain.contract_id            
            
#                 balance = balance + df_max_gain.margin_posted + df_max_gain.value_difference
#                 total_quantity_long = total_quantity_long - df_max_gain.contract_quantity
                
#                 n_active_long_contracts = n_active_long_contracts - 1
#                 summed_value_differences = summed_value_differences - df_max_gain.value_difference
#                 summed_margins = summed_margins - df_max_gain.margin_posted
                
#                 closed.append(df_max_gain.contract_id)
                
#         total_value_long = total_quantity_long * pit.price
#         df.loc[i, 'total_quantity_long'] = total_quantity_long
#         df.loc[i, 'total_value_long'] = total_value_long

#         total_value_short = total_quantity_short * pit.price
#         df.loc[i, 'total_quantity_short'] = total_quantity_short
#         df.loc[i, 'total_value_short'] = total_value_short
        
#         df.loc[i, 'n_active_long_contracts'] = n_active_long_contracts
#         df.loc[i, 'n_active_short_contracts'] = n_active_short_contracts
        
#         df.loc[i, 'balance'] = balance
        
#         df.loc[i, 'portfolio_value'] = balance + summed_margins + summed_value_differences
    
#     df_contract_tracking = pd.concat(contract_tracking).reset_index(drop=True)
#     df_contract_tracking = df_contract_tracking.sort_values(['contract_id', 'date']).reset_index(drop=True)
    
#     df['total_quantity_long'] = np.round(df['total_quantity_long'], 2)
#     df['total_quantity_short'] = np.round(df['total_quantity_short'], 2)
    
#     df['total_value_long'] = np.round(df['total_value_long'], 2)
#     df['total_value_short'] = np.round(df['total_value_short'], 2)
    
#     df['n_active_long_contracts'] = df['n_active_long_contracts'].astype(int)
#     df['n_active_short_contracts'] = df['n_active_short_contracts'].astype(int)
    
#     return df, df_contract_tracking

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
    
    def __init__(self, df, df_prices, search_space, optimiser, evals, hp_algo=tpe.suggest, n_cv_folds=1, price_var='close', avg_scores_method='mean'):
        
        self.df = df
        self.df_prices = df_prices
        self.search_space = search_space
        self.optimiser = optimiser
        self.evals = evals
        self.hp_algo = hp_algo
        self.n_cv_folds = n_cv_folds
        self.avg_scores_method = avg_scores_method
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
            drawdowns = []
            for df_fold in self.folds:
                df, df_contracts, portfolio_value, drawdown = long_short_strategy(df_fold, 
                                                                     self.df_prices,
                                                                     buy_threshold=params['buy_threshold'],
                                                                     sell_threshold=params['sell_threshold'], 
                                                                     flexible_quantities=params['flexible_quantities'], 
                                                                     adjusting_margins=params['adjusting_margins'], 
                                                                     base_exchange_prop=params['base_exchange_prop'], 
                                                                     stop_loss_prop=params['stop_loss_prop'], 
                                                                     base_leverage_ratio=params['base_leverage_ratio'], 
                                                                     price_var='close',  
                                                                     start_balance=params['start_balance'])
                portfolio_values.append(portfolio_value)
                drawdowns.append(drawdown)
            
            if self.avg_scores_method == 'hmean':
                scores = [0 if x < 0 else x for x in portfolio_values]
                objective_value = stats.hmean(scores)
                
                drawdowns_abs = [-x for x in drawdowns]
                objective_drawdown = - stats.hmean(drawdowns_abs)
            else:
                reversed_drawdowns = [1 - d for d in drawdowns]
                scores = [value * rev_drawdown for value, rev_drawdown in zip(portfolio_values, reversed_drawdowns)]
                objective = np.mean(portfolio_values)
                drawdowns_abs = [-x for x in drawdowns]
                objective_drawdown = np.mean(drawdowns_abs)
                
            cv_scores = [portfolio_values] + [drawdowns]

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
    
    def apply_strategy(self, df, params, plot=0):

        df = df.copy()

        df_applied_strategy, df_contracts, portfolio_value = long_short_strategy(df, 
                                                                self.df_prices,
                                                                buy_threshold=params['buy_threshold'],
                                                                sell_threshold=params['sell_threshold'], 
                                                                flexible_quantities=params['flexible_quantities'], 
                                                                adjusting_margins=params['adjusting_margins'], 
                                                                base_exchange_prop=params['base_exchange_prop'], 
                                                                stop_loss_prop=params['stop_loss_prop'], 
                                                                base_leverage_ratio=params['base_leverage_ratio'], 
                                                                price_var='close',  
                                                                start_balance=params['start_balance'])

        print('Final portfolio value: ' + str(portfolio_value))
        
        if plot==1:
            
#             _, axs = plt.subplots(1, 2, figsize=(15, 6))
#             ax = axs[0]
#             ax.plot(df_applied_strategy['date'], df_applied_strategy[['balance', 'portfolio_value']])
#             ax.set_title('Portfolio balance and total value through time')
            
#             self.df_prices[self.df_prices.date.isin(df_applied_strategy.date.values)].plot(x='date', y='close', ax=axs[1])
#             axs[1].set_title('Price of cotton through time')

            _, ax1 = plt.subplots(figsize=(15, 6))
            
            ax2 = ax1.twinx()
            
            ax1.plot(df_applied_strategy['date'], df_applied_strategy[['portfolio_value']], c='b', label='Portfolio Value')
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Portfolio Value', color='b')
            ax1.legend(loc="upper left")
            
            df = self.df_prices[self.df_prices.date.isin(df_applied_strategy.date.values)]
            ax2.plot(df['date'], df['close'], c='r', label='Futures Price')
            ax2.set_ylabel('Price', color='r')
            ax2.legend(loc="upper left")    
            ax2.grid(False)
            
            plt.title('Portfolio Value & Futures Price Through Time')
        
#             fig, ax1 = plt.subplots()

#             # Plot data on the primary y-axis
#             ax1.plot(x, y1, color='b', label='Sine Wave')
#             ax1.set_xlabel('X-axis')
#             ax1.set_ylabel('Sine', color='b')
#             ax1.tick_params(axis='y', labelcolor='b')

#             # Create a secondary y-axis sharing the same x-axis
#             ax2 = ax1.twinx()
#             ax2.plot(x, y2, color='r', label='Cosine Wave')
#             ax2.set_ylabel('Cosine', color='r')
#             ax2.tick_params(axis='y', labelcolor='r')

#             # Add legends
#             ax1.legend(loc="upper left")
#             ax2.legend(loc="upper right")

            # Add a title
            

            
#             ax.set_title('Portfolio balance and total value through time')
            
            
#             axs[1].set_title('Price of cotton through time')
            
        
# ax1.plot(x, y1, 'g-')
# ax2.plot(x, y2, 'b-')

# ax1.set_xlabel('X data')
# ax1.set_ylabel('Y1 data', color='g')
# ax2.set_ylabel('Y2 data', color='b')


        elif plot==2:
            buys = df_applied_strategy[df_applied_strategy.open_long > 0]
            sells = df_applied_strategy[df_applied_strategy.open_short > 0]

            Candle = go.Candlestick(x=df_applied_strategy.date,
                                open=df_applied_strategy.open,
                                high=df_applied_strategy.high,
                                low=df_applied_strategy.low,
                                close=df_applied_strategy.close 
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

        return df_applied_strategy

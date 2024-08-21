import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from plotly.subplots import make_subplots
import plotly.graph_objs as go
from plotly.offline import plot
from functools import reduce
import plotly.express as px
from datetime import datetime, timedelta
import matplotlib.ticker as mtick
import seaborn as sns
import multiprocessing as mp
import copy
import random
import math
from time import time
from dateutil.relativedelta import relativedelta
import pickle
import kds
import seaborn as sns
import types
from pandas.api.types import is_numeric_dtype
from sklearn import metrics
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score
from sklearn.inspection import PartialDependenceDisplay
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

from model_builder import OptimalModel, ModelBuilder, ModelPlots, ShapleyPlots

sys.path.insert(0, '/Users/thomaspile/Documents/GitHub/utilities')
from utilities import GenUtilities, GenPlots

plt.style.use('ggplot')


class RandomForestExplainer:
    
    def __init__(self, model):
        
        self.model = model
        self.features = list(model.__dict__['feature_names_in_'])
        
    def traverse_tree(self, tree_index, tree_, individual):
        
        data = []
        node_index = 0
        
        while tree_.children_left[node_index] != -1:
            try:
                left_child_index = tree_.children_left[node_index]
                right_child_index = tree_.children_right[node_index]

                split_variable_index = tree_.feature[node_index]
                split_threshold = tree_.threshold[node_index]

                left_child_class_volumes = tree_.value[left_child_index][0]
                right_child_class_volumes = tree_.value[right_child_index][0]
              
                left_child_class_rate = left_child_class_volumes[1] / sum(left_child_class_volumes)
                right_child_class_rate = right_child_class_volumes[1] / sum(right_child_class_volumes)
                
                if left_child_class_rate > right_child_class_rate:
                    direction_split_vote = 'left'
                else:
                    direction_split_vote = 'right'

                split_variable = self.features[split_variable_index]
                individual_value = individual[split_variable]

                if individual_value <= split_threshold:
                    direction_travelled = 'left'
                    node_index = left_child_index
                    if direction_split_vote == 'left':
                        split_vote = 1
                    else:
                        split_vote = -1
                else:
                    direction_travelled = 'right'
                    node_index = right_child_index
                    if direction_split_vote == 'right':
                        split_vote = 1
                    else:
                        split_vote = -1

                data.append({
                    'tree_number': tree_index,
                    'node_number': node_index,
                    'left_child_node': left_child_index,
                    'right_child_node': right_child_index,
                    'left_child_class_volumes': left_child_class_volumes,
                    'left_child_class_rate': left_child_class_rate,
                    'right_child_class_volumes': right_child_class_volumes,
                    'right_child_class_rate': right_child_class_rate,
                    'split_threshold': split_threshold,
                    'split_variable': split_variable,
                    'individuals_value': individual_value,
                    'direction_travelled': direction_travelled,
                    'split_vote': split_vote
                })  
            
            except Exception as e:
                print('node_index', node_index)    
                print('tree_.feature', tree_.feature)
                print('split_variable', split_variable)
                print('split_variable', split_variable)
                print('split_threshold', split_threshold)
                print('individual_value', individual_value)
                raise e
        
        return pd.DataFrame(data)
                
    def traverse_forest(self, individual):
        
        dfs = []
        for tree_index, tree in enumerate(self.model.estimators_): 
            tree_ = tree.tree_
            df_tree = self.traverse_tree(tree_index, tree_, individual)
            dfs.append(df_tree)
        
        self.df_trees = pd.concat(dfs)
        
        df_votes = self.df_trees.groupby('split_variable').split_vote.sum().sort_values()\
                                .reset_index().rename(columns={'split_vote':'vote_count'})
        df_votes['abs_vote_count'] = abs(df_votes['vote_count'])
        df_votes['abs_vote_prop'] = df_votes['abs_vote_count'] / df_votes['abs_vote_count'].sum()
        self.df_votes = df_votes.sort_values('abs_vote_prop', ascending=False).reset_index(drop=True)
        
        return self.df_trees, self.df_votes
        
    def plot_waterfall(self, individual, n_feats, orientation='v'):
        
        df_trees, df_votes = self.traverse_forest(individual)

        variables = df_votes.iloc[0:n_feats]['split_variable'].values
        vote_counts = df_votes.iloc[0:n_feats]['vote_count'].values
        
        values = individual[variables]
        
        if orientation == 'v':
            x = variables
            y = vote_counts
        else:
            x = vote_counts
            y = variables
            
        fig = go.Figure(go.Waterfall(
            name = "Contribution", 
            orientation = orientation,
            measure = ["relative", "relative", "relative", "relative", "relative"],
            x = x,
            textposition = "inside",
            text = [str(np.round(val, 2)) for val in values],
            y = y,
            connector = {"line":{"color":"rgb(63, 63, 63)"}},
        ))
        
        fig.update_layout(
                title = "Feature Contributions",
                showlegend = True,
                width = 1000, 
                height = 600,
        )

        fig.show()

        
class ModelValidation: 
    
    def __init__(self, model, features, target, id_var, df_train, df_test, df_feature_importance, 
                 feature_dictionary=None, bounds_dictionary=None, class_idx=1):
        
        self.model = model
        self.target = target
        self.features = features
        self.id_var = id_var
        self.df_train = df_train
        self.df_test = df_test
        self.df_feature_importance = df_feature_importance
        if feature_dictionary is None:
            self.feature_dictionary = {}
        else:
            self.feature_dictionary = feature_dictionary
        if bounds_dictionary is None:
            self.bounds_dictionary = {}
        else:
            self.bounds_dictionary = bounds_dictionary
        
        self.class_idx = class_idx
            
        self.df_train['prob'] = self.model.predict_proba(self.df_train[self.features])[:, self.class_idx]
        self.df_test['prob'] = self.model.predict_proba(self.df_test[self.features])[:, self.class_idx]
        
        self.prediction = 'prob'
        
        self.df_valid = None
        
    def plot_EDA(self, ylimits=None):
        
        print('Plotting EDA')
        
        if self.df_valid is None:
            self.df_valid = self.df_test
            print('Warning, producing validation on the In-Time test sample since no Out-Of-Time sample supplied (data_scenario_val=None)')
            
        if self.target == 'engagement':
            print('Rounding activity dates to month start for monthly analysis of performance')
            self.df_orders['date'] = self.df_orders['date'].apply(lambda date: date.replace(day=1, hour=0, minute=0, second=0))
            if self.df_valid is not None:
                self.df_valid['date'] = self.df_valid['date'].apply(lambda date: date.replace(day=1, hour=0, minute=0, second=0))        

        if self.df_valid is None:
            self.df_valid = self.df_orders
            
        self.df_orders.dset_type = 'Train'        
        self.df_valid.dset_type = 'Valid (OOT)'
        
        # world_region
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_distributions(self.df_orders, 'world_region', ax=axs[0], legend=False)
        GenPlots.plot_distributions(self.df_valid, 'world_region', ax=axs[1])        
        
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_rates(self.df_orders, self.target, groupby='world_region', ax=axs[0], ylimits=ylimits)
        GenPlots.plot_rates(self.df_valid, self.target, groupby='world_region', ax=axs[1], ylimits=ylimits)
        
        # days_since_last_purchase_evaldate
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_distributions(self.df_orders, 'days_since_last_purchase_evaldate', bounds=[50, 100, 150, 200, 250, 300], 
                                    labels=['050', '050-100', '100-150', '150-200', '200-250', '250-300'], other='>300', ax=axs[0],
                                    plot_mean=True, legend=False)
        GenPlots.plot_distributions(self.df_valid, 'days_since_last_purchase_evaldate', bounds=[50, 100, 150, 200, 250, 300], 
                                    labels=['050', '050-100', '100-150', '150-200', '200-250', '250-300'], other='>300',  ax=axs[1],
                                    plot_mean=True)

        df_orders = GenUtilities.dynamic_variable_definition(self.df_orders, variable='days_since_last_purchase_evaldate', 
                                                             boundaries=[50, 100, 150, 200, 250, 300], 
                                                             name='days_since_last_purchase_evaldate_grouped', 
                                                             labels=['050', '050-100', '100-150', '150-200', '200-250', '250-300'],
                                                             otherwise='>300')   
        df_valid = GenUtilities.dynamic_variable_definition(self.df_valid, variable='days_since_last_purchase_evaldate', 
                                                            boundaries=[50, 100, 150, 200, 250, 300], 
                                                            name='days_since_last_purchase_evaldate_grouped', 
                                                            labels=['050', '050-100', '100-150', '150-200', '200-250', '250-300'],
                                                            otherwise='>300')  
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_rates(df_orders, self.target, groupby='days_since_last_purchase_evaldate_grouped', ax=axs[0], ylimits=ylimits)
        GenPlots.plot_rates(df_valid, self.target, groupby='days_since_last_purchase_evaldate_grouped', ax=axs[1], ylimits=ylimits)

        # sum_totaldiscount
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_distributions(self.df_orders, 'sum_totaldiscount', bounds=[0, 10], labels=['£0', '£0-10'], 
                                    other='>£10', ax=axs[0], plot_mean=True, legend=False)
        GenPlots.plot_distributions(self.df_valid, 'sum_totaldiscount', bounds=[0, 10], labels=['£0', '£0-10'], 
                                    other='>£10',  ax=axs[1], plot_mean=True)

        df_orders = GenUtilities.dynamic_variable_definition(self.df_orders, variable='sum_totaldiscount', 
                                                             boundaries=[0, 10], name='sum_totaldiscount_grouped', 
                                                             labels=['£0', '£0-10'], otherwise='>£10')   
        df_valid = GenUtilities.dynamic_variable_definition(self.df_valid, variable='sum_totaldiscount', 
                                                            boundaries=[0, 10], name='sum_totaldiscount_grouped', 
                                                            labels=['£0', '£0-10'], otherwise='>£10')  
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_rates(df_orders, self.target, groupby='sum_totaldiscount_grouped', ax=axs[0], ylimits=ylimits)
        GenPlots.plot_rates(df_valid, self.target, groupby='sum_totaldiscount_grouped', ax=axs[1], ylimits=ylimits)
        
        # n_distinct_brands
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_distributions(self.df_orders, 'n_distinct_brands', bounds=[1, 2, 3], labels=['1', '2', '3'], other='>3', ax=axs[0], plot_mean=True, legend=False)
        GenPlots.plot_distributions(self.df_valid, 'n_distinct_brands', bounds=[1, 2, 3], labels=['1', '2', '3'], other='>3', ax=axs[1], plot_mean=True)   

        df_orders = GenUtilities.dynamic_variable_definition(self.df_orders, variable='n_distinct_brands', boundaries=[1, 2, 3], name='n_distinct_brands_grouped',
                                                             labels=['1', '2', '3'], otherwise='>3')   
        df_valid = GenUtilities.dynamic_variable_definition(self.df_valid, variable='n_distinct_brands', boundaries=[1, 2, 3], name='n_distinct_brands_grouped', 
                                                            labels=['1', '2', '3'], otherwise='>3')   
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_rates(df_orders, self.target, groupby='n_distinct_brands_grouped', ax=axs[0], ylimits=ylimits)
        GenPlots.plot_rates(df_valid, self.target, groupby='n_distinct_brands_grouped', ax=axs[1], ylimits=ylimits)
        
        # n_distinct_categories
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_distributions(self.df_orders, 'n_distinct_categories', bounds=[1, 2, 3], labels=['1', '2', '3'], other='>3', ax=axs[0], plot_mean=True,
                                    legend=False)
        GenPlots.plot_distributions(self.df_valid, 'n_distinct_categories', bounds=[1, 2, 3], labels=['1', '2', '3'], other='>3', ax=axs[1], plot_mean=True)        

        df_orders = GenUtilities.dynamic_variable_definition(self.df_orders, variable='n_distinct_categories', boundaries=[1, 2, 3],
                                                             name='n_distinct_categories_grouped', labels=['1', '2', '3'], otherwise='>3')   
        df_valid = GenUtilities.dynamic_variable_definition(self.df_valid, variable='n_distinct_categories', boundaries=[1, 2, 3], name='n_distinct_categories_grouped',
                                                            labels=['1', '2', '3'], otherwise='>3')   
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_rates(df_orders, self.target, groupby='n_distinct_categories_grouped', ax=axs[0], ylimits=ylimits)
        GenPlots.plot_rates(df_valid, self.target, groupby='n_distinct_categories_grouped', ax=axs[1], ylimits=ylimits)
        
        # age_at_purchase
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_distributions(self.df_orders, 'age_at_purchase', bounds=[0, 10, 20, 30, 50, 80, 100], labels=['0', '10', '20', '30', '50', '80', '100'],
                                    other='>£10', ax=axs[0], plot_mean=True, legend=False)
        GenPlots.plot_distributions(self.df_valid, 'age_at_purchase', bounds=[0, 10, 20, 30, 50, 80, 100], labels=['0', '10', '20', '30', '50', '80', '100'],
                                    other='>£10',  ax=axs[1], plot_mean=True)    
        
        # has_givex
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_rates(self.df_orders, 'has_givex', groupby=self.target, ax=axs[0], ylimits=ylimits)
        GenPlots.plot_rates(self.df_valid, 'has_givex', groupby=self.target, ax=axs[1], ylimits=ylimits)
        
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_rates(self.df_orders, self.target, groupby='has_givex', ax=axs[0], ylimits=ylimits)
        GenPlots.plot_rates(self.df_valid, self.target, groupby='has_givex', ax=axs[1], ylimits=ylimits)
        
        # contactable
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_rates(self.df_orders, 'contactable', groupby=self.target, ax=axs[0], ylimits=ylimits)
        GenPlots.plot_rates(self.df_valid, 'contactable', groupby=self.target, ax=axs[1], ylimits=ylimits)
        
        _, axs = plt.subplots(1, 2, figsize=(28, 10))
        GenPlots.plot_rates(self.df_orders, self.target, groupby='contactable', ax=axs[0], ylimits=ylimits)
        GenPlots.plot_rates(self.df_valid, self.target, groupby='contactable', ax=axs[1], ylimits=ylimits)    
        
        dropvars = [col for col in self.df_orders.columns if 'grouped' in col]
        self.df_orders = self.df_orders.drop(dropvars, axis=1)
        self.df_valid = self.df_valid.drop(dropvars, axis=1)
        
    def produce_auc(self):
        
        if self.df_valid is not None:
            assert self.prediction in self.df_valid.columns, 'Must predict on validation set before producing AUC summary'
            print('Producing AUC summary on validation sample')
        else:
            assert self.prediction in self.df_test.columns, 'Must predict on test set before producing AUC summary'
            print('Producing AUC summary on test sample')
            
        auc_summary = []
        if self.population == '1':
            orders_list = ['1']
        elif self.population == '2+':
            orders_list = ['2+']
        elif self.population == '3+':
            orders_list = ['3+']            
        else:
            orders_list = ['total', '1', '2+', '3+']
            
        for date in sorted(self.df_valid.date.unique()):
            for eval_before in [240, np.Inf]:
                for n_orders in orders_list:
                    if n_orders == 'total':
                        df = self.df_valid[(self.df_valid.date==date) & (self.df_valid.days_since_last_purchase_evaldate<eval_before)]
                    elif n_orders == '1':
                        df = self.df_valid[(self.df_valid.date==date) & (self.df_valid.days_since_last_purchase_evaldate<eval_before) & (self.df_valid.n_orders==1)]
                    elif n_orders == '2+':
                        df = self.df_valid[(self.df_valid.date==date) & (self.df_valid.days_since_last_purchase_evaldate<eval_before) & (self.df_valid.n_orders>=2)]                        
                    else:
                        df = self.df_valid[(self.df_valid.date==date) & (self.df_valid.days_since_last_purchase_evaldate<eval_before) & (self.df_valid.n_orders>=3)]

                    if df[self.target].nunique() > 1:
                        valid_auc = roc_auc_score(df[self.target], df[self.prediction])  
                        precision, recall, _ = metrics.precision_recall_curve(df[self.target], df[self.prediction])
                        valid_auc_pr = metrics.auc(recall, precision)
                    else:
                        valid_auc, valid_auc_pr = np.nan, np.nan

                    df_auc = pd.DataFrame({'date':date, 'eval_before':eval_before, 'n_orders':n_orders, 'auc':valid_auc, 'auc_pr':valid_auc_pr}, index=[0])
                    auc_summary.append(df_auc)
                    
        df_auc_summary = pd.concat(auc_summary).reset_index(drop=True) 
        self.df_auc_summary = df_auc_summary.pivot_table(index=['date'], columns=['eval_before', 'n_orders'], values='auc').reset_index()
        self.df_auc_pr_summary = df_auc_summary.pivot_table(index=['date'], columns=['eval_before', 'n_orders'], values='auc_pr').reset_index()
        
        if 'total' in orders_list:
            self.df_auc_summary.columns = ['date', 'excl_ra_1', 'excl_ra_2plus', 'excl_ra_3plus', 'excl_ra_total', '1', '2plus', '3plus', 'total']
            self.df_auc_pr_summary.columns = ['date', 'excl_ra_1', 'excl_ra_2plus', 'excl_ra_3plus', 'excl_ra_total', '1', '2plus', '3plus', 'total']
        else:
            self.df_auc_summary.columns = ['date', 'auc_excl_ra', 'auc']
            self.df_auc_pr_summary.columns = ['date', 'auc_excl_ra', 'auc']
            
    def create_grouped_lapse_summary(self, grouping, boundaries, on='valid'):

        grouped_vars = [var + '_grouping' for var in grouping]
        labelings = [['<=' + str(l) if l >= 100 else '<=0' + str(l) if l >= 10 else '<=00' + str(l) for l in bounds] for bounds in boundaries]

        if on == 'valid':
            df = self.df_valid.copy()
        else:
            df = self.df_orders.copy()

        for var, name, bounds, labels in zip(grouping, grouped_vars, boundaries, labelings):
            df = GenUtilities.dynamic_variable_definition(df, variable=var, boundaries=bounds, name=name, labels=labels, 
                                                          otherwise=f'>{bounds[-1]}')     

        df_summary = df.groupby(grouped_vars)['lapse_outcome'].agg({len, np.mean}).sort_values(grouped_vars)\
                                                                               .rename(columns={'mean':'lapse_rate'})
        df_summary_pred = df.groupby(grouped_vars)['prob_lapse'].agg({np.mean}).sort_values(grouped_vars)\
                                                                                 .rename(columns={'mean':'predicted_lapse_rate'})
        df_summary = df_summary.merge(df_summary_pred, on=grouped_vars, how='left')
        df_summary['lapse_volume'] = (df_summary['lapse_rate'] * df_summary['len']).astype(int)
        df_summary['predicted_lapse_volume'] = np.round(df_summary['predicted_lapse_rate'] * df_summary['len'], 0).astype(int)
        df_summary = df_summary.reset_index()

        group_cols = [col for col in df_summary.columns if 'grouping' in col]
        for i, row in df_summary.iterrows():
            df_filt = df.copy()
            for grouper in group_cols:
                df_filt = df_filt[df_filt[grouper] == row[grouper]]
            if df_filt["lapse_outcome"].nunique() == 1:
                df_summary.loc[i, 'auc'] = np.nan
            else:
                df_summary.loc[i, 'auc'] = roc_auc_score(df_filt["lapse_outcome"], df_filt["prob_lapse"])
                precision, recall, _ = metrics.precision_recall_curve(df_filt["lapse_outcome"], df_filt["prob_lapse"])
                df_summary.loc[i, 'auc_pr'] = metrics.auc(recall, precision)
            df_summary.loc[i, 'auc_volume'] = df_filt.shape[0]

        df_summary['auc_volume'] = df_summary['auc_volume'].astype(int)

        return df_summary
  
    def plot_ave(self, id_var, date=None, qcut=False, filter=None, dataset='test', n_cuts=10):
        
        if dataset == 'valid':
            df = self.df_valid.copy()
        elif dataset == 'test':
            df = self.df_test.copy()
        else:
            df = self.df_orders.copy()
        
        if filter is not None:
            df = df.filter(filter)
            
        if date is not None:
            df = df[df.date == date]     
            
        style = 'ggplot'
        plt.style.use(style)

        plt.rcParams.update({'font.size': 16})

        figs, axs = plt.subplots(1, 2, figsize=(35, 10))      

        for df, title, ax in zip([self.df_train, self.df_test], ['Train', 'Test'], axs):

            labels = [f'q{n}' for n in range(0, n_cuts)]
            if qcut:
                df['cut'] = pd.qcut(df[self.prediction], q=n_cuts)
            else:
                df['cut'] = pd.cut(df[self.prediction], bins=n_cuts)
            summary = df.groupby('cut', observed=False).agg({id_var: len, self.prediction: 'mean', 
                                                             self.target: 'mean'}).reset_index()\
                                       .rename(columns={id_var: 'volume', self.prediction: f'mean_{self.prediction}', 
                                                        self.target: 'actual_rate'})

            ax_twin = ax.twinx()

            summary.plot('cut', 'volume', kind='bar', ax=ax, color='red')
            summary.plot('cut', [f'mean_{self.prediction}', 'actual_rate'], ax=ax_twin, color=['blue', 'green'], linewidth=5)

            ax.set_xticklabels(sorted(summary['cut'].unique()), rotation=45)
            ax.set_title(f'Model Probability and Volumes per Cut, {title} Set')
            ax.set_xlabel('Probability Band', fontsize=18, y=1.5)
            ax.set_ylabel('Volume', fontsize=18)
            ax_twin.set_ylabel('Probability', fontsize=18)
            ax_twin.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            ax_twin.set_ylim([0, 1])
            ax.grid(False)
            ax_twin.grid(True)
            ax.legend(['Volume'], loc=2)
            ax_twin.legend(['Mean Probability of Outcome', 'Actual Outcome Rate'], loc=2, bbox_to_anchor=(0, 0.95));      
        
    def plot_monthly_auc(self, fontsize=16):
      
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(18, 12))

        df = self.df_auc_summary.copy()
        
        if 'total' in df.columns:
            df = df.rename(columns={'total': 'auc', 'excl_ra_total': 'auc_excl_ra'})
        
        df = df[df.date != 'overall']
        df['difference'] = df.auc - df.auc_excl_ra

        date_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        dates = df['date'].apply(lambda x: date_dict[pd.to_datetime(x).month])

        ax.bar(dates, df.auc_excl_ra, label='Excluding Red Alert', color='red')
        ax.bar(dates, df.difference, bottom=df.auc_excl_ra, label='All Individuals', color='blue')

        loc = mtick.MultipleLocator(base=0.01)

        ymin = df.auc_excl_ra.min() * 0.95
        ymax = df.auc.max() * 1.01
        ax.set_xticklabels(dates, rotation=0)
        ax.yaxis.set_major_locator(loc)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel('AUC')
        ax.set_xlabel('Date')
        ax.set_title(f'AUC through {pd.to_datetime(df.date[0]).year}')
        if not self.exclude_red_alert:
            ax.legend()

        plt.show()

    def plot_monthly_auprc(self, fontsize=16):
      
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(18, 12))

        df = self.df_auc_pr_summary.copy()
        
        if 'total' in df.columns:
            df = df.rename(columns={'total': 'auc', 'excl_ra_total': 'auc_excl_ra'})
            
        df = df[df.date != 'overall']
        df['difference'] = df.auc - df.auc_excl_ra
        
        date_dict = {1:'Jan', 2:'Feb', 3:'Mar', 4:'Apr', 5:'May', 6:'Jun', 7:'Jul', 8:'Aug', 9:'Sep', 10:'Oct', 11:'Nov', 12:'Dec'}
        dates = df['date'].apply(lambda x: date_dict[pd.to_datetime(x).month])

        ax.bar(dates, df.auc_excl_ra, label='Excluding Red Alert', color='red')
        ax.bar(dates, df.difference, bottom=df.auc_excl_ra, label='All Individuals', color='blue')

        loc = mtick.MultipleLocator(base=0.01)

        ymin = df.auc_excl_ra.min() * 0.95
        ymax = df.auc.max() * 1.01
        ax.set_xticklabels(dates, rotation=0)
        ax.yaxis.set_major_locator(loc)
        ax.set_ylim(ymin, ymax)
        ax.set_ylabel('AUC')
        ax.set_xlabel('Date')
        ax.set_title(f'AUPRC through {pd.to_datetime(df.date[0]).year}')
        if not self.exclude_red_alert:
            ax.legend()

        plt.show()
        
    def plot_risk_segments(self, df=None, on=None, date=None, risk_segs=['Very Low', 'Low', 'Medium', 'High', 'Very High'],
                           fontsize=16, title='Model Risk Segment Volumes'):
        
        if df is None:
            df = self.df_orders
            
        if on == 'valid':
            df = self.df_valid
        
        if 'risk_segment' not in df.columns:
              df = self.define_risk_segments(df, return_df=True)
            
        if date is not None:
            df = df[df.date==date].copy()
        else:
            df = df[df.date==df.date.max()].copy()
            
        df['red_alert'] = np.where(df.days_since_last_purchase_evaldate >= 240, True, False)
        
        sums = df.groupby(['risk_segment', 'red_alert']).individualid.count().reset_index().rename(columns={'individualid': 'volume'})
        
        red_alerts = sums[sums.red_alert==True]
        non_red_alerts = sums[sums.red_alert==False]
        
        for seg in risk_segs:
            if seg not in red_alerts.risk_segment.values:
                new_frame = pd.DataFrame({'red_alert': [True], 'risk_segment': [seg], 'volume': [0]})
                red_alerts = pd.concat([red_alerts, new_frame], axis=0)
        
        non_red_alerts = non_red_alerts.set_index('risk_segment').reindex(index=risk_segs).reset_index()
        red_alerts = red_alerts.set_index('risk_segment').reindex(index=risk_segs).reset_index()
 
        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.bar(risk_segs, non_red_alerts.volume, label='Normal')
        ax.bar(risk_segs, red_alerts.volume, bottom=non_red_alerts.volume, label='Red Alert', color='black')
        
        max_vol = sums.groupby('risk_segment').volume.sum().max()
        
        ax.set_ylabel('Volume')
        ax.set_xlabel('Probability Segment')
        ax.set_title(title + f'- {str(df.date.max())[0:10]}')
        ax.legend()
        ax.set_ylim(0, 1.05 * max_vol);
    
    def plot_classes(self):
      
        plt.rcParams.update({'font.size': 14})
        fig, axs = plt.subplots(1, 3, figsize=(35, 8))
        
        target = GenUtilities.capitalise_strings(self.target)
        
        ax1 = sns.countplot(x=self.target, data=self.df_orders, ax=axs[0])
        ax1.set_title(f'Distribution of {target} on Train/Test')

        ax2 = sns.countplot(x=self.target, data=self.df_valid, ax=axs[1])
        ax2.set_title(f'Distribution of {target} on Valid (OOT)')
        
        ax3 = sns.countplot(x=self.target, data=self.df_valid[self.df_valid.date==self.df_valid.date.max()], ax=axs[2])
        ax3.set_title(f'Distribution of {target}, {str(self.df_valid.date.max())[0:10]}')           

    def plot_lapse_rates(self, on=None, by=['pop', 'world_region'], fontsize=16, title=None, ax=None):

        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(12, 8))

        if on=='valid':
            df = self.df_valid.copy()
        else:
            df = self.df_orders.copy()

        df['pop'] = np.where(df.n_orders == 1, '1', '2+')
        
        if 'world_region' in by:
            if 'world_region' not in df.columns:
                by = ['pop', 'uk_market']
            else:
                df['world_region'] = np.where(df.world_region.isin(['UNITED KINGDOM', 'UK']), 'UK',
                                np.where(df.world_region == 'UNITED STATES', 'US',
                                         np.where(df.world_region.isin(['FRANCE', 'SPAIN', 'GERMANY', 'IRELAND', 'Rest of Europe']), 'Rest Of Europe',
                                                  'Rest Of World')))

        groupby = ['date'] + by + [self.target]
        lapse_rates = df.groupby(groupby).individualid.count().reset_index().rename(columns={'individualid':'volume'})
        sums = df.groupby(['date'] + by).individualid.count().reset_index().rename(columns={'individualid':'total_volume'})
        lapse_rates = lapse_rates.merge(sums, on=['date'] + by)
        lapse_rates['lapse_rate'] = lapse_rates['volume']/lapse_rates['total_volume']
        summary = lapse_rates[lapse_rates[self.target]==1].sort_values(['date'] + by)
        
        if 'world_region' in by:
            sns.lineplot(data=summary, x='date', y='lapse_rate', hue='world_region')
            plt.xlabel('Date')
            plt.ylabel('Lapse Rate')
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            ax.legend(prop={'size': 12})
        
        else:
            df = summary[(summary['pop'] == '1') & (summary.uk_market == 0)]
            ax.plot(df.date, df.lapse_rate, color='red')
            df = summary[(summary['pop'] == '1') & (summary.uk_market == 1)]
            ax.plot(df.date, df.lapse_rate, color='purple')
            df = summary[(summary['pop'] == '2+') & (summary.uk_market == 0)]
            ax.plot(df.date, df.lapse_rate, color='green')
            df = summary[(summary['pop'] == '2+') & (summary.uk_market == 1)]
            ax.plot(df.date, df.lapse_rate, color='blue')

            ax.set_ylabel('Lapse Rate')
            ax.set_xlabel('Date')
            ax.legend(['1 Purchase, US', '1 Purchase, UK', '2+ Purchases, US', '2+ Purchases, UK'])
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
            
        if title is None:
            if self.exclude_red_alert:
                ra_status = 'Excluded'
            else:
                ra_status = 'Included'
            plt.title(f'Lapse Rates, Population {self.population}, Red Alerts {ra_status}')
        else:
            plt.title(title)    
            
    def plot_importance(self, direction='horizontal', n_features=5, rotation=45, fontsize=16, return_ax=False):

        plt.rcParams.update({'font.size': fontsize})
        fig, ax = plt.subplots(figsize=(15, 9))
        df = self.df_feature_importance.sort_values('imp_rank').reset_index()
        df = df[0:n_features]

        if direction == 'horizontal':
            df = df.sort_values('imp_rank', ascending=False)
            ax.barh(df.feature, df.prop)
            plt.xlabel('Importance Proportion')
            plt.ylabel('Feature')
            plt.title(f'Model Feature Importances')
            ax.set_yticklabels(df.feature, rotation=0)
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        else:
            ax.bar(df.feature, df.prop)
            plt.ylabel('Importance Proportion')
            plt.xlabel('Feature')
            plt.title(f'Model Feature Importances')
            ax.set_xticklabels(df.feature, rotation=rotation)
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        
        if return_ax:
            return ax
          
    def produce_metrics_table(self):
        i = 0
        summaries = []
        
        dataframes = [self.df_train, self.df_test]
        labels = ['train', 'test']
        
        for df, label in zip(dataframes, labels):
            if df is not None:
                
                fpr, tpr, _ = metrics.roc_curve(df[self.target], df[self.prediction])
                auc = round(metrics.auc(fpr, tpr), 4)   
                
                df[f'{self.target}_inverse'] = np.where(df[self.target]==1, 0, 1)
                precision_class0, recall_class0, thresholds = metrics.precision_recall_curve(df[f'{self.target}_inverse'], 1 - df[self.prediction])
                auc_pr_class0 = round(metrics.auc(recall_class0, precision_class0), 4)

                precision_class1, recall_class1, thresholds = metrics.precision_recall_curve(df[self.target], df[self.prediction])
                auc_pr_class1 = round(metrics.auc(recall_class1, precision_class1), 4)  
                
                # Get labels and probabilities for both classes, not just positive class
                y_true = df[[f'{self.target}_inverse', self.target]].values
                y_scores = self.model.predict_proba(df[self.features])
            
                average_precision_class0 = metrics.average_precision_score(df[f'{self.target}_inverse'], 1 - df[self.prediction])
                average_precision_class1 = metrics.average_precision_score(df[self.target], df[self.prediction])
                
                average_precisions = average_precision_score(y_true, y_scores, average=None)  
                average_precision_micro_avg = average_precision_score(y_true, y_scores, average="micro")  
                average_precision_macro_avg = average_precision_score(y_true, y_scores, average="macro") 
                average_precision_weighted_avg = average_precision_score(y_true, y_scores, average="weighted") 
                
                class0_prop = df[df[self.target] == 0].shape[0] / df.shape[0]
                class1_prop = df[df[self.target] == 1].shape[0] / df.shape[0]
                
                assert class0_prop + class1_prop == 1, 'Class proportions must total 1'
                
                ap_to_random_ratio_class0 = average_precision_class0 / class0_prop
                ap_to_random_ratio_class1 = average_precision_class1 / class1_prop
                
                ap_to_random_ratio_weighted_avg = (ap_to_random_ratio_class0 * class0_prop) + (ap_to_random_ratio_class1 * class1_prop)
                
                summary = pd.DataFrame(dict(dset=label, class1_prop=class1_prop, auc=auc, 
                                            auc_pr_class0=auc_pr_class0, auc_pr_class1=auc_pr_class1, 
                                            avg_precision_class0=average_precision_class0,
                                            avg_precision_class1=average_precision_class1, 
                                            avg_precision_micro_avg=average_precision_micro_avg,
                                            average_precision_macro_avg=average_precision_macro_avg,
                                            ap_to_random_ratio_class0=ap_to_random_ratio_class0,
                                            ap_to_random_ratio_class1=ap_to_random_ratio_class1,
                                            ap_to_random_ratio_weighted_avg=ap_to_random_ratio_weighted_avg, 
                                            ap_to_random_ratio_weighted_avg_scor = ap_to_random_ratio_weighted_avg/2), index=[i])
                summaries.append(summary)
                i += 1
                
        self.metrics_summary = pd.concat(summaries, axis=0)
        self.metrics_summary_simple = self.metrics_summary[['dset', 'auc', 'avg_precision_micro_avg']]
        
    def plot_curves(self, on='test', figsize=(30, 10), fontsize=16):
        
        if isinstance(on, pd.DataFrame):
            df = on
            on = 'Supplied DF'
        elif on == 'valid':
            df = self.df_valid
        elif on == 'test':
            df = self.df_test
        elif on == 'train':
            df = self.df_train
            
        target_label = GenUtilities.capitalise_strings(self.target)

        fpr, tpr, _ = metrics.roc_curve(df[self.target], df[self.prediction])
        auc = round(metrics.auc(fpr, tpr), 4)

        df[f'{self.target}_inverse'] = np.where(df[self.target]==1, 0, 1)
        precision_class0, recall_class0, thresholds = metrics.precision_recall_curve(df[f'{self.target}_inverse'], 1 - df[self.prediction])
        auc_pr_class0 = round(metrics.auc(recall_class0, precision_class0), 4)

        precision_class1, recall_class1, thresholds = metrics.precision_recall_curve(df[self.target], df[self.prediction])
        auc_pr_class1 = round(metrics.auc(recall_class1, precision_class1), 4)

        plt.rcParams.update({'font.size': fontsize})
        fig, axs = plt.subplots(1, 3, figsize=figsize)

        axs[0].plot(fpr, tpr, linewidth=5)
        axs[0].set_title(f'Receiver Operator Curve, {on.title()}')
        axs[0].set_xlabel('False Positive Rate')
        axs[0].set_ylabel('True Positive Rate');
        axs[0].annotate(f'Area Under Curve: {auc}',
          xy     = (0.2, 0.1),
          color  = 'black',
          fontsize = 24
        )

        axs[1].plot(recall_class1, precision_class1, linewidth=5)
        axs[1].set_title(f'Precision Recall Curve, {on.title()}, {target_label}')
        axs[1].set_xlabel('True Positive Rate')
        axs[1].set_ylabel('Precision')
        axs[1].set_ylim([0, 1])
        axs[1].annotate(f'Area Under Curve: {auc_pr_class1}',
          xy     = (0.2, 0.1),
          color  = 'black',
          fontsize = 24
        )

        axs[2].plot(recall_class0, precision_class0, linewidth=5)
        axs[2].set_title(f'Precision Recall Curve, {on.title()}, Non-{target_label}')
        axs[2].set_xlabel('True Positive Rate')
        axs[2].set_ylabel('Precision')
        axs[2].set_ylim([0, 1])
        axs[2].annotate(f'Area Under Curve: {auc_pr_class0}',
          xy     = (0.2, 0.1),
          color  = 'black',
          fontsize = 24
        )
        
    def plot_prc(self, on='valid'):
         
        if on == 'valid':
            df = self.df_valid
        else:
            df = self.df_orders
            
        precision, recall, thresholds = metrics.precision_recall_curve(df[self.target], df[self.prediction])
        auc_pr = metrics.auc(recall, precision)
        
        plt.rcParams.update({'font.size': 16})
        fig, ax = plt.subplots(figsize=(12, 8))

        ax.plot(recall, precision, linewidth=5)
        
        ax.set_title(f'Precision Recall Curve')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        
        ax.annotate(f'Area Under Curve: {auc_pr}',
          xy     = (0.2, 0.2),
#           xytext = (1.02*df1.n_rules_found.values[-1], df1.n_gt_1pct.values[-1]),
          color  = 'black',
          fontsize = 16
        );        
        
    def produce_cutoff_summary(self, cutoff_freq=0.05):

        df = self.df_test.copy()

        range_ = np.arange(0, 1 + cutoff_freq, cutoff_freq)

        frames = []
        for cutoff in range_:

            # print('\r', f'Assessing cutoff: {cutoff}', end='')

            df['prediction'] = np.where(df[self.prediction] >= cutoff, 1, 0)

            tp = df[(df.prediction == 1) & (df[self.target] == 1)].shape[0]
            fp = df[(df.prediction == 1) & (df[self.target] == 0)].shape[0]
            tn = df[(df.prediction == 0) & (df[self.target] == 0)].shape[0]
            fn = df[(df.prediction == 0) & (df[self.target] == 1)].shape[0]

            n_positive_predictions = (tp + fp)
            n_negative_predictions = (tn + fn)
            n_positive_actuals = (tp + fn)
            n_negative_actuals = (tn + fp)

            accuracy = (tp + tn) / (tp + fp + tn + fn)

            positive_vol_pct = n_positive_predictions / (n_positive_actuals + n_negative_actuals)
            negative_vol_pct = n_negative_predictions / (n_positive_actuals + n_negative_actuals)
            true_positive_rate = recall = sensitivity = tp / n_positive_actuals if n_positive_actuals != 0 else np.nan  # Percentage of positive actuals that are correctly predicted positive
            true_negative_rate = specificity = tn / n_negative_actuals if n_negative_actuals != 0 else np.nan  # Percentage of negative actuals that are correctly predicted negative        
            false_positive_rate = fp / n_negative_actuals if n_negative_actuals != 0 else np.nan  # Percentage of negative actuals that are incorrectly predicted positive
            false_negative_rate = fn / n_positive_actuals if n_positive_actuals != 0 else np.nan  # Percentage of positive actuals that are incorrectly predicted negative        
            true_positive_prediction_rate = precision = tp / n_positive_predictions if n_positive_predictions != 0 else np.nan # Percentage of positive predictions that are correctly predicted positive
            true_negative_prediction_rate = tn / n_negative_predictions if n_negative_predictions != 0 else np.nan # Percentage of negative predictions that are correctly predicted negative
            false_positive_prediction_rate = fp / n_positive_predictions if n_positive_predictions != 0 else np.nan # Percentage of positive predictions that are incorrectly predicted positive
            false_negative_prediction_rate = fn / n_negative_predictions if n_negative_predictions != 0 else np.nan # Percentage of negative predictions that are incorrectly predicted negative

            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) != 0 else np.nan
            g_mean = np.sqrt(recall * specificity)

            frames.append(pd.DataFrame({'cutoff': cutoff, 'positive_vol_pct':positive_vol_pct,
                                        'negative_vol_pct':negative_vol_pct,
                                        'n_positive_actuals':n_positive_actuals, 
                                        'n_negative_actuals':n_negative_actuals,
                                        'tp':tp, 'fp':fp, 'accuracy':accuracy,
                                        'f1_score':f1_score, 'g_mean':g_mean, 
                                        'true_positive_rate':true_positive_rate, 'true_negative_rate':true_negative_rate,
                                        'false_positive_rate':false_positive_rate, 'false_negative_rate':false_negative_rate,
                                        'true_positive_prediction_rate':true_positive_prediction_rate, 
                                        'true_negative_prediction_rate':true_negative_prediction_rate,
                                        'false_positive_prediction_rate':false_positive_prediction_rate,
                                        'false_negative_prediction_rate':false_negative_prediction_rate}, index=[0]))

        df_report = pd.concat(frames) 

        actual_positive_rate = (df[self.target] == 1).sum() / df.shape[0] # The actual rate of positives un the dataset
        f1_coin = 2 * actual_positive_rate / (actual_positive_rate + 1) # F1 score of a random coin toss on the dataset

        df_report['always_positive_accuracy'] = actual_positive_rate
        df_report['always_negative_accuracy'] = 1 - actual_positive_rate
        df_report['random_coin_accuracy'] = 0.5
        df_report['random_coin_f1_score'] = f1_coin
        
        print('')
        
        frames = []
        columns = [x for x in df_report.columns if x not in ['cutoff', 'always_positive_accuracy', 'always_negative_accuracy', 'random_coin_accuracy', 'random_coin_f1_score']]
        for measure in columns:
            for func, label in zip([np.nanmin, np.nanmax],['min', 'max']):

                value = func(df_report[measure])
                cutoff = df_report.loc[df_report[measure]== value, 'cutoff']
                frame = pd.DataFrame({'function': label, 'measure': measure, 'value':value, 'cutoff': cutoff})
                frame = frame.drop_duplicates(['function', 'measure', 'value'], keep='first')
                frames.append(frame)

        self.df_cutoff_summary = pd.concat(frames)   
        self.df_cutoff_report = df_report.reset_index(drop=True)
        
    def plot_precision(self):
      
        plt.rcParams.update({'font.size': 16})
        _, axs = plt.subplots(1, 2, figsize=(15,10))
        
        ax = axs[0]
        ax.plot(self.df_cutoff_report.positive_vol_pct, self.df_cutoff_report.true_positive_prediction_rate)
        ax.set_xlabel('Volume Percentage of Sample Included (Top X Pct)', fontsize=16)
        ax.set_ylabel('Precision', fontsize=16)
        ax.set_title('Precision by Volume % of Population Ranked by Probability', fontsize=16)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1]);
        
        ax = axs[1]
        ax.plot((1 - self.df_cutoff_report.cutoff), self.df_cutoff_report.true_positive_prediction_rate)
        ax.set_xlabel('Probability Cutoff', fontsize=16)
        ax.set_ylabel('Precision', fontsize=16)
        ax.set_title('Precision by Probability Cutoff', fontsize=16)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xticklabels([str(x) + '%' for x in np.arange(100, -10, -10)])
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1));
        
    def plot_cumulative_response_binary(self, plot_tpr=True):

        plt.rcParams.update({'font.size': 16})
        fig, axs = plt.subplots(1, 2, figsize=(25, 6))
        df = self.df_test
            
        target = GenUtilities.capitalise_strings(self.target)
        
        random_response_positive = df[self.target].mean()
        random_response_negative = 1 - df[self.target].mean()

        ax=axs[0]
        ax.plot(self.df_cutoff_report.positive_vol_pct, self.df_cutoff_report.true_positive_prediction_rate)
        ax.invert_xaxis()
        ax.axhline(random_response_positive, linestyle='--', color='black')
        if plot_tpr:
            ax.plot(self.df_cutoff_report.positive_vol_pct, self.df_cutoff_report.true_positive_rate)
            ax.legend(['Precision (% of Positive Predictions Correct)', 'Precision of Random', f'Recall (% of Positive {target}s Captured)'])
        else:
            ax.legend(['Precision', 'Precision of Random']) 
        ax.set_xlabel('Volume Percentage of Sample Included (Top X Pct)', fontsize=16)
        ax.set_ylabel('Precision', fontsize=16)
        ax.set_title('Precision by Volume % of Population Ranked by Probability', fontsize=16)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1]);
        
        ax=axs[1]
        ax.plot(self.df_cutoff_report.negative_vol_pct, self.df_cutoff_report.true_negative_prediction_rate)
        ax.invert_xaxis()
        ax.axhline(random_response_negative, linestyle='--', color='black')
        if plot_tpr:
            ax.plot(self.df_cutoff_report.negative_vol_pct, self.df_cutoff_report.true_negative_rate)
            ax.legend(['Precision (% of Negative Predictions Correct)', 'Precision of Random', f'Recall (% of Negative {target}s Captured)'])
        else:
            ax.legend(['Precision', 'Precision of Random']) 
        ax.set_xlabel('Volume Percentage of Sample Included (Top X Pct)', fontsize=16)
        ax.set_ylabel('Precision', fontsize=16)
        ax.set_title('Precision by Volume % of Population Ranked by Probability', fontsize=16)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1]);

        fig, axs = plt.subplots(1, 2, figsize=(25, 6))
        ax=axs[0]
        ax.plot(self.df_cutoff_report['cutoff'], self.df_cutoff_report['positive_vol_pct'], linestyle='--')
        ax.plot(self.df_cutoff_report['cutoff'], self.df_cutoff_report['true_positive_prediction_rate'])
        # ax.legend()
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_ylim([0,1])
        ax.set_title('Precision by Cutoff')
        ax.set_xlabel('Cutoff')
        ax.set_ylabel('Percentage')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))        
        
        ax=axs[1]
        ax.plot(self.df_cutoff_report.cutoff, self.df_cutoff_report.accuracy)
        ax.plot(self.df_cutoff_report.cutoff, self.df_cutoff_report.positive_vol_pct)
        # ax.axhline(random_response, linestyle='--', color='black')
        # if plot_tpr:
        #     ax.plot(self.df_cutoff_report.positive_vol_pct, self.df_cutoff_report.true_positive_rate)
        #     ax.legend(['Precision', 'Precision of Random', f'% of {target}s Captured (True Positive Rate)'])
        # else:
        #     ax.legend(['Precision', 'Precision of Random']) 
        ax.set_xlabel('Cutoff', fontsize=16)
        ax.set_ylabel('Accuracy', fontsize=16)
        ax.set_title('Accuracy and Positive Percentage Classified', fontsize=16)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.legend(['Accuracy', 'Percentage Classified as Positive']); 
        
        kds.metrics.report(df[self.target], df[self.prediction], labels=False, title_fontsize=20, text_fontsize=12, figsize=(25, 12))

    

        # ax=axs[1]
        # ax.plot((1 - self.df_cutoff_report.cutoff), self.df_cutoff_report.true_positive_prediction_rate)
        # if plot_tpr:
        #     ax.plot((1 - self.df_cutoff_report.cutoff), self.df_cutoff_report.true_positive_rate)
        #     ax.legend(['Precision', 'True Positive Rate'])
        # ax.set_xlabel('Probability Cutoff', fontsize=16)
        # ax.set_ylabel('Precision', fontsize=16)
        # ax.set_title('Precision by Probability Cutoff', fontsize=16)
        # ax.set_xlim([0,1])
        # ax.set_ylim([0,1])
        # ax.set_xticklabels([str(x) + '%' for x in np.arange(100, -10, -10)])
        # ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        # ax.set_xticks(np.arange(0, 1.1, 0.1))
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(1));      
        
    def plot_cumulative_response(self, plot_tpr=True):

        plt.rcParams.update({'font.size': 16})
        fig, axs = plt.subplots(1, 2, figsize=(25, 6))
        df = self.df_test
            
        target = GenUtilities.capitalise_strings(self.target)
        
        random_response = df[self.target].mean()

        ax=axs[0]
        ax.plot(self.df_cutoff_report.positive_vol_pct, self.df_cutoff_report.true_positive_prediction_rate)
        ax.axhline(random_response, linestyle='--', color='black')
        if plot_tpr:
            ax.plot(self.df_cutoff_report.positive_vol_pct, self.df_cutoff_report.true_positive_rate)
            ax.legend(['Precision', 'Precision of Random', f'% of {target}s Captured (True Positive Rate)'])
        else:
            ax.legend(['Precision', 'Precision of Random']) 
        ax.set_xlabel('Volume Percentage of Sample Included (Top X Pct)', fontsize=16)
        ax.set_ylabel('Precision', fontsize=16)
        ax.set_title('Precision by Volume % of Population Ranked by Probability', fontsize=16)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1]);

        ax = axs[1] 
        kds.metrics.plot_lift(df[self.target], df[self.prediction])
        ax.set_xticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(10))
        ax.set_xlabel('Volume Percentage of Sample Included (Top X Pct Ranked by Probability)', fontsize=16)
        ax.set_ylabel('Lift', fontsize=16)
        ax.set_title(f'Lift Plot (Ratio of Correct Positive {target} Predictions to Random Sampling) ', fontsize=16)
        ax.set_xticks(np.arange(1, 11, 1));
        start, end = ax.get_ylim()
        ax.yaxis.set_ticks(np.arange(start, end, 0.5))
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter('%0.01f'));

        fig, axs = plt.subplots(1, 2, figsize=(25, 6))
        ax=axs[0]
        ax.plot((1 - self.df_cutoff_report.cutoff), self.df_cutoff_report.true_positive_prediction_rate)
        if plot_tpr:
            ax.plot((1 - self.df_cutoff_report.cutoff), self.df_cutoff_report.true_positive_rate)
            ax.legend(['Precision', 'True Positive Rate'])
        ax.set_xlabel('Probability Cutoff', fontsize=16)
        ax.set_ylabel('Precision', fontsize=16)
        ax.set_title('Precision by Probability Cutoff', fontsize=16)
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.set_xticklabels([str(x) + '%' for x in np.arange(100, -10, -10)])
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1));

        ax = axs[1]
        kds.metrics.plot_cumulative_gain(df[self.target], df[self.prediction])
        ax.set_xticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(10))
        ax.set_xlabel('Volume Percentage of Sample Included (Top X Pct)', fontsize=16)
        ax.set_ylabel(f'% of Captured Actual {target}s', fontsize=16)
        ax.set_title(f'Cumulative Gain Plot (% of {target}s Captured)', fontsize=16)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(100));
        ax.set_yticks(np.arange(0, 110, 10))
        ax.set_xticks(np.arange(0, 11, 1));

        fig, axs = plt.subplots(1, 2, figsize=(25, 6))
        ax=axs[0]
        ax.plot(self.df_cutoff_report['cutoff'], self.df_cutoff_report['positive_vol_pct'], linestyle='--')
        ax.plot(self.df_cutoff_report['cutoff'], self.df_cutoff_report['true_positive_prediction_rate'])
        ax.legend([f'% of Total Volume Predicted Positive {target}',f'% of Predicted Positive {target} that are Correct'])
        ax.set_yticks(np.arange(0, 1.1, 0.1))
        ax.set_ylim([0,1])
        ax.set_title('Precision by Cutoff')
        ax.set_xlabel('Cutoff')
        ax.set_ylabel('Percentage')
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))        

        ax=axs[1]
        kds.metrics.plot_ks_statistic(df[self.target], df[self.prediction])
        ax.set_xticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(10))
        ax.set_ylabel(f'% of Captured {target}s/Non-{target}s', fontsize=16)
        ax.set_xlabel('Volume Percentage of Sample Included (Top X Pct)', fontsize=16)
        ax.set_title(f'KS Statistic Plot \n (% of Responders/Non-Responders ({target}s/Non-{target}s) \n by Volume % of Population)', fontsize=16)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
        ax.set_yticks(np.arange(0, 110, 10))
        ax.set_xticks(np.arange(0, 11, 1));
        
        fig, axs = plt.subplots(1, 2, figsize=(25, 6))
        ax=axs[0]
        ax.plot(self.df_cutoff_report.cutoff, self.df_cutoff_report.accuracy)
        ax.plot(self.df_cutoff_report.cutoff, self.df_cutoff_report.positive_vol_pct)
        # ax.axhline(random_response, linestyle='--', color='black')
        # if plot_tpr:
        #     ax.plot(self.df_cutoff_report.positive_vol_pct, self.df_cutoff_report.true_positive_rate)
        #     ax.legend(['Precision', 'Precision of Random', f'% of {target}s Captured (True Positive Rate)'])
        # else:
        #     ax.legend(['Precision', 'Precision of Random']) 
        ax.set_xlabel('Cutoff', fontsize=16)
        ax.set_ylabel('Accuracy', fontsize=16)
        ax.set_title('Accuracy and Positive Percentage Classified', fontsize=16)
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax.set_yticks(np.arange(0.1, 1.1, 0.1))
        ax.set_xticks(np.arange(0, 1.1, 0.1))
        ax.set_xlim([0,1])
        ax.set_ylim([0,1])
        ax.legend(['Accuracy', 'Percentage Classified as Positive']) ;     

        # ax=axs[1]
        # kds.metrics.plot_ks_statistic(df[self.target], df[self.prediction])
        # ax.set_xticklabels([0, 0, 0.2, 0.4, 0.6, 0.8, 1])
        # ax.xaxis.set_major_formatter(mtick.PercentFormatter(10))
        # ax.set_ylabel(f'% of Captured {target}s/Non-{target}s', fontsize=16)
        # ax.set_xlabel('Volume Percentage of Sample Included (Top X Pct)', fontsize=16)
        # ax.set_title(f'KS Statistic Plot \n (% of Responders/Non-Responders ({target}s/Non-{target}s) \n by Volume % of Population)', fontsize=16)
        # ax.yaxis.set_major_formatter(mtick.PercentFormatter(100))
        # ax.set_yticks(np.arange(0, 110, 10))
        # ax.set_xticks(np.arange(0, 11, 1));        
        
    def produce_interpretability(self, variables=None):
        
        print('Producing interpretability report')
                
        if variables is not None:
            if isinstance(variables[0], list):
                assert len(variables[0]) == 2, 'If feeding variable sublists, must be fed as pairs'
                for var_pair in variables:
                    plt.rcParams.update({'font.size': 16})
                    _, axs = plt.subplots(1, 2, figsize=(28, 10))
                    display = PartialDependenceDisplay.from_estimator(self.model, 
                                            self.df_orders[self.features], 
                                            var_pair,
                                            n_jobs=None, 
                                            grid_resolution=20,
                                            pd_line_kw={"color": "tab:orange", "linestyle": "--"}, ax=axs)
                    axs[0].set_title(GenUtilities.capitalise_strings(var_pair[0]))
                    axs[1].set_title(GenUtilities.capitalise_strings(var_pair[1]))
                    display.figure_.suptitle(
                        f"Partial dependence of {GenUtilities.capitalise_strings(self.target)} Probability vs Predictors"
                    )
            else:
                for var in variables:
                    plt.rcParams.update({'font.size': 16})
                    _, axs = plt.subplots(figsize=(28, 10))
                    display = PartialDependenceDisplay.from_estimator(self.model, 
                                            self.df_orders[self.features], 
                                            var,
                                            n_jobs=None, 
                                            grid_resolution=20,
                                            pd_line_kw={"color": "tab:orange", "linestyle": "--"}, ax=axs)

                    axs.set_title(GenUtilities.capitalise_strings(var))
                    display.figure_.suptitle(
                        f"Partial dependence of {GenUtilities.capitalise_strings(self.target)} Probability vs Predictors"
                    )
        else:             
            plt.rcParams.update({'font.size': 16})
            _, axs = plt.subplots(1, 2, figsize=(28, 10))
            display = PartialDependenceDisplay.from_estimator(self.model, 
                                    self.df_orders[self.features], 
                                    ['days_since_last_purchase_evaldate', 'has_givex'],
                                    n_jobs=None, 
                                    grid_resolution=20,
                                    pd_line_kw={"color": "tab:orange", "linestyle": "--"}, ax=axs)

            axs[0].set_title('Days Since Last Purchase')
            axs[1].set_title('Loyalty Status')
            display.figure_.suptitle(
                "Partial dependence of Lapse Probability vs Predictors"
            )

            _, axs = plt.subplots(1, 2, figsize=(28, 10))
            display = PartialDependenceDisplay.from_estimator(self.model, 
                                    self.df_orders[self.features], 
                                    ['n_distinct_categories', 'n_distinct_brands'],
                                    n_jobs=None, 
                                    grid_resolution=20,
                                    pd_line_kw={"color": "tab:orange", "linestyle": "--"}, ax=axs)

            axs[0].set_title('N Distinct Categories')
            axs[1].set_title('N Distinct Brands')
    
    def produce_validation(self, cutoff_freq=0.05):
      
        if self.df_valid is None:
            self.df_valid = self.df_test
            # print('Warning, producing validation on the In-Time test sample since no Out-Of-Time sample supplied (data_scenario_val=None)')
            
        if self.target == 'engagement':
            print('Rounding activity dates to month start for monthly analysis of performance')
            self.df_orders['date'] = self.df_orders['date'].apply(lambda date: date.replace(day=1, hour=0, minute=0, second=0))
            self.df_valid['date'] = self.df_valid['date'].apply(lambda date: date.replace(day=1, hour=0, minute=0, second=0))
            
        # self.produce_auc()
        self.produce_cutoff_summary(cutoff_freq=cutoff_freq)
        
    def plot_customer_lapse_journeys(self, from_day='present', n_cust=3, query=None):
        
        if query is None:
            individuals = self.df_orders.individualid.values[0:n_cust]
        else:
            individuals = self.df_orders.query(query).individualid.values[0:n_cust]
        df_customers = self.df_orders[self.df_orders.individualid.isin(individuals)].copy()

        if from_day =='present':
            df_customers['all_days'] = df_customers['days_since_last_purchase_evaldate'].apply(lambda x: np.arange(x, 366, 1))
        else:
            if self.scaler is not None:
                df_customers['all_days'] = df_customers['days_since_last_purchase_evaldate']\
                                            .apply(lambda x: sorted(self.df_orders['days_since_last_purchase_evaldate'].unique()))
            else:
                df_customers['all_days'] = df_customers['days_since_last_purchase_evaldate'].apply(lambda x: np.arange(from_day, 366, 1))
               
        df_customers = df_customers.explode('all_days')
        df_customers['days_since_last_purchase_evaldate'] = df_customers['all_days'].astype(int)
        df_customers = df_customers.drop(columns='all_days')
        df_customers = self.predict(df_customers)

        n_rows = math.ceil(n_cust / 3)
        figure_height = n_rows * 6 

        _, axs = plt.subplots(n_rows, 3, figsize=(30, figure_height))
        for i, ax in enumerate(axs.flatten()):  
            df = df_customers[df_customers.individualid==individuals[i]]
            n_orders = df.n_orders.values[0]
            n_orders_l2y = df.n_orders_l2y.values[0]
            n_orders_l6m = df.n_orders_l6m.values[0]
            _ = df.plot(x='days_since_last_purchase_evaldate', y='prob_lapse', ax=ax);
            ax.set_title(f'{individuals[i]}, {n_orders} orders, {n_orders_l2y} orders l2y, {n_orders_l6m} orders l6m');  
            ax.set_ylim([0, 1]);
        
    def plot_validation(self, fontsize=16, reduced=False, n_features=8, feats_only=False, with_shapely=False, figsize=(25, 6), 
                        ylim=[0, 1]):
        
        self.df_train.dset_type = 'Train'
        self.df_test.dset_type = 'Test'
        
        target_name = GenUtilities.capitalise_strings(self.target)
        
        self.produce_metrics_table()
           
        if with_shapely:
            shapley = ShapleyPlots(self.model, self.df_test, self.features, observation_id='individualid', 
                                   target=self.target, feature_dictionary=self.feature_dictionary)
            
        if feats_only:
            ModelPlots.feature_importance(self.df_feature_importance, imp_type='gain', feature_dict=self.feature_dictionary,
                                          n_features=n_features, figsize=figsize, fontsize=fontsize, 
                                          title=f'Feature Importance: {target_name}') 
            if with_shapely:
                shapley.plot_importance(n_features=n_features, feature_dict=self.feature_dictionary, figsize=figsize,
                                        fontsize=fontsize)
            ModelPlots.target_interactions(self.df_orders, self.df_feature_importance.loc[0:n_features - 1, 'feature'], self.target, 
                                           feature_dict=self.feature_dictionary, bounds_dictionary=self.bounds_dictionary, ylim=ylim)
        elif reduced:
            ModelPlots.feature_importance(self.df_feature_importance, imp_type='gain', figsize=figsize, fontsize=fontsize,
                                          feature_dict=self.feature_dictionary, n_features=8) 
            if with_shapely:
                shapley.plot_importance(n_features=n_features, feature_dict=self.feature_dictionary, figsize=figsize,
                                        fontsize=fontsize)
                
            ModelPlots.target_interactions(self.df_train, self.df_feature_importance.loc[0:n_features - 1, 'feature'],
                                           self.target, idvar='date', feature_dict=self.feature_dictionary,
                                           bounds_dictionary=self.bounds_dictionary, ylim=ylim)

            self.plot_ave(id_var=self.id_var, qcut=False, n_cuts=5)
            self.plot_cumulative_response_binary()
            self.plot_curves()
        else:
            self.plot_risk_segments(on='valid', title=f'Model Segments: {target_name}')
            _, axs = plt.subplots(1, 2, figsize=(35, 10))
            ModelPlots.feature_importance(self.df_feature_importance, imp_type='gain', ax=axs[0], 
                                          feature_dict=self.feature_dictionary, n_features=n_features)
            ModelPlots.feature_importance(self.df_feature_importance, imp_type='splits', ax=axs[1], 
                                          feature_dict=self.feature_dictionary, n_features=n_features)
            if with_shapely:
                shapley.plot_importance(n_features=n_features, feature_dict=self.feature_dictionary, figsize=figsize,
                                        fontsize=fontsize)
            ModelPlots.target_interactions(self.df_orders, self.df_feature_importance.loc[0:n_features - 1, 'feature'], self.target,
                                           feature_dict=self.feature_dictionary, bounds_dictionary=self.bounds_dictionary, ylim=ylim)
            LapseMonitoring.plot_feature_stability(self.df_orders, self.df_valid, self.df_feature_importance, self.features)
            try:
                self.plot_ave(n_cuts=7)
            except:
                pass
            self.plot_cumulative_response()
            self.plot_lapse_rates()
            self.plot_classes()
            self.plot_curves()
            self.plot_monthly_auc()
            self.plot_monthly_auprc()
            self.plot_sliding_auc()
            
            


# class RandomForestExplainer:
    
#     def __init__(self, model):
        
#         self.model = model
#         self.features = list(model.__dict__['feature_names_in_'])
        
#     def traverse_tree(self, tree_index, tree_):
        
#         data = []
#         for node_index in range(tree_.node_count):
#             if tree_.children_left[node_index] != -1:
#                 left_child_index = tree_.children_left[node_index]
#                 right_child_index = tree_.children_right[node_index]

#                 split_variable = tree_.feature[node_index]
#                 split_threshold = tree_.threshold[node_index]

#                 left_child_class_volumes = tree_.value[left_child_index][0]
#                 right_child_class_volumes = tree_.value[right_child_index][0]

#                 left_child_class_rate = left_child_class_volumes[1] / (left_child_class_volumes[0] + left_child_class_volumes[1])
#                 right_child_class_rate = right_child_class_volumes[1] / (right_child_class_volumes[0] + right_child_class_volumes[1])

#                 data.append({
#                     'tree_number': tree_index,
#                     'node_number': node_index,
#                     'left_child_node': left_child_index,
#                     'right_child_node': right_child_index,
#                     'left_child_class_volumes': left_child_class_volumes,
#                     'left_child_class_rate': left_child_class_rate,
#                     'right_child_class_volumes': right_child_class_volumes,
#                     'right_child_class_rate': right_child_class_rate,
#                     'split_threshold': split_threshold,
#                     'split_variable': feat_vars[split_variable]
#                 })  
        
#         return pd.DataFrame(data)
                
#     def traverse_forest(self):
        
#         dfs = []
#         for tree_index, tree in enumerate(model.estimators_): 
#             tree_ = tree.tree_
#             df_tree = self.traverse_tree(tree_index, tree_)
#             dfs.append(df_tree)
        
#         self.df_trees = pd.concat(dfs)
        
#         self.df_trees['direction_for_class_1'] = np.where(self.df_trees['left_child_class_rate'] > \
#                                                           self.df_trees['right_child_class_rate'], 
#                                                           'left', 
#                                                           'right')



# data = []
# for tree_index, tree in enumerate(model.estimators_):
#     tree_ = tree.tree_
#     for node_index in range(tree_.node_count):
#         if tree_.children_left[node_index] != -1:  # Check if it's not a leaf node
#             left_child_index = tree_.children_left[node_index]
#             right_child_index = tree_.children_right[node_index]
            
#             split_variable = tree_.feature[node_index]
#             split_threshold = tree_.threshold[node_index]
            
#             left_child_class_volumes = tree_.value[left_child_index][0]
#             right_child_class_volumes = tree_.value[right_child_index][0]
            
#             left_child_class_rate = left_child_class_volumes[1] / (left_child_class_volumes[0] + left_child_class_volumes[1])
#             right_child_class_rate = right_child_class_volumes[1] / (right_child_class_volumes[0] + right_child_class_volumes[1])
            
#             data.append({
#                 'tree_number': tree_index,
#                 'node_number': node_index,
#                 'left_child_node': left_child_index,
#                 'right_child_node': right_child_index,
#                 'left_child_class_volumes': left_child_class_volumes,
#                 'left_child_class_rate': left_child_class_rate,
#                 'right_child_class_volumes': right_child_class_volumes,
#                 'right_child_class_rate': right_child_class_rate,
#                 'split_threshold': split_threshold,
#                 'split_variable': feat_vars[split_variable]
#             })


# df = pd.DataFrame(data)

# df['direction_for_class_1'] = np.where(df['left_child_class_rate'] > df['right_child_class_rate'], 'left', 'right')

# df





# from sklearn.tree import _tree

# class RandomForestTraversal:
    
#     def __init__(self, model):
        
#         self.model = model
#         self.features = model
        
        
# def tree_traversal(tree_, feature_names, tree_number):

#     feature_names = [
#         feature_names[i] if i != _tree.TREE_UNDEFINED else "undefined!"
#         for i in tree_.feature
#     ]

#     def recurse(node, df_all_nodes):
  
#         if tree_.feature[node] != _tree.TREE_UNDEFINED:
#             name = feature_names[node]
#             threshold = tree_.threshold[node]
#             df_node = pd.DataFrame({'node':node, 'feature':name, 'threshold':threshold}, index=[node])
            
#             df_all_nodes = pd.concat([df_all_nodes, df_node])
#             left_node = tree_.children_left[node]
#             right_node = tree_.children_right[node]
            
#             # is_split_node = tree_.children_left[node_id] != tree_.children_right[node_id]
            
#             if tree_.feature[left_node] != -2:
#                 df_left_nodes = recurse(left_node, df_all_nodes)
#                 df_all_nodes = pd.concat([df_all_nodes, df_left_nodes])
#             if tree_.feature[right_node] != -2:
#                 df_right_nodes = recurse(right_node, df_all_nodes)
#                 df_all_nodes = pd.concat([df_all_nodes, df_right_nodes])
         
#         return df_all_nodes

#     df = recurse(0, pd.DataFrame())
    
#     df['tree'] = tree_number
    
#     return df[['tree', 'node', 'feature', 'threshold']]

# dfs = []
# for i, estimator in enumerate(model.estimators_):
#     tree = estimator.__dict__['tree_']
#     df = tree_traversal(tree, feat_vars, i)
#     dfs.append(df)
    
# df_trees = pd.concat(dfs)
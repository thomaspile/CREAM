import pandas as pd
import numpy as np
import copy
import json
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, KFold
# from sklearn.utils.testing import ignore_warnings
# from sklearn.exceptions import ConvergenceWarning
from xgboost.sklearn import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
import lightgbm as lgb
from sklearn.metrics import classification_report, roc_auc_score, mean_squared_error, explained_variance_score
from sklearn import metrics
import random
from time import time
# from synapse.ml.lightgbm import LightGBMClassifier
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from hyperopt import hp, STATUS_OK, tpe, atpe, fmin, Trials, SparkTrials, pyll
import multiprocessing as mp
import warnings
# import shap
import math
from functools import partial

# from utilities import GenUtilities, GenPlots, PlotUtilities, StandardDictionaries

warnings.filterwarnings("ignore", message="Found `num_iterations` in params. Will use it instead of argument")

def wape(actual, pred):
    abs_diff = np.abs(actual - pred).sum()
    sum_actual = actual.sum()
    wape = abs_diff/sum_actual
    return wape

class ModelBuilder:
  
    def build_rf(self, X_train, y_train, X_test, y_test, X_valid, y_valid, params, n_jobs=1, seed=123, outcome_type='classification'):

        random.seed(seed)
        if params is None:
            params = {}
        
        if outcome_type == 'classification':
            model = RandomForestClassifier(n_jobs=n_jobs, random_state=seed)
        else:
            model = RandomForestRegressor(n_jobs=n_jobs, random_state=seed)
        model.set_params(**params)

        #t0 = time()
        model.fit(X_train, y_train)
        #print(f'Best RF built in {round(time() - t0, 3)} seconds')

        if X_valid.shape[0]==0:
            X_valid = None
            y_valid = None 

        if outcome_type == 'classification':
            errors = self.evaluate_classifier(model, X_train, y_train, X_test, y_test, X_valid, y_valid, ret=True)
        else:
            errors = self.evaluate_regressor(model, X_train, y_train, X_test, y_test, X_valid, y_valid, ret=True)

        importance = self.feature_importances(model, X_train.columns, model_type='rf')

        return model, importance, errors

    def build_xgb(self, X_train, y_train, X_test, y_test, X_valid, y_valid, params, n_jobs=1, seed=123, outcome_type='classification'):

        random.seed(seed)
        if params is None:
            params = {}
          
        if outcome_type == 'classification':
            model = XGBClassifier(n_jobs=n_jobs)
        else:
            model = XGBRegressor(n_jobs=n_jobs)
            
        model.set_params(**params)

        if 'early_stopping_rounds' in params or 'early_stopping_round' in params:
            try:
                es = params['early_stopping_round']
            except:
                es = params['early_stopping_rounds']

            model.fit(X_train, 
                      y_train,
                      early_stopping_rounds=es,
    #                   eval_metric='auc',
                      eval_set=[(X_test, y_test)],
                      verbose=False)
        else:
            model.fit(X_train, y_train)

        if X_valid.shape[0]==0:
            X_valid = None
            y_valid = None 

        if outcome_type == 'classification':
            errors = self.evaluate_classifier(model, X_train, y_train, X_test, y_test, X_valid, y_valid, ret=True)
        else:
            errors = self.evaluate_regressor(model, X_train, y_train, X_test, y_test, X_valid, y_valid, ret=True)
        
        try:
            importance = self.feature_importances(model, X_train.columns, model_type='xgb')
        except Exception as e:
            print(e)
            importance = None
            
        return model, importance, errors
      
    def build_lgb_spark(self):
        
        random.seed(seed)
        
        if params is None:
            params = {}
        params = params.copy()
        
        lgb = LightGBMClassifier(**params)
        model = lgb.fit(train)

    def build_lgb(self, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None, params=None, n_jobs=1, seed=123, early_stopping_rounds=False, outcome_type='classification'):

        random.seed(seed)

        if params is None:
            params = {}
        params = params.copy()

        # Make sure parameters that need to be integers are integers
        for parameter_name in ['max_depth', 'min_child_samples', 'num_leaves', 'subsample_for_bin']:
            try:
                params[parameter_name] = int(params[parameter_name])
            except:
                pass
              
        if 'monotone_constraints' in params.keys():
            if isinstance(params['monotone_constraints'], dict):
                monotone_constraints = [params['monotone_constraints'][col] if col in params['monotone_constraints'].keys() else 0 for col in X_train.columns]
                params['monotone_constraints'] = monotone_constraints             
            elif isinstance(params['monotone_constraints'], list):
                monotone_constraints = [1 if col in params['monotone_constraints'] else 0 for col in X_train.columns]
                params['monotone_constraints'] = monotone_constraints         
            elif isinstance(params['monotone_constraints'], str):
                monotone_constraints = [1 if col == params['monotone_constraints'] else 0 for col in X_train.columns]
                params['monotone_constraints'] = monotone_constraints   
                
        if 'focal_loss' in params.keys():
            focal_loss = params['focal_loss']
            del params['focal_loss']
        else:
            focal_loss = False
            
        if 'rank' in params.keys():
            if params['rank']:
                model = lgb.LGBMRanker(importance_type='gain')
            else:
                if outcome_type == 'classification':
                    model = LGBMClassifier(n_jobs=n_jobs)
                else:
                    model = LGBMRegressor(n_jobs=n_jobs)
        else:
            if outcome_type == 'classification':
                if focal_loss:
                    model = LGBMClassifier(n_jobs=n_jobs, objective=focal_loss_lgb)
                else:
                    model = LGBMClassifier(n_jobs=n_jobs)
            else:
                model = LGBMRegressor(n_jobs=n_jobs)

        params.update({'n_jobs': n_jobs})

        t0 = time()    

        if 'early_stopping_rounds' in params.keys() or 'early_stopping_round' in params.keys():
            try:
                rounds = params['early_stopping_round']
                del params['early_stopping_round']
                print('Fitting model with early stopping')  
            except:
                rounds = params['early_stopping_rounds']
                del params['early_stopping_rounds']

            early_stopping = lgb.early_stopping(stopping_rounds=rounds, verbose=True)
        else:
            early_stopping = False
                
        if focal_loss:
            eval_metric = focal_loss_lgb_eval_error
            print('Fitting model with focal loss objective')
        elif 'eval_metric' in params.keys():
            eval_metric = params['eval_metric']
            del params['eval_metric']
        else:
            if outcome_type == 'classification':
                eval_metric = 'auc'
            else:
                eval_metric = 'rmse'  

        model.set_params(**params)
                
        if early_stopping != False:  
            model.fit(X_train, 
                      y_train,
                      callbacks=[early_stopping],
                      eval_metric=eval_metric,
                      eval_set=[(X_test, y_test)])
             
        else:
            model.set_params(**params)
            model.fit(X_train, y_train)

        print(f'LGB built in {round(time() - t0, 3)} seconds')
        
        if X_valid is None:
            y_valid = None
        elif X_valid.shape[0]==0:
            X_valid = None
            y_valid = None 
        
        if outcome_type == 'classification':
            errors = self.evaluate_classifier(model, X_train, y_train, X_test, y_test, X_valid, y_valid, ret=True)
        else:
            errors = self.evaluate_regressor(model, X_train, y_train, X_test, y_test, X_valid, y_valid, ret=True)

        importance = self.feature_importances(model, X_train.columns)

        return model, importance, errors
      
    def build_lasso(self, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None, params=None):
        
        model = Lasso()
        if params is not None:
            model.set_params(**params)
        
        model.fit(X_train, y_train)
        
        if X_valid is not None:
            if X_valid.shape[0]==0:
                X_valid = None
                y_valid = None 
        
        errors = self.evaluate_regressor(model, X_train, y_train, X_test, y_test, X_valid, y_valid, ret=True)
        
        importance = self.feature_importances_linear(model, X_train.columns)
        
        return model, importance, errors
      
    def build_elasticnet(self, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None, params=None):
        
        model = ElasticNet()
        if params is not None:
            model.set_params(**params)
        
        model.fit(X_train, y_train)

        if X_valid is not None:
            if X_valid.shape[0]==0:
                X_valid = None
                y_valid = None        
        
        errors = self.evaluate_regressor(model, X_train, y_train, X_test, y_test, X_valid, y_valid, ret=True)
        
        importance = self.feature_importances_linear(model, X_train.columns)
        
        return model, importance, errors

    def build_ridge(self, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None, params=None):
        
        model = Ridge()
        if params is not None:
            model.set_params(**params)
        
        model.fit(X_train, y_train)
        
        if X_valid is not None:
            if X_valid.shape[0]==0:
                X_valid = None
                y_valid = None 

        errors = self.evaluate_regressor(model, X_train, y_train, X_test, y_test, X_valid, y_valid, ret=True)
        
        importance = self.feature_importances_linear(model, X_train.columns)
        
        return model, importance, errors
       
    def evaluate_classifier(self, model, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None, ret=False):

        try: 
            y_pred_train = model.predict_proba(X_train)[:, 1]
            train_auc = roc_auc_score(y_train, y_pred_train)
            precision, recall, _ = metrics.precision_recall_curve(y_train, y_pred_train)
            train_auc_pr = metrics.auc(recall, precision)

            print('Train AUC: ' + str(train_auc))
            print('Train AUC_PR: ' + str(train_auc_pr))
        except Exception as e:
            train_auc = None
            print('Unable to predict on train set')
            print(f'Error returned: {e}')
        classification_metrics = {'train_auc': train_auc, 'train_auc_pr':train_auc_pr}

        if X_test is not None:
            try: 
                y_pred_test = model.predict_proba(X_test)[:, 1]
                test_auc = roc_auc_score(y_test, y_pred_test)
                precision, recall, _ = metrics.precision_recall_curve(y_test, y_pred_test)
                test_auc_pr = metrics.auc(recall, precision)
                
                print('Test AUC: ' + str(test_auc))
                print('Test AUC_PR: ' + str(test_auc_pr))
            except Exception as e:
                test_auc = None
                test_auc_pr = None
                print('Unable to predict on test set')
                print(f'Error returned: {e}')
            classification_metrics.update({'test_auc': test_auc, 'test_auc_pr':test_auc_pr})
        else:
            classification_metrics.update({'test_auc': None})

        if X_valid is not None:
            try:  
                y_pred_valid = model.predict_proba(X_valid)[:, 1]
                valid_auc = roc_auc_score(y_valid, y_pred_valid)
                precision, recall, _ = metrics.precision_recall_curve(y_valid, y_pred_valid)
                valid_auc_pr = metrics.auc(recall, precision)
                print('Valid AUC: ' + str(valid_auc_pr))
            except Exception as e:
                valid_auc = None
                valid_auc_pr = None
                print('Unable to predict on validation set')
                print(f'Error returned: {e}')
                print(f'X_valid shape: {X_valid.shape}')
            classification_metrics.update({'valid_auc': valid_auc, 'valid_auc_pr':valid_auc_pr})
        else:
            classification_metrics.update({'valid_auc': None})

        if ret:
            return classification_metrics
          
    def evaluate_regressor(self, model, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None, ret=False):

        try: 
            y_pred_train = model.predict(X_train).astype('float32')  
            train_error = np.sqrt(mean_squared_error(y_train, y_pred_train))
            train_variance = explained_variance_score(y_train, y_pred_train)
            train_wape = wape(y_train, y_pred_train)
        except Exception as e:
            train_error = None
            train_variance = None
            train_wape = None
            print('Unable to predict on train set')
            print(f'Error returned: {e}')
        regression_metrics = {'train_error': train_error, 'train_explained_variance': train_variance, 'train_wape':train_wape}

        try: 
            y_pred_test = model.predict(X_test).astype('float32')  
            test_error = np.sqrt(mean_squared_error(y_test, y_pred_test))
            test_variance = explained_variance_score(y_test, y_pred_test)
            test_wape = wape(y_test, y_pred_test)
        except Exception as e:
            test_error = None
            test_variance = None
            test_wape = None
            print('Unable to predict on test set')
            print(f'Error returned: {e}')
        regression_metrics.update({'test_error': test_error, 'test_explained_variance': test_variance, 'test_wape':test_wape})

        if X_valid is not None:
            try:  
                y_pred_valid = model.predict(X_valid).astype('float32')  
                try:
                    valid_error = np.sqrt(mean_squared_error(y_valid, y_pred_valid))
                    valid_variance = explained_variance_score(y_valid, y_pred_valid)
                    valid_wape = wape(y_valid, y_pred_valid)
                except Exception as e:
                    valid_error = None
                    valid_variance = None
                    valid_wape = None
                    print('Unable to calculate rmse on validation set')
                    print(f'Error returned: {e}')
                    print(f'y_valid shape: {y_valid.shape}')
                    print(f'y_pred_valid shape: {y_pred_valid.shape}')
            except Exception as e:
                valid_error = None
                valid_variance = None
                print('Unable to predict on validation set')
                print(f'Error returned: {e}')
                print(f'X_valid shape: {X_valid.shape}')
            regression_metrics.update({'valid_error': valid_error, 'valid_explained_variance': valid_variance, 'valid_wape':valid_wape})
        else:
            regression_metrics.update({'valid_error': None, 'valid_explained_variance': None, 'valid_wape':None})

        print('Train error: ' + str(train_error))
        print('Test error: ' + str(test_error))   
        if X_valid is not None:
            print('Valid error: ' + str(valid_error))         
        print('Train explained variance: ' + str(train_variance))
        print('Test explained variance: ' + str(test_variance))
        if X_valid is not None:
            print('Valid explained variance: ' + str(valid_variance))
        print('Train WAPE: ' + str(train_wape))
        print('Test WAPE: ' + str(test_wape))
        if X_valid is not None:
            print('Valid WAPE: ' + str(valid_wape))

        if ret:
            return regression_metrics
      
    def split_train_test_val(self, df, target, test_size=0.25, dt_var=None, split_samples_by=None, stratify=None, val_date_from=None, seed=123):
      
        def train_test_split_by_key(data, target, usecols, key, test_size, random_state=123):
            """
            Split a dataset into training and testing set by a certain category key.
            Returns the same output as sklearn's train_test_split
            """
            train = data[data[key].isin(pd.Series(data[key].unique()).sample(frac=1-test_size, random_state=random_state))][usecols+[target]+[key]]
            test = data[~data[key].isin(train[key].unique())][usecols+[target]+[key]]
            print('Test Size: ', test.shape[0] / data.shape[0])
            return train.drop([target, key], axis=1), test.drop([target, key], axis=1), train[target], test[target]

        if val_date_from is not None:
            df_train_test = df[df[dt_var]<val_date_from] 
            df_val = df[df[dt_var]>=val_date_from]
        else:
            df_train_test = df.copy()

        if split_samples_by == 'day':

            print('Splitting train and test by day')

            df_train_test_c = df_train_test.copy()
            df_train_test_c['date_key'] = pd.to_datetime(df_train_test[dt_var].dt.date)

            use_cols = [col for col in df_train_test_c.columns if col not in [target, 'date_key']]

            X_train, X_test, y_train, y_test = train_test_split_by_key(df_train_test_c,
                                                                       target=target,
                                                                       usecols=use_cols,
                                                                       key='date_key',
                                                                       test_size=test_size,
                                                                       random_state=seed)

        elif split_samples_by =='week':

            print('Splitting train and test by week')

            df_train_test_c = df_train_test.copy()
            df_train_test_c['date_key'] = df_train_test_c[dt_var].dt.year.astype(str) + df_train_test_c[dt_var].dt.isocalendar().week.astype(str)
            use_cols = [col for col in df_train_test_c.columns if col not in [target, 'date_key']]

            X_train, X_test, y_train, y_test = train_test_split_by_key(df_train_test_c,
                                                                       target=target,
                                                                       usecols=use_cols,
                                                                       key='date_key',
                                                                       test_size=test_size,
                                                                       random_state=seed)

        elif isinstance(split_samples_by, str):
            df_train_test['key'] = df_train_test[split_samples_by]
            use_cols = [col for col in df_train_test.columns if col not in [target, 'key']]
            X_train, X_test, y_train, y_test = train_test_split_by_key(df_train_test,
                                                               target=target,
                                                               usecols=use_cols,
                                                               key='key',
                                                               test_size=test_size,
                                                               random_state=seed)
        else:
            if stratify not in [None, False]:
                stratify = df_train_test[stratify]
            X_train, X_test, y_train, y_test = train_test_split(df_train_test.drop(columns=target), 
                                                                df_train_test[target], 
                                                                test_size=test_size, 
                                                                random_state=seed,
                                                                stratify=stratify)

        if val_date_from is not None:
            X_valid = df_val.drop(columns=target)
            y_valid = df_val[target]
        else:
            X_valid = pd.DataFrame(dict(zip(X_train.columns, [[] for x in X_train.columns])))
            y_valid = pd.DataFrame()

        return X_train, X_test, X_valid, y_train, y_test, y_valid
      
    def feature_importances_linear(self, model, features, sort=True):

        importance = pd.DataFrame({'feature':features, 
                                  'coef':model.coef_, 
                                  'prop':abs(model.coef_)/abs(model.coef_).sum()})
        if sort:
            importance = importance.sort_values('prop', ascending=False).reset_index(drop=True)

        return importance
            
    def feature_importances(self, model, features, model_type='lgb', sort=True):
#         importance = pd.DataFrame({'feature':features, 
#                                    'imp':model.feature_importances_, 
#                                    'prop':model.feature_importances_/model.feature_importances_.sum()})
        
        if model_type == 'lgb':
            gain = model.booster_.feature_importance(importance_type='gain')
            splits = model.booster_.feature_importance(importance_type='split')
            importance = pd.DataFrame({'feature':features, 
                                       'gain':gain, 
                                       'splits':splits,
                                       'prop_gain':gain/gain.sum(),
                                       'prop_splits':splits/splits.sum()})     
        elif model_type == 'rf':
            gain = model.feature_importances_
            importance = pd.DataFrame({'feature':features, 
                            'gain':gain, 
                            'prop_gain':gain/gain.sum()})     
        else:
            if not json.loads(model.get_booster().save_config())['learner']['gradient_booster']['name'] == 'gblinear':
                splits = np.array(list(model.get_booster().get_score(importance_type="weight").values()))
                features = np.array(list(model.get_booster().get_score(importance_type="weight").keys()))              
                gain = np.array(list(model.get_booster().get_score(importance_type="gain").values()))
                importance = pd.DataFrame({'feature':features, 
                                           'gain':gain, 
                                           'splits':splits,
                                           'prop_gain':gain/gain.sum(),
                                           'prop_splits':splits/splits.sum()})                
            else:
                weight = np.array(list(model.get_booster().get_score(importance_type="weight").values()))
                features = np.array(list(model.get_booster().get_score(importance_type="weight").keys()))                              
                importance = pd.DataFrame({'feature':features, 
                                           'weight':weight, 
                                           'prop_weight':weight/weight.sum()})
                
        if 'prop_splits' in importance.columns:
            importance['rank_splits'] = importance.prop_splits.rank(ascending=False)             
        if 'prop_gain' in importance.columns:
            importance['rank_gain'] = importance.prop_gain.rank(ascending=False)      
            if sort:
                importance = importance.sort_values('prop_gain', ascending=False).reset_index(drop=True)
        else:
            if sort:
                importance = importance.sort_values('weight', ascending=False).reset_index(drop=True)        
            importance['rank_weight'] = importance.weight.rank(ascending=False)
     
        return importance

    def plot_importance(self, df, n_features=10, rotation=90, fontsize=12, model_type='XGBoost'):

        fig, ax = plt.subplots(figsize=(15, 9))
        df = df.copy().sort_values('rank_gain').reset_index()
        df = df[0:n_features]

        ax.bar(df.feature, df.gain)
        plt.ylabel('Importance Proportion')
        plt.xlabel('Feature')
        plt.title(f'Model Feature Importances, {model_type}')
        ax.set_xticklabels(df.feature, rotation=rotation, fontsize=fontsize)

    def replace_value(self, df, columns, replace=np.nan, with_val=99999):

        if np.isnan(replace):
            print('Replacing missing values')
            for var in columns:
                df[var] = np.where(df[var].isnull(), with_val, df[var])
        else:
            for var in columns:
                df[var] = np.where(df[var] == replace, with_val, df[var])      

        return df

      
def focal_loss_lgb(y_pred, dtrain, alpha, gamma):
    a,g = alpha, gamma
    y_true = dtrain.label
    def fl(x,t):
        p = 1/(1+np.exp(-x))
        return -( a*t + (1-a)*(1-t) ) * (( 1 - ( t*p + (1-t)*(1-p)) )**g) * ( t*np.log(p)+(1-t)*np.log(1-p) )
    partial_fl = lambda x: fl(x, y_true)
    grad = derivative(partial_fl, y_pred, n=1, dx=1e-6)
    hess = derivative(partial_fl, y_pred, n=2, dx=1e-6)
    return grad, hess

def focal_loss_lgb_eval_error(y_pred, dtrain, alpha, gamma):
    a,g = alpha, gamma
    y_true = dtrain.label
    p = 1/(1+np.exp(-y_pred))
    loss = -( a*y_true + (1-a)*(1-y_true) ) * (( 1 - ( y_true*p + (1-y_true)*(1-p)) )**g) * ( y_true*np.log(p)+(1-y_true)*np.log(1-p) )
    # (eval_name, eval_result, is_higher_better)
    return 'focal_loss', np.mean(loss), False
  

class ShapleyPlots:
  
    def __init__(self, model, df_valid, features, observation_id, class_idx=1, n_observations=None, target=None, feature_dictionary=None):
        
        self.class_idx = 1
        self.model = model
        self.observation_id = observation_id
        self.df_valid = df_valid
        if n_observations is not None:
            self.df_valid = self.df_valid.iloc[0: n_observations]
        self.features = features
        self.target = target
        self.feature_dictionary = feature_dictionary
        self.explainer = shap.TreeExplainer(self.model.booster_)
        self.shap_values = self.explainer.shap_values(self.df_valid[self.features])[self.class_idx]
        
    def plot_summary(self, n_features=20):
        
        shap.summary_plot(self.shap_values, self.df_valid[self.features], max_display=n_features)
        
    def plot_importance(self, n_features=10, feature_dict=None, ax=None, figsize=(7, 8), fontsize=14):

        global_importances = np.abs(self.shap_values).mean(0)[:-1]
        inds = np.argsort(-global_importances)
        inds2 = np.flip(inds[:n_features], 0)

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)
        
        if feature_dict is not None:
            features = [feature_dict[feat] if feat in feature_dict.keys() else feat for feat in list(self.df_valid[self.features].columns[inds2])]
        else:
            features = self.df_valid[self.features].columns[inds2]
        
        ax.barh(features, global_importances[inds2], align='center')
        ax.set_title('Shapely Feature Importance')
        ax.set_xlabel('Mean abs. SHAP value (impact on model output)', fontsize=fontsize)
        ax.set_ylabel('Feature Name', fontsize=fontsize)
    
    def plot_dependencies(self, features=None, n_cols=3):
        
        if isinstance(features, str):
            features = [features]
        n_plots = len(features)
        
        if n_plots == 1:
            shap.dependence_plot(features, self.shap_values, self.df_valid[self.features], show=False)
            
        else:
            n_rows = n_plots/n_cols
            n_rows = math.ceil(n_rows)
            _, axs = plt.subplots(n_rows, n_cols, figsize=(n_cols * 7, n_rows * 5))
            for feature, ax in zip(features, axs.flatten()):
                shap.dependence_plot(feature, self.shap_values, self.df_valid[self.features], ax=ax, show=False)
                
    def explain_observation(self, observation_id=None, contribution_threshold=0.05, matplotlib=True):
        
        if observation_id is not None:
            idx = list(self.df_valid[self.observation_id].values).index(observation_id)
        
        df = self.df_valid.iloc[idx]
        shap_values = self.shap_values[idx, :]
        
        assert df[self.observation_id] == observation_id, 'Not finding correct ID from idx'
        
        if self.target is not None:
            target_capitalised = GenUtilities.capitalise_strings(self.target)
            id_capitalised = GenUtilities.capitalise_strings(self.observation_id)
            print(f'{id_capitalised}: {observation_id}')
            print(f'{target_capitalised}: {np.round(df[self.target], 2)}%')
        
        if self.feature_dictionary is not None:
            feature_names = [self.feature_dictionary[feat] if feat in self.feature_dictionary.keys() else feat for feat in self.features]
            shap.force_plot(self.explainer.expected_value[self.class_idx], shap_values, df[self.features], 
                            feature_names=feature_names, contribution_threshold=contribution_threshold, matplotlib=matplotlib)  
        else:
            shap.force_plot(self.explainer.expected_value[self.class_idx], shap_values, df[self.features], 
                            contribution_threshold=contribution_threshold, matplotlib=matplotlib)  

class ModelPlots:
  
    def feature_importance(df_feature_importance, direction='horizontal', imp_type='gain', n_features=5, figsize=(15, 9), rotation=45, fontsize=16, ax=None, return_ax=False, hide_feature_label=True, feature_dict=None, title=None):

        plt.rcParams.update({'font.size': fontsize})
        
        if imp_type == 'gain':
            prop_type = 'prop_gain'
            rank_type = 'rank_gain'
        else:
            prop_type = 'prop_splits'
            rank_type = 'rank_splits'            
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        df = df_feature_importance.sort_values(rank_type).reset_index()
        df = df[0:n_features]
        
        if feature_dict is not None:
          
            def get_feature_label(feature, feature_dict):
              
                if feature in feature_dict.keys(): 
                    return feature_dict[feature]
                else:
                    return feature
                
            df['feature'] = df['feature'].apply(lambda feature: get_feature_label(feature, feature_dict))

        if direction == 'horizontal':
            df = df.sort_values(rank_type, ascending=False)
            ax.barh(df.feature, df[prop_type])
            plt.xlabel('Importance Proportion')
            if not hide_feature_label:
                plt.ylabel('Feature')
            if title is not None:
                ax.set_title(title)
            else:                
                ax.set_title(f'Model Feature Importances, {GenUtilities.capitalise_strings(imp_type)}')
            # ax.set_yticklabels(df.feature, rotation=0)
            # ax.xaxis.set_major_locator(mtick.FixedLocator(ax.get_xticks().tolist()))
            ax.xaxis.set_major_formatter(mtick.PercentFormatter(1))
        else:
            ax.bar(df.feature, df[prop_type])
            plt.ylabel('Importance Proportion')
            if not hide_feature_label:
                plt.xlabel('Feature')
            if title is not None:
                ax.set_title(title)
            else:
                ax.set_title(f'Model Feature Importances, {GenUtilities.capitalise_strings(imp_type)}')
            ax.set_xticklabels(df.feature, rotation=rotation)
            # ax.yaxis.set_major_locator(mtick.FixedLocator(ax.get_yticks().tolist()))
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1))

        if return_ax:
            return ax
          
    def target_interactions(df, features, target, idvar='individualid', bounds_dictionary=None, feature_dict=None, figsize=(33, 10), fontsize=18, ylim=[0, 1]):
        
        target_formatted = GenUtilities.capitalise_strings(target)
        
        if feature_dict is not None:
          
            def get_feature_label(feature, feature_dict):
                
                if feature in feature_dict.keys(): 
                    return feature_dict[feature]
                else:
                    return feature
                
            features_formatted = [get_feature_label(feat, feature_dict) for feat in features]
        else:
            features_formatted = features
        
        assert len(features)%2 == 0, 'Must supply even number of features so plots format correctly'
        features = [[features[i], features[i+1]] for i in range(len(features)) if i%2 == 0]
        features_formatted = [[features_formatted[i], features_formatted[i+1]] for i in range(len(features_formatted)) if i%2 == 0]
        
        for feats, feats_strings in zip(features, features_formatted):
            _, axs = plt.subplots(1, 2, figsize=figsize)
            if bounds_dictionary is not None:
                boundaries = bounds_dictionary.get(feats[0])
            else:
                boundaries = None
            GenPlots.plot_interactions(df, feats[0], feats_strings[0], target, boundaries=boundaries, idvar=idvar, remove_outliers=False, n_cuts=5, fontsize=fontsize,
                                       title=f'{feats_strings[0]} vs {target_formatted} Rate', 
                                       ylabel=f'{target_formatted} Rate', ax=axs[0], ylim=ylim)
            boundaries = bounds_dictionary.get(feats[1])       
            GenPlots.plot_interactions(df, feats[1], feats_strings[1], target, boundaries=boundaries, idvar=idvar, remove_outliers=False, n_cuts=5, fontsize=fontsize,
                                       title=f'{feats_strings[1]} vs {target_formatted} Rate', 
                                       ylabel=f'{target_formatted} Rate', ax=axs[1], ylim=ylim)
            
    def ave_reg(df_predictions, outcome, predicted, n_cuts=10, ax=None, df_label=None):

        df = df_predictions.copy()
        labels = [f'q{n}' for n in range(0, n_cuts)]
        df['cut'] = pd.qcut(df[predicted], n_cuts, labels=labels)
        summary = df.groupby('cut').agg({'individualid': len, predicted: 'mean', outcome: 'mean'}).reset_index()\
                    .rename(columns={'individualid': 'volume', predicted: f'mean_{predicted}', outcome: f'actual_{outcome}'})

        style = 'ggplot'
        plt.style.use(style)

        plt.rcParams.update({'font.size': 16})
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 8))

        ax2 = ax.twinx()

        summary.plot('cut', 'volume', kind='bar', ax=ax, color='red')
        summary.plot('cut', [f'mean_{predicted}', f'actual_{outcome}'], ax=ax2, color=['blue', 'green'], linewidth=5)
        
        if df_label is None:
            title = 'Model Predictions and Volumes per Cut'
        else:
            title = f'Model Predictions and Volumes per Cut, {df_label}'
        
        line_labels = GenUtilities.capitalise_strings([f'mean_{predicted}', f'actual_{outcome}'])
        
        ax.set_xticklabels(labels, rotation=0)
        ax.set_title(title)
        ax.set_xlabel('Cut', fontsize=18, y=1.5)
        ax.set_ylabel('Volume', fontsize=18)
        ax2.set_ylabel('Future Value', fontsize=18)
        ax.grid(False)
        ax2.grid(True)
        ax.legend(['Volume'], loc=2)
        ax2.legend([line_labels[0], line_labels[1]], loc=2, bbox_to_anchor=(0, 0.95));     
          
    def ave(df_predictions, target='lapse_outcome', probabilities='prob_lapse', n_cuts=10, ax=None):

        df = df_predictions.copy()
        labels = [f'q{n}' for n in range(0, n_cuts)]
        df['cut'] = pd.cut(df[probabilities], bins=n_cuts, labels=labels)
        summary = df.groupby('cut').agg({'individualid': len, probabilities: 'mean', target: 'mean'}).reset_index()\
                    .rename(columns={'individualid': 'volume', probabilities: f'mean_{probabilities}', target: 'actual_rate'})

        style = 'ggplot'
        plt.style.use(style)

        plt.rcParams.update({'font.size': 16})
        if ax is None:
            fig, ax = plt.subplots(figsize=(20, 8))

        ax2 = ax.twinx()

        summary.plot('cut', 'volume', kind='bar', ax=ax, color='red')
        summary.plot('cut', [f'mean_{probabilities}', f'actual_rate'], ax=ax2, color=['blue', 'green'], linewidth=5)

        ax.set_xticklabels(labels, rotation=0)
        ax.set_title('Model Probability and Volumes per Cut')
        ax.set_xlabel('Cut', fontsize=18, y=1.5)
        ax.set_ylabel('Volume', fontsize=18)
        ax2.set_ylabel('Probability', fontsize=18)
        ax2.yaxis.set_major_formatter(mtick.PercentFormatter(1))
        ax2.set_ylim([0, 1])
        ax.grid(False)
        ax2.grid(True)
        ax.legend(['Volume'], loc=2)
        ax2.legend(['Mean Probability', 'Actual Rate'], loc=2, bbox_to_anchor=(0, 0.95));


class ConceptDrift:

    '''
        Concept Drift measures how the relationship between model predictive feature and model outcome varies between 2 samples
        PSI (Population Stability Index) methodology is used, replacing volumes in buckets with the mean outcome value 
    '''
    
    def __init__(self, df1, df2, feature, outcome, n_cuts=10, df_labels=None, method='psi', normalise=True):

        self.feature = feature
        self.outcome = outcome   
        self.n_cuts = n_cuts
        self.method = method
        self.normalise = normalise
        self.df1 = df1[df1[feature].notnull()].copy()
        self.df2 = df2[df2[feature].notnull()].copy()
        self.df_labels = df_labels

    def bucket_population(self):

        if self.df1[self.feature].nunique() < self.n_cuts:
            self.df1_buckets= self.df1.groupby(self.feature)[self.outcome].mean().reset_index()
            self.df2_buckets = self.df2.groupby(self.feature)[self.outcome].mean().reset_index()
            self.df1_buckets = self.df1_buckets.rename(columns={self.feature:'bucket'})
            self.df2_buckets = self.df2_buckets.rename(columns={self.feature:'bucket'})
        else:
            self.df1['bucket'] = pd.qcut(self.df1[self.feature], q=self.n_cuts, duplicates='drop')
            self.df1_buckets = self.df1.groupby('bucket')[self.outcome].mean().reset_index()

            self.df2['bucket'] = pd.cut(self.df2[self.feature], pd.IntervalIndex(self.df1_buckets['bucket']))
            self.df2_buckets = self.df2.groupby('bucket')[self.outcome].mean().reset_index()

    def get_concept_values(self, normalised=True):
        
        if normalised:
            self.df1_buckets['concept_value'] = self.df1_buckets[self.outcome] / self.df1_buckets[self.outcome].sum()
            self.df2_buckets['concept_value'] = self.df2_buckets[self.outcome] / self.df2_buckets[self.outcome].sum()
        else:
            self.df1_buckets['concept_value'] = self.df1_buckets[self.outcome]
            self.df2_buckets['concept_value'] = self.df2_buckets[self.outcome]

    def calculate_concept_drift(self, method='psi'):

        assert method in ['psi', 'mae'], 'method must be set to one of "psi" or "mae"'

        values1 = self.df1_buckets['concept_value']
        values2 = self.df2_buckets['concept_value']
        
        if method == 'psi':
            values1 = np.clip(values1, a_min=0.0001, a_max=None)
            values2 = np.clip(values2, a_min=0.0001, a_max=None)
            self.concept_drift = sum((values2 - values1) * np.log(values2 / values1))
        elif method == 'mae':
            self.concept_drift = np.mean(abs(values2 - values1))

    def calculate(self):

        self.bucket_population()
        self.get_concept_values(normalised=self.normalise)
        self.calculate_concept_drift(method=self.method)  

    def plot(self, ax=None, figsize=(10, 10), kind='line', fontsize=12):

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)

        if self.df_labels is not None:
            df1_label = self.df_labels[0]
            df2_label = self.df_labels[1]
        else:
            df1_label = 'df1'
            df2_label = 'df2'
        
        if self.normalise == True:
            ax.bar(self.df1_buckets['bucket'].astype(str), self.df1_buckets['concept_value'], color='blue', 
                   alpha=0.4, label=df1_label + '_concept')
            ax.bar(self.df2_buckets['bucket'].astype(str), self.df2_buckets['concept_value'], color='red',
                   alpha=0.4, label=df2_label+'_concept')                    
            ax.plot(self.df1_buckets['bucket'].astype(str), self.df1_buckets[self.outcome], c='purple', label=df1_label+'_rate')
            ax.plot(self.df2_buckets['bucket'].astype(str), self.df2_buckets[self.outcome], c='green', label=df2_label+'_rate')  
            # THIS IS CLEARLY A BUG IN MATPLOTLIB. THE BARS ARE NOT BEING LABELLED CORRECTLY AS IN ax.bar UNLESS OVERWRITTEN BELOW
            ax.legend([df1_label + '_rate', df2_label + '_rate', df2_label + '_concept', df1_label + '_concept'])            
        elif kind == 'line':
            ax.plot(self.df1_buckets['bucket'].astype(str), self.df1_buckets[self.outcome])
            ax.plot(self.df2_buckets['bucket'].astype(str), self.df2_buckets[self.outcome])
            ax.legend([df1_label + '_rate', df2_label + '_rate'])
        else:
            ax.bar(self.df1_buckets['bucket'].astype(str), self.df1_buckets[self.outcome], alpha=0.7)
            ax.bar(self.df2_buckets['bucket'].astype(str), self.df2_buckets[self.outcome], alpha=0.7)
            ax.legend([df1_label + '_rate', df2_label + '_rate'])

        ax.set_title(f'Concept Drift for {self.feature}', fontsize=fontsize)
        ax.set_xlabel(self.feature + ' bucket', fontsize=fontsize)
        ax.set_xticklabels(self.df1_buckets['bucket'].astype(str), rotation=25, fontsize=int(fontsize * 0.7))
        ax.set_ylabel(f'Mean {self.outcome} / Concept Values', fontsize=fontsize)
        ax.set_ylim([0, 1])        
        ax.annotate(f'Concept Drift: {np.round(self.concept_drift, 4)}',
          xy     = (ax.get_xlim()[1] * 0.15, ax.get_ylim()[1] * 0.7),
          color  = 'black',
          fontsize = fontsize
        )
        

class ModelConceptDrift:

    def __init__(self, df1, df2, df_importance, features, outcome, method='psi', normalise=False):

        self.df1 = df1
        self.df2 = df2
        self.df_importance = df_importance
        self.features = features
        self.outcome = outcome
        self.method = method
        self.normalise = normalise
        
    def calculate_concept_drift(self):

        cds = []
        for i, feat in enumerate(self.features):
            print(f'{i}/{len(self.features)} completed', end='\r')
            try:
                cd = ConceptDrift(self.df1, self.df2, feat, self.outcome, method=self.method, normalise=self.normalise)
                cd.calculate()
                df = pd.DataFrame(dict(feature=feat, concept_drift=cd.concept_drift), index=[0])
                cds.append(df)
            except Exception as e:
                print(feat)  
                print(e)   

        df_concept_drift = pd.concat(cds).sort_values('concept_drift', ascending=False).reset_index(drop=True)

        return df_concept_drift

    def plot_feature_concept_drift(self, df_concept_drift, df_labels=['train', 'valid'], fontsize=16):
        
        _, axs = plt.subplots(1, 2, figsize=(40, 10))

        GenPlots.plot_psi(df_concept_drift, var='concept_drift', ax=axs[0])

        ModelPlots.feature_importance(self.df_importance, ax=axs[1])
        PlotUtilities.axis_addition(axs[1], df_concept_drift, 'feature', 'concept_drift', direction='horizontal', colour='b', linestyle='--', legend=['Concept Drift'])
        PlotUtilities.annotations(axs[1], graph_object=axs[1].lines[0], colour_dict=StandardDictionaries().PSI_colour_range_dict)      
        
        _, axs = plt.subplots(3, 3, figsize=(30, 35))
        for feat, ax in zip(df_concept_drift.head(9)['feature'], axs.flatten()):
            cd = ConceptDrift(self.df1, self.df2, feat, self.outcome, df_labels=df_labels, method=self.method, normalise=self.normalise)
            cd.calculate()
            cd.plot(ax, kind='line', fontsize=fontsize)


space_lgb = {
    'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 50, 750, 100)),
    'boosting_type': hp.choice('boosting_type', ['gbdt', 'dart', 'goss']),
    'learning_rate': hp.loguniform('learning_rate', np.log(0.01), np.log(0.5)),
    'max_depth': pyll.scope.int(hp.quniform('max_depth', 6, 20, 2)),
    'num_leaves': pyll.scope.int(hp.quniform('num_leaves', 2, 150, 1)),
    'min_child_samples': pyll.scope.int(hp.choice('min_child_samples', [10, 100, 500, 1000, 5000])),
#     'reg_alpha': hp.choice('reg_alpha', [0, 0.001, 0.01, 0.1, 0.2]),
#     'reg_lambda': hp.choice('reg_lambda', [0, 0.001, 0.01, 0.1, 0.2]),
#     'feature_fraction': hp.uniform('feature_fraction', 0.1, 1.0),
#     'subsample': hp.uniform('subsample', 0.1, 1.0),
#     'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
#    'min_data_in_leaf': hp.quniform('min_child_samples', 10, 500, 5),
    #,'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
}

space_rf = {
    'n_estimators': pyll.scope.int(hp.quniform('n_estimators', 50, 750, 100)),
    'max_depth': pyll.scope.int(hp.quniform('max_depth', 6, 20, 2)),
    'min_samples_split': pyll.scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
    'num_leaves': pyll.scope.int(hp.quniform('num_leaves', 2, 150, 1)),
    'max_features': hp.choice('boosting_type', ['sqrt'])
    # 'min_child_samples': pyll.scope.int(hp.choice('min_child_samples', [10, 100, 500, 1000, 5000])),
#     'reg_alpha': hp.choice('reg_alpha', [0, 0.001, 0.01, 0.1, 0.2]),
#     'reg_lambda': hp.choice('reg_lambda', [0, 0.001, 0.01, 0.1, 0.2]),
#     'feature_fraction': hp.uniform('feature_fraction', 0.1, 1.0),
#     'subsample': hp.uniform('subsample', 0.1, 1.0),
#     'subsample_for_bin': hp.quniform('subsample_for_bin', 20000, 300000, 20000),
#    'min_data_in_leaf': hp.quniform('min_child_samples', 10, 500, 5),
    #,'colsample_bytree': hp.uniform('colsample_bytree', 0.6, 1.0)
}

space_lasso = {
  'alpha': hp.uniform('alpha', 0, 1)
}

space_ridge = {
  'alpha': hp.uniform('alpha', 0, 1),
}

space_elasticnet = {
  'alpha': hp.uniform('alpha', 0, 1),
  'l1_ratio': hp.uniform('l1_ratio', 0, 1)
}

class OptimalModel:

    def __init__(self, cols, model_type, evals, opt_lib, outcome_var=None, df_train=None, df_test=None, df_valid=None, search_space=None, how_to_tune='test', n_jobs=-1, seed=123, hp_algo=tpe.suggest, debug=False, cat_vars=None, plot=False, print_params=False, outcome_type='classification', eval_metric='rmse', k=5, stratify_kfold=False):
        
        self.df_train = None
        self.df_test = None
        self.df_valid = None
        
        self.df_train = df_train.copy()
        if df_test is not None:
            self.df_test = df_test.copy()
        if df_valid is not None:
            self.df_valid = df_valid.copy()
            
        self.cols = cols
        self.model_type = model_type
        self.evals = evals
        self.opt_lib = opt_lib
        self.outcome_var = outcome_var
        self.search_space = search_space
        self.how_to_tune = how_to_tune
        self.n_jobs = n_jobs
        self.seed = seed
        self.hp_algo = hp_algo
        self.debug=debug
        self.cat_vars=cat_vars
        self.plot=plot 
        self.print_params=print_params 
        self.outcome_type=outcome_type
        self.eval_metric = eval_metric
        self.k = k
        self.stratify_kfold = stratify_kfold

        if self.df_valid is None:
            self.df_valid = pd.DataFrame().reindex_like(self.df_train).dropna()
        
        if self.search_space is not None:
            self.outcomes_in_search_space = [x for x in ['target', 'outcome', 'outcome_var'] if x in self.search_space.keys()]

            if len(self.outcomes_in_search_space) > 0:
                if self.df_train is None or self.df_test is None:
                    raise ValueError('If outcomes are suppled in the search space, df_train, df_test must be supplied')
                print('Performing model optimisation with an outcome optimisation')
            else:
                if self.outcome_var is None:
                    raise ValueError('outcome_var must be supplied in either the search space or as a class parameter')
                print('Performing model optimisation')

    def build_optimal_model(self):
            
        bayes_trials = Trials()
        
        t0 = time()
        
        if self.model_type not in ['lgb', 'xgboost', 'rf', 'dnn', 'lasso', 'ridge', 'elasticnet']:
            raise ValueError('model_type must be one of "lgb", "rf", "xgboost", "dnn", "lasso", "ridge" or "elasticnet"')
        if self.model_type not in ['lgb', 'xgboost'] and self.how_to_tune == 'early_stopping':
            raise ValueError('model_type must be one of "lgb", "xgboost" if how_to_tune == "early_stopping"')
        if self.opt_lib not in ['opt', 'hp', 'ga']:
            raise ValueError('opt_lib must be one of "opt", "hp" or "ga"')
                    
        if self.opt_lib == 'hp':  
            self.debug_out(f'hp param setup', self.debug)
            if self.model_type == 'lgb':
                space = space_lgb.copy()
                objective_fn = self.objective_lgb_hp
            elif self.model_type == 'xgboost':
                space = space_xgboost.copy()
                objective_fn = self.objective_xgboost_hp
            elif self.model_type == 'rf':
                space = space_rf.copy()
                objective_fn = self.objective_rf_hp
            elif self.model_type == 'dnn':
                space = space_keras.copy()
                objective_fn = self.objective_dnn_hp            
            elif self.model_type == 'lasso':
                warnings.filterwarnings("ignore")
                space = space_lasso.copy()
                objective_fn = self.objective_lasso_hp 
            elif self.model_type == 'ridge':
                warnings.filterwarnings("ignore")
                space = space_ridge.copy()
                objective_fn = self.objective_ridge_hp                 
            elif self.model_type == 'elasticnet': 
                space = space_elasticnet.copy()
                objective_fn = self.objective_elasticnet_hp            
            
            if self.search_space is not None:
                space = self.search_space.copy()
            
            self.debug_out(f'hp {self.model_type}', self.debug)  
            best = fmin(fn=objective_fn, space=space, algo=self.hp_algo, max_evals=self.evals, trials=bayes_trials)
            
            params = pd.DataFrame(sorted(bayes_trials.results, key = lambda x: x['loss'])).params[0]    
            cv_scores = pd.DataFrame(sorted(bayes_trials.results, key = lambda x: x['loss'])).cv_scores[0]
            trials = bayes_trials
            
        elif self.opt_lib == 'opt':
            study = optuna.create_study()
            if self.model_type == 'lgb':
                study.optimize(objective_lgb_optuna, n_trials=evals)
            elif self.model_type == 'rf':
                study.optimize(objective_rf_optuna, n_trials=evals)
                
            params = study.best_params
            trials = study
            
        elif self.opt_lib == 'ga':
            
            ga_param = {'max_num_iteration': self.evals,
                'population_size':20,
                'mutation_probability':0.1,
                'elit_ratio': 0.01,
                'crossover_probability': 0.5,
                'parents_portion': 0.3,
                'crossover_type':'uniform',
                'max_iteration_without_improv':4}
            
            if self.search_space is not None:
                space = search_space.copy()
            elif self.model_type == 'lgb': 
                space = space_lgb_ga.copy()
            elif self.model_type == 'rf':
                space = space_rf_ga.copy()
                
            if self.model_type == 'lgb':  
                
                vartypes_lgb = np.array([['real'],['int'],['int'],['int'],['real'],['real'],['int'],['int'],['real'],['real'],['real']])
                
                self.debug_out('ga lgb', self.debug)
                
                best=ga(function=objective_lgb_ga, dimension=len(space_lgb_ga), variable_type_mixed=vartypes_lgb, 
                        variable_boundaries=space_lgb_ga, algorithm_parameters=ga_param, function_timeout=10000)
                best.run()
                
                params = {'learning_rate': best.output_dict['variable'][0],
                          'max_depth': best.output_dict['variable'][1],
                          'num_leaves': best.output_dict['variable'][2],
                          'min_data_in_leaf': best.output_dict['variable'][3],
                          'feature_fraction': best.output_dict['variable'][4],
                          'subsample': best.output_dict['variable'][5],
                          'subsample_for_bin': best.output_dict['variable'][6],
                          'min_child_samples': best.output_dict['variable'][7],
                          'reg_alpha': best.output_dict['variable'][8],
                          'reg_lambda': best.output_dict['variable'][9],
                          'colsample_bytree': best.output_dict['variable'][10]}
                
            elif self.model_type == 'rf':
                
                vartypes_rf = np.array([['real'],['real'],['real'],['int']])
                
                self.debug_out('ga rf', self.debug)
                        
                best=ga(function=objective_rf_ga, dimension=len(space_rf_ga), variable_type=vartypes_rf, 
                        variable_boundaries=space_rf_ga, algorithm_parameters=ga_param)
                best.run()
                
                params = {'bootstrap': True,
                          'max_depth': best.output_dict['variable'][0],
                          'max_features': 'auto',
                          'min_samples_leaf': best.output_dict['variable'][1],
                          'min_samples_split': best.output_dict['variable'][2],
                          'min_weight_fraction_leaf': best.output_dict['variable'][3],
                          'n_estimators': 25}
                
            trials = best
              
        print(f'{self.evals} rounds of {self.opt_lib} {self.how_to_tune} optimisation ran in {round(time() - t0, 3)} seconds')
        
        if self.print_params:
            print(params)
        
        self.X_train = self.df_train[self.cols]
        self.X_test = self.df_test[self.cols]
        self.X_valid = self.df_valid[self.cols]
        
        if 'outcome_var' in params.keys():
            outcome_var = params['outcome_var']
        else:
            outcome_var = self.outcome_var
            
        self.y_train = self.df_train[outcome_var]
        if self.df_test is not None:
            self.y_test = self.df_test[outcome_var]
        if self.df_valid is not None:
            self.y_valid = self.df_valid[outcome_var]
        
        model_params = copy.deepcopy(params)
        
        if 'outcome_var' in model_params.keys():
            del model_params['outcome_var']
        if 'cutoff' in model_params.keys():
            del model_params['cutoff']
                
        builder = ModelBuilder()
        
        try:
            if self.model_type == 'lgb':
                model, importance, errors = builder.build_lgb(self.X_train[self.cols], self.y_train, 
                                                              self.X_test[self.cols], self.y_test, 
                                                              self.X_valid[self.cols], self.y_valid, 
                                                              model_params, outcome_type=self.outcome_type, 
                                                              n_jobs=self.n_jobs, seed=self.seed)
            elif self.model_type == 'xgboost':
                model, importance, errors = builder.build_xgboost(self.X_train[self.cols], self.y_train,
                                                                  self.X_test[self.cols], self.y_test, 
                                                                  self.X_valid[self.cols], self.y_valid, 
                                                                  model_params, outcome_type=self.outcome_type, 
                                                                  n_jobs=self.n_jobs, seed=self.seed)
            elif self.model_type == 'rf':
                model, importance, errors = builder.build_rf(self.X_train[self.cols], self.y_train, 
                                                             self.X_test[self.cols], self.y_test, 
                                                             self.X_valid[self.cols], self.y_valid, 
                                                             model_params, outcome_type=self.outcome_type, 
                                                             n_jobs=self.n_jobs, seed=self.seed)
            elif self.model_type == 'dnn':
                model, errors = builder.build_dnn(X_train[cols], y_train, X_test[cols], y_test, X_valid[cols], y_valid, model_params)
                importance = pd.DataFrame()
            elif self.model_type == 'lasso':
                model, importance, errors = builder.build_lasso(self.X_train[self.cols], self.y_train, 
                                                                self.X_test[self.cols], self.y_test, 
                                                                self.X_valid[self.cols], self.y_valid, model_params)
            elif self.model_type == 'ridge':
                model, importance, errors = builder.build_ridge(self.X_train[self.cols], self.y_train, 
                                                                self.X_test[self.cols], self.y_test, 
                                                                self.X_valid[self.cols], self.y_valid, model_params)               
            elif self.model_type == 'elasticnet':
                model, importance, errors = builder.build_elasticnet(self.X_train[self.cols], self.y_train, 
                                                                     self.X_test[self.cols], self.y_test, 
                                                                     self.X_valid[self.cols], self.y_valid, model_params)
        except Exception as e:
            print(f'Failed to build best model after hp optimisation. Error: {e}')
            print(f'The best params were:')
            print(params)
          
            model = []
            importance = pd.DataFrame()
            errors = {}
                
        print(f'Best parameters found: {params}')
        if cv_scores is not None:
            print(f'Cross validation scores: {cv_scores}')
        
        return model, params, trials, importance, errors, cv_scores

    def train_and_evaluate(self, model, X_train, y_train, X_test, y_test, how_to_tune, outcome_type, eval_metric='rmse', k=5, stratify_kfold=False):
        
        if outcome_type == 'classification':
            def pred_func(X, model):
                return model.predict_proba(X)[:, 1]   
            def cross_val_loss_func(scores):
                return np.mean(scores) 
            
            if callable(eval_metric):
                cv_scoring = eval_metric
                eval_func = eval_metric
            elif eval_metric == 'auc':
                cv_scoring = 'roc_auc'    
                def eval_func(y, y_hat):
                    return - roc_auc_score(y, y_hat)     
            elif eval_metric == 'auc_pr':
                def cv_scoring(estimator, X, y):
                    y_hat = estimator.predict_proba(X)[:, 1]  
                    precision, recall, _ = metrics.precision_recall_curve(y, y_hat)
                    auc_pr = metrics.auc(recall, precision)
                    return - auc_pr   
                
                def eval_func(y, y_hat):
                    precision, recall, _ = metrics.precision_recall_curve(y, y_hat)
                    auc_pr = metrics.auc(recall, precision)
                    return - auc_pr   
            else:
                raise ValueError('eval_metric must be supplied as a callable function or one of "auc" and "auc_pr"')
        else:
            if callable(eval_metric):
                cv_scoring = eval_metric
                eval_func = eval_metric
            elif eval_metric == 'wape':
                def cv_scoring(estimator, X, y):
                    y_pred = estimator.predict(X)
                    return wape(y, y_pred)
                def cross_val_loss_func(scores):
                    return np.mean(scores)
                def pred_func(X, model):
                    return model.predict(X)
                eval_func = wape                
            else:
                cv_scoring = "neg_mean_squared_error"
                def cross_val_loss_func(scores):
                    if self.debug:
                        print(scores)
                    rsmes = np.array([np.sqrt(-x) for x in scores])
                    if self.debug:
                        print(rsmes)
                    loss = np.mean(rsmes)
                    return loss
                def pred_func(X, model):
                    return model.predict(X)
                def eval_func(y, y_hat):
                    return np.sqrt(mean_squared_error(y, y_hat))

        random.seed(self.seed)
        scores = None

        if how_to_tune == 'cross_val':
            self.debug_out('hp tuning with cross validation on trainset', self.debug)
            if outcome_type == 'classification':
                if stratify_kfold:
                    cv = StratifiedKFold(n_splits=k)
                else:
                    cv = KFold(n_splits=k, shuffle=False)
            else:
                cv = KFold(n_splits=k) 
            scores = cross_val_score(model, X_train, y_train, cv=cv, n_jobs=self.n_jobs, scoring=cv_scoring)
            loss = cross_val_loss_func(scores)
        elif how_to_tune == 'cross_val_total':
            self.debug_out('hp tuning with cross validation', self.debug)
            X = pd.concat([X_train, X_test])
            y = pd.concat([y_train, y_test])
            if outcome_type == 'classification':
                if stratify_kfold:
                    cv = StratifiedKFold(n_splits=k)
                else:
                    cv = KFold(n_splits=k, shuffle=False)
            else:
                cv = KFold(n_splits=k) 
            scores = cross_val_score(model, X, y, cv=cv, n_jobs=1, scoring=cv_scoring)
            loss = cross_val_loss_func(scores)          
        elif how_to_tune == 'train+test': 
            self.debug_out('hp tuning on train and test sets', self.debug)
            model.fit(X_train, y_train)

            y_pred_train = pred_func(X_train, model)
            self.debug_out('predict on testset', self.debug)
            y_pred_test = pred_func(X_test, model)       

            loss_train = eval_func(y_train, y_pred_train)
            loss_test = eval_func(y_test, y_pred_test)
            loss = (loss_train + loss_test) / 2
        elif how_to_tune == 'early_stopping':
            self.debug_out('hp tuning with early stopping on test set', self.debug)
            early_stopping = lgb.early_stopping(stopping_rounds=25, verbose=True)
            model.fit(X_train, 
                      y_train,
                      callbacks=[early_stopping],
                      eval_metric=eval_metric,
                      eval_set=[(X_test, y_test)])

            self.debug_out('predict on test set', self.debug)
            y_pred_test = pred_func(X_test, model)     
            loss = eval_func(y_test, y_pred_test)

        else:
            self.debug_out('hp tuning on testset only', self.debug)
            model.fit(X_train, y_train)
            self.debug_out('predict on testset', self.debug)
            y_pred_test = pred_func(X_test, model)
            self.debug_out('calculate loss', self.debug)
            loss = eval_func(y_test, y_pred_test)
            
        return loss, scores  
    
    def objective_lgb_hp(self, params):

        """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
        
        random.seed(self.seed)
        
        # Make sure parameters that need to be integers are integers
        for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
            try:
                params[parameter_name] = int(params[parameter_name])
            except:
                pass
        
        if self.debug:
            print(params)
        
        if self.outcome_type == 'classification':
            model = LGBMClassifier(n_jobs=self.n_jobs, verbosity=-100)
        else:
            model = LGBMRegressor(n_jobs=self.n_jobs, verbosity=-100)
        
        if 'cutoff' in params.keys() and callable(self.eval_metric):
            evaluation = partial(self.eval_metric, cutoff=params['cutoff'])
        else:
            evaluation = self.eval_metric
            
        if 'outcome_var' in params.keys():
            outcome_var = params['outcome_var']
        else:
            outcome_var = self.outcome_var
            
        X_train = self.df_train[self.cols]
        y_train = self.df_train[outcome_var]
        if self.df_test is not None:
            X_test = self.df_test[self.cols]
            y_test = self.df_test[outcome_var]
        
        model_params = copy.deepcopy(params)
        if 'outcome_var' in model_params.keys():
            del model_params['outcome_var']
        if 'cutoff' in model_params.keys():
            del model_params['cutoff']
            
        model.set_params(**model_params)

        self.debug_out('hp lgb train and evaluate', self.debug)
        loss, cv_scores = self.train_and_evaluate(model, 
                                                  X_train, 
                                                  y_train, 
                                                  X_test, 
                                                  y_test, 
                                                  self.how_to_tune, 
                                                  self.outcome_type,
                                                  evaluation,
                                                  self.k,
                                                  self.stratify_kfold)

        return {'loss': loss, 'params': params, 'status': STATUS_OK, 'cv_scores':cv_scores}
    
    def objective_lgb_hp_old(self, params):

        """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
        
        random.seed(self.seed)

    #     # Retrieve the subsample if present otherwise set to 1.0
    #     subsample = params['boosting_type'].get('subsample', 1.0)

    #     # Extract the boosting type
    #     params['boosting_type'] = params['boosting_type']['boosting_type']
    #     params['subsample'] = subsample
        
        # Make sure parameters that need to be integers are integers
        for parameter_name in ['num_leaves', 'subsample_for_bin', 'min_child_samples']:
            try:
                params[parameter_name] = int(params[parameter_name])
            except:
                pass
        
        if self.debug:
            print(params)
        
    #     if 2**params['max_depth'] > params['num_leaves']:
    #         loss = 100000
    #     else:
        
        if self.outcome_type == 'classification':
            model = LGBMClassifier(n_jobs=self.n_jobs, verbose=-1)
        else:
            model = LGBMRegressor(n_jobs=self.n_jobs, verbose=-1)
            
        model.set_params(**params)

        self.debug_out('hp lgb train and evaluate', self.debug)
        loss, cv_scores = self.train_and_evaluate(model, 
                                                  self.X_train[self.cols], 
                                                  self.y_train, 
                                                  self.X_test[self.cols], 
                                                  self.y_test, 
                                                  self.how_to_tune, 
                                                  self.outcome_type,
                                                  self.eval_metric,
                                                  self.k,
                                                  self.stratify_kfold)

        return {'loss': loss, 'params': params, 'status': STATUS_OK, 'cv_scores':cv_scores}
      
    def objective_lasso_hp(self, params_in):

        """Objective function for Lasso Regression Hyperparameter Tuning"""
        
        random.seed(self.seed)
        
        params = params_in.copy()
        
        model = Lasso()
        model.set_params(**params)
        
        self.debug_out('hp lasso train and evaluate', self.debug)
        loss, cv_scores = self.train_and_evaluate(model, 
                                                  self.df_train[self.cols], 
                                                  self.df_train[self.outcome_var], 
                                                  self.df_test[self.cols], 
                                                  self.df_test[self.outcome_var], 
                                                  self.how_to_tune, 
                                                  self.outcome_type,
                                                  self.eval_metric,
                                                  self.k,
                                                  self.stratify_kfold)
            
        return {'loss': loss, 'params': params, 'status': STATUS_OK, 'cv_scores':cv_scores}

    def objective_ridge_hp(self, params_in):

        """Objective function for Ridge Regression Hyperparameter Tuning"""
        
        random.seed(self.seed)
        
        params = params_in.copy()
        
        model = Ridge()
        model.set_params(**params)
        
        self.debug_out('hp lasso train and evaluate', self.debug)
        loss, cv_scores = self.train_and_evaluate(model, 
                                                  self.X_train[self.cols], 
                                                  self.y_train, 
                                                  self.X_test[self.cols], 
                                                  self.y_test, 
                                                  self.how_to_tune, 
                                                  self.outcome_type,
                                                  self.eval_metric,
                                                  self.k,
                                                  self.stratify_kfold)
            
        return {'loss': loss, 'params': params, 'status': STATUS_OK, 'cv_scores':cv_scores}

    def objective_elasticnet_hp(self, params_in):

        """Objective function for ElasticNet Regression Hyperparameter Tuning"""
        
        random.seed(self.seed)
        
        params = params_in.copy()
        
        model = ElasticNet()
        model.set_params(**params)
        
        self.debug_out('hp lasso train and evaluate', self.debug)
        loss, cv_scores = self.train_and_evaluate(model, 
                                                  self.X_train[self.cols], 
                                                  self.y_train, 
                                                  self.X_test[self.cols], 
                                                  self.y_test, 
                                                  self.how_to_tune, 
                                                  self.outcome_type,
                                                  self.eval_metric,
                                                  self.k,
                                                  self.stratify_kfold)
            
        return {'loss': loss, 'params': params, 'status': STATUS_OK, 'cv_scores':cv_scores}            
      
    def objective_xgboost_hp(self, params_in):

        """Objective function for Gradient Boosting Machine Hyperparameter Tuning"""
        
        random.seed(self.seed)
        
        #If tiered search space with different boosters
        try:
            booster_dict = params_in['booster']
            params = params_in.copy()
            del params['booster']
            params.update(booster_dict)
        except:
            params = params_in.copy()
        
        model = XGBRegressor(n_jobs=n_jobs)
        model.set_params(**params)
        
        self.debug_out('hp lgb train and evaluate', self.debug)
        loss, cv_scores = self.train_and_evaluate(model, 
                                                  self.X_train[self.cols], 
                                                  self.y_train, 
                                                  self.X_test[self.cols], 
                                                  self.y_test, 
                                                  self.how_to_tune, 
                                                  self.outcome_type,
                                                  self.eval_metric,
                                                  self.k,
                                                  self.stratify_kfold)
            
        return {'loss': loss, 'params': params, 'status': STATUS_OK, 'cv_scores':cv_scores}
        
    def objective_rf_hp(self, params):

        """Objective function for Random Forest Hyperparameter Tuning"""
        
        random.seed(self.seed)    

        model = RandomForestRegressor(n_jobs=self.n_jobs)
        model.set_params(**params)
        
        loss, cv_scores = self.train_and_evaluate(model, 
                                                  self.X_train[self.cols], 
                                                  self.y_train, 
                                                  self.X_test[self.cols], 
                                                  self.y_test, 
                                                  self.how_to_tune, 
                                                  self.outcome_type,
                                                  self.eval_metric,
                                                  self.k,
                                                  self.stratify_kfold)

        return {'loss': loss, 'params': params, 'status': STATUS_OK, 'cv_scores':cv_scores}
      
    def debug_out(self, step, debug, simple=False):
        if debug:
            if simple:
                print(str(step))
            else:
                print('Commencing step ' + str(step))


# from pyspark.ml.evaluation import ClassificationEvaluator
# from mmlspark import ComputeModelStatistics
# from mmlspark import LightGBMClassifier


#     def build_lgb(self, X_train, y_train, X_test, y_test, X_valid=None, y_valid=None, params=None, n_jobs=16, seed=123, early_stopping_rounds=False, type='classifier'):

#         random.seed(seed)

#         if params is None:
#           params = {}
#         params = params.copy()

#         # Make sure parameters that need to be integers are integers
#         for parameter_name in ['max_depth', 'min_child_samples', 'num_leaves', 'subsample_for_bin']:
#             try:
#                 params[parameter_name] = int(params[parameter_name])
#             except:
#                 pass

#         if 'rank' in params.keys():
#             if params['rank']:
#                 model = lgb.LGBMRanker(importance_type='gain')
#             else:
#                 if type == 'classifier':
#                     model = LGBMClassifier(n_jobs=n_jobs)
#                 else:
#                     model = LGBMRegressor(n_jobs=n_jobs)
#         else:
#             if type == 'classifier':
#                 model = LGBMClassifier(n_jobs=n_jobs)
#             else:
#                 model = LGBMRegressor(n_jobs=n_jobs)

#         params.update({'n_jobs': n_jobs})

#         t0 = time()    

#         if 'early_stopping_rounds' in params.keys() or 'early_stopping_round' in params.keys():
#             try:
#                 rounds = params['early_stopping_round']
#                 del params['early_stopping_round']
#             except:
#                 rounds = params['early_stopping_rounds']
#                 del params['early_stopping_rounds']

#             early_stopping = lgb.early_stopping(stopping_rounds=rounds, verbose=True)

#             if 'eval_metric' in params.keys():
#                 eval_metric = params['eval_metric']
#                 del params['eval_metric']
#             else:
#                 if type == 'classifier':
#                     eval_metric = 'auc'
#                 else:
#                     eval_metric = 'rmse'

#             print('Fitting model with early stopping')    

#             model.set_params(**params)

#             model.fit(X_train, 
#                       y_train,
#                       callbacks=[early_stopping],
#                       eval_metric=eval_metric,
#                       eval_set=[(X_test, y_test)])
#         else:
#             model.set_params(**params)
#             model.fit(X_train, y_train)

#         print(f'LGB built in {round(time() - t0, 3)} seconds')

#         if X_valid.shape[0]==0:
#             X_valid = None
#             y_valid = None 
        
#         if type == 'classifier':
#             errors = self.evaluate_classifier(model, X_train, y_train, X_test, y_test, X_valid, y_valid, ret=True)
#         else:
#             errors = self.evaluate_regressor(model, X_train, y_train, X_test, y_test, X_valid, y_valid, ret=True)

#         importance = self.feature_importances(model, X_train.columns)

#         return model, importance, errors
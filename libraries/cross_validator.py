import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LinearRegression
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.metrics import log_loss, accuracy_score


class CrossValidator:
    
    def __init__(self, estimator, df, target, cv, features=None, n_jobs=-1):
        
        self.estimator = estimator
        self.df = df.reset_index(drop=True)
        self.features = features
        self.target = target 
        self.cv = cv
        self.n_jobs = n_jobs
        
        self.indicies = cv.split(self.df)
        
    def score(self, fit_and_score):
        
        scores = Parallel(n_jobs=self.n_jobs)(  
                delayed(fit_and_score)(
                                        estimator=clone(self.estimator),
                                        df_train=self.df.loc[train_idx, :], 
                                        df_test=self.df.loc[test_idx, :],
                                        features=self.features,
                                        target=self.target
                                       )
                for train_idx, test_idx in self.indicies)
        
        return list(scores)

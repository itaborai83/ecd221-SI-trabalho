 # -*- coding: utf-8 -*-
import io
import os
import sys
import abc
import pandas as pd
import warnings
import telchurn.util as util
from typing import List, Dict, Tuple
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

LOGGER = util.get_logger('data_loader')

class HyperParamTunner(abc.ABC):
    
    DEFAULT_K_FOLDS = 10
    DEFAULT_METRIC  = "f1"
    
    def __init__(self, k_folds, random_seed=None):
        self.kfold = StratifiedKFold(
            n_splits      = k_folds
        ,   shuffle       = True
        ,   random_state  = random_seed
        )     
        
    @abc.abstractmethod
    def tune(self, pipeline: Pipeline, param_grid: List[Dict], grid_name: str, scoring_metric: str, X_train_df: pd.DataFrame, y_train_df: pd.DataFrame) -> Tuple[RandomizedSearchCV, pd.DataFrame]:
        raise NotImplementedError
        
class HyperParamTunnerImpl(HyperParamTunner):
    
    def __init__(self, k_folds, random_seed=None):
        self.kfold = StratifiedKFold(
            n_splits      = k_folds
        ,   shuffle       = True
        ,   random_state  = random_seed
        )     
                
    def tune(self, pipeline: Pipeline, param_grid: List[Dict], grid_name: str, num_iterations: int, scoring_metric: str, X_train_df: pd.DataFrame, y_train_df: pd.DataFrame) -> Tuple[RandomizedSearchCV, pd.DataFrame]:
        LOGGER.info(f'tunning pipeline {grid_name} using scoring metric {scoring_metric}')
        grid = RandomizedSearchCV(
            estimator           = pipeline
        ,   param_distributions = param_grid
        ,   scoring             = scoring_metric
        ,   cv                  = self.kfold
        ,   n_iter              = num_iterations
        ,   return_train_score  = False
        )
        warnings.filterwarnings("ignore")
        util.silence_warnings()
        grid.fit(X_train_df, y_train_df)
        LOGGER.info(f'Best {scoring_metric}: {grid.best_score_}')
        LOGGER.info(f'Best estimator: {grid.best_score_} -> {grid.best_estimator_}')
        results_df = pd.DataFrame(grid.cv_results_)
        return grid, results_df

 # -*- coding: utf-8 -*-
import abc
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
from imblearn.over_sampling import ADASYN, SMOTE
from sklearn.metrics import make_scorer, fbeta_score

import telchurn.util as util
from telchurn.data_loader import DataLoader
from telchurn.pipeline_factory import PipelineFactory
from telchurn.hyper_param_tunner import HyperParamTunner
from telchurn.param_grids import ParamGridsImpl
from telchurn.model_repository import ModelRepository

LOGGER = util.get_logger('trainer')

class Trainer(abc.ABC):
    
    DEFAULT_TEST_PCT_SIZE   = 0.3 # 30% do conjunto de dados
    DEFAULT_RANDOM_STATE    = 42
    TARGET_VARIABLE         = "churn"
        
    @classmethod
    def train_test_split(klass, churn_df, seed: int, test_split_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        LOGGER.info('splitting data set into train and test sets')
        all_but_target = churn_df.columns.difference([klass.TARGET_VARIABLE])
        X_df = churn_df[all_but_target]
        y = churn_df[klass.TARGET_VARIABLE]
        X_train, X_test, y_train, y_test = train_test_split(
            X_df.values
        ,   y
        ,   test_size     = test_split_pct
        ,   shuffle       = True
        ,   random_state  = seed
        ,   stratify      = y # com estratificação
        )
        
        #smote = SMOTE(random_state=seed)
        #X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
        #adasyn = ADASYN(random_state=seed)
        #X_train_resampled, y_train_resampled = adasyn.fit_resample(X_train, y_train)
        #ros = RandomOverSampler(random_state=seed)
        #X_train_resampled, y_train_resampled = ros.fit_resample(X_train, y_train)

        X_train_df = pd.DataFrame(X_train, columns=X_df.columns)
        #X_train_df = pd.DataFrame(X_train_resampled, columns=X_df.columns)
        X_test_df = pd.DataFrame(X_test, columns=X_df.columns)
        y_train_df = pd.DataFrame(y_train, columns=[klass.TARGET_VARIABLE])
        #y_train_df = pd.DataFrame(y_train_resampled, columns=[klass.TARGET_VARIABLE])
        y_test_df = pd.DataFrame(y_test, columns=[klass.TARGET_VARIABLE])
        return X_train_df, X_test_df, y_train_df, y_test_df
        
    @abc.abstractmethod
    def train(self, input_file : str, seed: int, test_split_pct: float, k_folds: float) -> None:
        raise NotImplementedError
        
class TrainerImpl(Trainer):
    
    #SCORING_METHOD = "recall"
    SCORING_METHOD = "balanced_accuracy"
    
    def __init__(self, data_loader: DataLoader, pipeline_factory: PipelineFactory, hp_tunner: HyperParamTunner, repo: ModelRepository):
        self.data_loader = data_loader
        self.pipeline_factory = pipeline_factory
        self.hp_tunner = hp_tunner
        self.repo = repo
    
    def get_param_grids(self):
        return ParamGridsImpl().get_parameter_grids()

    def train(self, input_file: str, seed: int, test_split_pct: float, k_folds: float) -> None:
        LOGGER.info('starting telco churn model training')
        churn_df = self.data_loader.load_cleansed(input_file)
        util.report_df(LOGGER, churn_df)
        pipeline = self.pipeline_factory.build_pipeline_for(churn_df)
        X_train_df, X_test_df, y_train_df, y_test_df = Trainer.train_test_split(churn_df, seed, test_split_pct)
        param_grids = param_grids = self.get_param_grids()
        for param_grid in param_grids:
            name        = param_grid["name"]
            iterations  = param_grid["iterations"]
            grid        = param_grid["param_grid"]
            rand_search_cv, train_df = self.hp_tunner.tune(
                pipeline        = pipeline
            ,   param_grid      = grid
            ,   grid_name       = name
            ,   num_iterations  = iterations
            ,   scoring_metric  = self.SCORING_METHOD
            ,   X_train_df      = X_train_df
            ,   y_train_df      = y_train_df
            )
            grid_name = name + ".pkl"
            self.repo.save_grid(rand_search_cv, grid_name)
        

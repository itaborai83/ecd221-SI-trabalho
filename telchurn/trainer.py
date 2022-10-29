 # -*- coding: utf-8 -*-
import abc
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV
import telchurn.util as util
from telchurn.data_loader import DataLoader
from telchurn.pipeline_factory import PipelineFactory
from telchurn.hyper_param_tunner import HyperParamTunner
from telchurn.param_grids import ParamGridsImpl
from telchurn.grid_repository import GridRepository

LOGGER = util.get_logger('trainer')

class Trainer(abc.ABC):
    
    DEFAULT_TEST_PCT_SIZE   = 0.3 # 30% do conjunto de dados
    DEFAULT_RANDOM_STATE    = 42
    
    @abc.abstractmethod
    def train(self, input_file : str, seed: int, test_split_pct: float, k_folds: float) -> None:
        raise NotImplementedError
        
class TrainerImpl(abc.ABC):
    
    TARGET_VARIABLE = "churn"
    
    def __init__(self, data_loader: DataLoader, pipeline_factory: PipelineFactory, hp_tunner: HyperParamTunner, repo: GridRepository):
        self.data_loader = data_loader
        self.pipeline_factory = pipeline_factory
        self.hp_tunner = hp_tunner
        self.repo = repo
    
    def get_param_grids(self):
        return ParamGridsImpl().get_parameter_grids()
    
    def train(self, input_file: str, seed: int, test_split_pct: float, k_folds: float, metric: str) -> None:
        LOGGER.info('starting telco churn model training')
        churn_df = self.data_loader.load_cleansed(input_file)
        util.report_df(LOGGER, churn_df)
        pipeline = self.pipeline_factory.build_pipeline_for(churn_df)
        X_train_df, X_test_df, y_train_df, y_test_df = self.__train_test_split(churn_df, seed, test_split_pct)
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
            ,   scoring_metric  = metric
            ,   X_train_df      = X_train_df
            ,   y_train_df      = y_train_df
            )
            
            grid_name = name + ".pkl"
            self.repo.save_grid(rand_search_cv, grid_name)
        
    def __train_test_split(self, churn_df, seed: int, test_split_pct: float) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        LOGGER.info('splitting data set into train and test sets')
        all_but_target = churn_df.columns.difference([self.TARGET_VARIABLE])
        X_df = churn_df[all_but_target]
        y = churn_df[self.TARGET_VARIABLE]        
        X_train, X_test, y_train, y_test = train_test_split(
            X_df.values
        ,   y
        ,   test_size     = test_split_pct
        ,   shuffle       = True
        ,   random_state  = seed
        ,   stratify      = y # com estratificação
        )

        X_train_df = pd.DataFrame(X_train, columns=X_df.columns)
        X_test_df = pd.DataFrame(X_test, columns=X_df.columns)
        y_train_df = pd.DataFrame(y_train, columns=[self.TARGET_VARIABLE])
        y_test_df = pd.DataFrame(y_test, columns=[self.TARGET_VARIABLE])
        return X_train_df, X_test_df, y_train_df, y_test_df
 # -*- coding: utf-8 -*-
import abc
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split

import telchurn.util as util
from telchurn.data_loader import DataLoader
from telchurn.pipeline_factory import PipelineFactory
from telchurn.hyper_param_tunner import HyperParamTunner
from telchurn.param_grids import ParamGridsImpl
from telchurn.model_repository import ModelRepository
from telchurn.data_splitter import DataSplitter

LOGGER = util.get_logger('trainer')

class Trainer(abc.ABC):
    
    #DEFAULT_TEST_PCT_SIZE   = 0.3 # 30% do conjunto de dados
    #DEFAULT_RANDOM_STATE    = 42
                
    @abc.abstractmethod
    def train(self, input_file: str, splitter: DataSplitter) -> None:
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

    def train(self, input_file: str, splitter: DataSplitter) -> None:
        LOGGER.info('starting telco churn model training')
        churn_df = self.data_loader.load_cleansed(input_file)
        target = churn_df.columns[-1]
        util.report_df(LOGGER, churn_df)
        pipeline = self.pipeline_factory.build_pipeline_for(churn_df)
        (X_train_df, y_train_df), (X_test_df, y_test_df) = splitter.split(churn_df, target)
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
        

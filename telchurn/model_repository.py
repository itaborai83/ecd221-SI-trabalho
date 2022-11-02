# -*- coding: utf-8 -*-
import abc
import os.path
import glob
import pickle
from typing import Tuple, List, Dict
import pandas as pd
import telchurn.util as util
from mlxtend.classifier import EnsembleVoteClassifier

LOGGER = util.get_logger('model_repository')

class ModelRepository(abc.ABC):
    
    @abc.abstractmethod
    def save_grid(self, grid: Dict, file_name: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load_grid(self, file_name: str) -> Dict:
        raise NotImplementedError

    @abc.abstractmethod
    def save_final_model(self, model: EnsembleVoteClassifier, file_name: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load_final_model(self, file_name: str) -> EnsembleVoteClassifier:
        raise NotImplementedError
        
class ModelRepositoryImpl(ModelRepository):
    
    GRID_PREFIX = "grid_"
    GRID_GLOB = "grid_*.pkl"
    
    def __init__(self, repo_dir: str):
        self.repo_dir = repo_dir
    
    def list_grids(self):
        LOGGER.info(f'listing saved grids on {self.repo_dir}')
        path = os.path.join(self.repo_dir, self.GRID_GLOB)
        def get_grid_name(path):
            # f = lambda path: os.path.split(path)[-1].replace(self.GRID_PREFIX, "") # eita, gambiarra danada!
            path_parts = os.path.split(path)
            filename = path_parts[-1]
            filename = filename.replace(self.GRID_PREFIX, "")
            return filename
        return list([ get_grid_name(path) for path in glob.glob(path) ])
        
    def save_grid(self, grid: Dict, file_name: str) -> None:
        path = os.path.join(self.repo_dir, self.GRID_PREFIX + file_name)
        LOGGER.info(f'saving grid search results to {path}')
        data = {
            "best_score_"     : grid.best_score_
        ,   "best_params_"    : grid.best_params_
        ,   "best_estimator_" : grid.best_estimator_
        ,   "cv_results_"     : grid.cv_results_
        ,   "grid"            : grid
        }  
        with open(path, "wb") as fh:
            pickle.dump(data, fh)
        
    def load_grid(self, file_name: str) -> Dict:
        path = os.path.join(self.repo_dir, self.GRID_PREFIX + file_name)
        LOGGER.info(f'loading grid search results from {path}')
        with open(path, "rb") as fh:
            data                  = pickle.load(fh)
        grid                  = data["grid"]
        grid.best_score_      = data["best_score_"]
        grid.best_params_     = data["best_params_"]
        grid.best_estimator_  = data["best_estimator_"]
        grid.cv_results_      = data["cv_results_"]
        return grid
    
    def save_final_model(self, model: EnsembleVoteClassifier, file_name: str) -> None:
        path = os.path.join(self.repo_dir, file_name)
        LOGGER.info(f'saving final model to {path}')
        with open(path, "wb") as fh:
            pickle.dump(model, fh)

    def load_final_model(self, file_name: str) -> EnsembleVoteClassifier:
        path = os.path.join(self.repo_dir, file_name)
        LOGGER.info(f'loading final model from {path}')
        with open(path, "rb") as fh:
            return pickle.load(fh)
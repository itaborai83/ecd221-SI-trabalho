# -*- coding: utf-8 -*-
import abc
import pickle
from typing import Tuple, List, Dict
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import telchurn.util as util




LOGGER = util.get_logger('grid_repository')

class GridRepository(abc.ABC):
    
    @abc.abstractmethod
    def save_grid(self, grid: Dict, file_name: str) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def load_grid(self, file_name: str) -> Dict:
        raise NotImplementedError
        
class GridRepositoryImpl(GridRepository):
    
    def save_grid(self, grid: Dict, file_name: str) -> None:
        LOGGER.info(f'saving grid search results to {file_name}')
        import pickle
        data = {
            "best_score_"     : grid.best_score_
        ,   "best_params_"    : grid.best_params_
        ,   "best_estimator_" : grid.best_estimator_
        ,   "cv_results_"     : grid.cv_results_
        ,   "grid"            : grid
        }  
        with open(file_name, "wb") as fh:
            pickle.dump(data, fh)
        
    def load_grid(self, file_name: str) -> Dict:
        LOGGER.info(f'loading grid search results from {file_name}')
        with open(file_name, "rb") as fh:
            data                  = pickle.load(fh)
        grid                  = data["grid"]
        grid.best_score_      = data["best_score_"]
        grid.best_params_     = data["best_params_"]
        grid.best_estimator_  = data["best_estimator_"]
        grid.cv_results_      = data["cv_results_"]
        return grid
        
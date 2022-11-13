# -*- coding: utf-8 -*-
import abc
import os.path
import pickle
import math
from itertools import combinations
from typing import Tuple, List, Dict
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, balanced_accuracy_score, f1_score, confusion_matrix
from mlxtend.classifier import EnsembleVoteClassifier

import telchurn.util as util

LOGGER = util.get_logger('model_evaluator')

class ModelEvaluator(abc.ABC):
    
    @abc.abstractmethod
    def report_results(self, estimator, X, y):
        raise NotImplementedError
    
class ModelEvaluatorImpl(ModelEvaluator):
    
    def report_results(self, estimator: EnsembleVoteClassifier, X_df: pd.DataFrame, y_df: pd.DataFrame) -> None:
        y_hat           = estimator.predict(X_df)
        # Confusion matrix whose i-th row and j-th column entry indicates the number of 
        # samples with true label being i-th class and predicted label being j-th class.
        accuracy        = accuracy_score(y_df, y_hat)
        precision       = precision_score(y_df, y_hat)
        recall          = recall_score(y_df, y_hat)
        balanced_acc    = balanced_accuracy_score(y_df, y_hat)
        f1              = f1_score(y_df, y_hat)
        conf_matrix     = confusion_matrix(y_df, y_hat)
        true_negative   = conf_matrix[0][0]
        false_positive  = conf_matrix[0][1]
        false_negative  = conf_matrix[1][0]
        true_positive   = conf_matrix[1][1]
        
        LOGGER.info(f"accuracy score        : {accuracy}")
        LOGGER.info(f"precision score       : {precision}")
        LOGGER.info(f"recall score          : {recall}")
        LOGGER.info(f"balanced acc. score   : {balanced_acc}")
        LOGGER.info(f"f1 score              : {f1}")
        LOGGER.info(f"confusion matrix") 
        LOGGER.info(f"\tTrue  Negative : {true_negative}") 
        LOGGER.info(f"\tFalse Positive : {false_positive}") 
        LOGGER.info(f"\tFalse Negative : {false_negative}") 
        LOGGER.info(f"\tTrue  Positive : {true_positive}") 
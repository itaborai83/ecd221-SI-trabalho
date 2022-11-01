# -*- coding: utf-8 -*-
import abc
import os.path
import pickle
import math
from itertools import combinations
from typing import Tuple, List, Dict
import pandas as pd
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score, fbeta_score, confusion_matrix
from mlxtend.classifier import EnsembleVoteClassifier
import telchurn.util as util

LOGGER = util.get_logger('ensembler')

class Ensembler(abc.ABC):
        
    @abc.abstractmethod
    def ensemble_models(self, grids: List[RandomizedSearchCV], y_train_df: pd.DataFrame, X_test_df: pd.DataFrame, y_test_df: pd.DataFrame) -> EnsembleVoteClassifier:
        raise NotImplementedError
    
        
class EnsemblerImpl(Ensembler):
    
     # soft voting é aquele no qual o estimador com mais "certeza" sobre a classificação vence
    VOTING_TYPE     = 'soft'
    MIN_ESTIMATORS  = 1
    BETA = 2.0    

    def __init__(self):
        self.top10_scores = [(0.0, 0, "")] * 10
    
    def update_scores(self, score, num_estimators, voting_type):
        self.top10_scores.append((score, num_estimators, voting_type))
        self.top10_scores.sort(reverse=True)
        self.top10_scores.pop(10)
        LOGGER.info(f"current top scores")
        for i, (score, num_estimators, voting_type) in enumerate(self.top10_scores):
            if num_estimators == 0:
                continue
            LOGGER.info(f"\t{i+1} - {score} with {num_estimators} estimators with {voting_type} voting")
    
    def compute_estimator_weights(self, grids: List[RandomizedSearchCV]) -> List[Tuple[RandomizedSearchCV, float]]:
        LOGGER.info('computing estimator weights')
        result = list([ 
            #(grid.best_estimator_, grid.best_score_) 
            (grid.best_estimator_, math.exp(1.0+grid.best_score_))
            for grid in grids 
        ])
        result.sort(key=lambda x: x[1])
        return result
    
    def compute_score(self, estimator, X_test_df, y_test):
      y_test_hat = estimator.predict(X_test_df)
      #train_score = fbeta_score(y_test, y_test_hat, beta=self.BETA)
      train_score = f1_score(y_test, y_test_hat)
      return train_score
      
    def ensemble_models(self, grids: List[RandomizedSearchCV], y_train_df: pd.DataFrame, X_test_df: pd.DataFrame, y_test_df: pd.DataFrame) -> EnsembleVoteClassifier:
        top10_scores = [0.0] * 10
        estimators_and_weights = self.compute_estimator_weights(grids)
        total_estimators = len(estimators_and_weights)
        assert self.MIN_ESTIMATORS <= total_estimators
        num_combined_estimators = range(self.MIN_ESTIMATORS, total_estimators + 1) # final de range é não inclusivo
        best_score          = -99999
        best_estimator      = None
        best_voting_type    = None
        util.silence_warnings()
        for voting_type in ['soft', 'hard']:
            for num_estimators in num_combined_estimators:
                for comb_estimators_weights in combinations(estimators_and_weights, num_estimators):
                    # https://stackoverflow.com/questions/13635032/what-is-the-inverse-function-of-zip-in-python
                    comb_estimators, weights = zip(*comb_estimators_weights)
                    # em versões mais antigas da biblioteca, o parâmetro fit_base_estimators chamava-se refit
                    classifier = EnsembleVoteClassifier(
                        clfs                = comb_estimators
                    ,   weights             = weights
                    ,   voting              = voting_type
                    ,   fit_base_estimators = False
                    )
                    classifier.fit(None, y_train_df) # nenhum dado é necessário pois fit_base_estimators=False
                    score = self.compute_score(classifier, X_test_df, y_test_df)
                    if score > best_score:
                        best_score          = score
                        best_estimator      = classifier
                        best_voting_type    = voting_type
                        self.update_scores(best_score,  num_estimators, best_voting_type)
                    
        LOGGER.info(f"best combination of estimators: ")
        for clf in best_estimator.clfs:
            LOGGER.info(f"\t{clf}")
        LOGGER.info(f"estimators weights: {classifier.weights}")
        y_test_hat      = best_estimator.predict(X_test_df)
        conf_matrix     = confusion_matrix(y_test_df, y_test_hat)
        true_positive   = conf_matrix[0][0]
        false_negative  = conf_matrix[0][1]
        false_positive  = conf_matrix[1][0]
        true_negative   = conf_matrix[1][1]
        LOGGER.info(f"confusion matrix: ") 
        LOGGER.info(f"\tTrue Positive   : {true_positive}") 
        LOGGER.info(f"\tFalse Negative  : {false_negative}") 
        LOGGER.info(f"\tFalse Positve   : {false_positive}") 
        LOGGER.info(f"\tTrue Negative   : {true_negative}") 
        return best_estimator
  

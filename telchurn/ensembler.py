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
from telchurn.data_splitter import DataSplitter
from telchurn.model_evaluator import ModelEvaluator
import telchurn.util as util

LOGGER = util.get_logger('ensembler')

class Ensembler(abc.ABC):
        
    @abc.abstractmethod
    def ensemble_models(self, grids: List[RandomizedSearchCV], churn_df: pd.DataFrame) -> EnsembleVoteClassifier:
        raise NotImplementedError
            
class EnsemblerImpl(Ensembler):
    
     # soft voting é aquele no qual o estimador com mais "certeza" sobre a classificação vence
    MIN_ESTIMATORS  = 1

    def __init__(self, splitter: DataSplitter, evaluator: ModelEvaluator):
        self.splitter = splitter
        self.evaluator = evaluator
        self.top10_scores = [(0.0, 0, "")] * 10
    
    def update_scores(self, score, num_estimators, voting_type):
        self.top10_scores.append((score, num_estimators, voting_type))
        self.top10_scores.sort(reverse=True)
        self.top10_scores.pop(10)
        LOGGER.info(f"current top scores")
        for i, (score, num_estimators, voting_type) in enumerate(self.top10_scores):
            if num_estimators == 0:
                continue
            LOGGER.info(f"\t{i+1} - {score} with {num_estimators} estimators and {voting_type} voting")
    
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
      train_score = balanced_accuracy_score(y_test, y_test_hat)
      return train_score
            
    def ensemble_models(self, grids: List[RandomizedSearchCV], churn_df: pd.DataFrame) -> EnsembleVoteClassifier:
        target = churn_df.columns[-1]
        (X_train_df, y_train_df), (X_test_df, y_test_df) = self.splitter.split(churn_df, target) 
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
                        clfs                 = comb_estimators
                    ,   weights              = weights
                    ,   voting               = voting_type
                    ,   fit_base_estimators = False
                    #,   refit                = False
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
        LOGGER.info("Train Results")
        self.evaluator.report_results(best_estimator, X_train_df, y_train_df)
        LOGGER.info("Test Results")
        self.evaluator.report_results(best_estimator, X_test_df, y_test_df)
        
        LOGGER.info('refiting base classifiers on whole data set')
        X = pd.concat([X_train_df, X_test_df])
        y = pd.concat([y_train_df, y_test_df])
        for clf in best_estimator.clfs:
            clf.fit(X, y)
        best_estimator.fit(None, y) # nenhum dado é necessário pois fit_base_estimators=False
        LOGGER.warn("Whole data set results (has data leakage)")
        self.evaluator.report_results(best_estimator, X, y)
        return best_estimator
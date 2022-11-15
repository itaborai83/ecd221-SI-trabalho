# -*- coding: utf-8 -*-
import abc
import argparse
from typing import Tuple, List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import telchurn.util as util


from telchurn.data_loader import DataLoader

LOGGER = util.get_logger('feature_ranker')

class FeatureRanker(abc.ABC):
    
    @abc.abstractmethod
    def rank_features(self, df: pd.DataFrame, target_variable: str) -> Tuple[str, float]:
        raise NotImplementedError
    
class FeatureRankerImpl(FeatureRanker):
    
    N_ESTIMATORS = 150
    
    def __init__(self, random_seed=None):
        self.random_seed = random_seed

    def rank_features(self, df: pd.DataFrame, target_variable: str) -> List[Tuple[str, float]]:
        LOGGER.info('ranking features')
        all_but_target = df.columns.difference([target_variable])
        X_df = df[all_but_target]
        y = df[target_variable]

        classifier = RandomForestClassifier(
            n_estimators  = self.N_ESTIMATORS
        ,   bootstrap     = True
        ,   class_weight  = "balanced_subsample"
        ,   random_state  = self.random_seed
        )
        LOGGER.info('fitting random forest classifier')
        classifier.fit(X_df, y)
        
        importances_df = pd.DataFrame({
            "feature"       : X_df.columns
        ,   "importance"    : classifier.feature_importances_
        })
        importances_df.sort_values("importance", ascending=False, inplace=True)
        result = list([
            (row.feature, row.importance) for row in importances_df.itertuples()
        ])
        LOGGER.info('Ranked Features')
        for feature, importance in result:
            LOGGER.info(f'\t-> {feature}: {importance}')
        return result

  # -*- coding: utf-8 -*-
import abc
import argparse
import pandas as pd
import telchurn.util as util
from telchurn.feature_ranker import FeatureRanker
from telchurn.data_loader import DataLoader

LOGGER = util.get_logger('feature_selector')

class FeatureSelector(abc.ABC):
       
    @abc.abstractmethod
    def select_features(self, top_k: int, target_variable: str, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
        
class FeatureSelectorImpl(abc.ABC):
        
    def __init__(self, feature_ranker: FeatureRanker):
        self.feature_ranker = feature_ranker
        
    def select_features(self, top_k: int, target_variable: str, df: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info(f'selecting top {top_k} features')
        df = df.copy()
        rankings = self.feature_ranker.rank_features(df, target_variable)
        feature_names = []
        for i in range(top_k):
            feature_name, importance = rankings[i]
            feature_names.append(feature_name)
        if target_variable not in feature_names:
            feature_names.append(target_variable)
        df = df[ feature_names ].copy()
        return df
        
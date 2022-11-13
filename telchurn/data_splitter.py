 # -*- coding: utf-8 -*-
import abc
import pandas as pd
from typing import Tuple, List
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler, ADASYN, SMOTE
import telchurn.util as util


LOGGER = util.get_logger('data_splitter')

class DataSplitter(abc.ABC):
    
    DEFAULT_TEST_PCT_SIZE   = 0.3 # 30% do conjunto de dados
    DEFAULT_RANDOM_STATE    = 42
    
    def __init__(self, seed: int=None, test_split_pct: float=None):
        self.seed = seed if seed else self.DEFAULT_RANDOM_STATE
        self.test_split_pct = test_split_pct if test_split_pct else self.DEFAULT_TEST_PCT_SIZE        
        
    @abc.abstractmethod
    def split(self, df: pd.DataFrame, target: str) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
        raise NotImplementedError
            
class DataSplitterImpl(DataSplitter):
    
    def __init__(self, seed: int=None, test_split_pct: float=None):
        super().__init__(seed, test_split_pct)
        
    def _oversample(self, X_train, y_train):
        return X_train.copy(), y_train.copy()

    def split(self, df: pd.DataFrame, target: str) -> Tuple[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame]]:
        LOGGER.info('splitting data set into train and test sets')
        all_but_target = df.columns.difference([target])
        X_df = df[all_but_target]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(
            X_df.values
        ,   y
        ,   test_size     = self.test_split_pct
        ,   shuffle       = True
        ,   random_state  = self.seed
        ,   stratify      = y # com estratificação
        )
        X_train_resampled, y_train_resampled = self._oversample(X_train, y_train)
        X_train_df = pd.DataFrame(X_train_resampled, columns=X_df.columns)
        y_train_df = pd.DataFrame(y_train_resampled, columns=[target])
        X_test_df = pd.DataFrame(X_test, columns=X_df.columns)
        y_test_df = pd.DataFrame(y_test, columns=[target])
        return (X_train_df, y_train_df), (X_test_df, y_test_df)
        
class DataSplitterOverSamplerImpl(DataSplitterImpl):

    def __init__(self, seed: int=None, test_split_pct: float=None):
        super().__init__(seed, test_split_pct)

    def _oversample(self, X_train, y_train):
        ros = RandomOverSampler(random_state=self.seed)
        return ros.fit_resample(X_train, y_train)
        
class DataSplitterSmoteImpl(DataSplitterImpl):
    
    def __init__(self, seed: int=None, test_split_pct: float=None):
        super().__init__(seed, test_split_pct)

    def _oversample(self, X_train, y_train):
        smote = SMOTE(random_state=self.seed)
        return smote.fit_resample(X_train, y_train)
        
class DataSplitterAdasynImpl(DataSplitterImpl):
    
    def __init__(self, seed: int=None, test_split_pct: float=None):
        super().__init__(seed, test_split_pct)

    def _oversample(self, X_train, y_train):
        adasyn = ADASYN(random_state=self.seed)
        return adasyn.fit_resample(X_train, y_train)        
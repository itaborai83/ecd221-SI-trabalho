 # -*- coding: utf-8 -*-
import abc
import pandas as pd
from sklearn.model_selection import train_test_split
import telchurn.util as util
from telchurn.data_loader import DataLoader, DataLoaderImpl
from telchurn.feature_processor import FeatureProcessor, FeatureProcessorImpl
from telchurn.feature_selector import FeatureSelector, FeatureSelectorImpl


from telchurn.data_loader import DataLoader

LOGGER = util.get_logger('trainer')

class Trainer(abc.ABC):
       
    @abc.abstractmethod
    def train(self, input_file : str) -> None:
        raise NotImplementedError
        
class TrainerImpl(abc.ABC):
    
    TOP_K_FEATURES  = 16
    TARGET_VARIABLE = "churn"
    TEST_PCT_SIZE   = 0.3 # 30% do conjunto de dados
    RANDOM_STATE    = 42
    
    def __init__(self, data_loader: DataLoader, feature_processor: FeatureProcessor, feature_selector: FeatureSelector):
        self.data_loader = data_loader
        self.feature_processor = feature_processor
        self.feature_selector = feature_selector
    
    def train(self, input_file : str) -> None:
        LOGGER.info('starting telco churn model training')
        churn_df = self.data_loader.load(input_file)
        churn_df = self.feature_processor.handle_categorical_features(churn_df)
        churn_df = self.feature_processor.engineer_features(churn_df)
        churn_df = self.feature_selector.select_features(self.TOP_K_FEATURES, self.TARGET_VARIABLE, churn_df)
        util.report_df(LOGGER, churn_df)
        X_train_df, X_test_df, y_train_df, y_test_df = self.__train_test_split(churn_df)
        
    def __train_test_split(self, churn_df):
        LOGGER.info('splitting data set into train and test sets')
        all_but_target = churn_df.columns.difference([self.TARGET_VARIABLE])
        X_df = churn_df[all_but_target]
        y = churn_df[self.TARGET_VARIABLE]        
        X_train, X_test, y_train, y_test = train_test_split(
            X_df.values
        ,   y
        ,   test_size     = self.TEST_PCT_SIZE
        ,   shuffle       = True
        ,   random_state  = self.RANDOM_STATE
        ,   stratify      = y # com estratificação
        )

        X_train_df = pd.DataFrame(X_train, columns=X_df.columns)
        X_test_df = pd.DataFrame(X_test, columns=X_df.columns)
        y_train_df = pd.DataFrame(y_train, columns=[self.TARGET_VARIABLE])
        y_test_df = pd.DataFrame(y_test, columns=[self.TARGET_VARIABLE])
        return X_train_df, X_test_df, y_train_df, y_test_df
        

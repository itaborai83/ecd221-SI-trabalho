 # -*- coding: utf-8 -*-
import abc
import argparse
from telchurn.data_loader import DataLoader, DataLoaderImpl
from telchurn.feature_processor import FeatureProcessor, FeatureProcessorImpl
import telchurn.util as util

from telchurn.data_loader import DataLoader

LOGGER = util.get_logger('trainer')

class Trainer(abc.ABC):
       
    @abc.abstractmethod
    def train(self, input_file : str) -> None:
        raise NotImplementedError
        
class TrainerImpl(abc.ABC):
    
    def __init__(self, data_loader: DataLoader, feature_processor: FeatureProcessor):
        self.data_loader = data_loader
        self.feature_processor = feature_processor
    
    def train(self, input_file : str) -> None:
        LOGGER.info('starting telco churn model training')
        churn_df = self.data_loader.load(input_file)
        churn_df = self.feature_processor.handle_categorical_features(churn_df)
        churn_df = self.feature_processor.engineer_features(churn_df)
        
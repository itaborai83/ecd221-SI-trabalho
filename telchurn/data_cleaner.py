 # -*- coding: utf-8 -*-
import io
import sys
import abc
import pandas as pd
import telchurn.util as util
from telchurn.data_loader import DataLoader
from telchurn.feature_processor import FeatureProcessor
from telchurn.feature_selector import FeatureSelector


LOGGER = util.get_logger('data_loader')

class DataCleaner(abc.ABC):
    
    DEFAULT_TOP_K_FEATURES  = 16
    DEFAULT_SEED = 42
    
    @abc.abstractmethod
    def clean(self, input_file_name_or_url: str, output_file_name_or_url: str, top_k_features: int=None, fields:str=None) -> None:
        raise NotImplementedError
        
class DataCleanerImpl(DataCleaner):
    
    TARGET_VARIABLE = "churn"
    
    def __init__(self, data_loader: DataLoader, feature_processor: FeatureProcessor, feature_selector: FeatureSelector):
        self.data_loader = data_loader
        self.feature_processor = feature_processor
        self.feature_selector = feature_selector
        
    def clean(self, input_file_name_or_url: str, output_file_name_or_url: str, top_k_features: int=None, fields:str=None) -> None:
        if top_k_features is None:
            top_k_features = self.DEFAULT_TOP_K_FEATURES        
        LOGGER.info(f'starting data cleaner')
        churn_df = self.data_loader.load(input_file_name_or_url)
        
        # transforma a variável target em uma variável numérica
        churn_df["churn"] = churn_df["churn"].map({"No": 0, "Yes": 1})

        # excluindo a variável customer_id
        del churn_df["customer_id"]
        
        # coluna total_charges possui registros vazios com valor ' '
        def convert_total_charges(value):
            return 0.0 if value == ' ' else value
        churn_df["total_charges"] = churn_df["total_charges"].map(convert_total_charges).astype(float)
        
        # diferente das outras colunas, senior_citizem possui valores 1 ou 0 ao invés de "Yes" or "No"
        churn_df["senior_citizen"] = churn_df["senior_citizen"].map({1: "Yes", 0: "No"})
        
        # removendo a feature de sexo que se mostrou irrelevante durante a análise exploratória
        del churn_df["gender"]
        
        churn_df = self.feature_processor.handle_categorical_features(churn_df)
        churn_df = self.feature_processor.engineer_features(churn_df)
        if fields is None:
            churn_df = self.feature_selector.select_features(top_k_features, self.TARGET_VARIABLE, churn_df)
        else:
            column_names = fields.split(',')
            churn_df = churn_df[column_names]
        self.save_cleansed(output_file_name_or_url, churn_df)
    
    def save_cleansed(self, file_name_or_url: str, churn_df: pd.DataFrame) -> None:
        LOGGER.info(f'saving cleansed dataframe to {file_name_or_url}')
        churn_df.to_csv(file_name_or_url, sep=DataLoader.DELIMITER, index=False)

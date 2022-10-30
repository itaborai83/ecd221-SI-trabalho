 # -*- coding: utf-8 -*-
import io
import sys
import abc
import pandas as pd
import telchurn.util as util

LOGGER = util.get_logger('data_loader')

class DataLoader(abc.ABC):
    
    DELIMITER = ','
    
    @abc.abstractmethod
    def load(self, file_name_or_url: str) -> pd.DataFrame:
        raise NotImplementedError
        
    @abc.abstractmethod
    def load_cleansed(self, file_name_or_url: str) -> pd.DataFrame:
        raise NotImplementedError
        
        
class DataLoaderImpl(DataLoader):
    FIELD_SEPARATOR             = ","
    IMPORT_COLUMN_NAMES         = [
        "customer_id"
    ,   "gender"
    ,   "senior_citizen"
    ,   "partner"
    ,   "dependents"
    ,   "tenure"
    ,   "phone_service"
    ,   "multiple_lines"
    ,   "internet_service"
    ,   "online_security"
    ,   "online_backup"
    ,   "device_protection"
    ,   "tech_support"
    ,   "streaming_tv"
    ,   "streaming_movies"
    ,   "contract"
    ,   "paperless_billing"
    ,   "payment_method"
    ,   "monthly_charges"
    ,   "total_charges"
    ,   "churn"
    ]
    SKIP_ROWS = 1
    
    def load(self, file_name_or_url: str) -> pd.DataFrame:
        LOGGER.info(f'loading dataframe from {file_name_or_url}')
        churn_df = pd.read_csv(
            file_name_or_url
        ,   names     = self.IMPORT_COLUMN_NAMES
        ,   skiprows  = 1
        ,   delimiter = self.DELIMITER
        )
        util.report_df(LOGGER, churn_df)
        return churn_df
        
    def load_cleansed(self, file_name_or_url: str) -> pd.DataFrame:
        LOGGER.info(f'loading cleansed dataframe from {file_name_or_url}')
        churn_df = pd.read_csv(
            file_name_or_url
        ,   delimiter = self.DELIMITER
        )
        return churn_df

    

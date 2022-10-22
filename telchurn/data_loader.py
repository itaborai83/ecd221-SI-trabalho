 # -*- coding: utf-8 -*-
import io
import sys
import abc
import pandas as pd
import telchurn.util as util

LOGGER = util.get_logger('data_loader')

class DataLoader(abc.ABC):
    
    @abc.abstractmethod
    def load(self, file_name_or_url: str) -> pd.DataFrame:
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
    BOOLEAN_MAP = {"No": 0, "Yes": 1}
    
    def load(self, file_name_or_url: str) -> pd.DataFrame:
        LOGGER.info(f'loading datframe from {file_name_or_url}')
        churn_df = pd.read_csv(
            file_name_or_url
        ,   names     = self.IMPORT_COLUMN_NAMES
        ,   skiprows  = 1
        ,   delimiter = ','
        )
    
        # transforma a variável target em uma variável numérica
        churn_df["churn"] = churn_df["churn"].map(self.BOOLEAN_MAP)

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
        
        self.__report(churn_df)
        return churn_df
    
    def __report(self, churn_df: pd.DataFrame) -> None:
        buffer = io.StringIO()
        churn_df.info(verbose=True, buf=buffer)
        buffer.seek(0)
        LOGGER.info(buffer.read())
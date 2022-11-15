 # -*- coding: utf-8 -*-
import io
import sys
import abc
import pandas as pd
import telchurn.util as util

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.svm import SVC

class PipelineFactory(abc.ABC):

    @abc.abstractmethod
    def build_pipeline_for(self, churn_df: pd.DataFrame) -> Pipeline:
        raise NotImplementedError

class PipelineFactoryImpl(PipelineFactory):
    
    NUMERICAL_FEATURES = [
        'tenure'
    ,   'monthly_charges'
    ,   'total_charges'
    ,   'client_factor'
    ,   'internet_factor'
    ,   'financial_factor'
    ,   'multi_factor'    
    ]
    
    def select_numerical_features(self, churn_df):
        # seleciona as variáveis numéricas que existem no data frame
        return list([nf for nf in self.NUMERICAL_FEATURES if nf in churn_df.columns])
        
    def build_pipeline_for(self, churn_df: pd.DataFrame) -> Pipeline:
        numerical_features = self.select_numerical_features(churn_df)
        # Configuração do pipeline

        # Os transformadores numéricos são utilizado spara processamento de todas as variáveis não categóricas.
        numeric_transformer = Pipeline([
          ("scaler", StandardScaler())    
        ])

        column_transformer = ColumnTransformer(
          transformers = [
            ("num", numeric_transformer, numerical_features)
          ],
          # importante usar passthrough quando nem todos os atributos forem processados
          remainder="passthrough" 
        )

        # Este pipelie será ajustado diversas vezes durante o processo de otimização dos hiper parâmetros.
        pipeline = Pipeline([
            # a primeira fase consiste no pré-processamento das variáveis numéricas
            ("feature_scaling", column_transformer),
            # redução de dimesionalidade
            ("reduce_dim", PCA()),
            # O algoritmo de regressão e seus parâmetros serão configurados via gridsearch
            ("classifier", SVC())
        ])
        return pipeline

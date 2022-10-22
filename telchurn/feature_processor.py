 # -*- coding: utf-8 -*-
import io
import abc
import numpy as np
import pandas as pd
import telchurn.util as util

LOGGER = util.get_logger('feature_processor')

class FeatureProcessor(abc.ABC):
    
    @abc.abstractmethod
    def handle_categorical_features(self, churn_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError

    @abc.abstractmethod
    def engineer_features(self, churn_df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError
        
class FeatureProcessorImpl(FeatureProcessor):
    
    BOOLEAN_FEATURES = [
        "senior_citizen"
    ,   "partner"
    ,   "dependents"
    ,   "phone_service"
    ,   "paperless_billing"
    ]    
    
    CATEGORICAL_FEATURES = [
        "multiple_lines"
    ,   "internet_service"
    ,   "online_security"
    ,   "online_backup"
    ,   "device_protection"
    ,   "tech_support"
    ,   "streaming_tv"
    ,   "streaming_movies"
    ,   "contract"
    ,   "payment_method"
    ]
    
    NUMERICAL_FEATURES = [
        "tenure"
    ,   "monthly_charges"
    ,   "total_charges"
    ]
    
    TARGET_VARIABLE = "churn"
    
    BOOLEAN_MAP = {"No": 0, "Yes": 1}
    
    # ruído a ser adicionado para evitar overfitting e fazer as variáveis novas parecerem numéricas
    NOISE_STD = 0.1
    
    def handle_categorical_features(self, churn_df: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info('handling categorical features')
        # copia o data frame para não estragar os dados originais
        churn_df = churn_df.copy()
        
        # transformando variáveis booleanas em numéricas (dummy encoding não é necessário)
        for feature in self.BOOLEAN_FEATURES:
          churn_df[feature] = churn_df[feature].map(self.BOOLEAN_MAP)

        # realizando o dummy encoding usando pandas
        dummy_df = pd.get_dummies(
            data        = churn_df[self.CATEGORICAL_FEATURES]
        ,   prefix      = self.CATEGORICAL_FEATURES
        ,   prefix_sep  = "="
        )

        # concatenando as variáveis boleanas, categóricas codificadas, numéricas e variável target num novo dataset
        churn_df = pd.concat([
            churn_df[ self.BOOLEAN_FEATURES ]
        ,   dummy_df
        ,   churn_df[ self.NUMERICAL_FEATURES ]
        ,   churn_df[ self.TARGET_VARIABLE ]
        ], axis=1)

        # removendo variáveis equivalentes internet_service=No = 1
        del churn_df[ "device_protection=No internet service" ]
        del churn_df[ "streaming_tv=No internet service"      ]
        del churn_df[ "tech_support=No internet service"      ]
        del churn_df[ "online_backup=No internet service"     ]
        del churn_df[ "streaming_movies=No internet service"  ]
        del churn_df[ "online_security=No internet service"   ]

        # removendo variáveis equivalentes phone_service=0
        del churn_df[ "multiple_lines=No phone service" ]

        # removendo variáveis codificada tornadas redundantes pelas deleções acima
        del churn_df[ "multiple_lines=No"    ]
        del churn_df[ "online_security=No"   ]
        del churn_df[ "online_backup=No"     ]
        del churn_df[ "device_protection=No" ]
        del churn_df[ "tech_support=No"      ]
        del churn_df[ "streaming_tv=No"      ]
        del churn_df[ "streaming_movies=No"  ]
        del churn_df[ "internet_service=No"  ]        
        
        new_column_names = {
            'multiple_lines=Yes'                       : 'multiple_lines'
        ,   'internet_service=DSL'                     : 'dsl'
        ,   'internet_service=Fiber optic'             : 'fiber_optic'
        ,   'online_security=Yes'                      : 'online_security'
        ,   'online_backup=Yes'                        : 'online_backup'
        ,   'device_protection=Yes'                    : 'device_protection'
        ,   'tech_support=Yes'                         : 'tech_support'
        ,   'streaming_tv=Yes'                         : 'streaming_tv'
        ,   'streaming_movies=Yes'                     : 'streaming_movies'
        ,   'contract=Month-to-month'                  : 'monthly_contract'
        ,   'contract=One year'                        : 'one_year_contract'
        ,   'contract=Two year'                        : 'two_year_contract'
        ,   'payment_method=Bank transfer (automatic)' : 'bank_transfer'
        ,   'payment_method=Credit card (automatic)'   : 'credit_card'
        ,   'payment_method=Electronic check'          : 'electronic_check'
        ,   'payment_method=Mailed check'              : 'mailed_check'
        }
        churn_df.rename(columns=new_column_names, inplace=True)                
        self.__report(churn_df)
        return churn_df
    
    def engineer_features(self, churn_df: pd.DataFrame) -> pd.DataFrame:
        LOGGER.info('engineering new features')
        churn_df = churn_df.copy()
        
        # o primeiro quartil da variável tenure conforme análise univariada anterior.
        # a probabilidade de rotatividade é inversamente proporcional à variável tenure
        tenure_1st_quartile = churn_df['tenure'].quantile(0.25)
        
        # o terceiro quartil da variável monthly_charges conforme análise univariada anterior.
        # a probabilidade de rotatividade é  proporcional à variável monthly_charges
        charges_3rd_quartile = churn_df['monthly_charges'].quantile(0.75)

        # quantidade de linhas necesário para criação do ruído
        rows, cols = churn_df.shape
        
        LOGGER.info('creating client factor')
        # client factor
        # A análise das tabulações cruzadas revelou que a existência de parceiro e dependentes tendem a fidelizar o cliente.
        # Em contrapartida, observou-se que clientes na terceira idade proporcionalmente tendem a cancelar os serviços
        # de maneira mais frequente.
        # A expectativa é de que quanto maior for o client_factor, maior a probabilidade de que ele venha a cancelar o seu contrato
        noise_term = np.random.normal(loc=0.0, scale=self.NOISE_STD, size=rows)
        churn_df["client_factor"] = ((
            np.exp(churn_df["senior_citizen"]) # senior_citizen=1 aumenta a rotatividade
        +   np.exp(np.abs(1-churn_df["partner"]))
        +   np.exp(np.abs(1-churn_df["dependents"]))
        +   np.exp((churn_df["tenure"] < tenure_1st_quartile).astype(float))
        +   np.exp((churn_df["monthly_charges"] > charges_3rd_quartile).astype(float))
        ) / 3.0 + noise_term)
        
        LOGGER.info('creating internet factor')
        # internet factor
        # A análise das tabulações cruzadas revelou que a existência a contratação dos
        # serviços de suporte técnico e de segurança online tendem a indicar que um usuário
        # encontra-se fidelizado. A contratação da internet de fibra ótica, ao elevar o valor
        # mensalmente cobrado, contribui com a rotatividade do cliente. Por outro lado,
        # os clientes com internet DSL tendem a permanecer como cliente devido a valor 
        # comparativamente mais baixo sendo cobrado.
        # A expectativa é de que quanto maior for o internet_factor, maior a probabilidade de que ele venha a cancelar o seu contrato
        noise_term = np.random.normal(loc=0.0, scale=self.NOISE_STD, size=rows)
        churn_df["internet_factor"] = ((
            np.exp(np.abs(1-churn_df["tech_support"])) 
        +   np.exp(np.abs(1-churn_df["online_security"]))  
        +   np.exp(churn_df["fiber_optic"]) # senior_citizen=1 aumenta a rotatividade
        -   np.exp(churn_df["dsl"]) # dsl=1 diminui a rotatividade
        +   np.exp((churn_df["tenure"] < tenure_1st_quartile).astype(float))
        +   np.exp((churn_df["monthly_charges"] > charges_3rd_quartile).astype(float))
        ) / 3.0 + noise_term)
        
        LOGGER.info('creating financial factor')
        # financial factor
        # A análise das tabulações cruzadas revelou que a existência o uso de cobrança digital,
        # o uso de contratos mensais e o pagamento via cheque eletrônico são fatores que
        # contribuem com a rotatividade dos clientes
        # A expectativa é de que quanto maior for o financial_factor, maior a probabilidade de que ele venha a cancelar o seu contrato
        noise_term = np.random.normal(loc=0.0, scale=self.NOISE_STD, size=rows)
        churn_df["financial_factor"] = ((
            np.exp(churn_df["monthly_contract"]) 
        +   np.exp(churn_df["electronic_check"])
        +   np.exp(churn_df["paperless_billing"])
        +   np.exp((churn_df["tenure"] < tenure_1st_quartile).astype(float))
        +   np.exp((churn_df["monthly_charges"] > charges_3rd_quartile).astype(float))
        ) / 3.0 + noise_term)
        
        LOGGER.info('combineing factors into one')
        # por último, criamos um fator combinando todos usados anteriormente
        noise_term = np.random.normal(loc=0.0, scale=self.NOISE_STD, size=rows)
        churn_df["multi_factor"] = ((
            np.exp(churn_df["senior_citizen"]) # senior_citizen=1 piora as p
        +   np.exp(np.abs(1-churn_df["partner"]))
        +   np.exp(np.abs(1-churn_df["dependents"]))
        +   np.exp(np.abs(1-churn_df["tech_support"])) 
        +   np.exp(np.abs(1-churn_df["online_security"])) 
        -   np.exp(churn_df["dsl"]) 
        +   np.exp(churn_df["fiber_optic"])
        +   np.exp(churn_df["monthly_contract"]) 
        +   np.exp(churn_df["electronic_check"])
        +   np.exp(churn_df["paperless_billing"])
        +   np.exp((churn_df["tenure"] < tenure_1st_quartile).astype(float))
        +   np.exp((churn_df["monthly_charges"] > charges_3rd_quartile).astype(float))
        ) / 9.0 + noise_term)

        # reposiciona a variável target ao final
        churn_df["churn"] = churn_df.pop("churn")
        self.__report(churn_df)

    def __report(self, churn_df: pd.DataFrame) -> None:
        buffer = io.StringIO()
        churn_df.info(verbose=True, buf=buffer)
        buffer.seek(0)
        LOGGER.info(buffer.read())        
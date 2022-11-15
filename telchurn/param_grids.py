 # -*- coding: utf-8 -*-
import abc
import pandas as pd
import telchurn.util as util
from typing import List, Dict

from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.feature_selection import RFECV
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix, plot_confusion_matrix, plot_precision_recall_curve, plot_roc_curve
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer # transformador de colunas, usado para tratamento das variáveis
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier

LOGGER = util.get_logger('param_grids')

QUICK_RUN   = False
USE_LOGREG  = True
USE_KNN     = True
USE_NB      = True
USE_DT      = True
USE_SVM     = False
USE_ADA     = True
USE_GB      = True
USE_RF      = True



class ParamGrids(abc.ABC):
    
    @abc.abstractmethod
    def get_parameter_grids(self) -> List[Dict]:
        raise NotImplementedError
    
class ParamGridsImpl(abc.ABC):
    
    def get_parameter_grids(self) -> List:
        return []                   \
        +   (self.get_logreg_grid() if USE_LOGREG else []) \
        +   (self.get_knn_grid()    if USE_KNN else []) \
        +   (self.get_nb_grid()     if USE_NB  else []) \
        +   (self.get_dt_grid()     if USE_DT  else []) \
        +   (self.get_svm_grid()    if USE_SVM else []) \
        +   (self.get_ada_grid()    if USE_ADA else []) \
        +   (self.get_gb_grid()     if USE_GB  else []) \
        +   (self.get_rf_grid()     if USE_RF  else [])
        
    def get_logreg_grid(self):
        LOGGER.info('creating parameter grids for logistic Regression - LOGREG')
        param_grid = [{
            # Logistic Regression
            "feature_scaling__num__scaler"  : ["passthrough", MinMaxScaler(), StandardScaler()],
            "reduce_dim"                    : ["passthrough", PCA(n_components=3), PCA(n_components=5)],
            "classifier"                    : [LogisticRegression()],
            "classifier__n_jobs"            : [-1], # all cpus available
            "classifier__penalty"           : ["elasticnet"],
            "classifier__class_weight"      : ["balanced"],
            "classifier__solver"            : ["saga"],
            "classifier__l1_ratio"          : [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        },
        {
            "feature_scaling__num__scaler"  : ["passthrough", MinMaxScaler(), StandardScaler()],
            "reduce_dim"                    : ["passthrough", PCA(n_components=3), PCA(n_components=5)],
            "classifier"                    : [LogisticRegression()],
            "classifier__n_jobs"            : [-1], # all cpus available
            "classifier__penalty"           : ["none"],
            "classifier__class_weight"      : ["balanced"],
        }]
        return [{
            "name"          : "LOGREG"
        ,   "iterations"    : (200 if not QUICK_RUN else 20)
        ,   "param_grid"    : param_grid
        }]
    
    def get_knn_grid(self):
        LOGGER.info('creating parameter grids for K Nearest Neighbors - KNN')
        param_grid = [{
            # KNeighborsClassifier
            "feature_scaling__num__scaler"  : ["passthrough", MinMaxScaler(), StandardScaler()],
            "reduce_dim"                    : ["passthrough", PCA(n_components=3), PCA(n_components=5)],
            "classifier"                    : [KNeighborsClassifier()],
            "classifier__n_jobs"            : [-1], # all cpus available
            "classifier__algorithm"         : ["kd_tree"],
            "classifier__metric"            : ["minkowski"],
            "classifier__p"                 : [0.5, 1.0, 1.5, 2.0],
            "classifier__n_neighbors"       : [5, 7, 10, 13, 15, 17, 20],
            "classifier__weights"           : ["uniform", "distance"]
            }]
        return [{
            "name"          : "KNN"
        ,   "iterations"    : (200 if not QUICK_RUN else 20)
        ,   "param_grid"    : param_grid
        }]
    
    def get_nb_grid(self):
        LOGGER.info('creating parameter grids for Naive Bayes - nb')
        param_grid = [{
            # GaussianNB
            "feature_scaling__num__scaler"  : ["passthrough", MinMaxScaler(), StandardScaler()],
            "reduce_dim"                    : ["passthrough", PCA(n_components=3), PCA(n_components=5)],
            "classifier"                    : [GaussianNB()],
            "classifier__var_smoothing"     : [1e-9, 1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1],
        }]    
        return [{
            "name"          : "NB"
        ,   "iterations"    : (200 if not QUICK_RUN else 20)
        ,   "param_grid"    : param_grid
        }]
   
    def get_dt_grid(self):
        LOGGER.info('creating parameter grids for Decision Tree - DT')
        param_grid = [{
            # DecisionTreeClassifier{
            "feature_scaling__num__scaler"  : ["passthrough", MinMaxScaler(), StandardScaler()],
            "reduce_dim"                    : ["passthrough", PCA(n_components=3), PCA(n_components=5)],
            "classifier"                    : [DecisionTreeClassifier()],
            "classifier__class_weight"      : [None, "balanced"],
            "classifier__criterion"         : ["gini", "entropy"],
            "classifier__splitter"          : ["best", "random"],
            "classifier__max_features"      : [None, "auto", "sqrt", "log2"],
            "classifier__max_depth"         : [10, 25, 50]
        }]
        return [{
            "name"          : "DT"
        ,   "iterations"    : (200 if not QUICK_RUN else 20)
        ,   "param_grid"    : param_grid
        }]

    def get_svm_grid(self):
        LOGGER.info('creating parameter grids for Suport Vector Machine classifier - SVM')
        param_grid = param_grid = [{
            # SVC        
            "feature_scaling__num__scaler"  : [MinMaxScaler(), StandardScaler()], # SVC precisa ter os argumentos escalonados para uma melhor performance
            #"reduce_dim"                    : ["passthrough", PCA(n_components=3), PCA(n_components=5)],
            "reduce_dim"                    : [PCA(n_components=3), PCA(n_components=5), PCA(n_components=7)],
            "classifier"                    : [SVC(probability=True)],
            "classifier__kernel"            : ["linear","rbf"],
            "classifier__gamma"             : ["scale", "auto"],
            "classifier__class_weight"      : [None, "balanced"],
            "classifier__tol"               : [1e-2],
            "classifier__C"                 : [1, 0.5, 0.1],
            #"classifier__max_iter"          : [1000],

        }]
        return [{
            "name"          : "SVM"
        ,   "iterations"    : (200 if not QUICK_RUN else 20)
        ,   "param_grid"    : param_grid
        }]
    
    def get_ada_grid(self):
        LOGGER.info('creating parameter grids for AdaBoost classifier - ADA')
        param_grid = [{
        # AdaBoostClassifier
            "feature_scaling__num__scaler"    : ["passthrough", MinMaxScaler(), StandardScaler()],
            "reduce_dim"                      : ["passthrough", PCA(n_components=3), PCA(n_components=5)],
            "classifier"                      : [AdaBoostClassifier()],
            "classifier__n_estimators"        : [100, 250, 500],
            "classifier__learning_rate"       : [0.001, 0.01, 0.1, 1.0]
        }]
        return [{
            "name"          : "ADA"
        ,   "iterations"    : (200 if not QUICK_RUN else 20)
        ,   "param_grid"    : param_grid
        }]

    def get_gb_grid(self):
        LOGGER.info('creating parameter grids for Gradient Boosting classifier - GB')
        param_grid = [{
            # GradientBoostingClassifier        
            "feature_scaling__num__scaler"    : ["passthrough", MinMaxScaler(), StandardScaler()],
            "reduce_dim"                      : ["passthrough", PCA(n_components=3), PCA(n_components=5)],
            "classifier"                      : [GradientBoostingClassifier()],
            "classifier__loss"                : ["log_loss", "deviance", "exponential"],
            "classifier__n_estimators"        : [50, 75, 100, 150],
            "classifier__learning_rate"       : [0.1, 0.3, 0.5, 0.7, 1.0],
            "classifier__max_depth"           : [3, 5, 10],
            "classifier__max_features"        : [None, "sqrt", "log2"]
        }]
        return [{
            "name"          : "GB"
        ,   "iterations"    : (200 if not QUICK_RUN else 20)
        ,   "param_grid"    : param_grid
        }]
        
    def get_rf_grid(self):
        LOGGER.info('creating parameter grids for Random Forest classifier - RF')
        param_grid = [{
            # RandomForestClassifier
            "feature_scaling__num__scaler"    : ["passthrough", MinMaxScaler(), StandardScaler()],
            "reduce_dim"                      : ["passthrough"], #PCA(n_components=5), PCA(n_components=10)],
            "classifier"                      : [RandomForestClassifier()],
            "classifier__n_estimators"        : [50, 100, 150, 200],
            "classifier__criterion"           : ["gini", "entropy"],
            "classifier__bootstrap"           : [True, False],
            "classifier__n_jobs"              : [-1],
            "classifier__class_weight"        : ["balanced", "balanced_subsample"],
            "classifier__max_depth"           : [5, 10, 25]
        }]
        return [{
            "name"          : "RF"
        ,   "iterations"    : (200 if not QUICK_RUN else 20)
        ,   "param_grid"    : param_grid
        }]
        
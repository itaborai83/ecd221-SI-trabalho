# -*- coding: utf-8 -*-
import abc
import argparse
from typing import Tuple, List
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import telchurn.util as util


from telchurn.data_loader import DataLoader

LOGGER = util.get_logger('model_repository')

class ModelRepository(abc.ABC):
    pass    

class ModelRepositoryImpl(ModelRepository):
    pass
 # -*- coding: utf-8 -*-
import abc
import argparse
from telchurn.data_loader import DataLoader, DataLoaderImpl
from telchurn.feature_processor import FeatureProcessor, FeatureProcessorImpl
from telchurn.feature_ranker import FeatureRanker, FeatureRankerImpl
from telchurn.feature_selector import FeatureSelector, FeatureSelectorImpl
from telchurn.pipeline_factory import PipelineFactory, PipelineFactoryImpl
from telchurn.hyper_param_tunner import HyperParamTunner, HyperParamTunnerImpl
from telchurn.trainer import Trainer, TrainerImpl
import telchurn.util as util

from telchurn.data_loader import DataLoader

LOGGER = util.get_logger('train')

class App:
    
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
    
    def run(self, input_file_or_url: str) -> None:
        self.trainer.train(input_file_or_url)
        
def main(input_file: str, seed: int, testsplit: float, kfolds: int):
    data_loader = DataLoaderImpl()
    pipeline_factory = PipelineFactoryImpl()
    hp_tunner = HyperParamTunnerImpl(kfolds, seed)
    trainer = TrainerImpl(data_loader, pipeline_factory, hp_tunner)
    trainer.train(input_file, seed, testsplit, kfolds)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',  type=int, help='random seed', default=Trainer.DEFAULT_RANDOM_STATE)
    parser.add_argument('--testsplit',  type=float, help='test split percentage', default=Trainer.DEFAULT_TEST_PCT_SIZE)
    parser.add_argument('--kfolds', type=int, help='number of k folds', default=HyperParamTunner.DEFAULT_K_FOLDS)
    parser.add_argument('input_file',  type=str, help='input file name')
    args = parser.parse_args()
    main(args.input_file, args.seed, args.testsplit, args.kfolds)
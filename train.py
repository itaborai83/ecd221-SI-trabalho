 # -*- coding: utf-8 -*-
import abc
import argparse
from telchurn.data_loader import DataLoader, DataLoaderImpl
from telchurn.feature_processor import FeatureProcessor, FeatureProcessorImpl
from telchurn.feature_ranker import FeatureRanker, FeatureRankerImpl
from telchurn.feature_selector import FeatureSelector, FeatureSelectorImpl
from telchurn.pipeline_factory import PipelineFactory, PipelineFactoryImpl
from telchurn.hyper_param_tunner import HyperParamTunner, HyperParamTunnerImpl
from telchurn.model_repository import ModelRepository, ModelRepositoryImpl
import telchurn.param_grids as param_grids
from telchurn.trainer import Trainer, TrainerImpl
import telchurn.util as util

from telchurn.data_loader import DataLoader

LOGGER = util.get_logger('train')
        
def main(input_file: str, seed: int, testsplit: float, kfolds: int, model_dir: str, quick: bool):
    if quick:
        LOGGER.warn('activating quick run mode')
        param_grids.QUICK_RUN = True
    data_loader = DataLoaderImpl()
    pipeline_factory = PipelineFactoryImpl()
    hp_tunner = HyperParamTunnerImpl(kfolds, seed)
    repo = ModelRepositoryImpl(model_dir)
    trainer = TrainerImpl(data_loader, pipeline_factory, hp_tunner, repo)
    trainer.train(input_file, seed, testsplit, kfolds)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',       type=int,   help='random seed',             default=Trainer.DEFAULT_RANDOM_STATE)
    parser.add_argument('--testsplit',  type=float, help='test split percentage',   default=Trainer.DEFAULT_TEST_PCT_SIZE)
    parser.add_argument('--kfolds',     type=int,   help='number of k folds',       default=HyperParamTunner.DEFAULT_K_FOLDS)
    parser.add_argument('--quick',      action="store_true", help='quick run', default=False)
    parser.add_argument('input_file',   type=str,   help='input file name')
    parser.add_argument('model_dir',    type=str,   help='models output directory')
    args = parser.parse_args()
    main(args.input_file, args.seed, args.testsplit, args.kfolds, args.model_dir, args.quick)
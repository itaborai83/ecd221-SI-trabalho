# -*- coding: utf-8 -*-
import abc
import argparse
from telchurn.data_loader import DataLoader, DataLoaderImpl
from telchurn.model_repository import ModelRepository, ModelRepositoryImpl
from telchurn.ensembler import Ensembler, EnsemblerImpl
from telchurn.data_splitter import DataSplitter
from telchurn.hyper_param_tunner import HyperParamTunner
from telchurn.data_splitter import DataSplitterImpl
from telchurn.model_evaluator import ModelEvaluatorImpl
import telchurn.util as util

LOGGER = util.get_logger('ensembler')

class App:

    def __init__(self, data_loader: DataLoader, repo: ModelRepository, ensembler: Ensembler):
        self.data_loader = data_loader
        self.repo = repo
        self.ensembler = ensembler
    
    def read_grids(self):
        LOGGER.info('reading saved grids')
        grids = []
        for grid_name in self.repo.list_grids():
            grid = self.repo.load_grid(grid_name)
            grids.append(grid)
        return grids
        
    def run(self, input_file_or_url: str, model_name: str) -> None:
        LOGGER.info('starting ensembler')
        grids = self.read_grids()
        churn_df = self.data_loader.load_cleansed(input_file_or_url)
        util.report_df(LOGGER, churn_df)
        voting_classifier = self.ensembler.ensemble_models(grids, churn_df)
        self.repo.save_final_model(voting_classifier, model_name)
        
        
def main(input_file: str, seed: int, testsplit: float, kfolds: int, model_dir: str, model_name: str):
    data_loader = DataLoaderImpl()
    repo = ModelRepositoryImpl(model_dir)
    evaluator = ModelEvaluatorImpl()
    splitter = DataSplitterImpl(seed, testsplit)
    ensembler = EnsemblerImpl(splitter, evaluator)
    app = App(data_loader, repo, ensembler)
    app.run(input_file, model_name)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',       type=int,   help='random seed',             default=DataSplitter.DEFAULT_RANDOM_STATE)
    parser.add_argument('--testsplit',  type=float, help='test split percentage',   default=DataSplitter.DEFAULT_TEST_PCT_SIZE)
    parser.add_argument('--kfolds',     type=int,   help='number of k folds',       default=HyperParamTunner.DEFAULT_K_FOLDS)
    parser.add_argument('input_file',   type=str,   help='input file name')
    parser.add_argument('model_dir',    type=str,   help='models output directory')
    parser.add_argument('model_name',   type=str,   help='final model name')
    args = parser.parse_args()
    main(args.input_file, args.seed, args.testsplit, args.kfolds, args.model_dir, args.model_name)
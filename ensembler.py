# -*- coding: utf-8 -*-
import abc
import argparse
from telchurn.data_loader import DataLoader, DataLoaderImpl
from telchurn.grid_repository import GridRepository, GridRepositoryImpl
from telchurn.ensembler import Ensembler, EnsemblerImpl
from telchurn.trainer import Trainer
from telchurn.hyper_param_tunner import HyperParamTunner
import telchurn.util as util

LOGGER = util.get_logger('ensembler')

class App:

    def __init__(self, data_loader: DataLoader, repo: GridRepository, ensembler: Ensembler):
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
        
    def run(self, input_file_or_url: str, seed: int, test_split_pct: float) -> None:
        LOGGER.info('starting ensembler')
        grids = self.read_grids()
        churn_df = self.data_loader.load_cleansed(input_file_or_url)
        util.report_df(LOGGER, churn_df)
        X_train_df, X_test_df, y_train_df, y_test_df = Trainer.train_test_split(churn_df, seed, test_split_pct)
        voting_classifier = self.ensembler.ensemble_models(grids, y_train_df, X_test_df, y_test_df)
        
def main(input_file: str, seed: int, testsplit: float, kfolds: int, metric: str, model_dir: str, quick: bool):
    if quick:
        LOGGER.warn('activating quick run mode')
        param_grids.QUICK_RUN = True
    data_loader = DataLoaderImpl()
    repo = GridRepositoryImpl(model_dir)
    ensembler = EnsemblerImpl()
    app = App(data_loader, repo, ensembler)
    app.run(input_file, seed, testsplit)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed',       type=int,   help='random seed',             default=Trainer.DEFAULT_RANDOM_STATE)
    parser.add_argument('--testsplit',  type=float, help='test split percentage',   default=Trainer.DEFAULT_TEST_PCT_SIZE)
    parser.add_argument('--kfolds',     type=int,   help='number of k folds',       default=HyperParamTunner.DEFAULT_K_FOLDS)
    parser.add_argument('--metric',     type=str,   help='metric',                  default=HyperParamTunner.DEFAULT_METRIC)
    parser.add_argument('--quick',      action="store_true", help='quick run', default=False)
    parser.add_argument('input_file',   type=str,   help='input file name')
    parser.add_argument('model_dir',    type=str,   help='models output directory')
    args = parser.parse_args()
    main(args.input_file, args.seed, args.testsplit, args.kfolds, args.metric, args.model_dir, args.quick)
 # -*- coding: utf-8 -*-
import abc
import argparse
from telchurn.data_loader import DataLoader, DataLoaderImpl
from telchurn.feature_processor import FeatureProcessor, FeatureProcessorImpl
from telchurn.trainer import Trainer, TrainerImpl
import telchurn.util as util

from telchurn.data_loader import DataLoader

LOGGER = util.get_logger('train')

class App:
    
    def __init__(self, trainer: Trainer):
        self.trainer = trainer
    
    def run(self, input_file_or_url: str) -> None:
        self.trainer.train(input_file_or_url)
        
def main(input_file: str):
    data_loader = DataLoaderImpl()
    feature_processor = FeatureProcessorImpl()
    trainer = TrainerImpl(data_loader, feature_processor)
    trainer.train(input_file)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file',  type=str, help='input file name')
    args = parser.parse_args()
    main(args.input_file)
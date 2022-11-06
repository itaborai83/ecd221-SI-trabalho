 # -*- coding: utf-8 -*-
import abc
import argparse
from telchurn.data_loader import DataLoader, DataLoaderImpl
from telchurn.feature_processor import FeatureProcessor, FeatureProcessorImpl
from telchurn.feature_ranker import FeatureRanker, FeatureRankerImpl
from telchurn.feature_selector import FeatureSelector, FeatureSelectorImpl
from telchurn.data_cleaner import DataCleaner, DataCleanerImpl
import telchurn.util as util

from telchurn.data_loader import DataLoader

LOGGER = util.get_logger('dataclean')
        
def main(input_file: str, output_file: str, topk: int, seed: int):
    data_loader = DataLoaderImpl()
    feature_processor = FeatureProcessorImpl(seed)
    feature_ranker = FeatureRankerImpl(seed)
    feature_selector = FeatureSelectorImpl(feature_ranker)
    data_cleaner = DataCleanerImpl(data_loader, feature_processor, feature_selector)
    data_cleaner.clean(input_file, output_file, topk)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, help='number of features to keep', default=DataCleaner.DEFAULT_SEED)
    parser.add_argument('--topk', type=int, help='number of features to keep', default=DataCleaner.DEFAULT_TOP_K_FEATURES)
    parser.add_argument('input_file', type=str, help='input file name or url')
    parser.add_argument('output_file', type=str, help='output file name')
    args = parser.parse_args()
    main(args.input_file, args.output_file, args.topk, args.seed)
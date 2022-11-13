# -*- coding: utf-8 -*-
import abc
import argparse
import pandas as pd
from telchurn.data_loader import DataLoader, DataLoaderImpl
from telchurn.data_splitter import DataSplitter, DataSplitterImpl
import telchurn.util as util

LOGGER = util.get_logger('splitter')

def main(seed: int, testsplit: float, input_file: str, output_file1: str, output_file2: str) -> None:
    data_loader = DataLoaderImpl()
    data_splitter = DataSplitterImpl(seed, testsplit)
    LOGGER.info('starting data splitter')
    df = data_loader.load_cleansed(input_file) # GAMBIARRA: load cleansed does not alter the input file
    LOGGER.info(f"input file: {input_file}")
    util.report_df(LOGGER, df)
    target = df.columns[-1]
    LOGGER.info(f'using column "{target}" as target variable')
    (X_train_df, y_train_df), (X_test_df, y_test_df) = data_splitter.split(df, target)
    
    train_df = X_train_df.copy()
    train_df[target] = y_train_df[target].values
    train_df = train_df[ df.columns ] # reorder columns - don't know why this is needed
    
    test_df = X_test_df.copy()
    test_df[target] = y_test_df[target].values
    test_df = test_df[ df.columns ] # reorder columns - don't know why this is needed
    
    LOGGER.info(f"output file 1: {output_file1}")
    util.report_df(LOGGER, train_df)
    LOGGER.info(f"output file 2: {output_file2}")
    util.report_df(LOGGER, test_df)
    
    train_df.to_csv(output_file1, sep=DataLoader.DELIMITER, index=False)
    test_df.to_csv(output_file2, sep=DataLoader.DELIMITER, index=False)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int,   help='random seed', default=DataSplitter.DEFAULT_RANDOM_STATE)
    parser.add_argument('--testsplit', type=float, help='test split percentage', default=DataSplitter.DEFAULT_TEST_PCT_SIZE)
    parser.add_argument('input_file', type=str,   help='input file name')
    parser.add_argument('output_file1', type=str,   help='output file 1')
    parser.add_argument('output_file2', type=str,   help='output file 2')
    args = parser.parse_args()
    main(
        args.seed
    ,   args.testsplit
    ,   args.input_file
    ,   args.output_file1
    ,   args.output_file2
    )
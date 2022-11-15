# -*- coding: utf-8 -*-
import abc
import argparse
import pandas as pd
from telchurn.data_loader import DataLoader, DataLoaderImpl
import telchurn.util as util

LOGGER = util.get_logger('fieldnames')

def main(input_file: str, output_file: str) -> None:
    data_loader = DataLoaderImpl()
    LOGGER.info('starting data splitter')
    df = data_loader.load_cleansed(input_file) # GAMBIARRA: load cleansed does not alter the input file
    if output_file == "-":
        print(*df.columns, sep=DataLoader.DELIMITER)
    else:
        with open(output_file, "w") as fh:
            print(*df.columns, sep=DataLoader.DELIMITER, file=fh, end='')
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input_file', type=str,   help='input file name')
    parser.add_argument('output_file', type=str,   help='output file name')
    args = parser.parse_args()
    main(args.input_file, args.output_file)
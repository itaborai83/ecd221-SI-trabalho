 # -*- coding: utf-8 -*-
import argparse
import os.path
from telchurn.model_repository import ModelRepositoryImpl
from telchurn.data_loader import DataLoaderImpl
from telchurn.model_repository import ModelRepositoryImpl
from telchurn.model_evaluator import ModelEvaluatorImpl
import telchurn.util as util

LOGGER = util.get_logger('classify')
        
def main(model_path: str, input_file: str) -> None:
    LOGGER.info('starting classifier')
    model_dir = os.path.dirname(model_path)
    model_file = os.path.basename(model_path)
    repo = ModelRepositoryImpl(model_dir)
    loader = DataLoaderImpl()
    evaluator = ModelEvaluatorImpl()
    estimator = repo.load_final_model(model_file)
    churn_df = loader.load_cleansed(input_file)    
    target = churn_df.columns[-1]
    all_but_target = churn_df.columns.difference([target])
    X_df = churn_df[all_but_target]
    y_df = churn_df[target]
    evaluator.report_results(estimator, X_df, y_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('model_file',   type=str,   help='saved model file')
    parser.add_argument('input_file',   type=str,   help='input file name')
    args = parser.parse_args()
    main(args.model_file, args.input_file)
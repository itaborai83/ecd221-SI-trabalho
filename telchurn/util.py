import io
import os
import sys
import warnings
import logging
import datetime as dt

LOGGER_FORMAT = '%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d\n\t%(message)s\n'
LOGGER_FORMAT = '%(levelname)s - %(filename)s:%(funcName)s:%(lineno)s - %(message)s'
stdout_handler = logging.StreamHandler(stream=sys.stdout)
logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT, handlers=[stdout_handler])
logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)

def get_logger(name):
    return logging.getLogger(name)

def report_df(logger, df):
    buffer = io.StringIO()
    df.info(verbose=True, buf=buffer)
    buffer.seek(0)
    logger.info(buffer.read())

def silence_warnings():
    # to silence warnings of subprocesses
    if not sys.warnoptions:
        warnings.simplefilter("ignore")
        os.environ["PYTHONWARNINGS"] = "ignore::UserWarning,ignore::FutureWarning"
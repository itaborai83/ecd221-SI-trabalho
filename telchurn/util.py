import io
import logging
import datetime as dt

LOGGER_FORMAT = '%(asctime)s:%(levelname)s:%(filename)s:%(funcName)s:%(lineno)d\n\t%(message)s\n'
LOGGER_FORMAT = '%(levelname)s - %(filename)s:%(funcName)s:%(lineno)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=LOGGER_FORMAT)

def get_logger(name):
    return logging.getLogger(name)

def report_df(logger, df):
    buffer = io.StringIO()
    df.info(verbose=True, buf=buffer)
    buffer.seek(0)
    logger.info(buffer.read())
        
def shallow_equality_test(self, other, attrs):
    for attr in attrs:
        if not hasattr(self, attr) or not hasattr(other, attr):
            msg = f"objects {repr(self)} and {repr(other)} compared but one of them lack the attribute {attr}"
            raise ValueError(msg)
        elif getattr(self, attr) != getattr(other, attr):
            return False
    return True

def build_str(self, attrs, indent=True):
    result = []
    result.append("<")
    result.append(type(self).__name__)
    for i,attr in enumerate(attrs):
        value = getattr(self, attr)
        if i > 0:
            if indent:
                txt = f",\n\t{attr}={repr(value)}"
            else:
                txt = f", {attr}={repr(value)}"
        else:
            if indent:
                txt = f"\n\t{attr}={repr(value)}"
            else:
                txt = f" {attr}={repr(value)}"        
        result.append(txt)
    result.append(">")
    return "".join(result)

def now():
    return dt.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
def escape_sql_value(value):
    if value is None:
        return 'NULL'
    elif not isinstance(value, str):
        return str(value)
    else:
        escaped_value = value.strip().replace("'", "''")
        escaped_value = "'" + escaped_value  + "'"
        return escaped_value
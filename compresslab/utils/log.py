import logging
import colorlog

# modify the color of the log here
colors_config = {
    'DEBUG': 'white',
    'INFO': 'green',
    'WARNING': 'yellow',
    'ERROR': 'red',
    'CRITICAL': 'bold_red',
}


logger = logging.getLogger()
logger.setLevel(logging.INFO)

console_handler = logging.StreamHandler()
console_formatter = colorlog.ColoredFormatter(
    fmt='%(log_color)s[%(asctime)s.%(msecs)03d] %(filename)s -> %(funcName)s line:%(lineno)d [%(levelname)s] : %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    log_colors=colors_config
)
console_handler.setFormatter(console_formatter)

if not logger.handlers:
    logger.addHandler(console_handler)
    
console_handler.close()


def debug(msg, *args, **kwargs):
    logging.debug(msg, *args, **kwargs)

def info(msg, *args, **kwargs):
    logging.info(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    logging.warning(msg, *args, **kwargs)

def error(msg, *args, **kwargs):
    logging.error(msg, *args, **kwargs)

def critical(msg, *args, **kwargs):
    logging.critical(msg, *args, **kwargs)


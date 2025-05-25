import logging
import sys
from config import LOG_FILE_PATH

def get_basic_logger(name: str):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    if not logger.hasHandlers():
        file_handler = logging.FileHandler(LOG_FILE_PATH, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.DEBUG)

        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

    return logger
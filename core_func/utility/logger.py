# -*- coding: utf-8 -*-
import logging
import os
import time


def get_logger(logger_name='default logger', log_path='', format_string='', add_handler=True):

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)

    if add_handler: # and (not logger.hasHandlers()):
        if log_path:
            fh = logging.FileHandler(log_path, encoding='utf-8')
        else:
            current_file_abs_path = os.path.dirname(os.path.abspath(__file__))
            fh = logging.FileHandler(f'{current_file_abs_path}/log_{time.strftime("%Y%m%d")}.log', encoding='utf-8')
        fh.setLevel(logging.DEBUG)

        sh = logging.StreamHandler()
        sh.setLevel(logging.DEBUG)

        if format_string:
            formatter = logging.Formatter(format_string)
        else:
            formatter = logging.Formatter('%(asctime)s - %(filename)s:%(lineno)d - %(levelname)s - %(processName)s - %(threadName)s: %(message)s')

        fh.setFormatter(formatter)
        sh.setFormatter(formatter)

        logger.addHandler(fh)
        logger.addHandler(sh)

    return logger
"""
日志记录
"""
import logging
import os
from datetime import datetime


class Logger(object):
    def __init__(self, path_log):
        log_name = os.path.basename(path_log)  # log.log
        self.log_name = log_name if log_name else "root"
        self.out_path = path_log

        log_dir = os.path.dirname(self.out_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    def init_logger(self, file, console):
        logger = logging.getLogger(self.log_name)
        logger.setLevel(level=logging.INFO)

        # 配置文件Handler
        file_handler = logging.FileHandler(self.out_path, "w")
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)

        # 配置屏幕Handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))

        # 添加handler
        if file:
            print("file_hander")
            logger.addHandler(file_handler)
        if console:
            print("console_handler")
            logger.addHandler(console_handler)

        return logger


def build_default_logger():
    logger = logging.getLogger("DefaultLogger")
    logger.setLevel(logging.INFO)

    formatter = logging.Formatter("[%(levelname)s][%(filename)s:%(lineno)d][%(asctime)s]: %(message)s")
    sh_handler = logging.StreamHandler()
    sh_handler.setFormatter(formatter)
    logger.addHandler(sh_handler)
    return logger


def make_logger(out_dir, file=True, console=True, local_rank=0):
    now_time = datetime.now()
    time_str = datetime.strftime(now_time, "%m-%d_%H-%M")
    log_dir = os.path.join(out_dir, time_str)  # 根据config中的创建时间作为文件夹名
    if local_rank == 0:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    path_log = os.path.join(log_dir, "log.log")
    logger = Logger(path_log)
    logger = logger.init_logger(file=file, console=console)
    return logger


class SingleInstanceLogger:
    def __init__(self):
        self.logger = build_default_logger()

    def __getattr__(self, name):
        return getattr(self.logger, name)


def setup_single_instance_logger(path):
    global _single_instance_logger
    _single_instance_logger.logger = make_logger(path)


_single_instance_logger = SingleInstanceLogger()

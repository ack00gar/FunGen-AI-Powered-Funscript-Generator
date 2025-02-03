import logging
import os
import sys
from datetime import datetime
from colorama import Fore, Style, init
from typing import Literal

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

init(autoreset=True)

class ColorizedStreamHandler(logging.StreamHandler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING or record.levelno == logging.WARN:
            record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        super().emit(record)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
log_path = os.path.join(project_root, "logs")
ensure_path_exists(log_path)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(log_path, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"), mode="w", encoding='utf-8'),
        ColorizedStreamHandler(sys.stdout)
    ]
)

log = logging.getLogger("Main")
log_od = logging.getLogger("ObjectDetection")
log_tr = logging.getLogger("Tracking")
log_vid = logging.getLogger("Video")
log_fun = logging.getLogger("Funscript")


def set_log_level(level: LogLevel):
    level = level.upper()
    log_level = getattr(logging, level, logging.INFO)

    # Update all active loggers
    for logger_name in logging.root.manager.loggerDict.keys():
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)
        for handler in logger.handlers:
            handler.setLevel(log_level)

    log.info(f"Log level set to {level}")
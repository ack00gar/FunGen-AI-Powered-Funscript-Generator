from datetime import datetime
import logging
import os
import sys
from colorama import Fore, Style, init

init(autoreset=True)

class ColorizedStreamHandler(logging.StreamHandler):
    def emit(self, record):
        if record.levelno >= logging.ERROR:
            record.msg = f"{Fore.RED}{record.msg}{Style.RESET_ALL}"
        elif record.levelno == logging.WARNING or record.levelno == logging.WARN:
            record.msg = f"{Fore.YELLOW}{record.msg}{Style.RESET_ALL}"
        super().emit(record)

project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
    handlers=[
        logging.FileHandler(
            os.path.join(project_root, "logs", f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"), mode="w", encoding='utf-8'),
        ColorizedStreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("General")

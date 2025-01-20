import logging
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s|%(levelname)s|%(name)s|%(message)s",
    handlers=[
        logging.FileHandler("FSGenerator.log", mode="w"),
        ColorizedStreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger("General")

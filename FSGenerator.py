import os

from script_generator.debug.logger import logger
from script_generator.gui.app import start_app

if __name__ == "__main__":
    logger.info(f"Starting VR funscript generator GUI with process id (PID): {os.getpid()}")
    start_app()

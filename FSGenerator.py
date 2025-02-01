import os

from script_generator.debug.logger import log
from script_generator.gui.app import start_app

if __name__ == "__main__":
    log.info(f"Starting VR funscript generator GUI with process id (PID): {os.getpid()}")
    start_app()

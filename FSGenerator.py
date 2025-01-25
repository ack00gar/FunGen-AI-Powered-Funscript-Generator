import os
from script_generator.gui.app import start_app

if __name__ == "__main__":
    print(f"PID: {os.getpid()}")
    start_app()

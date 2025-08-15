import multiprocessing
import platform
import argparse
import sys
import logging


def _setup_bootstrap_logger():
    """Set up early bootstrap logger for startup phase before full logger initialization."""
    # Set up a minimal colored console handler for startup
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create colored formatter for bootstrap phase
    class BootstrapColoredFormatter(logging.Formatter):
        GREY = "\x1b[90m"
        GREEN = "\x1b[32m"
        YELLOW = "\x1b[33m"
        RED = "\x1b[31m"
        BOLD_RED = "\x1b[31;1m"
        RESET = "\x1b[0m"
        
        format_base = "%(levelname)-8s - %(message)s"
        
        FORMATS = {
            logging.DEBUG: GREY + format_base + RESET,
            logging.INFO: GREEN + format_base + RESET,
            logging.WARNING: YELLOW + format_base + RESET,
            logging.ERROR: RED + format_base + RESET,
            logging.CRITICAL: BOLD_RED + format_base + RESET
        }
        
        def format(self, record):
            log_fmt = self.FORMATS.get(record.levelno)
            formatter = logging.Formatter(log_fmt)
            return formatter.format(record)
    
    # Add console handler with bootstrap formatter
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(BootstrapColoredFormatter())
    logger.addHandler(console_handler)

def run_gui():
    """Initializes and runs the graphical user interface."""
    from application.logic.app_logic import ApplicationLogic
    from application.gui_components.app_gui import GUI
    core_app = ApplicationLogic(is_cli=False)
    gui = GUI(app_logic=core_app)
    core_app.gui_instance = gui
    gui.run()

def run_cli(args):
    """Runs the application in command-line interface mode."""
    from application.logic.app_logic import ApplicationLogic
    logger = logging.getLogger(__name__)
    logger.info("--- FunGen CLI Mode ---")
    core_app = ApplicationLogic(is_cli=True)
    # This new method in ApplicationLogic will handle the CLI workflow
    core_app.run_cli(args)
    logger.info("--- CLI Task Finished ---")

def main():
    """
    Main function to run the application.
    This function handles dependency checking, argument parsing, and starts either the GUI or CLI.
    """
    # Step 1: Initialize bootstrap logger for early startup logging
    _setup_bootstrap_logger()
    logger = logging.getLogger(__name__)
    
    # Step 2: Perform dependency check before importing anything else
    try:
        from application.utils.dependency_checker import check_and_install_dependencies
        check_and_install_dependencies()
    except ImportError as e:
        logger.error(f"Failed to import dependency checker: {e}")
        logger.error("Please ensure the file 'application/utils/dependency_checker.py' exists.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An unexpected error occurred during dependency check: {e}")
        sys.exit(1)

    # Step 3: Set platform-specific multiprocessing behavior
    if platform.system() != "Windows":
        multiprocessing.set_start_method('spawn', force=True)
    else:
        # On Windows, ensure proper console window management for multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
        # Note: Windows uses 'spawn' by default, but we ensure it's set explicitly
        # This helps maintain consistent behavior across different Python versions

    # Step 4: Parse command-line arguments
    parser = argparse.ArgumentParser(description="FunGen - Automatic Funscript Generation")
    parser.add_argument('input_path', nargs='?', default=None, help='Path to a video file or a folder of videos. If omitted, GUI will start.')
    parser.add_argument('--mode', choices=['2-stage', '3-stage', '3-stage-mixed', 'oscillation-detector'], default='3-stage', help='The processing mode to use for analysis.')
    parser.add_argument('--od-mode', choices=['current', 'legacy'], default='current', help='Oscillation detector mode to use in Stage 3 (current=experimental, legacy=f5ae40f).')
    parser.add_argument('--overwrite', action='store_true', help='Force processing and overwrite existing funscripts. Default is to skip videos with existing funscripts.')
    parser.add_argument('--no-autotune', action='store_false', dest='autotune', help='Disable applying the default Ultimate Autotune settings after generation.')
    parser.add_argument('--no-copy', action='store_false', dest='copy', help='Do not save a copy of the final funscript next to the video file (will save to output folder only).')
    parser.add_argument('--recursive', '-r', action='store_true', help='If input_path is a folder, process it recursively.')

    args = parser.parse_args()

    # Step 5: Start the appropriate interface
    if args.input_path:
        run_cli(args)
    else:
        run_gui()

if __name__ == "__main__":
    main()

import multiprocessing
import platform
import argparse
import sys
import logging


def _setup_bootstrap_logger():
    """Set up early bootstrap logger for startup phase before full logger initialization."""
    # Get git info for bootstrap logging with improved error handling
    try:
        import subprocess
        import os
        
        # Increase timeout and add better error handling
        branch = 'unknown'
        commit = 'unknown'
        
        try:
            # Try to get branch name
            branch_result = subprocess.run(['git', 'branch', '--show-current'], 
                                         capture_output=True, text=True, timeout=5,
                                         cwd=os.getcwd())
            if branch_result.returncode == 0 and branch_result.stdout.strip():
                branch = branch_result.stdout.strip()
            else:
                # Debug: Log why branch detection failed
                if os.environ.get('FUNGEN_DEBUG_GIT'):
                    print(f"DEBUG: Branch detection failed. Return code: {branch_result.returncode}")
                    print(f"DEBUG: Stdout: '{branch_result.stdout}'")
                    print(f"DEBUG: Stderr: '{branch_result.stderr}'")
                    print(f"DEBUG: Working directory: {os.getcwd()}")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            if os.environ.get('FUNGEN_DEBUG_GIT'):
                print(f"DEBUG: Branch detection exception: {type(e).__name__}: {e}")
            pass
        
        try:
            # Try to get commit hash
            commit_result = subprocess.run(['git', 'rev-parse', '--short', 'HEAD'], 
                                         capture_output=True, text=True, timeout=5,
                                         cwd=os.getcwd())
            if commit_result.returncode == 0 and commit_result.stdout.strip():
                commit = commit_result.stdout.strip()
            else:
                # Debug: Log why commit detection failed
                if os.environ.get('FUNGEN_DEBUG_GIT'):
                    print(f"DEBUG: Commit detection failed. Return code: {commit_result.returncode}")
                    print(f"DEBUG: Stdout: '{commit_result.stdout}'")
                    print(f"DEBUG: Stderr: '{commit_result.stderr}'")
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
            if os.environ.get('FUNGEN_DEBUG_GIT'):
                print(f"DEBUG: Commit detection exception: {type(e).__name__}: {e}")
            pass
        
        git_info = f"{branch}@{commit}"
    except Exception:
        git_info = "nogit@unknown"
    
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
        
        format_base = f"[{git_info}] - %(levelname)-8s - %(message)s"
        
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
    # First, try to install the most basic bootstrap dependencies if they're missing
    import subprocess
    import sys
    import importlib
    
    # Check if we have the required bootstrap packages
    bootstrap_packages = ['packaging', 'requests', 'tqdm', 'send2trash']
    missing_bootstrap = []
    
    for package in bootstrap_packages:
        try:
            importlib.import_module(package)
        except ImportError:
            missing_bootstrap.append(package)
    
    if missing_bootstrap:
        logger.warning(f"Bootstrap dependencies missing: {', '.join(missing_bootstrap)}")
        logger.info("Installing bootstrap dependencies...")
        try:
            result = subprocess.run([sys.executable, "-m", "pip", "install"] + missing_bootstrap, 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                logger.warning(f"Failed to install some bootstrap dependencies: {result.stderr}")
                logger.info("Attempting to install from core.requirements.txt...")
                result = subprocess.run([sys.executable, "-m", "pip", "install", "-r", "core.requirements.txt"], 
                                      capture_output=True, text=True)
                if result.returncode != 0:
                    logger.error(f"Failed to install core requirements: {result.stderr}")
                    logger.error("Please manually install the requirements using: pip install -r core.requirements.txt")
                    sys.exit(1)
        except Exception as install_error:
            logger.error(f"Failed to install bootstrap dependencies: {install_error}")
            logger.error("Please manually install the requirements using: pip install -r core.requirements.txt")
            sys.exit(1)
    
    # Now try to import and run the dependency checker
    try:
        from application.utils.dependency_checker import check_and_install_dependencies
        check_and_install_dependencies()
    except ImportError as e:
        logger.error(f"Failed to import dependency checker after bootstrap: {e}")
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
    parser = argparse.ArgumentParser(description="FunGen - Automatic Funscript Generation and Processing")
    parser.add_argument('input_path', nargs='?', default=None, help='Path to a video file, folder of videos, or funscript file. If omitted, GUI will start.')
    
    # Funscript filtering mode
    parser.add_argument('--funscript-mode', action='store_true', help='Process funscript files instead of videos. Apply filters to existing funscripts.')
    parser.add_argument('--filter', choices=['ultimate-autotune', 'rdp-simplify', 'savgol-filter', 'speed-limiter', 'anti-jerk', 'amplify', 'clamp', 'invert', 'keyframe'], 
                        help='Filter to apply to funscript(s). Only works with --funscript-mode.')
    
    # Dynamic mode selection - get available modes from discovery system
    try:
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        available_modes = discovery.get_supported_cli_modes()
        batch_modes = [info.cli_aliases[0] for info in discovery.get_batch_compatible_trackers() if info.cli_aliases]
        default_mode = batch_modes[0] if batch_modes else '3-stage'
        
        parser.add_argument('--mode', choices=available_modes, default=default_mode, 
                        help='The processing mode to use for analysis. Only works with video processing.')
    except Exception as e:
        # Fallback if discovery system fails
        parser.add_argument('--mode', default='3-stage', help='Processing mode (discovery system unavailable)')
    
    parser.add_argument('--od-mode', choices=['current', 'legacy'], default='current', help='Oscillation detector mode to use in Stage 3 (current=experimental, legacy=f5ae40f).')
    parser.add_argument('--overwrite', action='store_true', help='Force processing and overwrite existing funscripts. Default is to skip videos with existing funscripts.')
    parser.add_argument('--no-autotune', action='store_false', dest='autotune', help='Disable applying the default Ultimate Autotune settings after generation.')
    parser.add_argument('--no-copy', action='store_false', dest='copy', help='Do not save a copy of the final funscript next to the video file (will save to output folder only).')
    parser.add_argument('--generate-roll', action='store_true', help='Generate secondary axis (.roll.funscript) file for supported multi-axis devices.')
    parser.add_argument('--recursive', '-r', action='store_true', help='If input_path is a folder, process it recursively.')

    args = parser.parse_args()

    # Step 5: Validate arguments and start the appropriate interface
    if args.input_path:
        # Validate funscript mode arguments
        if args.funscript_mode and not args.filter:
            logger.error("--funscript-mode requires --filter to be specified")
            sys.exit(1)
        if args.filter and not args.funscript_mode:
            logger.error("--filter can only be used with --funscript-mode")
            sys.exit(1)
        
        run_cli(args)
    else:
        run_gui()

if __name__ == "__main__":
    main()
    
import multiprocessing
import platform
import argparse
import sys

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
    print("--- FunGen CLI Mode ---")
    core_app = ApplicationLogic(is_cli=True)
    # This new method in ApplicationLogic will handle the CLI workflow
    core_app.run_cli(args)
    print("--- CLI Task Finished ---")

def main():
    """
    Main function to run the application.
    This function handles dependency checking, argument parsing, and starts either the GUI or CLI.
    """
    # Step 1: Perform dependency check before importing anything else
    # try:
    #     from application.utils.dependency_checker import check_and_install_dependencies
    #     check_and_install_dependencies()
    #     pass
    # except ImportError as e:
    #     print(f"Failed to import dependency checker: {e}", file=sys.stderr)
    #     print("Please ensure the file 'application/utils/dependency_checker.py' exists.", file=sys.stderr)
    #     sys.exit(1)
    # except Exception as e:
    #     print(f"An unexpected error occurred during dependency check: {e}", file=sys.stderr)
    #     sys.exit(1)

    # Step 2: Set platform-specific multiprocessing behavior
    if platform.system() != "Windows":
        multiprocessing.set_start_method('spawn', force=True)
    else:
        # On Windows, ensure proper console window management for multiprocessing
        multiprocessing.set_start_method('spawn', force=True)
        # Note: Windows uses 'spawn' by default, but we ensure it's set explicitly
        # This helps maintain consistent behavior across different Python versions

    # Step 3: Parse command-line arguments
    parser = argparse.ArgumentParser(description="FunGen - Automatic Funscript Generation")
    parser.add_argument('input_path', nargs='?', default=None, help='Path to a video file or a folder of videos. If omitted, GUI will start.')
    parser.add_argument('--mode', choices=['2-stage', '3-stage', 'oscillation-detector'], default='3-stage', help='The processing mode to use for analysis.')
    parser.add_argument('--overwrite', action='store_true', help='Force processing and overwrite existing funscripts. Default is to skip videos with existing funscripts.')
    parser.add_argument('--no-autotune', action='store_false', dest='autotune', help='Disable applying the default Ultimate Autotune settings after generation.')
    parser.add_argument('--no-copy', action='store_false', dest='copy', help='Do not save a copy of the final funscript next to the video file (will save to output folder only).')
    parser.add_argument('--recursive', '-r', action='store_true', help='If input_path is a folder, process it recursively.')

    args = parser.parse_args()

    # Step 4: Start the appropriate interface
    if args.input_path:
        run_cli(args)
    else:
        run_gui()

if __name__ == "__main__":
    main()

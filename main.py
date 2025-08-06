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
    try:
        from application.utils.dependency_checker import check_and_install_dependencies
        # check_and_install_dependencies()
        pass
    except ImportError as e:
        print(f"Failed to import dependency checker: {e}", file=sys.stderr)
        print("Please ensure the file 'application/utils/dependency_checker.py' exists.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"An unexpected error occurred during dependency check: {e}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Set platform-specific multiprocessing behavior
    multiprocessing.set_start_method('spawn', force=True)
    
    # Windows-specific: Configure console window suppression for multiprocessing
    if platform.system() == "Windows":
        import os
        import subprocess
        
        # Set environment flag for subprocess calls to use CREATE_NO_WINDOW
        os.environ['SUBPROCESS_CREATE_NO_WINDOW'] = '1'
        
        # Additional Windows configuration to minimize console window creation
        os.environ['PYTHONHASHSEED'] = '0'  # Ensure reproducible behavior
        
        # Try to hide the current console window if it exists and is not being used interactively
        try:
            import ctypes
            from ctypes import wintypes
            
            # Get the console window handle
            kernel32 = ctypes.windll.kernel32
            user32 = ctypes.windll.user32
            
            # Check if we have a console window
            console_hwnd = kernel32.GetConsoleWindow()
            if console_hwnd:
                # Check if the console window is visible and not the main application window
                is_visible = user32.IsWindowVisible(console_hwnd)
                if is_visible:
                    # Get the current foreground window
                    foreground_hwnd = user32.GetForegroundWindow()
                    
                    # Only hide the console if it's not the main foreground window
                    # This prevents hiding legitimate console usage
                    if console_hwnd != foreground_hwnd:
                        # Try to minimize or hide the console window
                        # SW_MINIMIZE = 6, SW_HIDE = 0
                        user32.ShowWindow(console_hwnd, 6)  # Minimize instead of hide for safety
        except Exception:
            # Silently ignore if console manipulation fails
            pass
        
        # Windows-specific: Configure multiprocessing to suppress console windows
        try:
            # Use a more direct approach: patch the underlying subprocess creation
            from multiprocessing import popen_spawn_win32
            
            # Get the original Popen class
            OriginalPopen = popen_spawn_win32.Popen
            
            class SuppressedPopen(OriginalPopen):
                def __init__(self, process_obj):
                    # Patch subprocess.Popen to always use CREATE_NO_WINDOW on Windows
                    original_subprocess_popen = subprocess.Popen
                    
                    def patched_subprocess_popen(*args, **kwargs):
                        # Always add CREATE_NO_WINDOW for multiprocessing spawned processes
                        if 'creationflags' in kwargs:
                            kwargs['creationflags'] |= subprocess.CREATE_NO_WINDOW
                        else:
                            kwargs['creationflags'] = subprocess.CREATE_NO_WINDOW
                        return original_subprocess_popen(*args, **kwargs)
                    
                    # Temporarily replace subprocess.Popen
                    subprocess.Popen = patched_subprocess_popen
                    try:
                        # Call the original Popen constructor
                        super().__init__(process_obj)
                    finally:
                        # Restore original subprocess.Popen
                        subprocess.Popen = original_subprocess_popen
            
            # Replace the Windows multiprocessing Popen class
            popen_spawn_win32.Popen = SuppressedPopen
            
        except Exception as e:
            # If patching fails, continue without it but log the issue
            print(f"Warning: Failed to configure multiprocessing console suppression: {e}")
            pass

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

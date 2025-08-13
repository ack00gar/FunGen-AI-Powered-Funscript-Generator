import os
import threading
import logging
import time
from typing import Optional, Callable, List
from application.utils import tensorrt_compiler

class TensorRTCompilerLogic:
    """Operating logic for TensorRT compiler."""
    
    def __init__(self, logger=None):
        self.logger = logger or logging.getLogger(__name__)
        self.compiler = tensorrt_compiler.TensorRTCompiler()
        self._compile_lock = threading.Lock()
        self._should_stop = False
        
        # State
        self.selected_pt_path = ""
        self.selected_output_dir = ""
        self.status_message = ""
        self.is_compiling = False
        self.compile_thread = None
        
        # Copy functionality state
        self._original_status_message = ""
        self._copied_message_end_time = 0

    def set_model_path(self, path: str):
        """Set the model path and update output directory."""
        self.logger.info(f"[TensorRTCompilerLogic] Setting model path: {path}")
        self.selected_pt_path = path
        if path:
            self.selected_output_dir = os.path.dirname(path)
            # Clear validation cache when file changes to force re-validation
            self.compiler.validation_manager.clear_cache()

    def set_output_directory(self, path: str):
        """Set the output directory."""
        self.logger.info(f"[TensorRTCompilerLogic] Setting output directory: {path}")
        if path:
            self.selected_output_dir = path
            # Clear validation cache when directory changes to force re-validation
            self.compiler.validation_manager.clear_cache()

    def get_status_message(self) -> str:
        """Get current status message."""
        # Check if we should restore the original status message
        if self._copied_message_end_time > 0 and time.time() >= self._copied_message_end_time:
            self.status_message = self._original_status_message
            self._copied_message_end_time = 0
            self._original_status_message = ""
        
        return self.status_message

    def set_status_message(self, msg: str):
        """Set status message."""
        self.status_message = msg
        if self.logger:
            self.logger.info(msg)

    def get_compilation_status(self) -> bool:
        """Check if compilation is in progress."""
        return self.is_compiling

    def can_compile(self) -> bool:
        """Check if compilation can proceed."""
        return (
            self.selected_pt_path and 
            self.selected_output_dir and 
            not self.is_compiling and
            os.path.isfile(self.selected_pt_path) and
            os.path.isdir(self.selected_output_dir)
        )

    def get_output_path_preview(self) -> str:
        """Get the expected output path for preview."""
        if self.selected_pt_path:
            basename = os.path.splitext(os.path.basename(self.selected_pt_path))[0]
            return os.path.join(self.selected_output_dir or "", basename + ".engine")
        return ""

    def _progress_callback(self, msg: str):
        """Handle progress updates."""
        if not self._should_stop:
            self.status_message = msg

    def start_compilation(self):
        """Start compilation process."""
        with self._compile_lock:
            if self.is_compiling:
                return
            self.is_compiling = True
            self._should_stop = False
        
        self.status_message = "Starting compilation..."
        
        def run():
            try:
                if self._should_stop:
                    return
                    
                output_path = self.compiler.compile_yolo_to_tensorrt(
                    self.selected_pt_path,
                    self.selected_output_dir,
                    progress_callback=self._progress_callback)
                
                if not self._should_stop:
                    self.status_message = f"Success! Output: {output_path}"
                    # Clear validation cache after successful compilation to refresh file checks
                    self.compiler.validation_manager.clear_cache()
            except tensorrt_compiler.TensorRTCompilerError as e:
                if not self._should_stop:
                    self.status_message = f"Compilation Error: {e}"
            except Exception as e:
                if not self._should_stop:
                    self.status_message = f"Unexpected Error: {e}"
                    if self.logger:
                        self.logger.error(f"TensorRT compilation failed: {e}")
            finally:
                with self._compile_lock:
                    self.is_compiling = False
                    # Clear status message if compilation was stopped by user
                    if self._should_stop:
                        self.status_message = "Compilation stopped by user"
        
        self.compile_thread = threading.Thread(target=run, daemon=True, name="TensorRTCompileThread")
        self.compile_thread.start()

    def request_stop_compilation(self):
        """Request compilation to stop and update UI state."""
        with self._compile_lock:
            if self.is_compiling:
                self._should_stop = True
                self.compiler.stop_compilation()
                self.status_message = "Stopping compilation..."

    def reset_state(self):
        """Reset the compiler state."""
        self.selected_pt_path = ""
        self.selected_output_dir = ""
        self.status_message = ""
        self.is_compiling = False
        self.compile_thread = None
        self._should_stop = False
        self.compiler.validation_manager.clear_cache()

    def get_validation_results(self):
        """Get validation results from the compiler."""
        return self.compiler.validation_manager.validate_all(
            self.selected_pt_path, 
            self.selected_output_dir
        )

    def copy_output_to_clipboard(self):
        """Copy subprocess output to clipboard using cross-platform native methods."""
        try:
            output_lines = self.compiler.get_subprocess_output()
            
            if not output_lines:
                content = "No compilation output available"
            else:
                # Join all output lines with newlines
                content = '\n'.join(output_lines)
            
            # Detect platform and use appropriate clipboard command
            import subprocess
            import platform
            
            system = platform.system()
            
            if system == "Windows":
                # Windows: use clip command
                process = subprocess.Popen(['clip'], stdin=subprocess.PIPE, text=True, shell=True)
            elif system == "Darwin":  # macOS
                # macOS: use pbcopy command
                process = subprocess.Popen(['pbcopy'], stdin=subprocess.PIPE, text=True)
            elif system == "Linux":
                # Linux: try xclip first, fallback to xsel
                try:
                    process = subprocess.Popen(['xclip', '-selection', 'clipboard'], stdin=subprocess.PIPE, text=True)
                except FileNotFoundError:
                    try:
                        process = subprocess.Popen(['xsel', '--clipboard', '--input'], stdin=subprocess.PIPE, text=True)
                    except FileNotFoundError:
                        self.status_message = "Error: No clipboard utility found (install xclip or xsel)"
                        return
            else:
                self.status_message = f"Error: Clipboard not supported on {system}"
                return
            
            # Send content to clipboard and wait for completion
            stdout, stderr = process.communicate(input=content)
            
            if process.returncode == 0:
                # Store original status message and show copied message for 5 seconds
                self._original_status_message = self.status_message
                self._copied_message_end_time = time.time() + 4.0
                self.status_message = "Output copied to clipboard!"
            else:
                self.status_message = f"Error: Failed to copy to clipboard (code: {process.returncode})"
                
        except Exception as e:
            self.status_message = f"Error copying to clipboard: {e}"
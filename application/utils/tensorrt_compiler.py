import os
import shutil
import gc
import threading
import time
import subprocess
import sys
import json
import send2trash
from typing import Optional, Callable, List, Dict

from config.constants import SEND2TRASH_MAX_ATTEMPTS, SEND2TRASH_RETRY_DELAY

# Import dependencies once at module level to avoid duplicated imports
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    YOLO = None
    ULTRALYTICS_AVAILABLE = False

try:
    import tensorrt
    TENSORRT_AVAILABLE = True
except ImportError:
    tensorrt = None
    TENSORRT_AVAILABLE = False


class TensorRTCompilerError(Exception):
    """Base exception for TensorRT compiler errors."""
    pass

class ValidationResult:
    """Represents the result of a validation check."""
    def __init__(self, name: str, success: bool, version: str = None, 
                 details: str = None, file_path: str = None):
        self.name = name
        self.success = success
        self.version = version
        self.details = details
        self.file_path = file_path

class ValidationManager:
    """Centralized validation system with caching."""
    def __init__(self):
        self._validation_cache: Dict[str, ValidationResult] = {}
        self._version_cache: Dict[str, str] = {}
        self._lock = threading.Lock()

    def clear_cache(self):
        """Clear all cached validation results."""
        with self._lock:
            self._validation_cache.clear()
            self._version_cache.clear()

    def get_version(self, module_name: str) -> str:
        """Get version of a module, with caching."""
        with self._lock:
            if module_name in self._version_cache:
                return self._version_cache[module_name]

        try:
            module = __import__(module_name)
            version = getattr(module, '__version__', 'Unknown')
            with self._lock:
                self._version_cache[module_name] = version
            return version
        except ImportError:
            return 'Not installed'

    def validate_all(self, pt_path: str, output_dir: str) -> List[ValidationResult]:
        """Run all validation checks in logical dependency order."""
        results = []

        # Check base dependencies first
        pytorch_result = self._check_pytorch_installation()
        results.append(pytorch_result)

        # PyTorch-dependent checks
        results.extend([
            self._check_cuda_installation(),
            self._check_cudnn_installation(),
            self._check_cuda_device()
        ])

        # Independent checks
        results.extend([
            self._check_tensorrt_installation(),
            self._check_ultralytics_installation()
        ])

        # File and directory checks
        if output_dir:
            results.append(self._check_output_directory(output_dir))
            results.append(self._check_output_directory_writable(output_dir))

        if pt_path:
            results.append(self._check_onnx_file_exists(pt_path))
            results.append(self._check_engine_file_exists(pt_path))

        return results

    def _check_cuda_installation(self) -> ValidationResult:
        """Check CUDA installation."""
        cache_key = "cuda_installation"
        with self._lock:
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]

        if not TORCH_AVAILABLE:
            result = ValidationResult("CUDA Installation", False, details="PyTorch not installed")
        elif torch.cuda.is_available():
            cuda_version = torch.version.cuda
            result = ValidationResult("CUDA Installation", True, cuda_version)
        else:
            result = ValidationResult("CUDA Installation", False, details="CUDA not available")

        with self._lock:
            self._validation_cache[cache_key] = result
        return result

    def _check_tensorrt_installation(self) -> ValidationResult:
        """Check TensorRT installation."""
        cache_key = "tensorrt_installation"
        with self._lock:
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]

        if TENSORRT_AVAILABLE:
            version = self.get_version("tensorrt")
            result = ValidationResult("TensorRT Installation", True, version)
        else:
            result = ValidationResult("TensorRT Installation", False, details="TensorRT not installed")

        with self._lock:
            self._validation_cache[cache_key] = result
        return result

    def _check_pytorch_installation(self) -> ValidationResult:
        """Check PyTorch installation."""
        cache_key = "pytorch_installation"
        with self._lock:
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]

        if TORCH_AVAILABLE:
            version = self.get_version("torch")
            result = ValidationResult("PyTorch Installation", True, version)
        else:
            result = ValidationResult("PyTorch Installation", False, details="PyTorch not installed")

        with self._lock:
            self._validation_cache[cache_key] = result
        return result

    def _check_cudnn_installation(self) -> ValidationResult:
        """Check cuDNN installation."""
        cache_key = "cudnn_installation"
        with self._lock:
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]

        if not TORCH_AVAILABLE:
            result = ValidationResult("cuDNN Installation", False, details="PyTorch not installed")
        elif hasattr(torch.backends, 'cudnn') and torch.backends.cudnn.is_available():
            # Try to get cuDNN version
            try:
                cudnn_version = torch.backends.cudnn.version()
                if cudnn_version:
                    version_str = f"v{cudnn_version}"
                    result = ValidationResult("cuDNN Installation", True, version_str)
                else:
                    result = ValidationResult("cuDNN Installation", True, "Available")
            except (AttributeError, RuntimeError):
                # Fallback if version is not available
                result = ValidationResult("cuDNN Installation", True, "Available")
        else:
            result = ValidationResult("cuDNN Installation", False, details="cuDNN not available")

        with self._lock:
            self._validation_cache[cache_key] = result
        return result

    def _check_ultralytics_installation(self) -> ValidationResult:
        """Check ultralytics installation."""
        cache_key = "ultralytics_installation"
        with self._lock:
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]

        if ULTRALYTICS_AVAILABLE:
            version = self.get_version("ultralytics")
            result = ValidationResult("Ultralytics Installation", True, version)
        else:
            result = ValidationResult("Ultralytics Installation", False, details="Ultralytics not installed")

        with self._lock:
            self._validation_cache[cache_key] = result
        return result

    def _check_cuda_device(self) -> ValidationResult:
        """Check CUDA device availability."""
        cache_key = "cuda_device"
        with self._lock:
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]

        if not TORCH_AVAILABLE:
            result = ValidationResult("CUDA Device", False, details="PyTorch not installed")
        elif torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            result = ValidationResult("CUDA Device", True, details=device_name)
        else:
            result = ValidationResult("CUDA Device", False, details="No CUDA device available")

        with self._lock:
            self._validation_cache[cache_key] = result
        return result

    def _check_output_directory(self, output_dir: str) -> ValidationResult:
        """Check if output directory exists."""
        cache_key = f"output_dir_{output_dir}"
        with self._lock:
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]

        if os.path.isdir(output_dir):
            result = ValidationResult("Output Directory", True, details=output_dir)
        else:
            result = ValidationResult("Output Directory", False, details=f"Directory does not exist: {output_dir}")

        with self._lock:
            self._validation_cache[cache_key] = result
        return result

    def _check_output_directory_writable(self, output_dir: str) -> ValidationResult:
        """Check if output directory is writable."""
        cache_key = f"output_dir_writable_{output_dir}"
        with self._lock:
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]
        
        if os.path.isdir(output_dir):
            test_file = os.path.join(output_dir, ".test_write")
            try:
                with open(test_file, 'w') as f:
                    f.write("test")
                os.remove(test_file)
                result = ValidationResult("Output Directory Writable", True)
            except (IOError, OSError):
                result = ValidationResult("Output Directory Writable", False, details="Directory not writable")
        else:
            result = ValidationResult("Output Directory Writable", False, details="Directory does not exist")

        with self._lock:
            self._validation_cache[cache_key] = result
        return result

    def _check_onnx_file_exists(self, pt_path: str) -> ValidationResult:
        """Check if ONNX file exists in same directory as PT file."""
        cache_key = f"onnx_exists_{pt_path}"
        with self._lock:
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]

        if pt_path and os.path.isfile(pt_path):
            dir_path = os.path.dirname(pt_path)
            base_name = os.path.splitext(os.path.basename(pt_path))[0]
            onnx_path = os.path.join(dir_path, f"{base_name}.onnx")
            
            if os.path.isfile(onnx_path):
                result = ValidationResult("ONNX File Status", True, details="ONNX file already exists - will be overwritten", file_path=onnx_path)
            else:
                result = ValidationResult("ONNX File Status", True, details="ONNX file will be created during compilation")
        else:
            result = ValidationResult("ONNX File Status", False, details="PT file not selected")

        with self._lock:
            self._validation_cache[cache_key] = result
        return result

    def _check_engine_file_exists(self, pt_path: str) -> ValidationResult:
        """Check if Engine file exists in same directory as PT file."""
        cache_key = f"engine_exists_{pt_path}"
        with self._lock:
            if cache_key in self._validation_cache:
                return self._validation_cache[cache_key]

        if pt_path and os.path.isfile(pt_path):
            dir_path = os.path.dirname(pt_path)
            base_name = os.path.splitext(os.path.basename(pt_path))[0]
            engine_path = os.path.join(dir_path, f"{base_name}.engine")
            
            if os.path.isfile(engine_path):
                result = ValidationResult("Engine File Status", True, details="Engine file already exists - will be overwritten", file_path=engine_path)
            else:
                result = ValidationResult("Engine File Status", True, details="Engine file will be created during compilation")
        else:
            result = ValidationResult("Engine File Status", False, details="PT file not selected")
        
        with self._lock:
            self._validation_cache[cache_key] = result
        return result

def _safe_delete_file(file_path: str, max_attempts: int = None, retry_delay: float = None) -> bool:
    """
    Safely delete a file using send2trash with configurable retry logic.

    Args:
        file_path: Path to the file to delete
        max_attempts: Maximum number of deletion attempts (uses SEND2TRASH_MAX_ATTEMPTS if None)
        retry_delay: Delay in seconds between retry attempts (uses SEND2TRASH_RETRY_DELAY if None)

    Returns:
        True if file was successfully deleted, False otherwise.
    """
    if max_attempts is None:
        max_attempts = SEND2TRASH_MAX_ATTEMPTS
    if retry_delay is None:
        retry_delay = SEND2TRASH_RETRY_DELAY

    if not os.path.exists(file_path):
        return True

    for attempt in range(max_attempts):
        # Apply delay before retry attempts (not first attempt)
        if attempt > 0:
            time.sleep(retry_delay)

        try:
            send2trash.send2trash(file_path)
            time.sleep(0.5)  # Brief pause for file system
            if not os.path.exists(file_path):
                return True
        except Exception:
            if attempt == max_attempts - 1:  # Last attempt
                return False
            continue
    
    # Final check after all attempts
    return not os.path.exists(file_path)

class TensorRTCompiler:
    """Main TensorRT compiler class."""
    def __init__(self):
        self.validation_manager = ValidationManager()
        self._compile_lock = threading.Lock()
        self._should_stop = False
        self._export_process = None
        self._subprocess_output = []  # Store subprocess output lines
        self._output_lock = threading.Lock()

    def compile_yolo_to_tensorrt(
        self,
        pt_model_path: str,
        output_dir: str,
        progress_callback: Optional[Callable[[str], None]] = None
    ) -> str:
        """
        Compiles a YOLO .pt model to a TensorRT .engine file using ultralytics.
        Returns the path to the generated .engine file on success.
        Raises TensorRTCompilerError on failure.
        """
        model_basename = os.path.splitext(os.path.basename(pt_model_path))[0]
        output_path = os.path.join(output_dir, model_basename + ".engine")

        # Check for existing files and safely delete them FIRST
        existing_files = []
        if os.path.exists(output_path):
            existing_files.append(output_path)

        # Also check for ONNX file in the same directory as PT file
        pt_dir = os.path.dirname(pt_model_path)
        onnx_path = os.path.join(pt_dir, model_basename + ".onnx")
        if os.path.exists(onnx_path):
            existing_files.append(onnx_path)

        # Safely delete existing files before validation
        for file_path in existing_files:
            if not _safe_delete_file(file_path):
                raise TensorRTCompilerError(f"Failed to delete existing file: {file_path}")

        # Clear validation cache after successful deletion
        if existing_files:
            self.validation_manager.clear_cache()

        # Input validation (after file cleanup)
        self._validate_compilation_inputs(pt_model_path, output_dir)

        # Validate the model file
        if not self._validate_yolo_model(pt_model_path):
            raise TensorRTCompilerError("Selected file is not a valid YOLO model.")

        model = None
        try:
            if progress_callback:
                try:
                    progress_callback("Loading YOLO model...")
                except Exception:
                    pass  # Don't fail compilation if callback fails

            model = YOLO(pt_model_path)
            
            if progress_callback:
                try:
                    progress_callback("Exporting to TensorRT .engine (this may take a while)...")
                except Exception:
                    pass

            # Check for stop flag before export
            if self._should_stop:
                raise TensorRTCompilerError("Compilation was stopped by user")

            # Run export in subprocess to allow interruption
            export_script = os.path.join(os.path.dirname(__file__), "tensorrt_export_engine_model.py")
            if not os.path.exists(export_script):
                raise TensorRTCompilerError("Export script not found")

            # Start subprocess and monitor output
            stdout, stderr = self._run_export_subprocess(export_script, pt_model_path, output_dir)
            
            if self._export_process.returncode != 0:
                raise TensorRTCompilerError(f"Export failed: {stderr}")

            try:
                # Try to find JSON in the output (may be mixed with other text)
                json_start = stdout.rfind('{')
                json_end = stdout.rfind('}') + 1
                if json_start >= 0 and json_end > json_start:
                    json_str = stdout[json_start:json_end]
                    result = json.loads(json_str)
                    if not result.get('success'):
                        raise TensorRTCompilerError(f"Export failed: {result.get('error', 'Unknown error')}")

                    # Move the created engine file to the expected location
                    created_engine = result.get('engine_file')
                    if created_engine and os.path.exists(created_engine):
                        if os.path.abspath(created_engine) != os.path.abspath(output_path):
                            shutil.move(created_engine, output_path)
                    else:
                        raise TensorRTCompilerError("Engine file was not created")
                else:
                    # No JSON found, check if output file exists anyway
                    if not os.path.exists(output_path):
                        raise TensorRTCompilerError("No valid result and engine file not found")
                    
            except json.JSONDecodeError:
                # JSON parsing failed, check if output file exists
                if not os.path.exists(output_path):
                    raise TensorRTCompilerError(f"Invalid export result format. Output: {stdout[:200]}...")
                # If file exists, continue (some export scripts don't return JSON)

            if not os.path.exists(output_path):
                raise TensorRTCompilerError(".engine file was not created.")

            if progress_callback:
                try:
                    progress_callback(f"Model compiled successfully! Output: {output_path}")
                except Exception:
                    pass

            return output_path
        except Exception as e:
            if isinstance(e, TensorRTCompilerError):
                raise
            raise TensorRTCompilerError(f"TensorRT compilation failed: {e}")
        finally:
            # Clean up model and GPU memory
            if model is not None:
                del model
                gc.collect()
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

    def _validate_compilation_inputs(self, pt_model_path: str, output_dir: str):
        """Validate compilation inputs and dependencies."""
        # Input validation
        if not pt_model_path or not output_dir:
            raise TensorRTCompilerError("Model path and output directory must be provided.")
        if not os.path.isfile(pt_model_path):
            raise TensorRTCompilerError(f"Model file not found: {pt_model_path}")
        if not pt_model_path.lower().endswith('.pt'):
            raise TensorRTCompilerError("Selected file is not a .pt model file.")
        if not os.path.isdir(output_dir):
            raise TensorRTCompilerError(f"Output directory does not exist: {output_dir}")

        # Check dependencies using ValidationManager
        validation_results = self.validation_manager.validate_all(pt_model_path, output_dir)
        failed_checks = [result for result in validation_results if not result.success]
        if failed_checks:
            # Find the first critical dependency failure
            for result in failed_checks:
                if result.name in ["PyTorch Installation", "CUDA Installation", "TensorRT Installation", "Ultralytics Installation", "CUDA Device"]:
                    raise TensorRTCompilerError(f"{result.name} check failed: {result.details}")
            # If no critical dependency failure, raise general validation error
            raise TensorRTCompilerError(f"Validation failed: {failed_checks[0].name} - {failed_checks[0].details}")

    def _run_export_subprocess(self, export_script: str, pt_model_path: str, output_dir: str) -> tuple:
        """
        Run the export subprocess and monitor output in real-time.
        
        Returns:
            tuple: (stdout, stderr) - Combined output from the subprocess
        """
        # Start subprocess with unbuffered output
        # Windows fix: prevent terminal windows from spawning
        import sys
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        self._export_process = subprocess.Popen(
            [sys.executable, "-u", export_script, pt_model_path, output_dir],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,  # Combine stderr into stdout
            text=True,
            bufsize=1,  # Line buffered
            universal_newlines=True,
            creationflags=creation_flags
        )

        # Monitor process and capture output in real-time
        stdout_lines = []

        # Clear previous output
        with self._output_lock:
            self._subprocess_output.clear()
        
        while self._export_process.poll() is None:
            if self._should_stop:
                # Terminate the subprocess
                self._export_process.terminate()
                try:
                    self._export_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._export_process.kill()
                raise TensorRTCompilerError("Compilation was stopped by user")
            
            # Read available output
            try:
                if self._export_process.stdout:
                    line = self._export_process.stdout.readline()
                    if line:
                        line_str = line.strip()
                        if line_str:  # Only add non-empty lines
                            stdout_lines.append(line_str)
                            # Color code based on content
                            if "ERROR" in line_str.upper() or "FAILED" in line_str.upper():
                                self._add_output_line(f"[ERR] {line_str}")
                            else:
                                self._add_output_line(f"[OUT] {line_str}")
            except:
                pass  # Continue if reading fails
            
            time.sleep(0.1)  # Check more frequently for output

        # Get any remaining output
        remaining_output, _ = self._export_process.communicate()
        if remaining_output and remaining_output.strip():
            for line in remaining_output.strip().split('\n'):
                if line.strip():
                    stdout_lines.append(line.strip())
                    if "ERROR" in line.upper() or "FAILED" in line.upper():
                        self._add_output_line(f"[ERR] {line.strip()}")
                    else:
                        self._add_output_line(f"[OUT] {line.strip()}")

        # Reconstruct full output
        stdout = '\n'.join(stdout_lines)
        stderr = ""  # No separate stderr since we combined it

        return stdout, stderr

    def _validate_yolo_model(self, model_path: str) -> bool:
        """Validate that the file is actually a YOLO model."""
        try:
            # Check if Ultralytics is available using ValidationManager
            ultralytics_check = self.validation_manager._check_ultralytics_installation()
            if not ultralytics_check.success:
                return False
            
            model = YOLO(model_path)
            # Try to access model info to verify it's valid
            _ = model.info
            return True
        except Exception:
            return False
    
    def stop_compilation(self):
        """Stop the current compilation process."""
        with self._compile_lock:
            self._should_stop = True
            # Terminate the export subprocess if it's running
            if self._export_process and self._export_process.poll() is None:
                try:
                    self._export_process.terminate()
                    self._export_process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._export_process.kill()
                except Exception:
                    pass  # Ignore errors during termination

    def get_subprocess_output(self) -> List[str]:
        """Get captured subprocess output lines."""
        with self._output_lock:
            return self._subprocess_output.copy()
    
    def _add_output_line(self, line: str):
        """Add a line to subprocess output (thread-safe)."""
        with self._output_lock:
            self._subprocess_output.append(line)
            # Keep only last 100 lines to prevent memory issues
            if len(self._subprocess_output) > 100:
                self._subprocess_output.pop(0)

# Global validation manager instance
validation_manager = ValidationManager()
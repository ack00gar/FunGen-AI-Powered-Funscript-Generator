import imgui
import time
from application.utils import tensorrt_compiler

# Color palette for validation display
_GREEN = (0.2, 0.8, 0.2, 1.0)  # Success color
_RED = (0.8, 0.2, 0.2, 1.0)    # Failure color
_GRAY = (0.7, 0.7, 0.7, 1.0)   # Neutral/unavailable color

class ValidationPanel:
    """Panel for displaying validation results."""
    def __init__(self):
        self.validation_results = []
        self.last_pt_path = ""
        self.last_output_dir = ""
        self.initial_validation_done = False
        self.last_file_check_time = 0.0
        self.file_check_interval = 2.0  # Check every 2 seconds
    
    def update_validation(self, pt_path: str, output_dir: str, is_compiling: bool = False):
        """Update validation results when parameters change or periodically for file checks."""
        current_time = time.time()
        
        # Full validation if parameters changed or initial run
        if (pt_path != self.last_pt_path or 
            output_dir != self.last_output_dir or 
            not self.initial_validation_done):
            
            self.validation_results = tensorrt_compiler.validation_manager.validate_all(pt_path, output_dir)
            self.last_pt_path = pt_path
            self.last_output_dir = output_dir
            self.initial_validation_done = True
            self.last_file_check_time = current_time
        
        # Periodic file status refresh when not compiling
        elif (not is_compiling and 
              current_time - self.last_file_check_time >= self.file_check_interval and
              pt_path and output_dir):
            
            self._refresh_file_status(pt_path, output_dir)
            self.last_file_check_time = current_time
    
    def render(self):
        """Render the validation panel."""
        imgui.begin_child("ValidationPanel", 0, 200, border=True)
        imgui.text("Validation Results")
        imgui.separator()
        
        if not self.validation_results:
            imgui.text_colored("No validation results available", *_GRAY)
        else:
            for result in self.validation_results:
                self._render_validation_item(result)
        
        imgui.end_child()
    
    def _render_validation_item(self, result):
        """Render a single validation item."""
        # Status indicator and color
        if result.success:
            imgui.text_colored("[OK]", *_GREEN)
            imgui.same_line()
            imgui.text(f"{result.name}:")  # Default white for check labels
        else:
            imgui.text_colored("[FAIL]", *_RED)
            imgui.same_line()
            imgui.text(f"{result.name}:")  # Default white for check labels
        
        imgui.same_line()
        
        # Version or details - convert to string if needed
        if result.version:
            version_str = str(result.version) if result.version else ""
            color = _GREEN if result.success else _RED
            imgui.text_colored(version_str, *color)
        elif result.details:
            details_str = str(result.details) if result.details else ""
            color = _GREEN if result.success else _RED
            imgui.text_colored(details_str, *color)
        else:
            if result.success:
                imgui.text_colored("OK", *_GREEN)
            else:
                imgui.text_colored("Failed", *_RED)
        
        # File path if available
        if result.file_path:
            imgui.text_colored(f"  Path: {result.file_path}", *_GRAY)
    
    def _refresh_file_status(self, pt_path: str, output_dir: str):
        """Refresh only the file status checks without full validation."""
        if not self.validation_results:
            return
        
        # Clear validation cache for file checks to force refresh
        vm = tensorrt_compiler.validation_manager
        cache_keys_to_clear = [
            f"onnx_exists_{pt_path}",
            f"engine_exists_{pt_path}"
        ]
        with vm._lock:
            for key in cache_keys_to_clear:
                vm._validation_cache.pop(key, None)
        
        # Get fresh file status checks
        onnx_result = vm._check_onnx_file_exists(pt_path)
        engine_result = vm._check_engine_file_exists(pt_path)
        
        # Update existing results
        for i, result in enumerate(self.validation_results):
            if result.name == "ONNX File Status":
                self.validation_results[i] = onnx_result
            elif result.name == "Engine File Status":
                self.validation_results[i] = engine_result

    def reset_state(self):
        """Reset the validation panel state."""
        self.last_pt_path = ""
        self.last_output_dir = ""
        self.initial_validation_done = False
        self.validation_results = []
        self.last_file_check_time = 0.0 
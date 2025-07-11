import imgui
import os
import threading
from application.utils import tensorrt_compiler
import logging

class TensorRTCompilerWindow:
    def __init__(self, app, on_close_callback=None):
        self.app = app
        self.on_close_callback = on_close_callback
        self.is_open = True
        self.selected_pt_path = ""
        self.selected_output_dir = ""
        self.status_message = ""
        self.is_compiling = False
        self.compile_thread = None
        self.model_instance = None
        self.logger = getattr(app, 'logger', logging.getLogger(__name__))

    def _set_status(self, msg):
        self.status_message = msg
        if self.logger:
            self.logger.info(msg)

    def _on_file_selected(self, path):
        self.logger.info(f"[TensorRTCompilerWindow] File dialog callback: {path}")
        self.selected_pt_path = path
        if path:
            self.selected_output_dir = os.path.dirname(path)

    def _on_folder_selected(self, path):
        self.logger.info(f"[TensorRTCompilerWindow] Folder dialog callback: {path}")
        if path:
            self.selected_output_dir = path

    def _progress_callback(self, msg):
        self.status_message = msg

    def _start_compile(self):
        if self.is_compiling:
            return
        self.is_compiling = True
        self.status_message = "Starting compilation..."
        def run():
            try:
                output_path = tensorrt_compiler.compile_yolo_to_tensorrt(
                    self.selected_pt_path,
                    self.selected_output_dir,
                    progress_callback=self._progress_callback)
                self.status_message = f"Success! Output: {output_path}"
            except Exception as e:
                self.status_message = f"Error: {e}"
            finally:
                self.is_compiling = False
                self._unload_model()
        self.compile_thread = threading.Thread(target=run, daemon=True)
        self.compile_thread.start()

    def _unload_model(self):
        if self.model_instance is not None:
            tensorrt_compiler.unload_yolo_model(self.model_instance)
            self.model_instance = None

    def _close_window(self):
        self._unload_model()
        self.is_open = False
        if self.on_close_callback:
            self.on_close_callback()

    def render(self):
        if not self.is_open:
            return
        imgui.set_next_window_size(600, 320, condition=imgui.ONCE)
        # Use the return value from imgui.begin to detect window close (X button)
        is_open, should_show = imgui.begin("YOLO to TensorRT Compiler", True)
        if not should_show:
            self._close_window()
            imgui.end()
            return

        # File selection
        imgui.text("Select YOLO .pt Model:")
        imgui.same_line()
        if imgui.button("Browse##SelectPTModel"):
            if hasattr(self.app, 'gui_instance') and self.app.gui_instance:
                self.app.gui_instance.file_dialog.show(
                    title="Select YOLO .pt Model",
                    is_save=False,
                    callback=self._on_file_selected,
                    extension_filter="YOLO Model Files (*.pt),*.pt|All Files,*.*",
                    initial_path=os.path.dirname(self.selected_pt_path) if self.selected_pt_path else None
                )
        imgui.same_line()
        imgui.text(self.selected_pt_path or "[No file selected]")

        # Output folder selection
        imgui.text("Output Folder:")
        imgui.same_line()
        if imgui.button("Browse##SelectOutputFolder"):
            if hasattr(self.app, 'gui_instance') and self.app.gui_instance:
                self.app.gui_instance.file_dialog.show(
                    title="Select Output Folder",
                    is_save=False,
                    is_folder_dialog=True,
                    callback=self._on_folder_selected,
                    initial_path=self.selected_output_dir or os.path.dirname(self.selected_pt_path) if self.selected_pt_path else None
                )
        imgui.same_line()
        imgui.text(self.selected_output_dir or "[No folder selected]")

        # Output filename preview
        if self.selected_pt_path:
            basename = os.path.splitext(os.path.basename(self.selected_pt_path))[0]
            output_path = os.path.join(self.selected_output_dir or "", basename + ".engine")
            imgui.text(f"Output File: {output_path}")

        # Status area
        if self.status_message:
            imgui.text_wrapped(self.status_message)

        # CUDA/TensorRT checks
        cuda_ok = tensorrt_compiler.is_cuda_available()
        tensorrt_ok = tensorrt_compiler.is_tensorrt_installed()
        valid_pt = self.selected_pt_path.lower().endswith('.pt') if self.selected_pt_path else False
        can_compile = cuda_ok and tensorrt_ok and valid_pt and bool(self.selected_output_dir) and not self.is_compiling

        if not cuda_ok:
            imgui.text_colored("No CUDA-capable device detected. This tool requires an NVIDIA GPU.", 1, 0.5, 0.5, 1)
        if not tensorrt_ok:
            imgui.text_colored("TensorRT Python package is not installed. Please install it to use this tool.", 1, 0.5, 0.5, 1)
        if self.selected_pt_path and not valid_pt:
            imgui.text_colored("Please select a valid YOLO .pt model file.", 1, 0.5, 0.5, 1)

        # Compile button
        if not can_compile:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)
        if imgui.button("Compile"):
            self._start_compile()
        if not can_compile:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()
        imgui.same_line()
        if imgui.button("Close"):
            self._close_window()

        imgui.end() 
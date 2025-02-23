import tkinter as tk
from tkinter import ttk

from script_generator.debug.logger import set_log_level
from script_generator.gui.utils.widgets import Widgets
from script_generator.state.app_state import AppState


class HelpPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.grid_columnconfigure(0, weight=1)

        models_section = Widgets.frame(self, title="Model Compatibility", main_section=True, row=1, inner_padding=(10, 0))
        models_text = (
            "We support multiple model formats across Windows, macOS, and Linux.\n\n"
            "- **.pt (PyTorch)**: Requires CUDA (for NVIDIA GPUs) or ROCm (for AMD GPUs) for acceleration.\n"
            "- **.onnx (ONNX Runtime)**: Best for CPU users as it offers broad compatibility and efficiency.\n"
            "- **.engine (TensorRT)**: For NVIDIA GPUs: Provides significant efficiency improvements.\n"
            "- **.mlmodel (Core ML)**: Optimized for macOS users. Runs efficiently on Apple devices with Core ML.\n\n"
            "In most cases, the app will automatically detect the best model available at launch, but if the right model wasn't present\nor the right dependencies weren't installed, you might need to override it."
        )
        Widgets.label(models_section, models_text, align="left", sticky="w")
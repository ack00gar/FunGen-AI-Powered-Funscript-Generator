import imgui
import logging
import os
import time
from application.gui_components.engine_compiler.tensorrt_validation_panel import ValidationPanel
from application.logic.tensorrt_compiler_logic import TensorRTCompilerLogic
from application.utils import primary_button_style, destructive_button_style
from config.constants import TENSORRT_OUTPUT_DISPLAY_HEIGHT
from config.element_group_colors import CompilerToolColors

class TensorRTCompilerWindow:
    """Main TensorRT compiler window - GUI only."""
    def __init__(self, app, on_close_callback=None):
        self.app = app
        self.on_close_callback = on_close_callback
        self.is_open = True
        self.logger = getattr(app, 'logger', logging.getLogger(__name__))
        
        # Initialize components
        self.validation_panel = ValidationPanel()
        self.logic = TensorRTCompilerLogic(logger=self.logger)
        
        # Button debounce
        self.last_button_time = 0.0
        self.button_debounce_ms = 250

    def _close_window(self):
        """Close the window."""
        self.logic.request_stop_compilation()
        self.is_open = False
        if self.on_close_callback:
            self.on_close_callback()

    def _reset_state(self):
        """Reset window state when reopened."""
        self.logic.reset_state()
        self.validation_panel.reset_state()

    def _on_file_selected(self, file_path: str):
        """Handle file selection callback."""
        self.logic.set_model_path(file_path)

    def _on_folder_selected(self, folder_path: str):
        """Handle folder selection callback."""
        self.logic.set_output_directory(folder_path)

    def _render_file_selection(self):
        """Render file selection section."""
        imgui.text("Select YOLO .pt Model:")
        imgui.same_line()
        if imgui.button("Browse##SelectPTModel"):
            if hasattr(self.app, 'gui_instance') and self.app.gui_instance:
                self.app.gui_instance.file_dialog.show(
                    title="Select YOLO .pt Model",
                    is_save=False,
                    callback=self._on_file_selected,
                    extension_filter="YOLO Model Files (*.pt),*.pt|All Files,*.*",
                    initial_path=os.path.dirname(self.logic.selected_pt_path) if self.logic.selected_pt_path else None
                )
        imgui.same_line()
        imgui.text(self.logic.selected_pt_path or "[No file selected]")

        imgui.text("Output Folder:")
        imgui.same_line()
        if imgui.button("Browse##SelectOutputFolder"):
            if hasattr(self.app, 'gui_instance') and self.app.gui_instance:
                self.app.gui_instance.file_dialog.show(
                    title="Select Output Folder",
                    is_save=False,
                    is_folder_dialog=True,
                    callback=self._on_folder_selected,
                    initial_path=self.logic.selected_output_dir or os.path.dirname(self.logic.selected_pt_path) if self.logic.selected_pt_path else None
                )
        imgui.same_line()
        imgui.text(self.logic.selected_output_dir or "[No folder selected]")

        # Output filename preview
        output_path = self.logic.get_output_path_preview()
        if output_path:
            imgui.text(f"Output File: {output_path}")

    def _render_validation_section(self):
        """Render validation section."""
        # Update validation with compilation status for periodic file checks
        is_compiling = self.logic.get_compilation_status()
        self.validation_panel.update_validation(
            self.logic.selected_pt_path, 
            self.logic.selected_output_dir,
            is_compiling
        )
        self.validation_panel.render()

    def _render_compilation_controls(self):
        """Render compilation controls."""
        # Get compilation state from logic
        can_compile = self.logic.can_compile()
        is_compiling = self.logic.get_compilation_status()
        
        # Check debounce
        current_time = time.time() * 1000  # Convert to milliseconds
        button_enabled = (current_time - self.last_button_time) >= self.button_debounce_ms

        # Compile/Stop button
        if is_compiling:
            # Show Stop button when compiling (DESTRUCTIVE - stops process)
            if not button_enabled:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            with destructive_button_style():
                if imgui.button("Stop Compilation", width=120):
                    if button_enabled:
                        self.logic.request_stop_compilation()
                        self.last_button_time = current_time

            if not button_enabled:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()
        else:
            # Show Compile button when not compiling (PRIMARY - positive action)
            disabled = not can_compile or not button_enabled
            if disabled:
                imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
                imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

            with primary_button_style():
                if imgui.button("Compile", width=120):
                    if can_compile and button_enabled:
                        self.logic.start_compilation()
                        self.last_button_time = current_time

            if disabled:
                imgui.pop_style_var()
                imgui.internal.pop_item_flag()
        
        imgui.same_line()
        if imgui.button("Close", width=80):
            self._close_window()

    def _render_status_section(self):
        """Render status section."""
        status_message = self.logic.get_status_message()
        if status_message:
            imgui.separator()
            self._render_colored_status_message(status_message)
        
        # Always show subprocess output
        self._render_subprocess_output()
    
    def _render_colored_status_message(self, message: str):
        """Render status message with appropriate colors."""
        if "Error" in message or "Failed" in message:
            imgui.text_colored(message, *CompilerToolColors.ERROR)
        elif "Success" in message or "complete" in message.lower():
            imgui.text_colored(message, *CompilerToolColors.SUCCESS)
        elif "Starting" in message or "Loading" in message or "Exporting" in message:
            imgui.text_colored(message, *CompilerToolColors.STARTING)
        elif "Stopping" in message:
            imgui.text_colored(message, *CompilerToolColors.STOPPING)
        else:
            imgui.text_wrapped(message)
    
    def _render_subprocess_output(self):
        """Render subprocess output display."""
        
        # Get subprocess output from the compiler
        output_lines = self.logic.compiler.get_subprocess_output()
        
        # Create scrollable child window for output
        imgui.begin_child("SubprocessOutput", 0, TENSORRT_OUTPUT_DISPLAY_HEIGHT, border=True)
        
        if not output_lines:
            imgui.text_colored("Waiting for compilation output...", *CompilerToolColors.INFO)
        else:
            for line in output_lines:
                if line.startswith("[ERR]"):
                    imgui.text_colored(line, *CompilerToolColors.ERROR)
                elif line.startswith("[OUT]"):
                    imgui.text_colored(line, *CompilerToolColors.OUTPUT_TEXT)
                else:
                    # Default for other lines
                    imgui.text(line)
        
        # Auto-scroll to bottom
        if imgui.get_scroll_y() >= imgui.get_scroll_max_y():
            imgui.set_scroll_here_y(1.0)
        
        imgui.end_child()
        
        # Copy button below the output box
        if imgui.button("Copy Output"):
            self.logic.copy_output_to_clipboard()

    def render(self):
        """Render the main window."""
        if not self.is_open:
            return
        
        # Set window size and position
        imgui.set_next_window_size(700, 500, condition=imgui.ONCE)
        
        # Begin window
        is_open, should_show = imgui.begin("YOLO to TensorRT Compiler", True)
        if not should_show:
            self._close_window()
            imgui.end()
            return

        # Main content
        self._render_file_selection()
        imgui.spacing()
        self._render_validation_section()
        imgui.spacing()
        self._render_compilation_controls()
        self._render_status_section()

        imgui.end() 
import tkinter as tk
from tkinter import ttk

from script_generator.debug.logger import set_log_level
from script_generator.state.app_state import AppState
from script_generator.gui.utils.widgets import Widgets


class SettingsPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        self.state: AppState = controller.state
        c = self.state.config_manager
        self.grid_columnconfigure(0, weight=1)

        general_settings = Widgets.frame(self, title="Funscripts", main_section=True, row=0)

        Widgets.checkbox(
            general_settings,
            "Copy funscript to movie dir",
            state=self.state,
            attr="copy_funscript_to_movie_dir",
            default_value=False,
            tooltip_text="If enabled, the generated funscript will be copied to the same directory as the movie.",
            command=lambda val: c.save(),
            row=0
        )

        Widgets.file_selection(
            attr="funscript_output_dir",
            parent=general_settings,
            label_text="Copy funscript to folder",
            button_text="Browse",
            file_selector_title="Select Funscript Destination",
            file_types=[("Folders", "*")],
            state=self.state,
            tooltip_text="Set an optional output folder where the final funscript will be copied to.",
            command=lambda val: c.save(),
            select_folder=True,
            row=1
        )

        ffmpeg_settings = Widgets.frame(self, title="YOLO (restart to apply changes)", main_section=True, row=1)

        Widgets.file_selection(
            attr="yolo_model_path",
            parent=ffmpeg_settings,
            label_text="YOLO model",
            button_text="Browse",
            file_selector_title="Select YOLO model",
            file_types=[("YOLO Model Files", "*.onnx"), ("YOLO Model Files", "*.pt"), ("CoreML Model Files", "*.mlmodel"), ("All Files", "*.*")],
            state=self.state,
            tooltip_text="Path to the FFmpeg executable.",
            command=lambda val: c.save(),
            row=0
        )

        ffmpeg_settings = Widgets.frame(self, title="FFmpeg", main_section=True, row=2)

        Widgets.file_selection(
            attr="ffmpeg_path",
            parent=ffmpeg_settings,
            label_text="FFmpeg Path",
            button_text="Browse",
            file_selector_title="Select FFmpeg Executable",
            file_types=[("Executable Files", "*.exe"), ("All Files", "*.*")],
            state=self.state,
            tooltip_text="Path to the FFmpeg executable.",
            command=lambda val: c.save(),
            row=0
        )

        Widgets.file_selection(
            attr="ffprobe_path",
            parent=ffmpeg_settings,
            label_text="FFprobe Path",
            button_text="Browse",
            file_selector_title="Select FFprobe Executable",
            file_types=[("Executable Files", "*.exe"), ("All Files", "*.*")],
            state=self.state,
            tooltip_text="Path to the FFprobe executable.",
            command=lambda val: c.save(),
            row=1
        )

        Widgets.input(
            attr="ffmpeg_hwaccel",
            parent=ffmpeg_settings,
            label_text="Hardware acceleration",
            state=self.state,
            tooltip_text="Used hardware acceleration. Can be overwritten here or set to None to disable it when your\nexperiencing issues. However if you do, please let us know so we can better\nsupport you in the future.",
            command=lambda val: c.save(),
            row=2
        )

        dev_settings = Widgets.frame(self, title="Dev", main_section=True, row=3)

        def handle_log_level(level):
            set_log_level(level)
            c.save()

        Widgets.dropdown(
            attr="log_level",
            parent=dev_settings,
            label_text="Log level",
            options=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default_value=self.state.log_level,
            tooltip_text="Set the log level",
            state=self.state,
            command=handle_log_level
        )
import tkinter as tk
from tkinter import ttk

from script_generator.debug.logger import set_log_level
from script_generator.gui.utils.widgets import Widgets
from script_generator.state.app_state import AppState


class HelpDebugVideoPage(tk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        self.grid_columnconfigure(0, weight=1)

        general_debug_section = Widgets.frame(self, title="Play Debug Video", main_section=True, row=0, inner_padding=(10, 0))
        general_debug_text = (
            "When pressing the play debug video button, a video player popup will appear, allowing you to review the funscript generation\nprocess.\n\n"

            "Debugging Options:\n"
            "  - Position Debugging, Tracking Debugging, and Region Debugging can be toggled on or off to show more or less overlay\ninformation.\n\n"

            "Frame: Allows skipping to a specific frame in the video. The video will pause until you press the seekbar of clear the input.\n\n"

            "Max FPS: Limits the maximum FPS for the preview, which helps with slow-motion playback to better analyze movements.\n\n"

            "Mode: Selects what type of overlay information is displayed.\n"
            "  - Funscript: Shows details related to funscript generation.\n"
            "  - Detection: Displays bounding boxes with their confidence scores.\n\n"

            "Box Labels: Controls which metrics appear above bounding boxes.\n\n"

            "These settings also apply when exporting a debug video. See 'Generate Sharable Debug Video' in 'Share Debug Files' for more\ndetails."
        )

        Widgets.label(general_debug_section, general_debug_text, align="left", sticky="w")

        general_settings = Widgets.frame(self, title="Seekbar overlay", main_section=True, row=1, inner_padding=(10, 0))
        controls_text = (
            "The seekbar has various overlays to indicate segments\n"
            "Light blue markers: This indicates a segment where the penis is in a specific location\n"
            "Red line: This indicates a hard cut in the video\n"
            "Colors above light blue markers: These represent the sex act that is happening at that time. and are color coded to the class\n"
            "colors belonging to the major box involved. For instance a pussy class is red so a vaginal sex segment will also be marked red.\n"
        )
        Widgets.label(general_settings, controls_text, align="left", sticky="w")

        general_settings = Widgets.frame(self, title="Controls", main_section=True, row=2, inner_padding=(10, 0))
        controls_text = (
            "q = Quit video\n"
            "space = Pause video\n"
            "-> = Skip to next position\n"
            "<- = Skip to previous position\n"
            ", = Seek backward 2 seconds\n"
            ". = Seek forward 2 seconds\n"
            "[ = Seek backward 1 frame\n"
            "] = Seek forward 1 frame\n"
        )
        Widgets.label(general_settings, controls_text, align="left", sticky="w")

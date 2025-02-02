from tkinter import messagebox

from script_generator.debug.logger import log
from script_generator.funscript.create_funscript import create_funscript


def regenerate_funscript(state):
    if not state.video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    log.info("Regenerating Funscript with tweaked settings...")

    create_funscript(state)
    log.info("Funscript re-generation complete.")
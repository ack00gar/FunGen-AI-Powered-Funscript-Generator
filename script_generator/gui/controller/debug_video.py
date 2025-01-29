import os
import threading
from tkinter import messagebox

from script_generator.debug.video_player.play import play_debug_video
from script_generator.object_detection.util.utils import get_metrics_file_info
from script_generator.state.app_state import AppState
from script_generator.utils.file import get_output_file_path
from script_generator.debug.logger import logger

def debug_video(state: AppState):
    def run():
        if not state.video_path:
            messagebox.showerror("Error", "Please select a video file.")
            return

        state.set_video_info()

        exists, path, filename = get_metrics_file_info(state)
        if exists:
            play_debug_video(state=state, start_frame=state.frame_start)
        else:
            logger.error(f"Debug logs file not found: {path}")
            messagebox.showinfo("Info", f"Debug logs file not found: {filename}")

    processing_thread = threading.Thread(target=run)
    processing_thread.start()
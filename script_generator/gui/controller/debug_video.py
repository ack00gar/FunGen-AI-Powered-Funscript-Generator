import os
from tkinter import messagebox

from script_generator.debug.video_player.play import play_debug_video
from script_generator.state.app_state import AppState
from script_generator.utils.file import get_output_file_path
from script_generator.utils.logger import logger

# TODO this callback is called on every frame. This doesn't seem correct or can be optized (instantiating Debugger, checks etc.)
def debug_video(state: AppState):
    if not state.video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    state.set_video_info()

    logs_path, log_filename = get_output_file_path(state.video_path, "_debug_logs.json")
    if os.path.exists(logs_path):
        play_debug_video(state=state, start_frame=state.frame_start)
    else:
        logger.error(f"Debug logs file not found: {logs_path}")
        messagebox.showinfo("Info", f"Debug logs file not found: {log_filename}")

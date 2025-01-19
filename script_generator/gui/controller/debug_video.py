import os
from tkinter import messagebox

from script_generator.state.app_state import AppState
from script_generator.utils.file import get_output_file_path
from script_generator.utils.logger import logger
from utils.lib_Debugger import Debugger


def debug_video(state: AppState):
    if not state.video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    log_file, _ = get_output_file_path(state.video_path, "_debug_logs.json")
    state.debugger = Debugger(state.video_path, state.video_info.is_vr, state.video_reader, log_file=log_file)  # Initialize the debugger

    # if the debug_logs.json file exists, load it
    logs_path, _ = get_output_file_path(state.video_path, "_debug_logs.json")
    if os.path.exists(logs_path):
        state.debugger.load_logs()

        state.debugger.play_video(
            start_frame=state.frame_start,
            duration=state.debug_record_duration if state.debug_record_mode else 0,
            record=state.debug_record_mode,
            downsize_ratio=2
        )
    else:
        logger.error(f"Debug logs file not found: {log_file}")
        messagebox.showinfo("Info", f"Debug logs file not found: {log_file}")

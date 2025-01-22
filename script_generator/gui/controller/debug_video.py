import os
from tkinter import messagebox

from script_generator.state.app_state import AppState
from script_generator.utils.file import get_output_file_path
from script_generator.utils.logger import logger
from utils.lib_Debugger import Debugger


# TODO this callback is called on every frame. This doesn't seem correct or can be optized (instantiating Debugger, checks etc.)
def debug_video(state: AppState):
    if not state.video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    state.set_video_info()

    logger.info(f"Debugging video: {state.video_path}")
    logger.info(f"Video Reader: {state.video_reader}")
    logger.info(f"Debug Mode: {state.save_debug_file}")
    logger.info(f"Live Preview Mode: {state.live_preview_mode}")
    logger.info(f"Frame Start: {state.frame_start}")
    logger.info(f"Frame End: {state.frame_end}")

    log_file, _ = get_output_file_path(state.video_path, "_debug_logs.json")
    state.debugger = Debugger(state.video_path, state.video_info.is_vr, state.video_reader, log_file=log_file)  # Initialize the debugger

    # if the debug_logs.json file exists, load it
    logs_path, _ = get_output_file_path(state.video_path, "_debug_logs.json")
    if os.path.exists(logs_path):
        state.debugger.load_logs()

        state.debugger.play_video(
            state=state,
            start_frame=state.frame_start,
            duration=state.debug_video_duration if state.save_debug_video else 0,
            save_debug_video=state.save_debug_video,
            downsize_ratio=1
        )
    else:
        logger.error(f"Debug logs file not found: {log_file}")
        messagebox.showinfo("Info", f"Debug logs file not found: {log_file}")

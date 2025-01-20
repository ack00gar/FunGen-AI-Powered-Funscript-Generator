import threading
from tkinter import messagebox

from script_generator.gui.controller.tracking_analysis import tracking_analysis
from script_generator.gui.messages.messages import ProgressMessage
from script_generator.object_detection.utils import check_skip_object_detection
from script_generator.scripts.analyze_video import analyze_video
from script_generator.utils.file import get_output_file_path
from script_generator.utils.logger import logger
from utils.lib_Debugger import Debugger


def video_analysis(state, root):
    if not state.video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    logger.info(f"Processing video: {state.video_path}")
    logger.info(f"Video Reader: {state.video_reader}")
    logger.info(f"Debug Mode: {state.save_debug_file}")
    logger.info(f"Live Preview Mode: {state.live_preview_mode}")
    logger.info(f"Frame Start: {state.frame_start}")
    logger.info(f"Frame End: {state.frame_end}")
    logger.info(f"Reference Script: {state.reference_script}")

    def run():
        # Initialize the debugger
        log_file, _ = get_output_file_path(state.video_path, "_debug_logs.json")
        state.debugger = Debugger(state.video_path, log_file=log_file)

        skip_detection = check_skip_object_detection(state, root)

        if skip_detection:
            state.set_video_info()
            if state.update_ui:
                state.update_ui(ProgressMessage(
                    process="OBJECT_DETECTION",
                    frames_processed=state.video_info.total_frames,
                    total_frames=state.video_info.total_frames,
                    eta="Done"
                ))
        else:
            analyze_video(state)


        tracking_analysis(state)

    processing_thread = threading.Thread(target=run)
    processing_thread.start()
    # processing_thread.join()
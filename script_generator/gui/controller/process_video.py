import threading
from tkinter import messagebox

from script_generator.scripts.tracking_analysis import tracking_analysis
from script_generator.gui.messages.messages import ProgressMessage
from script_generator.object_detection.utils import check_skip_object_detection
from script_generator.scripts.analyze_video import analyze_video
from script_generator.utils.helpers import to_int_or_none
from script_generator.debug.logger import logger


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

    state.frame_start = to_int_or_none(state.frame_start)
    state.frame_end = to_int_or_none(state.frame_end)

    def run():
        try:
            choice = check_skip_object_detection(state, root)

            if choice == "cancel":
                return
            elif choice == "use_existing":
                state.set_video_info()
                if state.update_ui:
                    state.update_ui(ProgressMessage(
                        process="OBJECT_DETECTION",
                        frames_processed=state.video_info.total_frames,
                        total_frames=state.video_info.total_frames,
                        eta="Done"
                    ))
            elif choice == "generate":
                analyze_video(state)

            tracking_analysis(state)

        except Exception as e:
            logger.error(f"Error during video analysis: {e}")
            messagebox.showerror("Error", f"Could not process video:\n{e}")
            import traceback
            traceback.print_exc()

    processing_thread = threading.Thread(target=run)
    processing_thread.start()

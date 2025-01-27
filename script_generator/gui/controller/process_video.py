import threading
from tkinter import messagebox

from script_generator.scripts.tracking_analysis import tracking_analysis
from script_generator.gui.messages.messages import ProgressMessage, UpdateGUIState
from script_generator.object_detection.utils import check_skip_object_detection
from script_generator.scripts.analyze_video import analyze_video
from script_generator.state.app_state import AppState, log_state_settings
from script_generator.utils.helpers import to_int_or_none
from script_generator.debug.logger import logger


def video_analysis(state: AppState, root):
    if not state.video_path:
        messagebox.showerror("Error", "Please select a video file.")
        return

    state.set_video_info()
    log_state_settings(state)

    state.frame_start = to_int_or_none(state.frame_start)
    state.frame_end = to_int_or_none(state.frame_end)

    def run():
        try:
            choice = check_skip_object_detection(state, root)

            if choice == "cancel":
                return
            elif choice == "use_existing":
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

            state.update_ui(UpdateGUIState(attr="is_processing", value=False))

        except Exception as e:
            logger.error(f"Error during video analysis: {e}")
            messagebox.showerror("Error", f"Could not process video:\n{e}")
            import traceback
            traceback.print_exc()

    processing_thread = threading.Thread(target=run)
    processing_thread.start()

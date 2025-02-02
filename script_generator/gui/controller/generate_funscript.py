import threading
from tkinter import messagebox

from script_generator.scripts.tracking_analysis import tracking_analysis
from script_generator.gui.messages.messages import ProgressMessage, UpdateGUIState
from script_generator.object_detection.util.object_detection import check_skip_object_detection
from script_generator.scripts.analyze_video import analyze_video
from script_generator.state.app_state import AppState, log_state_settings
from script_generator.utils.helpers import to_int_or_none
from script_generator.debug.logger import log


def generate_funscript(state: AppState, root):
    configured, msg = state.is_configured()
    if not configured:
        log.warn(msg)
        messagebox.showerror("Error", msg)
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

            state.update_ui(ProgressMessage(
                process="TRACKING_ANALYSIS",
                frames_processed=0,
                total_frames=0,
                eta="Queued"
            ))

            if choice == "use_existing":
                if state.update_ui:
                    state.update_ui(ProgressMessage(
                        process="OBJECT_DETECTION",
                        frames_processed=state.video_info.total_frames,
                        total_frames=state.video_info.total_frames,
                        eta="Done"
                    ))
            elif choice == "generate":
                state.update_ui(ProgressMessage(
                    process="OBJECT_DETECTION",
                    frames_processed=0,
                    total_frames=0,
                    eta="Calculating..."
                ))
                analyze_video(state)

            if state.analyze_task and state.analyze_task.is_stopped:
                return

            state.update_ui(UpdateGUIState(attr="has_raw_yolo", value=True))

            tracking_analysis(state)

            state.update_ui(UpdateGUIState(attr="has_tracking_data", value=True))
            state.update_ui(UpdateGUIState(attr="is_processing", value=False))
            state.analyze_task = None

        except Exception as e:
            state.analyze_task = None
            state.update_ui(UpdateGUIState(attr="is_processing", value=False))
            log.error(f"Error during video analysis: {e}")
            messagebox.showerror("Error", f"Could not process video:\n{e}")
            import traceback
            traceback.print_exc()

    processing_thread = threading.Thread(target=run)
    processing_thread.start()

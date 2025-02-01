from script_generator.object_detection.utils import get_raw_yolo_file_info
from script_generator.scripts.analyze_video import analyze_video
from script_generator.scripts.tracking_analysis import tracking_analysis
from script_generator.state.app_state import AppState, log_state_settings
from script_generator.debug.logger import log
from script_generator.utils.helpers import to_int_or_none

def generate_funscript(state: AppState):
    try:
        configured, msg = state.is_configured()
        if not configured:
            log.warn(msg)
            return

        state.set_video_info()
        log_state_settings(state)

        state.frame_start = to_int_or_none(state.frame_start)
        state.frame_end = to_int_or_none(state.frame_end)

        # analyze video if required
        if not state.use_existing_raw_yolo or not get_raw_yolo_file_info(state):
            analyze_video(state)

        tracking_analysis(state)
    except Exception as e:
        log.error(f"Error during video analysis: {e}")
        import traceback
        traceback.print_exc()
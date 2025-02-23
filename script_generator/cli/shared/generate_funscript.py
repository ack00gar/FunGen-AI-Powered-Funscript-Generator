from script_generator.debug.logger import log
from script_generator.object_detection.util.data import load_yolo_data
from script_generator.scripts.analyze_video import analyze_video
from script_generator.state.app_state import AppState, log_state_settings
from script_generator.utils.helpers import to_int_or_none


def generate_funscript_cli(state: AppState):
    try:
        configured, msg = state.is_configured()
        if not configured:
            log.warn(msg)
            return

        state.set_video_info()
        log_state_settings(state)

        state.frame_start = to_int_or_none(state.frame_start)
        state.frame_end = to_int_or_none(state.frame_end)

        exists, yolo_data, _, _ = load_yolo_data(state)

        # analyze video if required
        if not state.use_existing_raw_yolo or not exists:
            analyze_video(state)
        else:
            log.info("Skipping yolo analysis as it was already generated")

        generate_funscript(state)
    except Exception as e:
        log.error(f"Error during video analysis: {e}")
        import traceback
        traceback.print_exc()
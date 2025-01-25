from script_generator.scripts.analyze_video import analyze_video
from script_generator.state.app_state import AppState
from script_generator.utils.logger import logger


def generate_funscript(state: AppState):
    try:
        results = analyze_video(state)
    except Exception as e:
        logger.error(f"Could not generate funscript: {e}")

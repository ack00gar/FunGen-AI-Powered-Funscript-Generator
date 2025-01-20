from script_generator.scripts.analyze_video import analyse_video
from script_generator.state.app_state import AppState


def generate_funscript(state: AppState):
    restults = analyse_video(state)
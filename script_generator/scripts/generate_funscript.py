from script_generator.scripts.analyse_video import analyse_video
from script_generator.state.app_state import AppState


def generate_funscript(state: AppState):
    restults = analyse_video(state)
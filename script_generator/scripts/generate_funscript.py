from script_generator.scripts.analyze_video import analyze_video
from script_generator.state.app_state import AppState


def generate_funscript(state: AppState):
    results = analyze_video(state)
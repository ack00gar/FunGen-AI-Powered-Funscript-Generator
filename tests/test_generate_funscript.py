import os

from script_generator.cli.shared.generate_funscript import generate_funscript
from script_generator.state.app_state import AppState

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app_state = AppState()
app_state.video_path = "C:/cvr/funscript-generator/test_medium.mp4"
app_state.video_reader = "FFmpeg"

generate_funscript(app_state)

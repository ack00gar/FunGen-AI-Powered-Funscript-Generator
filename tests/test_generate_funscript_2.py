import os

from script_generator.scripts.generate_funscript import generate_funscript
from script_generator.state.app_state import AppState

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

app_state = AppState(is_cli=True)
app_state.video_path = "C:/cvr/funscript-generator/test.mp4"
app_state.frame_start = 60*60*10
generate_funscript(app_state)

from script_generator.gui.utils.widgets import Widgets
from script_generator.scripts.generate_debug_video import generate_debug_video
from script_generator.state.app_state import AppState
from script_generator.utils.helpers import to_int_or_fallback


class RenderDebugVideoState:
    def __init__(self, video_info):
        self.video_info = video_info
        self.frame_start = 0
        self.frame_end = None
        self.ui_callback = None

def render_debug_video_popup(window, state: AppState):
    state.set_video_info()
    debug_state = RenderDebugVideoState(state.video_info)
    Widgets.label(window, "First stream/play your results locally (see Play debug video). Note the frame when the issue occurs. \nSubtract about 120 frames so we have some context. Then add a few seconds of recording (2 to 20 seconds\nshould do it)", column=0, sticky="w", padx=10, pady=10)
    p_container, p, p_label, p_perc = Widgets.labeled_progress(window, "Progress", row=1)
    Widgets.frames_input(window, "Start", state=debug_state, attr="frame_start", tooltip_text="Where to start processing the video. If you have a 60fps video starting at 10s would mean frame 600", row=2)
    Widgets.frames_input(window, "End", state=debug_state, attr="frame_end", tooltip_text="Where to end processing the video. If you have a 60fps video stopping  at 10s would mean frame 600", row=3)
    Widgets.button(
        window,
        "Generate",
        lambda: generate_debug_video(state, to_int_or_fallback(debug_state.frame_start, 0), to_int_or_fallback(debug_state.frame_end, state.video_info.total_frames)),
        row=4,
        column=0,
        tooltip_text="Render a debug video once funscript processing is complete that can be\neasily shared on Discord for showcasing issues or areas of improvement."
    )

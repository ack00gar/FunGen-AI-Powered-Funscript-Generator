from script_generator.gui.utils.widgets import Widgets
from script_generator.state.app_state import AppState
from script_generator.video.data_classes.video_info import VideoInfo


class EditVideoState:
    def __init__(self, video_info: VideoInfo):
        self.is_vr = video_info.is_vr
        self.is_fisheye = video_info.is_fisheye
        self.fov = video_info.fov

def render_video_edit_popup(window, state: AppState, on_update):
    state.set_video_info()
    edit_video_state = EditVideoState(state.video_info)
    def update():
        video_info = state.video_info
        video_info.is_vr = edit_video_state.is_vr
        video_info.is_fisheye = edit_video_state.is_fisheye
        video_info.fov = edit_video_state.fov
        on_update()

    Widgets.checkbox(window, "Is 3D", edit_video_state, "is_vr", row=2, command=lambda val: update())

    Widgets.checkbox(window, "Fish eye", edit_video_state, "is_fisheye", row=3, command=lambda val: update())
    Widgets.input(
        window,
        "FOV",
        state=edit_video_state,
        attr="fov",
        row=4,
        tooltip_text="The FOV of the VR video. Usually 180 but various formats can have\na different FOV. For instance MKX200 has 200 degrees.",
        width=200,
        command=lambda val: update()
    )


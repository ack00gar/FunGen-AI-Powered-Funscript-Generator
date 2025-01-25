import string
from typing import Literal

from script_generator.debug.debug_data import DebugData
from script_generator.tasks.tasks import AnalyzeVideoTask
from script_generator.utils.helpers import is_mac
from script_generator.video.ffmpeg.hwaccel import get_preferred_hwaccel
from script_generator.video.info.video_info import VideoInfo, get_video_info


class AppState:
    def __init__(self, is_cli):
        self.is_cli: bool = is_cli

        # Gui/settings general
        self.video_path: string = None
        self.frame_start: int = 0
        self.frame_end: int | None = None
        self.video_reader: Literal["FFmpeg", "FFmpeg + OpenGL (Windows)"] = "FFmpeg" if is_mac() else "FFmpeg + OpenGL (Windows)"

        # Gui/settings debug
        self.save_debug_file: bool = True
        # TODO REMOVE
        # self.save_debug_video: bool = False
        self.live_preview_mode: bool = False
        self.reference_script: string = None

        # Gui/settings Funscript Tweaking Variables
        self.boost_enabled: bool = True
        self.boost_up_percent: int = 10
        self.boost_down_percent: int = 15
        self.threshold_enabled: bool = True
        self.threshold_low: int = 10
        self.threshold_high: int = 90
        self.vw_simplification_enabled: bool = True
        self.vw_factor: float = 8.0
        self.rounding: int = 5

        # TODO move this to a batch task class (so parallel inference is possible)
        self.funscript_data = []
        self.funscript_frames = []
        self.funscript_distances = []
        self.offset_x: int = 0
        self.frame_start_track = 0
        self.current_frame_id = 0
        self.frame_area = 0

        # State
        self.video_info: VideoInfo | None = None
        self.analyze_task: AnalyzeVideoTask | None = None

        # App logic
        self.debug_data = DebugData(self)
        self.update_ui = None
        self.ffmpeg_hwaccel = get_preferred_hwaccel()

    def set_video_info(self):
        if self.video_info is None:
            self.video_info = get_video_info(self.video_path)


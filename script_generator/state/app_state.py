import string
from typing import Literal

from script_generator.tasks.tasks import AnalyseVideoTask
from script_generator.utils.helpers import is_mac
from script_generator.video.ffmpeg import is_cuda_supported, get_video_info
from script_generator.video.video_info import VideoInfo


class AppState:
    def __init__(self, is_cli):
        self.is_cli: bool = is_cli

        # Job
        self.video_info: VideoInfo | None = None
        self.analyse_task: AnalyseVideoTask | None = None
        self.video_path: string = None
        self.frame_start: int = 0
        self.frame_end: int | None = None

        # Detection & decoding
        self.video_reader: Literal["FFmpeg", "FFmpeg + OpenGL (Windows)"] = "FFmpeg" if is_mac() else "FFmpeg + OpenGL (Windows)"

        # Debug
        self.debug_mode: bool = False
        self.life_display_mode: bool = False
        self.reference_script: string = None
        self.save_debug_as_video: bool = False
        self.debug_record_mode: bool = False
        self.debug_record_duration: int = 0

        # TODO what's this
        self.funscript_data = []
        self.funscript_frames = []
        self.funscript_distances = []
        self.offset_x: int = 0

        # Funscript Tweaking Variables
        self.boost_enabled: bool = True
        self.boost_up_percent: int = 10
        self.boost_down_percent: int = 15
        self.threshold_enabled: bool = True
        self.threshold_low: int = 10
        self.threshold_high: int = 90
        self.vw_simplification_enabled: bool = True
        self.vw_factor: float = 8.0
        self.rounding: int = 5

        # App logic
        self.debugger = None
        self.update_ui = None
        self.ffmpeg_cuda_supported = is_cuda_supported()

    def set_video_info(self):
        if self.video_info is None:
            self.video_info = get_video_info(self.video_path)


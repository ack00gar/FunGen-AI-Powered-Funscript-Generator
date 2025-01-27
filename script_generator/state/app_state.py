import os.path
import string
from typing import Literal

from ultralytics import YOLO

from script_generator.constants import MODEL_PATH
from script_generator.debug.debug_data import DebugData
from script_generator.debug.logger import logger
from script_generator.object_detection.util.utils import get_metrics_file_info
from script_generator.object_detection.utils import get_raw_yolo_file_info
from script_generator.tasks.tasks import AnalyzeVideoTask
from script_generator.utils.file import get_output_file_path
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
        self.video_reader: Literal["FFmpeg", "FFmpeg + OpenGL (Windows)"] = "FFmpeg" # if is_mac() else "FFmpeg + OpenGL (Windows)"
        self.copy_funscript_to_movie_dir = True

        # Gui/settings debug
        self.save_debug_file: bool = True
        self.live_preview_mode: bool = False
        self.reference_script: string = None
        self.max_preview_fps = 60

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

        # TODO move this to a analyse results class
        self.funscript_data = []
        self.funscript_frames = []
        self.funscript_distances = []
        self.offset_x: int = 0
        self.frame_start_track = 0
        self.current_frame_id = 0
        self.frame_area = 0

        # Cli
        self.use_existing_raw_yolo = False

        # State
        self.video_info: VideoInfo | None = None
        self.analyze_task: AnalyzeVideoTask | None = None
        self.has_raw_yolo = False
        self.has_tracking_data = False
        self.is_processing = False

        # App logic
        self.debug_data = DebugData(self)
        self.update_ui = None
        self.ffmpeg_hwaccel = get_preferred_hwaccel()
        self.yolo_model = YOLO(MODEL_PATH, task="detect")

    def set_video_info(self):
        # If movie changed
        if self.video_info is None or self.video_info.path != self.video_path:
            if not self.video_path:
                self.video_info = None
                self.has_raw_yolo = False
                self.has_tracking_data = False
                self.max_preview_fps = 60
            else:
                self.video_info = get_video_info(self.video_path)
                self.has_raw_yolo, _, _ = get_raw_yolo_file_info(self.video_path)
                self.has_tracking_data, _, _ = get_metrics_file_info(self.video_path)
                self.max_preview_fps = int(self.video_info.fps)


def log_state_settings(state: AppState):
    settings = [
        ("Processing video", state.video_path),
        ("Video Reader", state.video_reader),
        ("Debug Mode", state.save_debug_file),
        ("Live Preview Mode", state.live_preview_mode),
        ("Frame Start", state.frame_start),
        ("Frame End", state.frame_end),
        ("Reference Script", state.reference_script),
    ]

    for setting_name, setting_value in settings:
        logger.info(f"{setting_name}: {setting_value}")

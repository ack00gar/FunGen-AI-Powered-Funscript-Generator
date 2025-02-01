import math
import os.path
import string
from tkinter import messagebox
from typing import Literal, Optional, TYPE_CHECKING

from script_generator.config.config_manager import ConfigManager
from script_generator.debug.debug_data import DebugData
from script_generator.debug.logger import log
from script_generator.object_detection.util.utils import get_metrics_file_info, load_yolo_model
from script_generator.object_detection.utils import get_raw_yolo_file_info
from script_generator.video.info.video_info import VideoInfo, get_video_info

if TYPE_CHECKING:
    from script_generator.tasks.analyze_video_task import AnalyzeVideoTask

class AppState:
    _instance: Optional["AppState"] = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AppState, cls).__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.is_cli = True
        self.config_manager = ConfigManager(self)
        c = self.config_manager

        # Gui/settings general
        self.video_path: string = None
        self.frame_start: int = 0
        self.frame_end: int | None = None
        self.video_reader: Literal["FFmpeg", "FFmpeg + OpenGL (Windows)"] = "FFmpeg" # if is_mac() else "FFmpeg + OpenGL (Windows)"
        self.copy_funscript_to_movie_dir = True
        self.copy_funscript_to_movie_dir = c.get("copy_funscript_to_movie_dir")
        self.funscript_output_dir = c.get("funscript_output_dir")
        self.make_funscript_backup = c.get("make_funscript_backup")
        self.ffmpeg_path = c.get("ffmpeg_path")
        self.ffprobe_path = c.get("ffprobe_path")
        self.yolo_model_path = c.get("yolo_model_path")

        # Gui/settings debug
        self.log_level = c.get("log_level")
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
        self.ffmpeg_hwaccel = c.get("ffmpeg_hwaccel")
        self.yolo_model = load_yolo_model(self.yolo_model_path)

    def set_is_cli(self, cli):
        self.is_cli = cli

    def is_configured(self):
        message_prefix = "Cannot process the video."
        checks = [
            (self.video_path, f"{message_prefix} Please select a valid video file."),
            (self.ffprobe_path, f"{message_prefix} FFprobe is missing. Please provide the correct path."),
            (self.ffmpeg_path, f"{message_prefix} FFMPEG is missing. Please provide the correct path."),
            (self.yolo_model, f"{message_prefix} YOLO model is not loaded. Please make sure to download the YOLO model to the models directory."),
        ]

        for path, error_message in checks:
            if not path:
                return False, error_message

        return True, None

    def set_video_info(self):
        # If movie changed
        if self.video_info is None or self.video_info.path != self.video_path:
            if not self.video_path:
                self.video_info = None
                self.has_raw_yolo = False
                self.has_tracking_data = False
                self.max_preview_fps = 60
            else:
                if os.path.exists(self.video_path) and not os.path.isdir(self.video_path):
                    try:
                        self.video_info = get_video_info(self.video_path)
                        self.has_raw_yolo, _, _ = get_raw_yolo_file_info(self)
                        self.has_tracking_data, _, _ = get_metrics_file_info(self)
                        self.max_preview_fps = math.ceil(self.video_info.fps)
                    except Exception as e:
                        if not self.is_cli:
                            messagebox.showerror("Error", e)
                    finally:
                        return

                self.video_path = None
                self.set_video_info()

    def reload_video_info(self):
        self.video_info = None
        self.set_video_info()


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
        log.info(f"{setting_name}: {setting_value}")


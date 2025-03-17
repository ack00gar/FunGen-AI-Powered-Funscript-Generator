import math
import os.path
import string
from tkinter import messagebox
from typing import Literal, Optional, TYPE_CHECKING

from script_generator.config.config_manager import ConfigManager
from script_generator.debug.debug_data import DebugData, get_metrics_file_info
from script_generator.debug.logger import log
from script_generator.funscript.util.check_existing_funscript import check_existing_funscript
from script_generator.object_detection.util.data import load_yolo_model, get_raw_yolo_file_info
from script_generator.video.data_classes.video_info import VideoInfo, get_video_info

if TYPE_CHECKING:
    from script_generator.tasks.data_classes.analyze_video_task import AnalyzeVideoTask

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
        self.root = None

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
        self.tracking_logic_version = c.get("tracking_logic_version")
        self.save_debug_file: bool = True
        self.live_preview_mode: bool = False
        self.reference_script: string = None
        self.max_preview_fps = 60
        self.static_debug_frame = None
        self.debug_mode: Literal["funscript", "detection"] = "funscript"
        self.debug_positions: bool = True
        self.debug_tracking: bool = True
        self.debug_regions: bool = True
        self.debug_outliers: bool = True

        # Gui/settings Funscript Tweaking Variables
        self.boost_enabled: bool = True
        self.boost_up_percent: int = 15
        self.boost_down_percent: int = 15
        self.threshold_enabled: bool = True
        self.threshold_low: int = 10
        self.threshold_high: int = 90
        self.vw_simplification_enabled: bool = True
        self.vw_factor: float = 4.0
        self.rounding: int = 5

        # TODO remove when swtiched to tracking logic v2
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

    def set_root(self, root):
        self.root = root

    def load_yolo(self):
        if self.yolo_model_path and not self.yolo_model:
            self.yolo_model = load_yolo_model(self.yolo_model_path)

    def is_configured(self):
        message_prefix = "Cannot process the video."
        checks = [
            (self.video_path, f"{message_prefix} Please select a valid video file."),
            (self.ffprobe_path, f"{message_prefix} FFprobe is missing. Please provide the correct path."),
            (self.ffmpeg_path, f"{message_prefix} FFMPEG is missing. Please provide the correct path."),
            (self.yolo_model_path, f"{message_prefix} YOLO model path not set. Please make sure to download the YOLO model to the models directory and that the path under settings is correct."),
            (self.yolo_model, f"{message_prefix} YOLO model is not loaded. Please make sure to download the YOLO model to the models directory."),
        ]

        for path, error_message in checks:
            if not path:
                return False, error_message

        return True, None

    def set_video_info(self):
        # If movie changed
        if self.video_info is None or self.video_info.path != self.video_path:

            # TODO move this to a task an remove this here
            self.funscript_data = []
            self.funscript_frames = []
            self.funscript_distances = []
            self.offset_x: int = 0
            self.frame_start_track = 0
            self.current_frame_id = 0
            self.frame_area = 0
            self.debug_data = DebugData(self)

            if not self.video_path:
                self.video_info = None
                self.has_raw_yolo = False
                self.has_tracking_data = False
                self.max_preview_fps = 60
                self.reference_script = None
            else:
                if os.path.exists(self.video_path) and not os.path.isdir(self.video_path):
                    try:
                        self.video_info = get_video_info(self.video_path)
                        self.has_raw_yolo, _, _ = get_raw_yolo_file_info(self)
                        self.has_tracking_data, _, _ = get_metrics_file_info(self)
                        self.max_preview_fps = math.ceil(self.video_info.fps)

                        # Auto load reference script when available
                        # TODO make a generic function
                        video_folder = os.path.dirname(self.video_path)
                        filename_base = os.path.splitext(os.path.basename(self.video_path))[0]
                        ref_funscript = os.path.join(video_folder, f"{filename_base}.funscript")
                        file_exists, is_ours, _, out_of_date = check_existing_funscript(ref_funscript, filename_base, False)
                        if file_exists and not is_ours:
                            self.reference_script = ref_funscript
                    except Exception as e:
                        if not self.is_cli:
                            messagebox.showerror("Error", str(e))
                    finally:
                        return

                self.video_path = None
                self.set_video_info()

    def reload_video_info(self):
        self.video_info = None
        self.set_video_info()


def log_state_settings(state: AppState):
    log.info(f"Processing video: {state.video_path}")
    state.video_info.log_stats()

    if state.video_info.bit_depth > 8:
        log.info(f"Note: This video has a bit depth of {state.video_info.bit_depth}, which limits hardware acceleration and may result in processing times up to 8x slower.")

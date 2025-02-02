import datetime
import json
import os
from dataclasses import dataclass, asdict
from typing import TYPE_CHECKING

from script_generator.constants import OBJECT_DETECTION_VERSION, TRACKING_VERSION, FUNSCRIPT_VERSION
from script_generator.utils.file import get_output_file_path
from script_generator.utils.json_utils import get_data_file_info
from script_generator.video.data_classes.video_info import VideoInfo

if TYPE_CHECKING:
    from script_generator.state.app_state import AppState


@dataclass
class MetaData:
    video_info: VideoInfo
    created_date: str
    updated_date: str
    yolo_model: str | None
    raw_yolo_version: str | None
    raw_yolo_date: str | None
    metrics_version: str | None
    metrics_date: str | None
    funscript_date: str | None
    tracking_version: str | None
    funscript_version: str | None

    def to_json(self) -> str:
        data = asdict(self)
        return json.dumps(data, indent=4)

    def finish_analyze_video(self, state: "AppState"):
        now_str = datetime.datetime.now().isoformat()
        self.yolo_model = os.path.basename(state.yolo_model_path)
        self.raw_yolo_date = self.updated_date = now_str
        self.raw_yolo_version = OBJECT_DETECTION_VERSION
        MetaData._write_meta(state, self)

    def finish_tracking_analysis(self, state: "AppState"):
        now_str = datetime.datetime.now().isoformat()
        self.metrics_date = self.funscript_date = self.updated_date = now_str
        self.metrics_version = TRACKING_VERSION
        self.funscript_version = FUNSCRIPT_VERSION
        MetaData._write_meta(state, self)

    @classmethod
    def from_json(cls, json_str: str) -> "MetaData":
        data = json.loads(json_str)

        # Re-created video info class
        video_info_data = data.pop("video_info")
        projection = video_info_data.pop("projection", None)
        fov = video_info_data.pop("fov", None)
        is_fisheye = video_info_data.pop("is_fisheye", None)
        video_info_obj = VideoInfo(**video_info_data)
        video_info_obj.projection = projection
        video_info_obj.fov = fov
        video_info_obj.is_fisheye = is_fisheye

        return cls(video_info=video_info_obj, **data)

    @staticmethod
    def get_create_meta(state: "AppState") -> "MetaData":
        exists, file_path, _ = get_data_file_info(state.video_path, ".json", "metadata")
        if exists:
            return MetaData.load_meta(file_path)
        else:
            now_str = datetime.datetime.now().isoformat()
            meta = MetaData(
                video_info=state.video_info,
                created_date=now_str,
                updated_date=now_str,
                raw_yolo_date=None,
                metrics_version=None,
                metrics_date=None,
                yolo_model=None,
                raw_yolo_version=None,
                tracking_version=None,
                funscript_version=None,
                funscript_date=None
            )
            MetaData._write_meta(state, meta)

            bat_path, _ = get_output_file_path(state.video_path, ".bat", "Restore GUI")
            MetaData._create_bat_file(bat_path)

            return meta

    @staticmethod
    def load_meta(file_path: str) -> "MetaData":
        with open(file_path, 'r', encoding='utf-8') as f:
            json_str = f.read()
        return MetaData.from_json(json_str)

    @staticmethod
    def _write_meta(state: "AppState", meta: "MetaData"):
        file_path, _ = get_output_file_path(state.video_path, ".json", "metadata")
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(meta.to_json())

    @staticmethod
    def _create_bat_file(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)

        bat_content = """@echo off
cd /d "%~dp0"
set "SCRIPT_DIR=%CD%"
set "CONDA_PATH=C:\\Users\\%USERNAME%\\miniconda3\\Scripts\\activate.bat"

if not exist "%CONDA_PATH%" (
    echo Conda not found at %CONDA_PATH%. Please check your installation.
    pause
    exit /b 1
)

call "%CONDA_PATH%" VRFunAIGen

cd ..
cd ..
python -m script_generator.cli.open_gui_from_meta "%SCRIPT_DIR%\\metadata.json"

pause
"""

        with open(path, "w", encoding="utf-8") as bat_file:
            bat_file.write(bat_content)
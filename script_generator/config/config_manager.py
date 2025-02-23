import json
import os

from script_generator.constants import CONFIG_FILE_PATH, CONFIG_VERSION, DEFAULT_CONFIG
from script_generator.debug.logger import log, set_log_level
from script_generator.object_detection.util.data import get_yolo_model_path
from script_generator.video.ffmpeg.hwaccel import get_preferred_hwaccel
from script_generator.video.util.ffmpeg import get_ffmpeg_paths


class ConfigManager:
    def __init__(self, app_state):
        self.app_state = app_state
        self.config = self.load()

    def load(self):
        if os.path.exists(CONFIG_FILE_PATH):
            try:
                with open(CONFIG_FILE_PATH, "r") as file:
                    self.config = json.load(file)

                self._ensure_defaults(self.config)
                set_log_level(self.config["log_level"])

                return self.config
            except (json.JSONDecodeError, IOError):
                log.warning("Config file is corrupted. Resetting to defaults.")

        return self._initialize_defaults()

    def save(self):
        self.config["config_version"] = CONFIG_VERSION

        # Sync settings from AppState
        for key in DEFAULT_CONFIG.keys():
            if hasattr(self.app_state, key):
                self.config[key] = getattr(self.app_state, key)

        with open(CONFIG_FILE_PATH, "w") as file:
            json.dump(self.config, file, indent=4)

    def get(self, key, default=None):
        return self.config.get(key, default)

    def _initialize_defaults(self):
        self.config = DEFAULT_CONFIG.copy()
        self._ensure_defaults(self.config)
        self.save()
        return self.config

    def _ensure_defaults(self, config):
        # Add any missing keys from DEFAULT_CONFIG with their default values.
        for key, default_value in DEFAULT_CONFIG.items():
            if key not in config:
                config[key] = default_value

        # additional checks to verify that certain keys have valid values,
        checks = [
            ("ffmpeg_path", lambda: get_ffmpeg_paths()[0], self._is_valid_path),
            ("ffprobe_path", lambda: get_ffmpeg_paths()[1], self._is_valid_path),
            ("yolo_model_path", get_yolo_model_path, self._is_valid_path),
            ("ffmpeg_hwaccel", lambda: get_preferred_hwaccel(config["ffmpeg_path"]) if config["ffmpeg_path"] else None, lambda val: val is not None)
        ]

        updated = False

        for key, get_default, check_func in checks:
            if not check_func(config.get(key)):
                config[key] = get_default()
                updated = True

        if updated:
            self.save()

    def _is_valid_path(self, value):
        return isinstance(value, str) and value.strip() and os.path.exists(os.path.abspath(value))

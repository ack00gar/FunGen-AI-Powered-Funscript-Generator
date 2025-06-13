import json
import os
import logging
from typing import Optional

SETTINGS_FILE = "settings.json"


class AppSettings:
    def __init__(self, settings_file_path=SETTINGS_FILE, logger: Optional[logging.Logger] = None):
        self.settings_file = settings_file_path
        self.data = {}
        # Logger setup
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__ + '_AppSettings_fallback')
            if not self.logger.handlers:
                handler = logging.StreamHandler()  # Default to console for fallback
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.WARNING)  # Set a default level
            self.logger.info("AppSettings using its own configured fallback logger.")

        self.load_settings()

    def get_default_settings(self):
        # The new constants file is now the source of truth.
        # This method now assembles the default dictionary from those constants.
        from config import constants  # Import the refactored constants module
        import platform

        # Select the correct shortcut dictionary based on the operating system
        #shortcuts = constants.DEFAULT_SHORTCUTS_MACOS if platform.system() == "Darwin" else constants.DEFAULT_SHORTCUTS_WINDOWS
        shortcuts = constants.DEFAULT_SHORTCUTS

        defaults = {
            # General
            "yolo_det_model_path": "",
            "yolo_pose_model_path": "",
            "output_folder_path": constants.DEFAULT_OUTPUT_FOLDER,
            "logging_level": "INFO",

            # UI & Layout
            "window_width": constants.DEFAULT_WINDOW_WIDTH,
            "window_height": constants.DEFAULT_WINDOW_HEIGHT,
            "ui_layout_mode": constants.DEFAULT_UI_LAYOUT,
            "global_font_scale": 1.0,
            "timeline_pan_speed_multiplier": 20,
            "show_funscript_interactive_timeline": True,
            "show_funscript_interactive_timeline2": False,
            "show_funscript_timeline": True,
            "show_heatmap": True,
            "show_stage2_overlay": True,
            "show_gauge_window": True,
            "show_lr_dial_graph": True,
            "show_chapter_list_window": False,

            # File Handling & Output
            "autosave_final_funscript_to_video_location": True,
            "generate_roll_file": True,
            "batch_mode_overwrite_strategy": 0,  # 0=Process All, 1=Skip Existing

            # Performance & System
            "num_producers_stage1": constants.DEFAULT_S1_NUM_PRODUCERS,
            "num_consumers_stage1": constants.DEFAULT_S1_NUM_CONSUMERS,
            "hardware_acceleration_method": "auto",
            "ffmpeg_path": "ffmpeg",

            # Autosave & Energy Saver
            "autosave_enabled": True,
            "autosave_interval_seconds": 60,
            "autosave_on_exit": True,
            "energy_saver_enabled": True,
            "energy_saver_threshold_seconds": 30.0,
            "energy_saver_fps": 1,
            "main_loop_normal_fps_target": 60,

            # Tracking & Processing
            "funscript_output_delay_frames": 3,
            "discarded_tracking_classes": constants.CLASSES_TO_DISCARD_BY_DEFAULT,
            "tracking_axis_mode": "both",
            "single_axis_output_target": "primary",

            # Auto Post-Processing
            "enable_auto_post_processing": False,
            "auto_post_processing_sg_window": constants.DEFAULT_AUTO_POST_SG_WINDOW,
            "auto_post_processing_sg_polyorder": constants.DEFAULT_AUTO_POST_SG_POLYORDER,
            "auto_post_processing_rdp_epsilon": constants.DEFAULT_AUTO_POST_RDP_EPSILON,
            "auto_post_processing_clamp_lower_threshold_primary": constants.DEFAULT_AUTO_POST_CLAMP_LOW,
            "auto_post_processing_clamp_upper_threshold_primary": constants.DEFAULT_AUTO_POST_CLAMP_HIGH,
            "auto_post_processing_amplification_config": constants.DEFAULT_AUTO_POST_AMP_CONFIG,

            # Shortcuts
            "funscript_editor_shortcuts": shortcuts,
        }

        return defaults

    def load_settings(self):
        defaults = self.get_default_settings()
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)

                # Merge defaults with loaded settings, ensuring all keys from defaults are present
                self.data = defaults.copy()  # Start with defaults
                self.data.update(loaded_settings)  # Override with loaded values

                # Special handling for nested dictionaries like shortcuts
                if "funscript_editor_shortcuts" in loaded_settings and isinstance(
                        loaded_settings["funscript_editor_shortcuts"], dict):
                    # Ensure default shortcuts are present if not in loaded file
                    default_shortcuts = defaults.get("funscript_editor_shortcuts", {})
                    merged_shortcuts = default_shortcuts.copy()
                    merged_shortcuts.update(loaded_settings["funscript_editor_shortcuts"])
                    self.data["funscript_editor_shortcuts"] = merged_shortcuts
                else:
                    self.data["funscript_editor_shortcuts"] = defaults.get("funscript_editor_shortcuts", {})
            else:
                self.data = defaults
                self.save_settings()  # Save defaults if no settings file exists
        except Exception as e:
            self.logger.error(f"Error loading settings from '{self.settings_file}': {e}. Using default settings.",
                              exc_info=True)
            self.data = defaults

    def save_settings(self):
        try:
            with open(self.settings_file, 'w') as f:
                json.dump(self.data, f, indent=4)
            self.logger.info(f"Settings saved to {self.settings_file}.")
        except Exception as e:
            self.logger.error(f"Error saving settings to '{self.settings_file}': {e}", exc_info=True)

    def get(self, key, default=None):
        # Ensure that if a key is missing from self.data (e.g. new setting added),
        # it falls back to the hardcoded default from get_default_settings()
        # then to the 'default' parameter of this get method.
        if key not in self.data:
            return default  # Fallback to the 'default' arg of this function
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save_settings()  # here for immediate saving

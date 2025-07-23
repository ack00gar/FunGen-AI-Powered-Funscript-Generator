import json
import os
import logging
from typing import Optional

from config import constants


class AppSettings:
    def __init__(self, settings_file_path=constants.SETTINGS_FILE, logger: Optional[logging.Logger] = None):
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

        self.is_first_run = False
        self.load_settings()

    def get_default_settings(self):
        shortcuts = constants.DEFAULT_SHORTCUTS

        defaults = {
            # General
            "yolo_det_model_path": "",
            "yolo_pose_model_path": "",
            "pose_model_artifacts_dir": "",
            "output_folder_path": constants.DEFAULT_OUTPUT_FOLDER,
            "logging_level": "INFO",

            # UI & Layout
            "ui_view_mode": "simple",  # can be 'expert' or 'simple'
            "full_width_nav": True,
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
            "show_gauge_window_timeline1": True,
            "show_gauge_window_timeline2": False,
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
            "autosave_interval_seconds": 120,
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

            # --- Live Tracker Settings ---
            "live_tracker_confidence_threshold": constants.DEFAULT_TRACKER_CONFIDENCE_THRESHOLD,
            "live_tracker_roi_padding": constants.DEFAULT_TRACKER_ROI_PADDING,
            "live_tracker_roi_update_interval": constants.DEFAULT_ROI_UPDATE_INTERVAL,
            "live_tracker_roi_smoothing_factor": constants.DEFAULT_ROI_SMOOTHING_FACTOR,
            "live_tracker_roi_persistence_frames": constants.DEFAULT_ROI_PERSISTENCE_FRAMES,
            "live_tracker_use_sparse_flow": False, # Assuming False is the default for a boolean
            "live_tracker_dis_flow_preset": constants.DEFAULT_DIS_FLOW_PRESET,
            "live_tracker_dis_finest_scale": constants.DEFAULT_DIS_FINEST_SCALE,
            "live_tracker_sensitivity": constants.DEFAULT_LIVE_TRACKER_SENSITIVITY,
            "live_tracker_base_amplification": constants.DEFAULT_LIVE_TRACKER_BASE_AMPLIFICATION,
            "live_tracker_class_amp_multipliers": constants.DEFAULT_CLASS_AMP_MULTIPLIERS,
            "live_tracker_flow_smoothing_window": constants.DEFAULT_FLOW_HISTORY_SMOOTHING_WINDOW,

            # Auto Post-Processing
            "enable_auto_post_processing": False,
            "auto_processing_use_chapter_profiles": True,
            "auto_post_proc_final_rdp_enabled": False,
            "auto_post_proc_final_rdp_epsilon": 10.0,
            "auto_post_processing_amplification_config": constants.DEFAULT_AUTO_POST_AMP_CONFIG,

            # Shortcuts
            "funscript_editor_shortcuts": shortcuts,

            # Recent Projects
            "last_opened_project_path": "",
            "recent_projects": [],

            # Updater Settings
            "updater_check_on_startup": True,
            "updater_check_periodically": True,
            "updater_suppress_popup": False,
        }
        return defaults

    def load_settings(self):
        defaults = self.get_default_settings()
        try:
            if os.path.exists(self.settings_file):
                with open(self.settings_file, 'r') as f:
                    loaded_settings = json.load(f)

                # Migration for old setting name
                if "show_gauge_window" in loaded_settings:
                    loaded_settings["show_gauge_window_timeline1"] = loaded_settings.pop("show_gauge_window")


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
                self.is_first_run = True
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
            defaults = self.get_default_settings()
            if key in defaults:
                self.data[key] = defaults[key]
                return defaults[key]
            return default
        return self.data.get(key, default)

    def set(self, key, value):
        self.data[key] = value
        self.save_settings()  # here for immediate saving

    def reset_to_defaults(self):
        self.data = self.get_default_settings()
        self.save_settings()
        self.logger.info("All application settings have been reset to their default values.")

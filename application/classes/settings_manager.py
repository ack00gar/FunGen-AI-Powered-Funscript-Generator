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
        default_cores = os.cpu_count() if os.cpu_count() else 4
        temp_initial_window_width = 1800
        temp_initial_gauge_width = 100
        defaults = {
            "yolo_det_model_path": "",
            "yolo_pose_model_path": "",
            "num_producers_stage1": 1,
            "num_consumers_stage1": max(default_cores // 2, 1),
            "autosave_enabled": True,
            "autosave_interval_seconds": 60,
            "autosave_on_exit": True,
            "funscript_editor_shortcuts": {
                "add_point_0": "#", "add_point_10": "1", "add_point_20": "2",
                "add_point_30": "3", "add_point_40": "4", "add_point_50": "5",
                "add_point_60": "6", "add_point_70": "7", "add_point_80": "8",
                "add_point_90": "9", "add_point_100": "0",
                "seek_next_frame": "RIGHT_ARROW",
                "seek_prev_frame": "LEFT_ARROW",
                "nudge_next_frame": "SHIFT+RIGHT_ARROW",
                "nudge_prev_frame": "SHIFT+LEFT_ARROW",
                "nudge_selection_pos_up": "UP_ARROW",
                "nudge_selection_pos_down": "DOWN_ARROW",
                "delete_selected_point": "DELETE",
                "delete_selected_point_alt": "BACKSPACE",
                "select_all_points": "CTRL+A",
                "undo_timeline1": "CTRL+Z",
                "redo_timeline1": "CTRL+Y",
                "undo_timeline2": "CTRL+ALT+Z",
                "redo_timeline2": "CTRL+ALT+Y",
                "copy_selection": "CTRL+C",
                "paste_selection": "CTRL+V",
                "toggle_playback": "SPACE"
            },
            "window_width": temp_initial_window_width,
            "window_height": 1000,
            "timeline_zoom_factor_ms_per_px": 20.0,
            "timeline_pan_offset_ms": 0.0,
            "show_gauge_window": True,
            "gauge_window_pos_x": temp_initial_window_width - temp_initial_gauge_width - 20,
            "gauge_window_pos_y": 35,
            "gauge_window_size_w": temp_initial_gauge_width,
            "gauge_window_size_h": 220,
            "show_lr_dial_graph": True,
            "lr_dial_window_pos_x": -1,
            "lr_dial_window_pos_y": 35,
            "lr_dial_window_size_w": 150,
            "lr_dial_window_size_h": 180,
            "main_menu_bar_height_for_gauge_default_y": 25,
            "show_funscript_interactive_timeline": True,
            "show_funscript_interactive_timeline2": False,
            "show_funscript_timeline": False,
            "show_heatmap": True,
            "show_stage2_overlay": True,
            "funscript_output_delay_frames": 3,
            # --- Energy Saver Settings ---
            "energy_saver_enabled": True,
            "energy_saver_threshold_seconds": 30.0,
            "energy_saver_fps": 1,
            "main_loop_normal_fps_target": 60,
            # --- UI Settings ---
            "ui_layout_mode": "fixed",  # "fixed" or "floating"
            "show_control_panel_window": True,
            "show_video_display_window": True,
            # --- Hardware Acceleration Settings ---
            "hardware_acceleration_method": "auto",  # "auto", "videotoolbox", "none"
            "ffmpeg_path": "ffmpeg",
            # --- Logging Settings ---
            "logging_level": "INFO",
            # --- Output Settings ---
            "output_folder_path": "output",
            "autosave_final_funscript_to_video_location": True,
            "batch_mode_overwrite_strategy": 0,  # 0=Process All, 1=Skip Existing
            "generate_roll_file": True,
            # --- Tracking Settings ---
            "discarded_tracking_classes": [],
            "tracking_axis_mode": "both",
            "single_axis_output_target": "primary",
            "enable_auto_post_processing": False,
            "auto_post_processing_sg_window": 7,  # Savitzky-Golay window length (odd number >= 3)
            "auto_post_processing_sg_polyorder": 3,  # Savitzky-Golay polynomial order ( < window)
            "auto_post_processing_rdp_epsilon": 15,  # RDP epsilon
            "auto_post_processing_clamp_lower_threshold_primary": 15,  # For primary: clamp values < this to 0
            "auto_post_processing_clamp_upper_threshold_primary": 85,  # For primary: clamp values > this to 100
            "auto_post_processing_amplification_config": {
                "Blowjob": {"scale_factor": 1.3, "center_value": 60},
                "Handjob": {"scale_factor": 1.3, "center_value": 60},
                "Cowgirl / Missionary": {"scale_factor": 1.1, "center_value": 50},
                "Rev. Cowgirl / Doggy": {"scale_factor": 1.1, "center_value": 50},
                "Boobjob": {"scale_factor": 1.2, "center_value": 55},
                "Footjob": {"scale_factor": 1.2, "center_value": 50},
                "Default": {"scale_factor": 1.0, "center_value": 50}  # Fallback
            }
        }

        import platform
        if platform.system() == "Darwin":  # macOS
            if "funscript_editor_shortcuts" in defaults:
                defaults["funscript_editor_shortcuts"]["undo_timeline1"] = "SUPER+Z"
                defaults["funscript_editor_shortcuts"]["redo_timeline1"] = "SUPER+SHIFT+Z"
                defaults["funscript_editor_shortcuts"]["undo_timeline2"] = "SUPER+ALT+Z"
                defaults["funscript_editor_shortcuts"]["redo_timeline2"] = "SUPER+ALT+SHIFT+Z"
                defaults["funscript_editor_shortcuts"]["copy_selection"] = "SUPER+C"
                defaults["funscript_editor_shortcuts"]["paste_selection"] = "SUPER+V"
                defaults["funscript_editor_shortcuts"]["select_all_points"] = "SUPER+A"
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

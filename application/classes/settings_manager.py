import json
import os
import logging
import subprocess
import platform
from datetime import datetime
from typing import Optional, List, Dict
from config import constants


# Settings keys to include in profiles (processing-related, not UI layout)
PROFILE_KEYS = [
    # Tracking
    "live_tracker_confidence_threshold",
    "live_tracker_roi_padding",
    "live_tracker_roi_update_interval",
    "live_tracker_roi_smoothing_factor",
    "live_tracker_roi_persistence_frames",
    "live_tracker_use_sparse_flow",
    "live_tracker_dis_flow_preset",
    "live_tracker_dis_finest_scale",
    "live_tracker_sensitivity",
    "live_tracker_base_amplification",
    "live_tracker_class_amp_multipliers",
    "live_tracker_flow_smoothing_window",
    "funscript_output_delay_frames",
    "discarded_tracking_classes",
    "tracking_axis_mode",
    "single_axis_output_target",
    # Oscillation Detector
    "oscillation_detector_grid_size",
    "oscillation_detector_sensitivity",
    "stage3_oscillation_detector_mode",
    "oscillation_enable_decay",
    "oscillation_hold_duration_ms",
    "oscillation_decay_factor",
    "oscillation_use_simple_amplification",
    "live_oscillation_dynamic_amp_enabled",
    "live_oscillation_amp_window_ms",
    # Post-Processing
    "enable_auto_post_processing",
    "auto_post_proc_final_rdp_enabled",
    "auto_post_proc_final_rdp_epsilon",
    "auto_post_processing_amplification_config",
    "auto_processing_use_chapter_profiles",
    # Signal Enhancement
    "enable_signal_enhancement",
    "signal_enhancement_motion_threshold_low",
    "signal_enhancement_motion_threshold_high",
    "signal_enhancement_signal_change_threshold",
    "signal_enhancement_strength",
    # Performance
    "num_producers_stage1",
    "num_consumers_stage1",
    "num_workers_stage2_of",
    "hardware_acceleration_method",
    "funscript_point_simplification_enabled",
    "adaptive_batch_tuning_enabled",
]


class AppSettings:
    def __init__(self, settings_file_path=constants.SETTINGS_FILE, logger: Optional[logging.Logger] = None):
        self.constants = constants
        self.settings_file = settings_file_path
        self.data = {}
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__ + '_AppSettings_fallback')
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.WARNING)  # Set a default level
            self.logger.info("AppSettings using its own configured fallback logger.")

        self.is_first_run = False
        self.shortcuts_were_reset = False  # Set by migration; GUI shows one-time notice
        self.load_settings()

        # Auto-detect and set hardware acceleration on first run
        if self.is_first_run:
            self.auto_detect_hardware_acceleration()

    def get_default_settings(self):
        constants = self.constants
        shortcuts = constants.DEFAULT_SHORTCUTS

        defaults = {
            # General
            "yolo_det_model_path": "",
            "yolo_pose_model_path": "",
            "output_folder_path": constants.DEFAULT_OUTPUT_FOLDER,
            "logging_level": "INFO",

            # UI & Layout
            "full_width_nav": True,
            "window_width": constants.DEFAULT_WINDOW_WIDTH,
            "window_height": constants.DEFAULT_WINDOW_HEIGHT,
            "ui_layout_mode": constants.DEFAULT_UI_LAYOUT,

            "global_font_scale": 1.0,
            "auto_system_scaling_enabled": True,  # Automatically detect and apply system scaling
            "timeline_pan_speed_multiplier": 20,
            "timeline_pan_drag_modifier": "ALT",  # Modifier key for left-drag pan (trackpad alternative to middle-mouse)
            "timeline_create_point_modifier": "SHIFT",  # Modifier key for click-to-create-point
            "timeline_marquee_modifier": "CTRL",  # Modifier key for box/marquee selection drag
            "tracker_show_legacy": False,        # Show legacy trackers in dropdown
            "tracker_show_experimental": True,   # Show experimental trackers in dropdown
            "tracker_show_community": True,      # Show community trackers in dropdown
            "show_funscript_interactive_timeline": True,
            "show_funscript_interactive_timeline2": False,
            "show_funscript_timeline": True,

            # Timeline Performance
            "show_timeline_optimization_indicator": False,  # Performance indicators hidden by default
            "timeline_performance_logging": False,  # Log timeline performance stats
            "show_heatmap": True,
            "use_simplified_funscript_preview": False,
            "show_stage2_overlay": True,
            "show_simulator_3d": True,
            "simulator_3d_overlay_mode": True,
            "show_chapter_list_window": False,
            "show_timeline_editor_buttons": False,
            "show_advanced_options": False,
            "show_toast_notifications": True,
            "show_video_feed": True,
            "hd_video_display": True,  # Decode at display resolution (up to 1920px) for sharper preview

            # File Handling & Output
            "autosave_final_funscript_to_video_location": True,
            "generate_roll_file": True,
            "export_raw_as_funscript": False,
            "batch_mode_overwrite_strategy": 0,  # 0=Process All, 1=Skip Existing
            "metadata_verbose": True,          # Include extended fields in funscript metadata
            "performance_metadata": False,          # Include extended performance fields in funscript metadata
            "metadata_creator_identity": "",  # Identity string embedded in funscript metadata

            # Performance & System
            "num_producers_stage1": constants.DEFAULT_S1_NUM_PRODUCERS,
            "num_consumers_stage1": constants.DEFAULT_S1_NUM_CONSUMERS,
            "num_workers_stage2_of": constants.DEFAULT_S2_OF_WORKERS,
            "adaptive_batch_tuning_enabled": True,  # Progressive pipeline thread optimization during batch processing
            "hardware_acceleration_method": "none",  # Default to CPU to avoid CUDA errors on non-NVIDIA systems
            "default_secondary_axis": "roll",  # Default secondary axis for dual-axis trackers (roll, twist, pitch, etc.)
            "ffmpeg_path": "ffmpeg",
            # VR Unwarp method: 'auto', 'metal', 'opengl', 'v360'
            # macOS: v360 is 26% faster than GPU unwarp due to optimized FFmpeg filter
            # Other platforms: auto selects best GPU backend
            "vr_unwarp_method": "v360" if platform.system() == "Darwin" else "auto",

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

            # --- Settings for the 2D Oscillation Detector ---
            "oscillation_detector_grid_size": 20,
            "oscillation_detector_sensitivity": 2.5,
            "stage3_oscillation_detector_mode": "current",  # "current", "legacy", or "hybrid"

            # Oscillation Detector Improvements
            "oscillation_enable_decay": True,  # Enable decay mechanism
            "oscillation_hold_duration_ms": 250,  # Hold duration before decay starts
            "oscillation_decay_factor": 0.95,  # Decay factor toward center
            "oscillation_use_simple_amplification": False,  # Use simple fixed multipliers

            "live_oscillation_dynamic_amp_enabled": True,
            "live_oscillation_amp_window_ms": 4000,  # 4-second analysis window

            # --- Funscript Generation Settings ---
            "funscript_point_simplification_enabled": True,  # Enable on-the-fly point simplification

            # --- Signal Enhancement Settings ---
            "enable_signal_enhancement": True,  # Enable frame difference based signal enhancement
            "signal_enhancement_motion_threshold_low": 12.0,  # Minimum motion for significant movement
            "signal_enhancement_motion_threshold_high": 30.0,  # High motion threshold for missing strokes
            "signal_enhancement_signal_change_threshold": 6,  # Minimum signal change to consider significant
            "signal_enhancement_strength": 0.25,  # Enhancement strength (0.0 - 1.0)

            # Auto Post-Processing
            "enable_auto_post_processing": False,

            # Database Management
            "retain_stage2_database": True,  # Keep SQLite database after processing (default: True for GUI, False for CLI)
            "auto_processing_use_chapter_profiles": True,

            # Chapter Management
            "chapter_auto_save_standalone": False,  # Auto-save chapters to standalone JSON files
            "chapter_backup_on_regenerate": True,  # Create backup before overwriting chapter files
            "chapter_skip_if_exists": False,  # Skip chapter creation if standalone file exists
            "overwrite_chapters_on_analysis": False,  # Allow analysis to overwrite existing chapters (default: preserve)

            # VR Streaming / Streamer
            "xbvr_host": "localhost",  # XBVR server host/IP
            "xbvr_port": 9999,  # XBVR server port
            "xbvr_enabled": True,  # Enable XBVR integration
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

            # Device Control Settings
            "device_control_enabled": True,
            "buttplug_server_address": "localhost",
            "buttplug_server_port": 12345,
            "buttplug_auto_connect": False,
            "device_control_preferred_backend": "buttplug",  # "buttplug", "osr", or "auto"
            "device_control_last_connected_device_type": "",  # Last successfully connected device type
            "device_control_max_rate_hz": 20.0,
            "device_control_selected_devices": [],  # List of selected device IDs
            "device_control_log_commands": False,

            # Audio Playback
            "audio_enabled": True,
            "audio_volume": 0.8,
            "audio_muted": False,
        }
        return defaults

    def load_settings(self):
        defaults = self.get_default_settings()
        settings_file = self.settings_file

        try:
            if os.path.exists(settings_file):
                with open(settings_file, 'r') as f:
                    loaded_settings = json.load(f)

                # Migration for old setting name
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

                # --- v0.7.0 migration: reset shortcuts if from older version ---
                stored_sc_version = loaded_settings.get("_shortcuts_version", "0.0.0")
                if stored_sc_version < "0.7.0":
                    self.data["funscript_editor_shortcuts"] = defaults.get("funscript_editor_shortcuts", {})
                    self.data["_shortcuts_version"] = constants.APP_VERSION
                    self.shortcuts_were_reset = True
                    self.save_settings()
                    self.logger.info("Shortcuts reset to v0.7.0 defaults (layout restructured).")
            else:
                self.is_first_run = True
                self.data = defaults
                self.data["_shortcuts_version"] = constants.APP_VERSION
                self.save_settings()  # Save defaults if no settings file exists
        except Exception as e:
            self.logger.error(f"Error loading settings from '{settings_file}': {e}. Using default settings.", exc_info=True)
            self.data = defaults

    def save_settings(self):
        settings_file = self.settings_file
        try:
            with open(settings_file, 'w') as f:
                json.dump(self.data, f, indent=4)
            self.logger.debug(f"Settings saved to {settings_file}.")
        except Exception as e:
            self.logger.error(f"Error saving settings to '{settings_file}': {e}", exc_info=True)

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

    def set_batch(self, **kwargs):
        """Set multiple keys at once and save only once at the end."""
        for key, value in kwargs.items():
            self.data[key] = value
        self.save_settings()

    def reset_to_defaults(self):
        self.data = self.get_default_settings()
        self.save_settings()
        self.logger.info("All application settings have been reset to their default values.")

    def auto_detect_hardware_acceleration(self):
        """
        Auto-detect available GPU and set appropriate hardware acceleration.
        Only runs on first launch or when settings are reset.
        """
        system = platform.system()
        detected_method = "none"  # Default to CPU

        try:
            # Check for NVIDIA GPU on Windows/Linux
            if system in ["Windows", "Linux"]:
                try:
                    result = subprocess.run(
                        ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader,nounits'],
                        capture_output=True, text=True, timeout=5
                    )
                    if result.returncode == 0 and result.stdout.strip():
                        detected_method = "auto"  # NVIDIA GPU detected, allow auto selection
                        self.logger.info(f"NVIDIA GPU detected: {result.stdout.strip()}")
                except (subprocess.TimeoutExpired, FileNotFoundError):
                    pass

                # Check for Intel QSV (Quick Sync)
                if detected_method == "none":
                    try:
                        # Check if ffmpeg supports qsv
                        result = subprocess.run(
                            ['ffmpeg', '-hide_banner', '-hwaccels'],
                            capture_output=True, text=True, timeout=5
                        )
                        if 'qsv' in result.stdout.lower():
                            detected_method = "qsv"
                            self.logger.info("Intel Quick Sync Video detected")
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass

            # macOS: VideoToolbox is SLOWER for sequential frame processing with filters
            # Benchmark shows CPU-only is 6x faster due to GPU→CPU transfer overhead
            # Keep "none" (CPU-only) as default for best performance
            elif system == "Darwin":
                detected_method = "none"  # CPU-only is faster for this workload
                self.logger.info("macOS detected, using CPU-only decoding (6x faster than VideoToolbox for filter chains)")

        except Exception as e:
            self.logger.warning(f"Error during GPU detection: {e}")

        # Update the setting
        if detected_method != self.data.get("hardware_acceleration_method"):
            self.logger.info(f"Setting hardware acceleration to: {detected_method}")
            self.data["hardware_acceleration_method"] = detected_method
            self.save_settings()

    # ------- Settings Profiles -------

    def get_profiles_dir(self) -> str:
        """Get the profiles directory path, creating it if needed."""
        settings_dir = os.path.dirname(self.settings_file)
        profiles_dir = os.path.join(settings_dir, "profiles")
        os.makedirs(profiles_dir, exist_ok=True)
        return profiles_dir

    def save_profile(self, name: str, keys: Optional[List[str]] = None) -> bool:
        """Save current processing settings as a named profile."""
        if not name or not name.strip():
            return False
        name = name.strip()
        profile_keys = keys or PROFILE_KEYS
        settings = {k: self.data[k] for k in profile_keys if k in self.data}
        profile_data = {
            "profile_name": name,
            "created_at": datetime.now().isoformat(),
            "version": 1,
            "settings": settings,
        }
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        file_path = os.path.join(self.get_profiles_dir(), "%s.json" % safe_name)
        try:
            with open(file_path, 'w') as f:
                json.dump(profile_data, f, indent=4)
            self.logger.info("Profile saved: %s" % name)
            return True
        except Exception as e:
            self.logger.error("Failed to save profile '%s': %s" % (name, e))
            return False

    def load_profile(self, name: str) -> bool:
        """Load a profile by name, updating matching settings keys."""
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        file_path = os.path.join(self.get_profiles_dir(), "%s.json" % safe_name)
        try:
            with open(file_path, 'r') as f:
                profile_data = json.load(f)
            settings = profile_data.get("settings", {})
            for k, v in settings.items():
                if k in PROFILE_KEYS:
                    self.data[k] = v
            self.save_settings()
            self.logger.info("Profile loaded: %s (%d settings)" % (name, len(settings)))
            return True
        except Exception as e:
            self.logger.error("Failed to load profile '%s': %s" % (name, e))
            return False

    def list_profiles(self) -> List[Dict]:
        """List all saved profiles."""
        profiles = []
        profiles_dir = self.get_profiles_dir()
        try:
            for filename in sorted(os.listdir(profiles_dir)):
                if not filename.endswith(".json"):
                    continue
                file_path = os.path.join(profiles_dir, filename)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                    profiles.append({
                        "name": data.get("profile_name", filename[:-5]),
                        "file": filename,
                        "created_at": data.get("created_at", ""),
                    })
                except Exception:
                    profiles.append({"name": filename[:-5], "file": filename, "created_at": ""})
        except Exception as e:
            self.logger.error("Failed to list profiles: %s" % e)
        return profiles

    def delete_profile(self, name: str) -> bool:
        """Delete a profile by name."""
        safe_name = "".join(c if c.isalnum() or c in (' ', '-', '_') else '_' for c in name)
        file_path = os.path.join(self.get_profiles_dir(), "%s.json" % safe_name)
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                self.logger.info("Profile deleted: %s" % name)
                return True
            return False
        except Exception as e:
            self.logger.error("Failed to delete profile '%s': %s" % (name, e))
            return False

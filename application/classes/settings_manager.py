import json
import os
import logging
import subprocess
import platform
from typing import Optional
from config import constants


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
            "pose_model_artifacts_dir": "",
            "output_folder_path": constants.DEFAULT_OUTPUT_FOLDER,
            "logging_level": "INFO",

            # UI & Layout
            "ui_view_mode": "simple",  # can be 'expert' or 'simple'
            "full_width_nav": True,
            "window_width": constants.DEFAULT_WINDOW_WIDTH,
            "window_height": constants.DEFAULT_WINDOW_HEIGHT,
            "ui_layout_mode": constants.DEFAULT_UI_LAYOUT,
            
            # Movement Bar (L/R Dial) Window Settings
            "lr_dial_window_size_w": 180,  # Width of movement bar window
            "lr_dial_window_size_h": 220,  # Height of movement bar window
            "lr_dial_window_pos_x": -1,    # X position (-1 = auto-calculate)
            "lr_dial_window_pos_y": -1,    # Y position (-1 = auto-calculate)
            "global_font_scale": 1.0,
            "auto_system_scaling_enabled": True,  # Automatically detect and apply system scaling
            "timeline_pan_speed_multiplier": 20,
            "show_funscript_interactive_timeline": True,
            "show_funscript_interactive_timeline2": False,
            "show_funscript_timeline": True,
            
            # Timeline Performance & GPU Settings
            "timeline_gpu_enabled": True,  # GPU rendering enabled by default for better performance
            "timeline_gpu_threshold_points": 5000,  # Use GPU above this point count
            "show_timeline_optimization_indicator": False,  # Performance indicators hidden by default
            "timeline_performance_logging": False,  # Log timeline performance stats
            "show_heatmap": True,
            "use_simplified_funscript_preview": False,
            "show_stage2_overlay": True,
            "show_gauge_window_timeline1": False,
            "show_gauge_window_timeline2": False,
            "show_lr_dial_graph": False,  # Movement Bar (rotating bar with up/down fill and roll angle)
            "show_simulator_3d": True, # 3D Simulator
            "show_3d_simulator_logo": True,  # Display logo texture on 3D simulator cylinder

            # Overlay mode settings (render as overlay on video display)
            "gauge_overlay_mode": False,  # Render gauges as video overlay
            "movement_bar_overlay_mode": False,  # Render movement bar as video overlay
            "simulator_3d_overlay_mode": True,  # Render 3D simulator as video overlay
            "show_chapter_list_window": False,
            "show_timeline_editor_buttons": False,
            "show_advanced_options": False,
            "show_video_feed": True,

            # File Handling & Output
            "autosave_final_funscript_to_video_location": True,
            "generate_roll_file": True,
            "batch_mode_overwrite_strategy": 0,  # 0=Process All, 1=Skip Existing

            # Performance & System
            "num_producers_stage1": constants.DEFAULT_S1_NUM_PRODUCERS,
            "num_consumers_stage1": constants.DEFAULT_S1_NUM_CONSUMERS,
            "num_workers_stage2_of": constants.DEFAULT_S2_OF_WORKERS,
            "hardware_acceleration_method": "none",  # Default to CPU to avoid CUDA errors on non-NVIDIA systems
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
            "device_control_max_rate_hz": 20.0,
            "device_control_selected_devices": [],  # List of selected device IDs
            "device_control_log_commands": False,
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
            self.logger.error(f"Error loading settings from '{settings_file}': {e}. Using default settings.", exc_info=True)
            self.data = defaults

    def save_settings(self):
        settings_file = self.settings_file
        try:
            with open(settings_file, 'w') as f:
                json.dump(self.data, f, indent=4)
            self.logger.info(f"Settings saved to {settings_file}.")
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

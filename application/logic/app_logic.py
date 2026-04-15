import time
import logging
import subprocess
import os
import platform
import threading
from typing import Optional, Dict, Tuple, List, Any
from datetime import datetime, timedelta
from ultralytics import YOLO

from video import VideoProcessor
from tracker.tracker_manager import create_tracker_manager

from application.classes import AppSettings, ProjectManager, ShortcutManager
from application.classes.undo_manager import UndoManager
from application.utils import AppLogger, check_write_access, AutoUpdater, VideoSegment
from application.utils.addon_update_checker import AddonUpdateChecker
from config.constants import DEFAULT_MODELS_DIR, FUNSCRIPT_METADATA_VERSION, PROJECT_FILE_EXTENSION, MODEL_DOWNLOAD_URLS, YOLO_INPUT_SIZE
from config.tracker_discovery import get_tracker_discovery
from pathlib import Path

from .app_state_ui import AppStateUI
from .app_file_manager import AppFileManager
from .app_stage_processor import AppStageProcessor
from .app_funscript_processor import AppFunscriptProcessor
from .app_event_handlers import AppEventHandlers
from .app_energy_saver import AppEnergySaver
from .app_utility import AppUtility
from .app_model_manager import AppModelManager
from .app_autotuner import AppAutotuner
from .app_roi_manager import AppROIManager
from .app_batch_processor import AppBatchProcessor, AdaptiveTuningState
from .app_cli_runner import (
    AppCLIRunner,
    cli_live_video_progress_callback,
    _create_cli_progress_bar,
    cli_stage1_progress_callback,
    cli_stage2_progress_callback,
    cli_stage3_progress_callback,
)

# Audio playback (optional, graceful degradation if sounddevice missing)
try:
    from video.audio_player import AudioPlayer, SOUNDDEVICE_AVAILABLE
    from video.audio_video_sync import AudioVideoSync
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# Import InteractiveFunscriptTimeline for type hinting
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from application.classes.interactive_timeline import InteractiveFunscriptTimeline


# AdaptiveTuningState moved to app_batch_processor.py (imported above for backward compat)

# Module-level CLI callback functions moved to app_cli_runner.py
# They are re-imported at the top of this file for backward compatibility.


class ApplicationLogic:
    def __init__(self, is_cli: bool = False):
        self.is_cli_mode = is_cli # Store the mode
        self.gui_instance = None
        self.app_settings = AppSettings(logger=None)

        # Initialize logging_level_setting before AppLogger uses it indirectly via AppSettings
        self.logging_level_setting = self.app_settings.get("logging_level", "INFO")

        self.cached_class_names: Optional[List[str]] = None

        status_log_config = {
            logging.INFO: 3.0, logging.WARNING: 6.0, logging.ERROR: 10.0, logging.CRITICAL: 15.0,
        }
        Path("logs").mkdir(exist_ok=True)
        self.app_log_file_path = 'logs/fungen.log'  # Define app_log_file_path

        # Log purge runs in background, non-critical for startup
        threading.Thread(
            target=self._purge_old_log_entries,
            args=(self.app_log_file_path,),
            daemon=True,
            name="LogPurge",
        ).start()

        self._logger_instance = AppLogger(
            app_logic_instance=self,
            status_level_durations=status_log_config,
            log_file=self.app_log_file_path,
            level=getattr(logging, self.logging_level_setting.upper(), logging.INFO)  # Use initial setting
        )
        self.logger = self._logger_instance.get_logger()
        self.app_settings.logger = self.logger  # Now provide the logger to AppSettings
        
        # Configure third-party logging to reduce startup noise
        self._configure_third_party_logging()

        # --- Initialize Auto-Updater ---
        self.updater = AutoUpdater(self)
        self.addon_checker = AddonUpdateChecker(self)

        # REFACTORED Defensive programming. Always make sure the type is a list of strings.
        discarded_tracking_classes = self.app_settings.get("discarded_tracking_classes", [])
        if discarded_tracking_classes is None:
            discarded_tracking_classes = []
        self.discarded_tracking_classes: List[str] = discarded_tracking_classes
        self.pending_action_after_tracking: Optional[Dict] = None

        self.app_state_ui = AppStateUI(self)
        self.utility = AppUtility(self)

        # --- State for first-run setup ---
        self.show_first_run_setup_popup = False
        self.first_run_progress = 0.0
        self.first_run_status_message = ""
        self.first_run_error = False
        self.first_run_thread: Optional[threading.Thread] = None

        # --- Autotuner State ---
        self.is_autotuning_active: bool = False
        self.autotuner_thread: Optional[threading.Thread] = None
        self.autotuner_status_message: str = "Idle"
        self.autotuner_results: Dict[Tuple[int, int, str], Tuple[float, str]] = {}
        self.autotuner_best_combination: Optional[Tuple[int, int, str]] = None
        self.autotuner_best_fps: float = 0.0
        self.autotuner_forced_hwaccel: Optional[str] = None
        self._autotuner_lock = threading.Lock()  # Protects autotuner_results, _best_combination, _best_fps, _status_message

        # --- Hardware Acceleration ---
        # Load cached hwaccel list from settings so validation works immediately;
        # background thread refreshes the cache from ffmpeg
        cached_hwaccels = self.app_settings.get("available_ffmpeg_hwaccels", None)
        self.available_ffmpeg_hwaccels = cached_hwaccels if cached_hwaccels else ["auto", "none"]
        self.hardware_acceleration_method = self.app_settings.get("hardware_acceleration_method", "auto")
        self._hwaccel_query_done = threading.Event()
        threading.Thread(
            target=self._query_hwaccels_background,
            daemon=True,
            name="HWAccelQuery",
        ).start()

        # --- Tracking Axis Configuration (ensure these are initialized before tracker if tracker uses them in __init__) ---
        self.tracking_axis_mode = self.app_settings.get("tracking_axis_mode", "both")
        self.single_axis_output_target = self.app_settings.get("single_axis_output_target", "primary")

        # --- Models ---
        self.yolo_detection_model_path_setting = self.app_settings.get("yolo_det_model_path")
        self.yolo_pose_model_path_setting = self.app_settings.get("yolo_pose_model_path")
        self.yolo_det_model_path = self.yolo_detection_model_path_setting
        self.yolo_pose_model_path = self.yolo_pose_model_path_setting
        self.yolo_input_size = YOLO_INPUT_SIZE

        # --- Undo/Redo ---
        self.undo_manager = UndoManager(max_history=100)

        # --- Initialize Tracker Manager ---
        # Yield before heavy YOLO loading to allow splash rendering
        time.sleep(0.001)  # 1ms yield - forces GIL release

        self.tracker = create_tracker_manager(
            app_logic_instance=self,
            tracker_model_path=self.yolo_detection_model_path_setting)

        # Yield after tracker creation (YOLO model loaded)
        time.sleep(0.001)  # 1ms yield - forces GIL release

        if self.tracker:
            self.tracker.show_stats = False  # Default internal tracker states
            self.tracker.show_funscript_preview = False

        # --- NOW Sync Tracker UI Flags as tracker and app_state_ui exist ---
        time.sleep(0.001)  # Yield before sync
        self.app_state_ui.sync_tracker_ui_flags()

        # --- Initialize Processor (after tracker and logger/app_state_ui are ready) ---
        # _check_model_paths can be called now before processor if it's critical for processor init
        time.sleep(0.001)  # Yield before model check
        self.model_manager = AppModelManager(self)
        self._check_model_paths()

        time.sleep(0.001)  # Yield before processor creation
        self.processor = VideoProcessor(self, self.tracker, yolo_input_size=self.yolo_input_size, cache_size=1000)

        # --- Modular Components Initialization ---
        self.file_manager = AppFileManager(self)
        self.stage_processor = AppStageProcessor(self)
        self.funscript_processor = AppFunscriptProcessor(self)
        self.event_handlers = AppEventHandlers(self)
        self.energy_saver = AppEnergySaver(self)
        self.utility = AppUtility(self)
        self.autotuner = AppAutotuner(self)
        self.roi_manager = AppROIManager(self)
        self.batch_processor = AppBatchProcessor(self)
        self.cli_runner = AppCLIRunner(self)

        # --- Streamer state (set by cp_streamer_ui when streamer starts/stops) ---
        self._streamer_active = False

        # --- Audio Playback (GUI only, no audio in CLI/batch mode) ---
        self._audio_player = None
        self._audio_sync = None
        if not self.is_cli_mode and SOUNDDEVICE_AVAILABLE and self.app_settings.get("audio_enabled", True):
            try:
                self._audio_player = AudioPlayer()
                self._audio_sync = AudioVideoSync(self.processor, self._audio_player, self)
                self._audio_player.set_volume(self.app_settings.get("audio_volume", 0.8))
                self._audio_player.set_mute(self.app_settings.get("audio_muted", False))
                self._audio_sync.start()
                self.logger.debug("Audio playback initialized")
            except Exception as e:
                self.logger.warning(f"Audio playback init failed: {e}", exc_info=True)
                self._audio_player = None
                self._audio_sync = None

        # --- mpv Playback Controller (fullscreen, Patreon supporter feature) ---
        self._mpv_controller = None
        self._mpv_binary_missing = False   # True = patreon folder present but mpv not installed
        if not self.is_cli_mode:
            try:
                import shutil
                from video.mpv_ipc_bridge import _find_mpv_binary
                mpv_bin = _find_mpv_binary()
                # Verify the binary actually exists (not the "mpv" fallback string)
                if shutil.which(mpv_bin) is None and not __import__('os').path.isfile(mpv_bin):
                    self._mpv_binary_missing = True
                    self.logger.warning(
                        "mpv binary not found, fullscreen mode unavailable. "
                        "Install mpv (macOS: brew install mpv)"
                    )
                else:
                    from video.mpv_playback_controller import MpvPlaybackController
                    self._mpv_controller = MpvPlaybackController(self)
                    self.logger.debug(f"MpvPlaybackController initialized (binary: {mpv_bin})")
            except Exception as e:
                self.logger.warning(f"MpvPlaybackController unavailable: {e}")

        # --- System Scaling Detection ---
        if not self.is_cli_mode:
            try:
                from application.utils.system_scaling import apply_system_scaling_to_settings, get_system_scaling_info
                scaling_applied = apply_system_scaling_to_settings(self.app_settings)
                if scaling_applied:
                    self.logger.info("System scaling applied to application settings")
                else:
                    # Log system scaling info for debugging even if not applied
                    try:
                        scaling_factor, dpi, platform = get_system_scaling_info()
                        self.logger.debug(f"System scaling info: {scaling_factor:.2f}x ({dpi:.0f} DPI on {platform})")
                    except Exception as e:
                        self.logger.debug(f"Could not get system scaling info: {e}")
            except Exception as e:
                self.logger.warning(f"Failed to apply system scaling: {e}")

        # --- Other Managers ---
        self.project_manager = ProjectManager(self)
        self.shortcut_manager = ShortcutManager(self)

        # Pattern Library
        try:
            from funscript.pattern_library import PatternLibrary
            self.pattern_library = PatternLibrary()
        except Exception:
            self.pattern_library = None
        self._shortcut_mapping_cache = {}  # Cache parsed shortcut mappings to avoid string parsing every frame

        # Initialize chapter type manager for custom chapter types
        from application.classes.chapter_type_manager import ChapterTypeManager, set_chapter_type_manager
        self.chapter_type_manager = ChapterTypeManager(self)
        set_chapter_type_manager(self.chapter_type_manager)  # Set global instance

        # Initialize chapter manager for standalone chapter file operations
        from application.classes.chapter_manager import ChapterManager, set_chapter_manager
        self.chapter_manager = ChapterManager(self)
        set_chapter_manager(self.chapter_manager)  # Set global instance

        self.project_data_on_load: Optional[Dict] = None
        self.s2_frame_objects_map_for_s3: Optional[Dict[int, Any]] = None
        self.s2_sqlite_db_path: Optional[str] = None

        # User Defined ROI
        self.is_setting_user_roi_mode: bool = False
        # --- State for chapter-specific ROI setting ---
        self.chapter_id_for_roi_setting: Optional[str] = None

        # Oscillation Area Selection
        self.is_setting_oscillation_area_mode: bool = False
        self.oscillation_grid_size = self.app_settings.get("oscillation_detector_grid_size")
        self.oscillation_sensitivity = self.app_settings.get("oscillation_detector_sensitivity")

        # --- Batch Processing ---
        self.batch_video_paths: List[str] = []
        self.show_batch_confirmation_dialog: bool = False
        self.batch_confirmation_videos: List[str] = []
        self.batch_confirmation_message: str = ""
        self.is_batch_processing_active: bool = False
        self.current_batch_video_index: int = -1
        self.batch_processing_thread: Optional[threading.Thread] = None
        self.stop_batch_event = threading.Event()
        # An event to signal when a single video's analysis is complete
        self.single_video_analysis_complete_event = threading.Event()
        # Event to ensure saving is complete before the next batch item
        self.save_and_reset_complete_event = threading.Event()
        # State to hold the selected batch processing method
        self.batch_processing_method_idx: int = 0
        self.batch_pipeline_preset: str = None
        self.batch_copy_funscript_to_video_location: bool = True
        self.batch_overwrite_mode: int = 0  # 0 for Process All, 1 for Skip Existing
        self.batch_generate_roll_file: bool = True
        self.batch_adaptive_tuning_enabled: bool = False
        self.adaptive_tuning_state: Optional[AdaptiveTuningState] = None
        self.pause_batch_event = threading.Event()

        # --- Audio waveform data ---
        self.audio_waveform_data = None
        self._waveform_lock = threading.Lock()  # Protects audio_waveform_data

        self.app_state_ui.show_timeline_selection_popup = False
        self.app_state_ui.show_timeline_comparison_results_popup = False
        self.app_state_ui.timeline_comparison_results = None
        self.app_state_ui.timeline_comparison_reference_num = 1 # Default to T1 as reference

        # --- Final Setup Steps ---
        self._apply_loaded_settings()
        if not self.is_cli_mode:
            self._load_last_project_on_startup()
        self.energy_saver.reset_activity_timer()

        # Check for updates on startup only if enabled
        if self.app_settings.get("updater_check_on_startup", True):
            self.updater.check_for_updates_async()
            self.addon_checker.check_for_updates_async()

        # Start WebSocket API server if enabled (opt-in)
        self._ws_api = None
        if not self.is_cli_mode and self.app_settings.get("ws_api_enabled", False):
            try:
                from common.ws_api import FunGenWSAPI
                api_port = self.app_settings.get("ws_api_port", 8769)
                self._ws_api = FunGenWSAPI(self, port=api_port)
                self._ws_api.start()
                # Bridge processor playback callbacks to WS event push.
                if self.processor is not None:
                    def _on_playback(is_playing: bool, current_time_ms: float):
                        try:
                            self._ws_api.emit_play(is_playing)
                            self._ws_api.emit_time(current_time_ms)
                        except Exception:
                            pass
                    self.processor.register_playback_state_callback(_on_playback)
            except Exception as e:
                self.logger.warning(f"WebSocket API failed to start: {e}")

        # First-run model download is now handled by the FirstRunWizard (step 5).
        # The wizard calls trigger_first_run_setup() when the user reaches that step.

        # --- Initialize tracker mode from persisted setting; default handled by AppStateUI ---
        if not self.is_cli_mode and self.tracker:
            tracker_name = self.app_state_ui.selected_tracker_name
            if not self.tracker.set_tracking_mode(tracker_name):
                from config.tracker_discovery import get_tracker_discovery, TrackerCategory
                discovery = get_tracker_discovery()
                live_trackers = discovery.get_trackers_by_category(TrackerCategory.LIVE)
                if live_trackers:
                    fallback = live_trackers[0].internal_name
                    self.logger.info(f"Tracker '{tracker_name}' unavailable, falling back to '{fallback}'")
                    self.app_state_ui.selected_tracker_name = fallback
                    self.tracker.set_tracking_mode(fallback)

    def get_timeline(self, timeline_num: int) -> Optional['InteractiveFunscriptTimeline']:
        """
        Retrieves the interactive timeline instance for the given timeline number.
        """
        if timeline_num == 1:
            return getattr(self, 'interactive_timeline1', None)
        elif timeline_num == 2:
            return getattr(self, 'interactive_timeline2', None)
        elif timeline_num >= 3 and hasattr(self, 'gui_instance') and self.gui_instance:
            return self.gui_instance._extra_timeline_editors.get(timeline_num)
        return None

    @staticmethod
    def _purge_old_log_entries(log_file_path: str):
        """Purge log entries older than 7 days. Runs in a background thread."""
        try:
            if not os.path.exists(log_file_path):
                return
            cutoff_date = datetime.now() - timedelta(days=7)
            with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
                all_lines = f.readlines()

            first_line_to_keep_index = -1
            for i, line in enumerate(all_lines):
                try:
                    line_date = datetime.strptime(line[:19], "%Y-%m-%d %H:%M:%S")
                    if line_date >= cutoff_date:
                        first_line_to_keep_index = i
                        break
                except (ValueError, IndexError):
                    continue

            lines_to_keep = all_lines[first_line_to_keep_index:] if first_line_to_keep_index != -1 else []
            with open(log_file_path, 'w', encoding='utf-8') as f:
                if lines_to_keep:
                    f.writelines(lines_to_keep)
        except Exception:
            pass  # Non-critical, swallow silently

    def _configure_third_party_logging(self):
        """Configure third-party library logging to reduce startup noise."""
        # Suppress/reduce noisy third-party library logging
        # Suppress scikit-learn warnings from CoreML Tools before any imports
        import warnings
        warnings.filterwarnings("ignore", message="scikit-learn version .* is not supported")
        
        third_party_loggers = {
            'coremltools': logging.ERROR,  # Only show critical errors from CoreML
            'ultralytics': logging.WARNING,  # Reduce ultralytics noise
            'torch': logging.WARNING,  # Reduce PyTorch noise
            'torchvision': logging.WARNING,  # Reduce torchvision noise
            'requests': logging.WARNING,  # Reduce requests noise
            'urllib3': logging.WARNING,  # Reduce urllib3 noise
            'PIL': logging.WARNING,  # Reduce Pillow noise
            'matplotlib': logging.WARNING,  # Reduce matplotlib noise
        }
        
        for logger_name, level in third_party_loggers.items():
            logging.getLogger(logger_name).setLevel(level)
        
        # Special handling for ultralytics model loading warnings
        ultralytics_logger = logging.getLogger('ultralytics')
        ultralytics_logger.setLevel(logging.ERROR)  # Only show errors from ultralytics
        
        self.logger.debug("Third-party logging configured for reduced startup noise")

    def trigger_first_run_setup(self):
        """Initiates the first-run model download process in a background thread."""
        if self.first_run_thread and self.first_run_thread.is_alive():
            return  # Already running
        self.show_first_run_setup_popup = True
        self.first_run_progress = 0
        self.first_run_status_message = "Starting setup..."
        self.first_run_thread = threading.Thread(target=self._run_first_run_setup_thread, daemon=True, name="FirstRunSetupThread")
        self.first_run_thread.start()

    def _run_first_run_setup_thread(self):
        """The actual logic for downloading and setting up models."""
        try:
            # 1. Create models directory
            models_dir = DEFAULT_MODELS_DIR
            os.makedirs(models_dir, exist_ok=True)
            self.first_run_status_message = f"Created directory: {models_dir}"
            self.logger.info(self.first_run_status_message)

            # 2. Check if user has already selected models
            user_has_detection_model = (self.yolo_detection_model_path_setting and 
                                      os.path.exists(self.yolo_detection_model_path_setting))
            user_has_pose_model = (self.yolo_pose_model_path_setting and 
                                 os.path.exists(self.yolo_pose_model_path_setting))

            # 3. Determine which models to download based on OS
            is_mac_arm = platform.system() == "Darwin" and platform.machine() == 'arm64'

            # --- Download and Process Detection Model (only if user hasn't selected one) ---
            if not user_has_detection_model:
                det_url = MODEL_DOWNLOAD_URLS["detection_pt"]
                det_filename_pt = os.path.basename(det_url)
                det_model_path_pt = os.path.join(models_dir, det_filename_pt)
                self.first_run_status_message = f"Downloading Detection Model: {det_filename_pt}..."
                success = self.utility.download_file_with_progress(det_url, det_model_path_pt, self._update_first_run_progress)

                if not success:
                    self.first_run_status_message = "Detection model download failed."
                    self.first_run_error = True
                    return

                final_det_model_path = det_model_path_pt
                if is_mac_arm:
                    self.first_run_status_message = "Converting detection model to CoreML format..."
                    self.logger.info(f"Running on macOS ARM. Converting {det_filename_pt} to .mlpackage")
                    try:
                        model = YOLO(det_model_path_pt)
                        model.export(format="coreml")
                        final_det_model_path = det_model_path_pt.replace('.pt', '.mlpackage')
                        self.logger.info(f"Successfully converted detection model to {final_det_model_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to convert detection model to CoreML: {e}", exc_info=True)
                        self.first_run_status_message = "Detection model conversion to CoreML failed. Using .pt format."
                        # Continue with the .pt file if conversion fails

                self.app_settings.set("yolo_det_model_path", final_det_model_path)
                self.yolo_detection_model_path_setting = final_det_model_path
                self.yolo_det_model_path = final_det_model_path
                self.logger.info(f"Detection model set to: {final_det_model_path}")
            else:
                self.logger.info(f"User already has detection model selected: {self.yolo_detection_model_path_setting}")

            # --- Download and Process Pose Model (only if user hasn't selected one) ---
            if not user_has_pose_model:
                self.first_run_progress = 0
                pose_url = MODEL_DOWNLOAD_URLS["pose_pt"]
                pose_filename_pt = os.path.basename(pose_url)
                pose_model_path_pt = os.path.join(models_dir, pose_filename_pt)
                self.first_run_status_message = f"Downloading Pose Model: {pose_filename_pt}..."
                success = self.utility.download_file_with_progress(pose_url, pose_model_path_pt, self._update_first_run_progress)

                if not success:
                    self.first_run_status_message = "Pose model download failed."
                    self.first_run_error = True
                    return

                final_pose_model_path = pose_model_path_pt
                if is_mac_arm:
                    self.first_run_status_message = "Converting pose model to CoreML format..."
                    self.logger.info(f"Running on macOS ARM. Converting {pose_filename_pt} to .mlpackage")
                    try:
                        model = YOLO(pose_model_path_pt)
                        model.export(format="coreml")
                        final_pose_model_path = pose_model_path_pt.replace('.pt', '.mlpackage')
                        self.logger.info(f"Successfully converted pose model to {final_pose_model_path}")
                    except Exception as e:
                        self.logger.error(f"Failed to convert pose model to CoreML: {e}", exc_info=True)
                        self.first_run_status_message = "Pose model conversion to CoreML failed. Using .pt format."
                        # Continue with the .pt file if conversion fails

                self.app_settings.set("yolo_pose_model_path", final_pose_model_path)
                self.yolo_pose_model_path_setting = final_pose_model_path
                self.yolo_pose_model_path = final_pose_model_path
                self.logger.info(f"Pose model set to: {final_pose_model_path}")
            else:
                self.logger.info(f"User already has pose model selected: {self.yolo_pose_model_path_setting}")

            self.first_run_status_message = "Setup complete! Please restart the application."
            self.logger.info("Default model setup complete.")
            self.first_run_progress = 100

        except Exception as e:
            self.first_run_status_message = f"An error occurred: {e}"
            self.logger.error(f"First run setup failed: {e}", exc_info=True)

    def _update_first_run_progress(self, percent, downloaded, total_size):
        """Callback to update the progress bar state from the download thread."""
        self.first_run_progress = percent

    def trigger_timeline_comparison(self):
        """
        Initiates the timeline comparison process by showing the selection popup.
        """
        # Reset previous results and open the first dialog
        self.app_state_ui.timeline_comparison_results = None
        self.app_state_ui.show_timeline_selection_popup = True
        self.logger.info("Timeline comparison process started.")

    def run_and_display_comparison_results(self, reference_timeline_num: int):
        """
        Executes the comparison and prepares the results for display.
        Called by the UI after the user selects the reference timeline.
        """
        target_timeline_num = 2 if reference_timeline_num == 1 else 1

        ref_axis = 'primary' if reference_timeline_num == 1 else 'secondary'
        target_axis = 'secondary' if reference_timeline_num == 1 else 'primary'

        self.logger.info(
            f"Running comparison: Reference=T{reference_timeline_num} ({ref_axis}), Target=T{target_timeline_num} ({target_axis})")

        ref_actions = self.funscript_processor.get_actions(ref_axis)
        target_actions = self.funscript_processor.get_actions(target_axis)

        if not ref_actions or not target_actions:
            self.logger.error("Cannot compare signals: one of the timelines has no actions.",
                              extra={'status_message': True})
            return

        comparison_stats = self.funscript_processor.compare_funscript_signals(
            actions_ref=ref_actions,
            actions_target=target_actions,
            prominence=5
        )

        if comparison_stats and comparison_stats.get("error") is None:
            # Store results along with which timeline is the target for applying the offset
            comparison_stats['target_timeline_num'] = target_timeline_num
            self.app_state_ui.timeline_comparison_results = comparison_stats
            self.app_state_ui.show_timeline_comparison_results_popup = True

        elif comparison_stats:
            self.logger.error(f"Funscript comparison failed: {comparison_stats.get('error')}",
                              extra={'status_message': True})
        else:
            self.logger.error("Funscript comparison returned no results.", extra={'status_message': True})

    def start_autotuner(self, force_hwaccel: Optional[str] = None):
        """Initiates the autotuning process in a background thread."""
        self.autotuner.start_autotuner(force_hwaccel)

    def _run_autotuner_thread(self):
        """The actual logic for the autotuning process."""
        self.autotuner._run_autotuner_thread()

    def notify(self, message: str, type: str = "info", duration: float = 4.0):
        """Send a toast notification to the GUI. type: 'success', 'error', 'warning', 'info'."""
        gui = getattr(self, 'gui_instance', None)
        if gui and hasattr(gui, 'notification_manager'):
            gui.notification_manager.add(message, type, duration)

    def get_waveform_data(self):
        """Thread-safe access to audio waveform data."""
        with self._waveform_lock:
            return self.audio_waveform_data

    def get_autotuner_snapshot(self) -> Dict:
        """Thread-safe snapshot of autotuner state for GUI rendering."""
        return self.autotuner.get_autotuner_snapshot()

    def trigger_ultimate_autotune_with_defaults(self, timeline_num: int):
        """
        Non-interactively runs the Ultimate Autotune pipeline with default settings.
        """
        self.autotuner.trigger_ultimate_autotune_with_defaults(timeline_num)

    def _run_post_analysis_pipeline(self, frame_range=None):
        """
        Run post-analysis processing after tracking completes.
        Checks auto_apply_post_processing setting, then runs:
        1. CLI --pipeline preset (if specified), OR
        2. GUI pipeline steps (if any configured), OR
        3. Ultimate Autotune (legacy batch flag)
        Returns True if any processing was applied.
        """
        # CLI pipeline preset takes priority
        pipeline_preset = getattr(self, 'batch_pipeline_preset', None)
        if pipeline_preset:
            self.logger.info(f"Applying CLI pipeline preset '{pipeline_preset}' after analysis.")
            from application.classes.plugin_pipeline import PluginPipeline
            pipeline = PluginPipeline(self)
            if pipeline.load_preset(pipeline_preset):
                funscript_obj = self.funscript_processor.get_funscript_obj()
                if funscript_obj:
                    success, errors = pipeline.run_with_target(funscript_obj)
                    for err in errors:
                        self.logger.warning(f"Pipeline: {err}")
                    return success
            else:
                self.logger.warning(f"Pipeline preset '{pipeline_preset}' not found.")
            return False

        # Batch mode: use batch autotune flag
        if self.is_batch_processing_active:
            if self.batch_apply_ultimate_autotune:
                self.logger.info("Applying Ultimate Autotune for batch processing.")
                self.trigger_ultimate_autotune_with_defaults(timeline_num=1)
                return True
            return False

        # Interactive mode: check auto_apply_post_processing setting
        if not self.app_settings.get("auto_apply_post_processing", True):
            self.logger.info("Auto post-processing disabled, skipping.")
            return False

        funscript_obj = self.funscript_processor.get_funscript_obj()
        if not funscript_obj:
            return False

        # Per-axis preset assignments (e.g. {"T1": "Full Enhancement", "T2": "Light Polish"})
        assignments = self.app_settings.get("auto_pipeline_assignments", {})
        if assignments:
            from application.classes.plugin_pipeline import PluginPipeline, timeline_label_to_axis
            any_applied = False
            for axis_label, preset_name in assignments.items():
                if not preset_name:
                    continue
                pipeline = PluginPipeline(self)
                if pipeline.load_preset(preset_name):
                    axis_name = timeline_label_to_axis(axis_label, funscript_obj)
                    self.logger.info(f"Auto pipeline: '{preset_name}' on {axis_label} ({axis_name})")
                    success, errors = pipeline.run(funscript_obj, axis=axis_name)
                    for err in errors:
                        self.logger.warning(f"Pipeline ({axis_label}): {err}")
                    any_applied = any_applied or success
                else:
                    self.logger.warning(f"Auto pipeline: preset '{preset_name}' not found for {axis_label}")
            if any_applied:
                return True

        # Check if GUI pipeline has steps configured (single pipeline with target_axis)
        gui = getattr(self, 'gui_instance', None)
        if gui and hasattr(gui, 'plugin_pipeline_ui'):
            pipeline = gui.plugin_pipeline_ui.pipeline
            enabled_steps = [s for s in pipeline.steps if s.enabled]
            if enabled_steps:
                self.logger.info(f"Running pipeline ({len(enabled_steps)} steps, target: {pipeline.target_axis}) after analysis.")
                success, errors = pipeline.run_with_target(funscript_obj)
                for err in errors:
                    self.logger.warning(f"Pipeline: {err}")
                return success

        # Fallback: run Ultimate Autotune as default post-processing
        self.logger.info("Running default Ultimate Autotune after analysis.")
        self.trigger_ultimate_autotune_with_defaults(timeline_num=1)
        return True

    def toggle_file_manager_window(self):
        """Toggles the visibility of the Generated File Manager window."""
        if hasattr(self, 'app_state_ui'):
            self.app_state_ui.show_generated_file_manager = not self.app_state_ui.show_generated_file_manager

    def unload_model(self, model_type: str):
        """Clears the path for a given model type and releases it from the tracker."""
        self.model_manager.unload_model(model_type)

    def generate_waveform(self):
        if not self.processor or not self.processor.is_video_open():
            self.logger.info("Cannot generate waveform: No video loaded.", extra={'status_message': True})
            return

        def _generate_waveform_thread():
            self.logger.info("Generating audio waveform...", extra={'status_message': True})

            # If subtitle audio cache exists, read it directly instead of re-extracting from video
            # (avoids the slow USB extraction that times out at 60s)
            waveform_data = None
            try:
                import os, numpy as np
                video_path = getattr(self.file_manager, 'video_path', '') or ''
                cached_audio = os.path.splitext(video_path)[0] + ".sub_audio.wav" if video_path else ''
                if cached_audio and os.path.exists(cached_audio) and os.path.getsize(cached_audio) > 10000:
                    import wave
                    with wave.open(cached_audio, 'rb') as wf:
                        sr = wf.getframerate()
                        raw = wf.readframes(wf.getnframes())
                    samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    # Downsample to 2000 points
                    chunk_size = max(1, len(samples) // 2000)
                    n = (len(samples) // chunk_size) * chunk_size
                    blocks = np.abs(samples[:n].reshape(-1, chunk_size))
                    waveform_data = np.max(blocks, axis=1).astype(np.float32)
                    self.logger.info(f"Waveform from cached subtitle audio ({len(waveform_data)} samples)")
            except Exception as e:
                self.logger.debug(f"Cached audio waveform failed: {e}")

            if waveform_data is None:
                waveform_data = self.processor.get_audio_waveform(num_samples=2000)

            with self._waveform_lock:
                self.audio_waveform_data = waveform_data

            if waveform_data is not None:
                self.logger.info("Audio waveform generated successfully.", extra={'status_message': True})
                self.app_state_ui.show_audio_waveform = True
            else:
                self.logger.error("Failed to generate audio waveform.", extra={'status_message': True})
                self.app_state_ui.show_audio_waveform = False

        thread = threading.Thread(target=_generate_waveform_thread, daemon=True, name="WaveformGenThread")
        thread.start()

    def toggle_waveform_visibility(self):
        if not self.app_state_ui.show_audio_waveform and self.get_waveform_data() is None:
            self.generate_waveform()
        else:
            self.app_state_ui.show_audio_waveform = not self.app_state_ui.show_audio_waveform
            status = "enabled" if self.app_state_ui.show_audio_waveform else "disabled"
            self.logger.info(f"Audio waveform display {status}.", extra={'status_message': True})

    # --- Batch Processing (delegated to AppBatchProcessor) ---

    def start_batch_processing(self, video_paths: List[str]):
        self.batch_processor.start_batch_processing(video_paths)

    def _initiate_batch_processing_from_confirmation(self):
        self.batch_processor._initiate_batch_processing_from_confirmation()

    def _cancel_batch_processing_from_confirmation(self):
        self.batch_processor._cancel_batch_processing_from_confirmation()

    def abort_batch_processing(self):
        self.batch_processor.abort_batch_processing()

    def pause_batch_processing(self):
        self.batch_processor.pause_batch_processing()

    def resume_batch_processing(self):
        self.batch_processor.resume_batch_processing()

    @property
    def is_batch_paused(self) -> bool:
        return self.batch_processor.is_batch_paused()

    def _run_batch_processing_thread(self):
        self.batch_processor._run_batch_processing_thread()

    # --- ROI Manager delegation (see app_roi_manager.py) ---

    def enter_set_user_roi_mode(self):
        self.roi_manager.enter_set_user_roi_mode()

    def exit_set_user_roi_mode(self):
        self.roi_manager.exit_set_user_roi_mode()

    def user_roi_and_point_set(self, roi_rect_video_coords: Tuple[int, int, int, int], point_video_coords: Tuple[int, int]):
        self.roi_manager.user_roi_and_point_set(roi_rect_video_coords, point_video_coords)

    def clear_all_overlays_and_ui_drawings(self) -> None:
        self.roi_manager.clear_all_overlays_and_ui_drawings()

    def enter_set_oscillation_area_mode(self):
        self.roi_manager.enter_set_oscillation_area_mode()

    def exit_set_oscillation_area_mode(self):
        self.roi_manager.exit_set_oscillation_area_mode()

    def oscillation_area_and_point_set(self, area_rect_video_coords: Tuple[int, int, int, int], point_video_coords: Tuple[int, int]):
        self.roi_manager.oscillation_area_and_point_set(area_rect_video_coords, point_video_coords)

    def set_pending_action_after_tracking(self, action_type: str, **kwargs):
        """Stores information about an action to be performed after tracking."""
        self.pending_action_after_tracking = {"type": action_type, "data": kwargs}
        self.logger.info(f"Pending action set after tracking: {action_type} with data {kwargs}")

    def clear_pending_action_after_tracking(self):
        """Clears any pending action."""
        if self.pending_action_after_tracking:
            self.logger.info(f"Cleared pending action: {self.pending_action_after_tracking.get('type')}")
        self.pending_action_after_tracking = None

    def on_offline_analysis_completed(self, payload: Dict):
        """
        Handles the finalization of a completed offline analysis run (2-Stage or 3-Stage).
        This includes saving raw and final funscripts, applying post-processing,
        and handling batch mode tasks.
        """
        video_path = payload.get("video_path")
        chapters_for_save_from_payload = payload.get("video_segments")

        if not video_path:
            self.logger.warning("Completion event is missing its video path. Cannot save funscripts.")
            # Still need to signal batch processing to avoid a hang
            if self.is_batch_processing_active:
                self.save_and_reset_complete_event.set()
            return

        # The chapter list is now the single source of truth from funscript_processor,
        # which was populated by the stage2_results_success event.
        chapters_for_save = self.funscript_processor.video_chapters

        # 1. SAVE THE RAW FUNSCRIPT
        self.logger.info("Offline analysis completed. Saving raw funscript before post-processing.")
        self.file_manager.save_raw_funscripts_after_generation(video_path)

        # 2. PROCEED WITH POST-PROCESSING
        any_processing_applied = self._run_post_analysis_pipeline()

        # Notify user of completion
        action_count = len(self.funscript_processor.get_actions('primary') or [])
        self.notify(f"Analysis complete - {action_count} points generated", "success")
        
        if any_processing_applied:
            self.logger.info("Saving final (post-processed) funscripts...")
            self.file_manager.save_final_funscripts(video_path, chapters=chapters_for_save)
        else:
            self.logger.info("No post-processing was applied. Saving raw funscript with .raw.funscript extension to video location.")
            self.file_manager.save_raw_funscripts_next_to_video(video_path)

        # 5. SAVE THE PROJECT
        self.logger.info("Saving project file for completed video...")
        project_filepath = self.file_manager.get_output_path_for_file(video_path, PROJECT_FILE_EXTENSION)
        self.project_manager.save_project(project_filepath)

        # 6. Signal batch loop to continue
        if self.is_batch_processing_active and hasattr(self, 'save_and_reset_complete_event'):
            self.logger.debug("Signaling batch loop to continue after offline analysis completion.")
            self.save_and_reset_complete_event.set()

        # If in CLI mode without a GUI, we must manually reset the project state for the next video
        if not self.gui_instance and self.is_batch_processing_active:
            self.logger.info("CLI Mode: Resetting project state for next video in batch.")
            self.reset_project_state(for_new_project=False)

    def on_processing_stopped(self, was_scripting_session: bool = False, scripted_frame_range: Optional[Tuple[int, int]] = None):
        """
        Called when video processing (tracking, playback) stops or completes.
        This now handles post-processing for live tracking sessions.
        """
        self.logger.debug(
            f"on_processing_stopped triggered. Was scripting: {was_scripting_session}, Range: {scripted_frame_range}")

        # Handle pending actions like merge-gap first
        if self.pending_action_after_tracking:
            action_info = self.pending_action_after_tracking
            self.clear_pending_action_after_tracking()
            self.clear_pending_action_after_tracking()
            self.logger.info(f"Processing pending action: {action_info['type']}")
            action_type = action_info['type']
            action_data = action_info['data']
            if action_type == 'finalize_gap_merge_after_tracking':
                chapter1_id = action_data.get('chapter1_id')
                chapter2_id = action_data.get('chapter2_id')
                if not all([chapter1_id, chapter2_id]):
                    self.logger.error(f"Missing data for finalize_gap_merge_after_tracking: {action_data}")
                    return
                if hasattr(self.funscript_processor, 'finalize_merge_after_gap_tracking'):
                    self.funscript_processor.finalize_merge_after_gap_tracking(chapter1_id, chapter2_id)
                else:
                    self.logger.error("FunscriptProcessor missing finalize_merge_after_gap_tracking method.")
            else:
                self.logger.warning(f"Unknown pending action type: {action_type}")

        # If this was a live scripting session, save the raw script first.
        if was_scripting_session:
            video_path = self.file_manager.video_path
            if video_path:
                # 1. SAVE THE RAW FUNSCRIPT
                self.logger.info("Live session ended. Saving raw funscript before post-processing.")
                self.file_manager.save_raw_funscripts_after_generation(video_path)

                # CRITICAL FIX: Ensure timeline cache reflects final live tracking data
                # This prevents UA "points disappearing" bug when clicked right after generation
                timeline1 = getattr(self, 'interactive_timeline1', None)
                if timeline1 and hasattr(timeline1, 'invalidate_cache'):
                    timeline1.invalidate_cache()
                    self.logger.debug("Timeline 1 cache invalidated after live session completion")
                timeline2 = getattr(self, 'interactive_timeline2', None)
                if timeline2 and hasattr(timeline2, 'invalidate_cache'):
                    timeline2.invalidate_cache()
                    self.logger.debug("Timeline 2 cache invalidated after live session completion")
                # Invalidate extra timeline caches (3+)
                if hasattr(self, 'gui_instance') and self.gui_instance:
                    for t_num, editor in getattr(self.gui_instance, '_extra_timeline_editors', {}).items():
                        if hasattr(editor, 'invalidate_cache'):
                            editor.invalidate_cache()
                            self.logger.debug(f"Timeline {t_num} cache invalidated after live session completion")

                # 2. PROCEED WITH POST-PROCESSING
                any_processing_applied = self._run_post_analysis_pipeline(frame_range=scripted_frame_range)

                # 3. SAVE THE FINAL FUNSCRIPT

                if any_processing_applied:
                    self.logger.info("Saving final (post-processed) funscript.")
                    chapters_for_save = self.funscript_processor.video_chapters
                    self.file_manager.save_final_funscripts(video_path, chapters=chapters_for_save)
                else:
                    self.logger.info("No post-processing was applied to live session. Saving raw funscript with .raw.funscript extension to video location.")
                    self.file_manager.save_raw_funscripts_next_to_video(video_path)

            else:
                self.logger.warning("Live session ended, but no video path is available to save the raw funscript.")

    def _cache_tracking_classes(self):
        """Temporarily loads the detection model to get class names, then unloads it."""
        self.model_manager._cache_tracking_classes()

    def get_available_tracking_classes(self) -> List[str]:
        """Gets the list of class names from the model (cached)."""
        return self.model_manager.get_available_tracking_classes()

    def set_status_message(self, message: str, duration: float = 3.0, level: int = logging.INFO):
        if hasattr(self, 'app_state_ui') and self.app_state_ui is not None:
            self.app_state_ui.status_message = message
            self.app_state_ui.status_message_time = time.time() + duration
        else:
            print(f"Debug Log (app_state_ui not set): Status: {message}")

    def _get_target_funscript_details(self, timeline_num: int) -> Tuple[Optional[object], Optional[str]]:
        """
        Returns the core Funscript object and the axis name ('primary' or 'secondary')
        based on the timeline number.
        This is used by InteractiveFunscriptTimeline to know which data to operate on.
        """
        if self.processor and self.processor.tracker and self.processor.tracker.funscript:
            funscript_obj = self.processor.tracker.funscript
            if timeline_num == 1:
                return funscript_obj, 'primary'
            elif timeline_num == 2:
                return funscript_obj, 'secondary'
        return None, None

    def _query_hwaccels_background(self):
        """Background thread: query ffmpeg, then validate the configured method."""
        queried = self._get_available_ffmpeg_hwaccels()
        self.available_ffmpeg_hwaccels = queried
        # Cache for next startup so _apply_loaded_settings can validate immediately
        self.app_settings.set("available_ffmpeg_hwaccels", queried)

        default_hw = "auto"
        if "auto" not in queried:
            default_hw = "none" if "none" in queried else (queried[0] if queried else "none")

        current = self.app_settings.get("hardware_acceleration_method", default_hw)
        if current not in queried:
            self.logger.warning(
                f"Configured hardware acceleration '{current}' not listed by ffmpeg "
                f"({queried}). Falling back to '{default_hw}'.")
            self.hardware_acceleration_method = default_hw
            self.app_settings.set("hardware_acceleration_method", default_hw)
        else:
            self.hardware_acceleration_method = current
        self._hwaccel_query_done.set()

    def _get_available_ffmpeg_hwaccels(self) -> List[str]:
        """Queries FFmpeg for available hardware acceleration methods."""
        try:
            ffmpeg_path = self.app_settings.get("ffmpeg_path") or "ffmpeg"
            result = subprocess.run(
                [ffmpeg_path, '-hide_banner', '-hwaccels'],
                capture_output=True, text=True, check=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            hwaccels = []
            if lines and "Hardware acceleration methods:" in lines[0]:
                hwaccels = [line.strip() for line in lines[1:] if line.strip() and line.strip() != "none"]

            standard_options = ["auto", "none"]
            unique_hwaccels = [h for h in hwaccels if h not in standard_options]
            final_options = standard_options + unique_hwaccels
            log_func = self.logger.debug if hasattr(self, 'logger') and self.logger else print
            log_func(f"Available FFmpeg hardware accelerations: {final_options}")
            return final_options
        except FileNotFoundError:
            log_func = self.logger.error if hasattr(self, 'logger') and self.logger else print
            log_func("ffmpeg not found. Hardware acceleration detection failed.")
            return ["auto", "none"]
        except Exception as e:
            log_func = self.logger.error if hasattr(self, 'logger') and self.logger else print
            log_func(f"Error querying ffmpeg for hwaccels: {e}")
            return ["auto", "none"]

    def _check_model_paths(self):
        """Checks essential model paths and auto-downloads if missing."""
        return self.model_manager._check_model_paths()

    def set_application_logging_level(self, level_name: str):
        """Sets the application-wide logging level (root logger + all handlers)."""
        numeric_level = getattr(logging, level_name.upper(), None)
        if numeric_level is not None and hasattr(self, '_logger_instance'):
            self._logger_instance.set_level(numeric_level)
            self.logging_level_setting = level_name
            self.logger.info(f"Logging level changed to: {level_name}", extra={'status_message': True})
        else:
            self.logger.warning(f"Failed to set logging level or invalid level: {level_name}")

    def _apply_loaded_settings(self):
        """Applies all settings from AppSettings to their respective modules/attributes."""
        self.logger.debug("Applying loaded settings...")
        defaults = self.app_settings.get_default_settings()

        self.discarded_tracking_classes = self.app_settings.get("discarded_tracking_classes", defaults.get("discarded_tracking_classes")) or []

        # Logging Level
        new_logging_level = self.app_settings.get("logging_level", defaults.get("logging_level")) or "INFO"
        if self.logging_level_setting != new_logging_level:
            self.set_application_logging_level(new_logging_level)

        # Hardware Acceleration, uses cached list from last run (or background
        # thread result if it finished first), so no blocking wait needed
        default_hw_accel_in_apply = "auto"
        if "auto" not in self.available_ffmpeg_hwaccels:
            default_hw_accel_in_apply = "none" if "none" in self.available_ffmpeg_hwaccels else \
                (self.available_ffmpeg_hwaccels[0] if self.available_ffmpeg_hwaccels else "none")
        loaded_hw_method = self.app_settings.get("hardware_acceleration_method", defaults.get("hardware_acceleration_method")) or default_hw_accel_in_apply
        if loaded_hw_method not in self.available_ffmpeg_hwaccels:
            self.logger.warning(
                f"Hardware acceleration method '{loaded_hw_method}' from settings is not currently available "
                f"({self.available_ffmpeg_hwaccels}). Resetting to '{default_hw_accel_in_apply}'.")
            self.hardware_acceleration_method = default_hw_accel_in_apply
        else:
            self.hardware_acceleration_method = loaded_hw_method

        # Models
        self.yolo_detection_model_path_setting = self.app_settings.get("yolo_det_model_path", defaults.get("yolo_det_model_path"))
        self.yolo_pose_model_path_setting = self.app_settings.get("yolo_pose_model_path", defaults.get("yolo_pose_model_path"))

        # Update actual model paths used by tracker/processor if they changed
        if self.yolo_det_model_path != self.yolo_detection_model_path_setting:
            self.yolo_det_model_path = self.yolo_detection_model_path_setting or ""
            if self.tracker: self.tracker.det_model_path = self.yolo_det_model_path
            self.logger.info(
                f"Detection model path updated from settings: {os.path.basename(self.yolo_det_model_path or '')}")
        if self.yolo_pose_model_path != self.yolo_pose_model_path_setting:
            self.yolo_pose_model_path = self.yolo_pose_model_path_setting or ""
            if self.tracker: self.tracker.pose_model_path = self.yolo_pose_model_path
            self.logger.info(
                f"Pose model path updated from settings: {os.path.basename(self.yolo_pose_model_path or '')}")

        # Inform sub-modules to update their settings
        # TODO: Refactor this to use tuple unpacking
        self.app_state_ui.update_settings_from_app()
        self.file_manager.update_settings_from_app()
        self.stage_processor.update_settings_from_app()
        self.energy_saver.update_settings_from_app()
        self.energy_saver.reset_activity_timer()

    def save_app_settings(self):
        """Saves current application settings to file via AppSettings."""
        self.logger.debug("Saving application settings...")

        # Core settings directly on AppLogic
        self.app_settings.set("hardware_acceleration_method", self.hardware_acceleration_method)
        self.app_settings.set("yolo_det_model_path", self.yolo_detection_model_path_setting)
        self.app_settings.set("yolo_pose_model_path", self.yolo_pose_model_path_setting)
        self.app_settings.set("discarded_tracking_classes", self.discarded_tracking_classes)

        # Call save methods on sub-modules
        # TODO: Refactor this to use tuple unpacking
        self.app_state_ui.save_settings_to_app()
        self.file_manager.save_settings_to_app()
        self.stage_processor.save_settings_to_app()
        self.energy_saver.save_settings_to_app()
        self.app_settings.save_settings()
        self.logger.info("Application settings saved.", extra={'status_message': True})
        self.energy_saver.reset_activity_timer()

    def _load_last_project_on_startup(self):
        """Checks for and loads the most recently used project on application start."""
        self.logger.debug("Checking for last opened project...")

        # Read from the new dedicated setting, not the recent projects list.
        last_project_path = self.app_settings.get("last_opened_project_path")

        if not last_project_path:
            self.logger.debug("No last project found to load. Starting fresh.")
            return

        if os.path.exists(last_project_path):
            try:
                self.logger.debug(f"Loading last opened project: {last_project_path}")
                self.project_manager.load_project(last_project_path)
            except Exception as e:
                self.logger.error(f"Failed to load last project '{last_project_path}': {e}", exc_info=True)
                # Clear the invalid path so it doesn't try again next time
                self.app_settings.set("last_opened_project_path", None)
        else:
                self.logger.warning(f"Last project file not found: '{last_project_path}'. Clearing setting.")
                # Clear the missing path so it doesn't try again next time
                self.app_settings.set("last_opened_project_path", None)
    

    def reset_project_state(self, for_new_project: bool = True):
        """Resets the application to a clean state for a new or loaded project."""
        self.logger.debug(f"Resetting project state ({'new project' if for_new_project else 'project load'})...")

        # Preserve current bar visibility states
        prev_show_heatmap = getattr(self.app_state_ui, 'show_heatmap', True)
        prev_show_funscript_timeline = getattr(self.app_state_ui, 'show_funscript_timeline', True)

        # Stop any active processing
        if self.processor and self.processor.is_processing: self.processor.stop_processing()
        if self.stage_processor.full_analysis_active: self.stage_processor.abort_stage_processing()  # Signals thread

        self.file_manager.close_video_action(clear_funscript_unconditionally=True, skip_tracker_reset=(not for_new_project))
        self.funscript_processor.reset_state_for_new_project()
        self.funscript_processor.update_funscript_stats_for_timeline(1, "Project Reset")
        self.funscript_processor.update_funscript_stats_for_timeline(2, "Project Reset")

        # Reset waveform data
        with self._waveform_lock:
            self.audio_waveform_data = None
        self.app_state_ui.show_audio_waveform = False

        # Reset UI states to defaults (or app settings defaults)
        app_settings_defaults = self.app_settings.get_default_settings()
        self.app_state_ui.timeline_pan_offset_ms = self.app_settings.get("timeline_pan_offset_ms", app_settings_defaults.get("timeline_pan_offset_ms", 0.0))
        self.app_state_ui.timeline_zoom_factor_ms_per_px = self.app_settings.get("timeline_zoom_factor_ms_per_px", app_settings_defaults.get("timeline_zoom_factor_ms_per_px", 20.0))

        self.app_state_ui.show_funscript_interactive_timeline = self.app_settings.get(
            "show_funscript_interactive_timeline",
            app_settings_defaults.get("show_funscript_interactive_timeline", True))
        self.app_state_ui.show_funscript_interactive_timeline2 = self.app_settings.get(
            "show_funscript_interactive_timeline2",
            app_settings_defaults.get("show_funscript_interactive_timeline2", False))
        self.app_state_ui.show_heatmap = self.app_settings.get("show_heatmap", app_settings_defaults.get("show_heatmap", True))
        self.app_state_ui.show_stage2_overlay = self.app_settings.get("show_stage2_overlay", app_settings_defaults.get("show_stage2_overlay", True))
        self.app_state_ui.reset_video_zoom_pan()

        # Reset model paths to current app_settings (in case project had different ones)
        self.yolo_detection_model_path_setting = self.app_settings.get("yolo_det_model_path")
        self.yolo_det_model_path = self.yolo_detection_model_path_setting
        self.yolo_pose_model_path_setting = self.app_settings.get("yolo_pose_model_path")
        self.yolo_pose_model_path = self.yolo_pose_model_path_setting
        if self.tracker:
            self.tracker.det_model_path = self.yolo_det_model_path
            self.tracker.pose_model_path = self.yolo_pose_model_path

        # Clear undo history for both timelines
        self.undo_manager.clear()
        self.app_state_ui.heatmap_dirty = True
        self.app_state_ui.funscript_preview_dirty = True
        self.app_state_ui.force_timeline_pan_to_current_frame = True

        # Restore previous bar visibility states
        if hasattr(self.app_state_ui, 'show_heatmap'):
            self.app_state_ui.show_heatmap = prev_show_heatmap
        if hasattr(self.app_state_ui, 'show_funscript_timeline'):
            self.app_state_ui.show_funscript_timeline = prev_show_funscript_timeline

        if for_new_project:
            self.logger.info("New project state initialized.", extra={'status_message': True})
        self.energy_saver.reset_activity_timer()

    def _map_shortcut_to_glfw_key(self, shortcut_string_to_parse: str) -> Optional[Tuple[int, dict]]:
        """
        Parses a shortcut string (e.g., "CTRL+SHIFT+A") into a GLFW key code
        and a dictionary of modifiers. Results are cached to avoid string
        parsing overhead on every frame.
        """
        if not shortcut_string_to_parse:
            return None

        # Check cache first (avoids string parsing every frame)
        if shortcut_string_to_parse in self._shortcut_mapping_cache:
            return self._shortcut_mapping_cache[shortcut_string_to_parse]

        # Parse the shortcut string
        parts = shortcut_string_to_parse.upper().split('+')
        modifiers = {'ctrl': False, 'alt': False, 'shift': False, 'super': False}
        main_key_str = None

        for part_val in parts:
            part_cleaned = part_val.strip()
            if part_cleaned == "CTRL":
                modifiers['ctrl'] = True
            elif part_cleaned == "ALT":
                modifiers['alt'] = True
            elif part_cleaned == "SHIFT":
                modifiers['shift'] = True
            elif part_cleaned == "SUPER":
                modifiers['super'] = True
            else:
                if main_key_str is not None:
                    self._shortcut_mapping_cache[shortcut_string_to_parse] = None
                    return None
                main_key_str = part_cleaned

        if main_key_str is None:
            self._shortcut_mapping_cache[shortcut_string_to_parse] = None
            return None

        if not self.shortcut_manager:
            return None

        glfw_key_code = self.shortcut_manager.name_to_glfw_key(main_key_str)
        if glfw_key_code is None:
            self._shortcut_mapping_cache[shortcut_string_to_parse] = None
            return None

        result = (glfw_key_code, modifiers)
        self._shortcut_mapping_cache[shortcut_string_to_parse] = result
        return result

    def invalidate_shortcut_cache(self):
        """Clear the shortcut mapping cache. Call this when shortcuts are modified."""
        self._shortcut_mapping_cache.clear()

    def get_effective_video_duration_params(self) -> Tuple[float, int, float]:
        """
        Retrieves effective video duration, total frames, and FPS.
        Uses processor.video_info if available, otherwise falls back to
        primary funscript data for duration.
        """
        duration_s: float = 0.0
        total_frames: int = 0
        fps_val: float = 30.0  # Default FPS

        if self.processor and self.processor.video_info:
            duration_s = self.processor.video_info.get('duration', 0.0)
            total_frames = self.processor.video_info.get('total_frames', 0)
            fps_val = self.processor.video_info.get('fps', 30.0)
            if fps_val <= 0: fps_val = 30.0
        elif self.processor and self.processor.tracker and self.processor.tracker.funscript and self.processor.tracker.funscript.primary_actions:
            try:
                duration_s = self.processor.tracker.funscript.primary_actions[-1]['at'] / 1000.0
            except Exception:
                duration_s = 0.0
        return duration_s, total_frames, fps_val


    def run_cli(self, args):
        """Handles the application's command-line interface logic. Delegated to AppCLIRunner."""
        return self.cli_runner.run_cli(args)

    def shutdown_app(self):
        """Gracefully shuts down application components."""
        self.logger.info("Shutting down application logic...")

        # Stop stage processing threads
        self.stage_processor.shutdown_app_threads()

        # Stop video processing if active
        if self.processor and self.processor.is_processing:
            self.processor.stop_processing(join_thread=True)  # Ensure thread finishes

        # Perform autosave on shutdown if enabled and dirty
        if self.app_settings.get("autosave_on_exit", True) and \
                self.app_settings.get("autosave_enabled", True) and \
                self.project_manager.project_dirty:
            self.logger.info("Performing final autosave on exit...")
            self.project_manager.perform_autosave()

        # Stop mpv review mode if active
        if self._mpv_controller and self._mpv_controller.is_active:
            self._mpv_controller.stop()

        # Stop audio playback and persist volume to settings
        if hasattr(self, '_audio_volume_live'):
            self.app_settings.set("audio_volume", self._audio_volume_live)
        if self._audio_sync:
            self._audio_sync.stop()
        if self._audio_player:
            self._audio_player.cleanup()

        # Any other cleanup (e.g. closing files, releasing resources)
        # self.app_settings.save_settings() # Settings usually saved explicitly by user or before critical changes

        self.logger.info("Application logic shutdown complete.")

    def download_default_models(self):
        """Manually download default models if they don't exist."""
        self.model_manager.download_default_models()

    def _run_funscript_cli_mode(self, args):
        """Handles CLI funscript processing mode. Delegated to AppCLIRunner."""
        return self.cli_runner._run_funscript_cli_mode(args)

    def _generate_filtered_funscript_path(self, original_path, filter_name, overwrite):
        """Generate output path for filtered funscript. Delegated to AppCLIRunner."""
        return self.cli_runner._generate_filtered_funscript_path(original_path, filter_name, overwrite)

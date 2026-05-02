import time
import logging
import os
import threading
from typing import Optional, Dict, Tuple, List, Any

from video import VideoProcessor
from tracker.tracker_manager import create_tracker_manager

from application.classes import AppSettings, ProjectManager, ShortcutManager
from application.classes.undo_manager import UndoManager
from application.utils import AppLogger, AutoUpdater
from application.utils.addon_update_checker import AddonUpdateChecker
from config.constants import YOLO_INPUT_SIZE
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
        self.logging_level_setting = self.app_settings.config.logging.level

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
        discarded_tracking_classes = self.app_settings.config.tracking.discarded_classes
        if discarded_tracking_classes is None:
            discarded_tracking_classes = []
        self.discarded_tracking_classes: List[str] = discarded_tracking_classes
        self.pending_action_after_tracking: Optional[Dict] = None

        # Sub-managers. Imported inline so their transitive imports don't
        # weigh on app_logic module load.
        from application.logic.tracking_lifecycle import TrackingLifecycleController
        from application.logic.settings_lifecycle import SettingsLifecycleController
        from application.logic.project_lifecycle import ProjectLifecycleController
        self._tracking_lifecycle = TrackingLifecycleController(self)
        self._settings_lifecycle = SettingsLifecycleController(self)
        self._project_lifecycle = ProjectLifecycleController(self)

        self.app_state_ui = AppStateUI(self)
        self.utility = AppUtility(self)

        # --- State for first-run setup ---
        self.show_first_run_setup_popup = False
        self.first_run_progress = 0.0
        self.first_run_status_message = ""
        self.first_run_error = False
        self.first_run_thread: Optional[threading.Thread] = None
        # Controller holds the download/convert thread logic. Import here
        # (not top of module) so `ultralytics.YOLO` stays off the startup path.
        from application.logic.first_run_setup import FirstRunSetupController
        self._first_run_setup = FirstRunSetupController(self)

        # --- Hardware Acceleration ---
        # Load cached hwaccel list from settings so validation works immediately;
        # background thread refreshes the cache from ffmpeg
        cached_hwaccels = self.app_settings.config.performance.available_ffmpeg_hwaccels
        self.available_ffmpeg_hwaccels = cached_hwaccels if cached_hwaccels else ["auto", "none"]
        self.hardware_acceleration_method = self.app_settings.config.performance.hardware_acceleration_method or "auto"
        self._hwaccel_query_done = threading.Event()
        from application.logic.hardware_accel import HardwareAccelController
        self._hwaccel = HardwareAccelController(self)
        threading.Thread(
            target=self._hwaccel.query_background,
            daemon=True,
            name="HWAccelQuery",
        ).start()

        # --- Tracking Axis Configuration (ensure these are initialized before tracker if tracker uses them in __init__) ---
        tracking_cfg = self.app_settings.config.tracking
        self.tracking_axis_mode = tracking_cfg.axis_mode
        self.single_axis_output_target = tracking_cfg.single_axis_output_target

        # --- Models ---
        self.yolo_detection_model_path_setting = self.app_settings.config.models.yolo_det_path
        self.yolo_pose_model_path_setting = self.app_settings.config.models.yolo_pose_path
        self.yolo_det_model_path = self.yolo_detection_model_path_setting
        self.yolo_pose_model_path = self.yolo_pose_model_path_setting
        self.yolo_input_size = YOLO_INPUT_SIZE

        # --- Undo/Redo ---
        self.undo_manager = UndoManager(max_history=100)

        # --- Initialize Tracker Manager ---
        self.tracker = create_tracker_manager(
            app_logic_instance=self,
            tracker_model_path=self.yolo_detection_model_path_setting)

        if self.tracker:
            self.tracker.show_stats = False  # Default internal tracker states
            self.tracker.show_funscript_preview = False

        # --- NOW Sync Tracker UI Flags as tracker and app_state_ui exist ---
        self.app_state_ui.sync_tracker_ui_flags()

        # --- Initialize Processor (after tracker and logger/app_state_ui are ready) ---
        self.model_manager = AppModelManager(self)
        self._check_model_paths()

        self.processor = VideoProcessor(self, self.tracker, yolo_input_size=self.yolo_input_size, cache_size=1000)

        # --- Modular Components Initialization ---
        # VideoSession before file_manager: file_manager.open_video_from_path
        # + .close_video_action are now thin delegators to video_session.
        from application.logic.video_session import VideoSession
        self.video_session = VideoSession(self)
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
        audio_cfg = self.app_settings.config.audio
        if not self.is_cli_mode and SOUNDDEVICE_AVAILABLE and audio_cfg.enabled:
            try:
                self._audio_player = AudioPlayer()
                self._audio_sync = AudioVideoSync(self.processor, self._audio_player, self)
                self._audio_player.set_volume(audio_cfg.volume)
                self._audio_player.set_mute(audio_cfg.muted)
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
        from application.logic.shortcut_mapper import ShortcutMapper
        self._shortcut_mapper = ShortcutMapper(self.shortcut_manager)

        # Pattern Library
        try:
            from funscript.pattern_library import PatternLibrary
            self.pattern_library = PatternLibrary()
        except Exception as e:
            self.logger.warning(f"PatternLibrary init failed: {e}")
            self.pattern_library = None

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
        self.oscillation_grid_size = self.app_settings.config.tracking.oscillation_grid_size
        self.oscillation_sensitivity = self.app_settings.config.tracking.oscillation_sensitivity

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
        if self.app_settings.config.updater.check_on_startup:
            self.updater.check_for_updates_async()
            self.addon_checker.check_for_updates_async()

        # Start WebSocket API server if enabled (opt-in)
        self._ws_api = None
        _ws = self.app_settings.config.ws_api
        if not self.is_cli_mode and _ws.enabled:
            try:
                from common.ws_api import FunGenWSAPI
                api_port = _ws.port
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
        # GUI startup uses lazy=True so the YOLO model load + warmup forward
        # (~1-2 s on MPS) is deferred until the user actually hits Start. CLI
        # mode still runs analysis immediately after init so keeps eager init.
        if not self.is_cli_mode and self.tracker:
            tracker_name = self.app_state_ui.selected_tracker_name
            if not self.tracker.set_tracking_mode(tracker_name, lazy=True):
                from config.tracker_discovery import get_tracker_discovery, TrackerCategory
                discovery = get_tracker_discovery()
                live_trackers = discovery.get_trackers_by_category(TrackerCategory.LIVE)
                if live_trackers:
                    fallback = live_trackers[0].internal_name
                    self.logger.info(f"Tracker '{tracker_name}' unavailable, falling back to '{fallback}'")
                    self.app_state_ui.selected_tracker_name = fallback
                    self.tracker.set_tracking_mode(fallback, lazy=True)

    @staticmethod
    def _purge_old_log_entries(log_file_path: str):
        """Delegator — see log_config.purge_old_log_entries."""
        from application.logic.log_config import purge_old_log_entries
        purge_old_log_entries(log_file_path)

    def _configure_third_party_logging(self):
        """Delegator — see log_config.configure_third_party_logging."""
        from application.logic.log_config import configure_third_party_logging
        configure_third_party_logging()
        self.logger.debug("Third-party logging configured for reduced startup noise")

    def trigger_first_run_setup(self):
        """Delegator — see FirstRunSetupController."""
        self._first_run_setup.trigger()

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

    def notify(self, message: str, type: str = "info", duration: float = 4.0):
        """Send a toast notification to the GUI. type: 'success', 'error', 'warning', 'info'."""
        gui = getattr(self, 'gui_instance', None)
        if gui and hasattr(gui, 'notification_manager'):
            gui.notification_manager.add(message, type, duration)

    def get_waveform_data(self):
        """Thread-safe access to audio waveform data."""
        with self._waveform_lock:
            return self.audio_waveform_data

    def trigger_ultimate_autotune_with_defaults(self, timeline_num: int):
        """
        Non-interactively runs the Ultimate Autotune pipeline with default settings.
        """
        self.autotuner.trigger_ultimate_autotune_with_defaults(timeline_num)

    def _run_post_analysis_pipeline(self, frame_range=None):
        """Delegator — see TrackingLifecycleController.run_post_analysis_pipeline."""
        return self._tracking_lifecycle.run_post_analysis_pipeline(frame_range=frame_range)

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
                    # ~144ms per bucket on a 2h video.
                    n_target = 50000
                    chunk_size = max(1, len(samples) // n_target)
                    n = (len(samples) // chunk_size) * chunk_size
                    blocks = np.abs(samples[:n].reshape(-1, chunk_size))
                    waveform_data = np.max(blocks, axis=1).astype(np.float32)
                    self.logger.info(f"Waveform from cached subtitle audio ({len(waveform_data)} samples)")
            except Exception as e:
                self.logger.debug(f"Cached audio waveform failed: {e}")

            if waveform_data is None:
                waveform_data = self.processor.get_audio_waveform(num_samples=50000)

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
        """Delegator — see TrackingLifecycleController.set_pending_action."""
        self._tracking_lifecycle.set_pending_action(action_type, **kwargs)

    def clear_pending_action_after_tracking(self):
        """Delegator — see TrackingLifecycleController.clear_pending_action."""
        self._tracking_lifecycle.clear_pending_action()

    def on_offline_analysis_completed(self, payload: Dict):
        """Delegator — see TrackingLifecycleController.on_offline_analysis_completed."""
        self._tracking_lifecycle.on_offline_analysis_completed(payload)

    def on_processing_stopped(self, was_scripting_session: bool = False,
                              scripted_frame_range: Optional[Tuple[int, int]] = None):
        """Delegator — see TrackingLifecycleController.on_processing_stopped."""
        self._tracking_lifecycle.on_processing_stopped(
            was_scripting_session=was_scripting_session,
            scripted_frame_range=scripted_frame_range)

    def _cache_tracking_classes(self):
        """Temporarily loads the detection model to get class names, then unloads it."""
        self.model_manager._cache_tracking_classes()

    def get_available_tracking_classes(self) -> List[str]:
        """Gets the list of class names from the model (cached)."""
        return self.model_manager.get_available_tracking_classes()

    def set_status_message(self, message: str, duration: float = 3.0, level: int = logging.INFO):
        if hasattr(self, 'app_state_ui') and self.app_state_ui is not None:
            self.app_state_ui.set_status(message, time.time() + duration)
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
        """Delegator — see HardwareAccelController."""
        self._hwaccel.query_background()

    def _get_available_ffmpeg_hwaccels(self) -> List[str]:
        """Delegator — see HardwareAccelController."""
        return self._hwaccel._query_ffmpeg()

    def _check_model_paths(self):
        """Checks essential model paths and auto-downloads if missing."""
        return self.model_manager._check_model_paths()

    def set_application_logging_level(self, level_name: str):
        """Delegator — see log_config.set_application_logging_level."""
        from application.logic.log_config import set_application_logging_level
        set_application_logging_level(self, level_name)

    def _apply_loaded_settings(self):
        """Delegator — see SettingsLifecycleController.apply_loaded."""
        self._settings_lifecycle.apply_loaded()

    def save_app_settings(self):
        """Delegator — see SettingsLifecycleController.save."""
        self._settings_lifecycle.save()

    def _load_last_project_on_startup(self):
        """Delegator — see ProjectLifecycleController.load_last_on_startup."""
        self._project_lifecycle.load_last_on_startup()

    def reset_project_state(self, for_new_project: bool = True):
        """Delegator — see ProjectLifecycleController.reset."""
        self._project_lifecycle.reset(for_new_project=for_new_project)

    def _map_shortcut_to_glfw_key(self, shortcut_string_to_parse: str) -> Optional[Tuple[int, dict]]:
        """Delegator — see ShortcutMapper.map."""
        return self._shortcut_mapper.map(shortcut_string_to_parse)

    def invalidate_shortcut_cache(self):
        """Delegator — see ShortcutMapper.invalidate."""
        self._shortcut_mapper.invalidate()

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
            except Exception as e:
                self.logger.warning(f"Failed to extract funscript duration: {e}")
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
        _as_cfg = self.app_settings.config.autosave
        if _as_cfg.on_exit and _as_cfg.enabled and self.project_manager.project_dirty:
            self.logger.info("Performing final autosave on exit...")
            self.project_manager.perform_autosave()

        # Stop mpv review mode if active
        if self._mpv_controller and self._mpv_controller.is_active:
            self._mpv_controller.stop()

        # Stop audio playback and persist volume to settings
        if hasattr(self, '_audio_volume_live'):
            self.app_settings.config.audio.volume = self._audio_volume_live
        if self._audio_sync:
            self._audio_sync.stop()
        if self._audio_player:
            self._audio_player.cleanup()

        # Any other cleanup (e.g. closing files, releasing resources)

        # Flush any debounced settings write so the last set() before shutdown
        # (e.g. audio_volume just above) isn't lost to the debounce window.
        self.app_settings.flush()

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

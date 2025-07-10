import time
import logging
import subprocess
import os
import threading
from typing import Optional, Dict, Tuple, List, Any

from video.video_processor import VideoProcessor
from tracker.tracker import ROITracker as Tracker

from application.classes import settings_manager, project_manager, shortcut_manager, undo_redo_manager
from application.utils.logger import AppLogger
from config.constants import *
from application.utils.write_access import check_write_access

from .app_state_ui import AppStateUI
from .app_file_manager import AppFileManager
from .app_stage_processor import AppStageProcessor
from .app_funscript_processor import AppFunscriptProcessor
from .app_event_handlers import AppEventHandlers
from .app_calibration import AppCalibration
from .app_energy_saver import AppEnergySaver
from .app_utility import AppUtility

class ApplicationLogic:
    def __init__(self):
        self.gui_instance = None
        self.app_settings = settings_manager.AppSettings(logger=None)

        # Initialize logging_level_setting before AppLogger uses it indirectly via AppSettings
        self.logging_level_setting = self.app_settings.get("logging_level", "INFO")

        status_log_config = {
            logging.INFO: 3.0, logging.WARNING: 6.0, logging.ERROR: 10.0, logging.CRITICAL: 15.0,
        }
        self.app_log_file_path = 'fungen.log'  # Define app_log_file_path

        # --- Start of Log Purge ---
        try:
            # Purge the log file if it's older than 7 days. This runs before the logger is attached to it.
            if os.path.exists(self.app_log_file_path):
                if (time.time() - os.path.getmtime(self.app_log_file_path)) > (7 * 24 * 60 * 60):
                    os.remove(self.app_log_file_path)
        except Exception:
            # If purging fails, it's not a critical error. The app can continue.
            pass

        self._logger_instance = AppLogger(
            app_logic_instance=self,
            status_level_durations=status_log_config,
            log_file=self.app_log_file_path,
            level=getattr(logging, self.logging_level_setting.upper(), logging.INFO)  # Use initial setting
        )
        self.logger = self._logger_instance.get_logger()
        self.app_settings.logger = self.logger  # Now provide the logger to AppSettings

        # REFACTORED Defensive programming. Always make sure the type is a list of strings.
        discarded_tracking_classes = self.app_settings.get("discarded_tracking_classes", [])
        if discarded_tracking_classes is None:
            discarded_tracking_classes = []
        self.discarded_tracking_classes: List[str] = discarded_tracking_classes
        self.pending_action_after_tracking: Optional[Dict] = None

        self.app_state_ui = AppStateUI(self)
        self.utility = AppUtility(self)

        # --- Hardware Acceleration
        # Query ffmpeg for available hardware accelerations
        self.available_ffmpeg_hwaccels = self._get_available_ffmpeg_hwaccels()

        # Get the hardware acceleration method from settings and validate it
        default_hw_accel = "auto"
        if "auto" not in self.available_ffmpeg_hwaccels:
            self.logger.warning("'auto' not in available hwaccels. Defaulting to 'none' or first available.")
            default_hw_accel = "none" if "none" in self.available_ffmpeg_hwaccels else \
                (self.available_ffmpeg_hwaccels[0] if self.available_ffmpeg_hwaccels else "none")

        current_hw_method_from_settings = self.app_settings.get("hardware_acceleration_method", default_hw_accel)

        if current_hw_method_from_settings not in self.available_ffmpeg_hwaccels:
            self.logger.warning(
                f"Configured hardware acceleration '{current_hw_method_from_settings}' "
                f"not listed by ffmpeg ({self.available_ffmpeg_hwaccels}). Falling back to '{default_hw_accel}'.")
            self.hardware_acceleration_method = default_hw_accel
            self.app_settings.set("hardware_acceleration_method", default_hw_accel)
        else:
            self.hardware_acceleration_method = current_hw_method_from_settings

        # --- Tracking Axis Configuration (ensure these are initialized before tracker if tracker uses them in __init__) ---
        self.tracking_axis_mode = self.app_settings.get("tracking_axis_mode", "both")
        self.single_axis_output_target = self.app_settings.get("single_axis_output_target", "primary")

        # --- Models ---
        self.yolo_detection_model_path_setting = self.app_settings.get("yolo_det_model_path")
        self.yolo_pose_model_path_setting = self.app_settings.get("yolo_pose_model_path")
        self.pose_model_artifacts_dir = self.app_settings.get("pose_model_artifacts_dir")
        self.pose_model_artifacts_dir = self.app_settings.get("pose_model_artifacts_dir")
        self.yolo_det_model_path = self.yolo_detection_model_path_setting
        self.yolo_pose_model_path = self.yolo_pose_model_path_setting
        self.yolo_input_size = 640

        # --- Undo/Redo Managers ---
        self.undo_manager_t1: Optional[undo_redo_manager.UndoRedoManager] = None
        self.undo_manager_t2: Optional[undo_redo_manager.UndoRedoManager] = None

        # --- Initialize Tracker ---
        self.tracker = Tracker(
            app_logic_instance = self,
            tracker_model_path = self.yolo_detection_model_path_setting,
            pose_model_path = self.yolo_pose_model_path_setting,
            logger = self.logger)
        if self.tracker:
            self.tracker.show_stats = False  # Default internal tracker states
            self.tracker.show_funscript_preview = False

        # --- NOW Sync Tracker UI Flags as tracker and app_state_ui exist ---
        self.app_state_ui.sync_tracker_ui_flags()

        # --- Initialize Processor (after tracker and logger/app_state_ui are ready) ---
        # _check_model_paths can be called now before processor if it's critical for processor init
        self._check_model_paths()
        self.processor = VideoProcessor(self, self.tracker, yolo_input_size=self.yolo_input_size)

        # --- Modular Components Initialization ---
        self.file_manager = AppFileManager(self)
        self.stage_processor = AppStageProcessor(self)
        self.funscript_processor = AppFunscriptProcessor(self)
        self.event_handlers = AppEventHandlers(self)
        self.calibration = AppCalibration(self)
        self.energy_saver = AppEnergySaver(self)

        # --- Other Managers ---
        self.project_manager = project_manager.ProjectManager(self)
        self.shortcut_manager = shortcut_manager.ShortcutManager(self)

        self.project_data_on_load: Optional[Dict] = None
        self.s2_frame_objects_map_for_s3: Optional[Dict[int, Any]] = None

        # User Defined ROI
        self.is_setting_user_roi_mode: bool = False
        # --- State for chapter-specific ROI setting ---
        self.chapter_id_for_roi_setting: Optional[str] = None

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
        self.batch_apply_post_processing: bool = True
        self.batch_copy_funscript_to_video_location: bool = True
        self.batch_overwrite_mode: int = 0  # 0 for Process All, 1 for Skip Existing
        self.batch_generate_roll_file: bool = True

        # --- Final Setup Steps ---
        self._apply_loaded_settings()
        self.funscript_processor._ensure_undo_managers_linked()
        self._check_for_autosave_restore()
        self.energy_saver.reset_activity_timer()

    def unload_model(self, model_type: str):
        """
        Clears the path for a given model type and releases it from the tracker.
        """
        if model_type == 'detection':
            self.yolo_detection_model_path_setting = ""
            self.app_settings.set("yolo_det_model_path", "")
            if self.tracker:
                self.tracker.unload_detection_model()
            self.logger.info("YOLO Detection Model unloaded.", extra={'status_message': True})
        elif model_type == 'pose':
            self.yolo_pose_model_path_setting = ""
            self.app_settings.set("yolo_pose_model_path", "")
            if self.tracker:
                self.tracker.unload_pose_model()
            self.logger.info("YOLO Pose Model unloaded.", extra={'status_message': True})
        else:
            self.logger.warning(f"Unknown model type '{model_type}' for unload.")

        self.project_manager.project_dirty = True
        self.energy_saver.reset_activity_timer()

    def generate_waveform(self):
        if not self.processor or not self.processor.is_video_open():
            self.logger.info("Cannot generate waveform: No video loaded.", extra={'status_message': True})
            return

        def _generate_waveform_thread():
            self.logger.info("Generating audio waveform...", extra={'status_message': True})
            waveform_data = self.processor.get_audio_waveform(num_samples=2000)

            self.audio_waveform_data = waveform_data

            if self.audio_waveform_data is not None:
                self.logger.info("Audio waveform generated successfully.", extra={'status_message': True})
                self.app_state_ui.show_audio_waveform = True
            else:
                self.logger.error("Failed to generate audio waveform.", extra={'status_message': True})
                self.app_state_ui.show_audio_waveform = False

        thread = threading.Thread(target=_generate_waveform_thread, daemon=True)
        thread.start()

    def toggle_waveform_visibility(self):
        if not self.app_state_ui.show_audio_waveform and self.audio_waveform_data is None:
            self.generate_waveform()
        else:
            self.app_state_ui.show_audio_waveform = not self.app_state_ui.show_audio_waveform
            status = "enabled" if self.app_state_ui.show_audio_waveform else "disabled"
            self.logger.info(f"Audio waveform display {status}.", extra={'status_message': True})

    def start_batch_processing(self, video_paths: List[str]):
        """
        Prepares for batch processing by creating a confirmation message and showing a dialog.
        """
        if not self._check_model_paths():
            return
        if self.is_batch_processing_active or self.stage_processor.full_analysis_active:
            self.logger.warning("Cannot start batch processing: A process is already active.",
                                extra={'status_message': True})
            return

        if not video_paths:
            self.logger.info("No videos provided for batch processing.", extra={'status_message': True})
            return

        # --- Prepare the confirmation message ---
        num_videos = len(video_paths)
        message_lines = [
            f"Found {num_videos} video{'s' if num_videos > 1 else ''} to script.",
            "Do you want to run batch processing?",
            ""  # Visual separator
        ]

        # Add conditional warnings
        if self.calibration.funscript_output_delay_frames == 0:
            message_lines.append("-> Warning: Optical flow delay is 0. Have you calibrated it?")

        if not self.app_settings.get("enable_auto_post_processing", False):
            message_lines.append("-> Warning: Automatic post-processing is currently disabled.")

        # Set the state to trigger the GUI dialog
        self.batch_confirmation_message = "\n".join(message_lines)
        self.batch_confirmation_videos = video_paths
        self.show_batch_confirmation_dialog = True
        self.energy_saver.reset_activity_timer()  # Ensure UI is responsive

    def _initiate_batch_processing_from_confirmation(self, selected_method_idx: int, apply_post_processing: bool,
                                                     copy_to_video_location: bool, overwrite_mode: int,
                                                     generate_roll: bool):
        """
        [Private] Called from the GUI when the user clicks 'Yes' in the confirmation dialog.
        This method starts the actual batch processing thread.
        """
        if not self._check_model_paths():
            return
        if self.is_batch_processing_active: return
        if not self.batch_confirmation_videos:
            self.logger.error("Batch confirmation accepted, but no videos were found in the list.")
            self._cancel_batch_processing_from_confirmation()
            return

        # --- Store all user choices from the dialog ---
        self.batch_processing_method_idx = selected_method_idx
        self.batch_apply_post_processing = apply_post_processing
        self.batch_copy_funscript_to_video_location = copy_to_video_location
        self.batch_overwrite_mode = overwrite_mode
        self.batch_generate_roll_file = generate_roll

        self.logger.info(
            f"User confirmed. Starting batch processing with method: {selected_method_idx}, post-proc: {apply_post_processing}, copy: {copy_to_video_location}, overwrite: {overwrite_mode}, gen_roll: {generate_roll}")

        # Update the main UI state to reflect the chosen batch mode, so the correct progress UI is displayed.
        batch_mode_map = {
            0: TrackerMode.OFFLINE_3_STAGE,
            1: TrackerMode.OFFLINE_2_STAGE
        }
        chosen_mode = batch_mode_map.get(selected_method_idx)
        if chosen_mode:
            self.app_state_ui.selected_tracker_mode = chosen_mode
            self.logger.info(f"Control panel UI mode updated to: {chosen_mode.name} for batch processing.")
        else:
            self.logger.warning(f"Could not map batch method index {selected_method_idx} to a valid tracker mode.")


        # Set up batch processing state from the confirmed data
        self.batch_video_paths = list(self.batch_confirmation_videos)
        self.is_batch_processing_active = True
        self.current_batch_video_index = -1
        self.stop_batch_event.clear()

        # Start the background thread
        self.batch_processing_thread = threading.Thread(target=self._run_batch_processing_thread, daemon=True)
        self.batch_processing_thread.start()

        # Clear the confirmation dialog state
        self.show_batch_confirmation_dialog = False
        self.batch_confirmation_videos = []
        self.batch_confirmation_message = ""

    def _cancel_batch_processing_from_confirmation(self):
        """
        [Private] Called from the GUI when the user clicks 'No' in the confirmation dialog.
        """
        self.logger.info("Batch processing cancelled by user.", extra={'status_message': True})
        # Clear the confirmation dialog state
        self.show_batch_confirmation_dialog = False
        self.batch_confirmation_videos = []
        self.batch_confirmation_message = ""

    def abort_batch_processing(self):
        if not self.is_batch_processing_active:
            return

        self.logger.info("Aborting batch processing...", extra={'status_message': True})
        self.stop_batch_event.set()
        # Also signal the currently running stage analysis (if any) to stop
        self.stage_processor.abort_stage_processing()
        self.single_video_analysis_complete_event.set()  # Release the wait lock

    def _run_batch_processing_thread(self):
        try:
            for i, video_path in enumerate(self.batch_video_paths):
                if self.stop_batch_event.is_set():
                    self.logger.info("Batch processing was aborted by user.")
                    break

                self.current_batch_video_index = i
                video_basename = os.path.basename(video_path)
                self.logger.info(f"Batch processing video {i + 1}/{len(self.batch_video_paths)}: {video_basename}")

                # --- Pre-flight checks for overwrite strategy ---
                # This is now the very first step for each video in the loop.
                path_next_to_video = os.path.splitext(video_path)[0] + ".funscript"

                funscript_to_check = None
                if os.path.exists(path_next_to_video):
                    funscript_to_check = path_next_to_video

                if funscript_to_check:
                    if self.batch_overwrite_mode == 1:
                        # Mode 1: Process only if funscript is missing (skip any existing funscript)
                        self.logger.info(
                            f"Skipping '{video_basename}': Funscript already exists at '{funscript_to_check}'.")
                        continue

                    if self.batch_overwrite_mode == 0:
                        # Mode 0: Process all except own matching version (skip if up-to-date FunGen funscript exists)
                        funscript_data = self.file_manager._get_funscript_data(funscript_to_check)
                        if funscript_data:
                            author = funscript_data.get('author', '')
                            metadata = funscript_data.get('metadata', {})
                            # Ensure metadata is a dict before calling .get() on it
                            version = metadata.get('version', '') if isinstance(metadata, dict) else ''
                            if author.startswith("FunGen") and version == FUNSCRIPT_METADATA_VERSION:
                                self.logger.info(
                                    f"Skipping '{video_basename}': Up-to-date funscript from this program version already exists.")
                                continue

                    if self.batch_overwrite_mode == 2:
                        # Mode 2: Process ALL videos, including up-to-date FunGen funscript. Do not skip for any reason.
                        self.logger.info(
                            f"Processing '{video_basename}': Mode 2 selected, will process regardless of funscript existence or version.")
                # --- End of pre-flight checks ---

                open_success = self.file_manager.open_video_from_path(video_path)
                if not open_success:
                    self.logger.error(f"Failed to open video, skipping: {video_path}")
                    continue

                time.sleep(1.0)
                if self.stop_batch_event.is_set(): break

                self.single_video_analysis_complete_event.clear()
                self.save_and_reset_complete_event.clear()
                self.stage_processor.start_full_analysis()

                self.single_video_analysis_complete_event.wait()
                if self.stop_batch_event.is_set(): break

                self.logger.debug("Batch loop: Waiting for save/reset signal from GUI thread...")
                self.save_and_reset_complete_event.wait(timeout=120)
                self.logger.debug("Batch loop: Save/reset signal received. Proceeding to next video.")

                if self.stop_batch_event.is_set(): break
        except Exception as e:
            self.logger.error(f"An error occurred during the batch process: {e}", exc_info=True)
        finally:
            self.is_batch_processing_active = False
            self.current_batch_video_index = -1
            self.batch_video_paths = []
            self.stop_batch_event.clear()
            self.logger.info("Batch processing finished.", extra={'status_message': True})

    def enter_set_user_roi_mode(self):
        if self.processor and self.processor.is_processing:
            self.processor.pause_processing()  # Pause if playing/tracking
            self.logger.info("Video paused to set User ROI.")

        self.is_setting_user_roi_mode = True
        if self.gui_instance and hasattr(self.gui_instance, 'video_display_ui'):  # Reset drawing state in UI
            self.gui_instance.video_display_ui.is_drawing_user_roi = False
            self.gui_instance.video_display_ui.drawn_user_roi_video_coords = None
            self.gui_instance.video_display_ui.waiting_for_point_click = False

        self.logger.info("Setting User Defined ROI: Draw rectangle on video, then click point inside.", extra={'status_message': True, 'duration': 5.0})
        self.energy_saver.reset_activity_timer()

    def exit_set_user_roi_mode(self):
        self.is_setting_user_roi_mode = False
        if self.gui_instance and hasattr(self.gui_instance, 'video_display_ui'):
            self.gui_instance.video_display_ui.is_drawing_user_roi = False
            self.gui_instance.video_display_ui.drawn_user_roi_video_coords = None
            self.gui_instance.video_display_ui.waiting_for_point_click = False

    def user_roi_and_point_set(self, roi_rect_video_coords: Tuple[int, int, int, int], point_video_coords: Tuple[int, int]):
        if self.chapter_id_for_roi_setting:
            # --- Logic for setting chapter-specific ROI ---
            target_chapter = next((ch for ch in self.funscript_processor.video_chapters if ch.unique_id == self.chapter_id_for_roi_setting), None)
            if target_chapter:
                target_chapter.user_roi_fixed = roi_rect_video_coords

                # Calculate the point's position relative to the new ROI
                rx, ry, _, _ = roi_rect_video_coords
                px_rel = float(point_video_coords[0] - rx)
                py_rel = float(point_video_coords[1] - ry)
                target_chapter.user_roi_initial_point_relative = (px_rel, py_rel)

                self.logger.info(
                    f"ROI and point set for chapter: {target_chapter.position_short_name} ({target_chapter.unique_id[:8]})", extra={'status_message': True})
                self.project_manager.project_dirty = True
            else:
                self.logger.error(f"Could not find the target chapter ({self.chapter_id_for_roi_setting}) to set ROI.", extra={'status_message': True})

            # Reset the state variable
            self.chapter_id_for_roi_setting = None

        else:
            if self.tracker and self.processor:
                current_display_frame = None
                # We need the raw frame buffer that corresponds to the video_coords.
                # processor.current_frame is usually the one passed to tracker (e.g. 640x640 BGR)
                with self.processor.frame_lock:
                    if self.processor.current_frame is not None:
                        current_display_frame = self.processor.current_frame.copy()

                if current_display_frame is not None:
                    self.tracker.set_user_defined_roi_and_point(roi_rect_video_coords, point_video_coords, current_display_frame)
                    # Tracker mode is usually set via UI combo, but ensure it if not already.
                    if self.tracker.tracking_mode != "USER_FIXED_ROI":
                        self.tracker.set_tracking_mode("USER_FIXED_ROI")
                    self.logger.info("User defined ROI and point have been set in the tracker.", extra={'status_message': True})
                else:
                    self.logger.error("Could not get current frame to set user ROI patch. ROI not set.", extra={'status_message': True})
            else:
                self.logger.error("Tracker or Processor not available to set user ROI.", extra={'status_message': True})

        self.exit_set_user_roi_mode()
        self.energy_saver.reset_activity_timer()

    def set_pending_action_after_tracking(self, action_type: str, **kwargs):
        """Stores information about an action to be performed after tracking."""
        self.pending_action_after_tracking = {"type": action_type, "data": kwargs}
        self.logger.info(f"Pending action set after tracking: {action_type} with data {kwargs}")

    def clear_pending_action_after_tracking(self):
        """Clears any pending action."""
        if self.pending_action_after_tracking:
            self.logger.info(f"Cleared pending action: {self.pending_action_after_tracking.get('type')}")
        self.pending_action_after_tracking = None

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

                # 2. PROCEED WITH POST-PROCESSING (if enabled)
                if self.app_settings.get("enable_auto_post_processing", False):
                    self.logger.info(
                        f"Triggering auto post-processing for live tracking session range: {scripted_frame_range}.")
                    if hasattr(self, 'funscript_processor') and hasattr(self.funscript_processor, 'apply_automatic_post_processing'):
                        try:
                            # Pass the specific frame range to the post-processing function
                            self.funscript_processor.apply_automatic_post_processing(frame_range=scripted_frame_range)

                            # 3. SAVE THE FINAL (POST-PROCESSED) FUNSCRIPT
                            self.logger.info("Saving final (post-processed) funscript.")
                            # The chapter list from the funscript processor is the most current after post-processing
                            chapters_for_save = self.funscript_processor.video_chapters
                            self.file_manager.save_final_funscripts(video_path, chapters=chapters_for_save)

                        except Exception as e_post:
                            self.logger.error(f"Error during automatic post-processing after live tracking: {e_post}", exc_info=True)
                else:
                    self.logger.info("Auto post-processing disabled, skipping.")
            else:
                self.logger.warning("Live session ended, but no video path is available to save the raw funscript.")

    def get_available_tracking_classes(self) -> List[str]:
        """Gets the list of class names from the loaded YOLO detection model."""
        if self.tracker and self.tracker.yolo and hasattr(self.tracker.yolo, 'names'):
            model_names = self.tracker.yolo.names
            if isinstance(model_names, dict):
                return sorted(list(model_names.values()))
            elif isinstance(model_names, list):
                return sorted(model_names)
            self.logger.warning("Tracker model names format not recognized for available classes.")
        return []

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

    def _get_available_ffmpeg_hwaccels(self) -> List[str]:
        """Queries FFmpeg for available hardware acceleration methods."""
        try:
            # Consider making ffmpeg_path configurable via app_settings
            ffmpeg_path = self.app_settings.get("ffmpeg_path") or "ffmpeg" # Without 'or' it would accept "" or None as valid values (Is what Cluade told me)
            result = subprocess.run(
                [ffmpeg_path, '-hide_banner', '-hwaccels'],
                capture_output=True, text=True, check=True, timeout=5
            )
            lines = result.stdout.strip().split('\n')
            hwaccels = []
            if lines and "Hardware acceleration methods:" in lines[0]:  #
                # Parse the methods, excluding 'none' if FFmpeg lists it, as we add it manually.
                hwaccels = [line.strip() for line in lines[1:] if line.strip() and line.strip() != "none"]

                # Ensure "auto" and "none" are always present and prioritized
            standard_options = ["auto", "none"]
            unique_hwaccels = [h for h in hwaccels if h not in standard_options]
            final_options = standard_options + unique_hwaccels
            log_func = self.logger.info if hasattr(self, 'logger') and self.logger else print
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
        """Checks essential model paths and logs errors if not found."""
        # Detection model remains essential
        if not self.yolo_det_model_path or not os.path.exists(self.yolo_det_model_path):
            self.logger.error(
                f"CRITICAL ERROR: YOLO Detection Model not found or path not set: '{self.yolo_det_model_path}'. Please check settings.",
                extra={'status_message': True, 'duration': 15.0})
            # Show GUI popup if available
            if self.gui_instance and hasattr(self.gui_instance, 'show_error_popup'):
                self.gui_instance.show_error_popup(
                    "Detection Model Missing",
                    "No valid Detection Model is set.\nPlease select a YOLO model file in the UI Configuration tab.",
                    action_label="Select Model",
                    action_callback=getattr(self.gui_instance, 'open_detection_model_dialog', None)
                )
            return False

        # Pose model is now optional
        if not self.yolo_pose_model_path or not os.path.exists(self.yolo_pose_model_path):
            self.logger.warning(
                f"Warning: YOLO Pose Model not found or path not set. Pose-dependent features will be disabled.",
                extra={'status_message': True, 'duration': 8.0})

        return True

    def set_application_logging_level(self, level_name: str):
        """Sets the application-wide logging level."""
        numeric_level = getattr(logging, level_name.upper(), None)
        if numeric_level is not None and hasattr(self, '_logger_instance') and hasattr(self._logger_instance, 'logger'):
            self._logger_instance.logger.setLevel(numeric_level)
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

        # Hardware Acceleration
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
        self.calibration.update_settings_from_app()
        self.energy_saver.update_settings_from_app()
        self.calibration.update_tracker_delay_params()
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
        self.calibration.save_settings_to_app()
        self.energy_saver.save_settings_to_app()
        self.app_settings.save_settings()
        self.logger.info("Application settings saved.", extra={'status_message': True})
        self.energy_saver.reset_activity_timer()

    def _check_for_autosave_restore(self):
        if os.path.exists(AUTOSAVE_FILE) and self.app_settings.get("autosave_enabled", True):
            try:
                self.project_manager.load_project(AUTOSAVE_FILE)
                self.project_manager.project_dirty = True
                self.logger.info(f"State restored from autosave: {AUTOSAVE_FILE}.", extra={'status_message': True})
            except Exception as e:
                self.logger.error(f"Failed to auto-restore from {AUTOSAVE_FILE}: {e}", exc_info=True, extra={'status_message': False})

    def reset_project_state(self, for_new_project: bool = True):
        """Resets the application to a clean state for a new or loaded project."""
        self.logger.info(f"Resetting project state ({'new project' if for_new_project else 'project load'})...")

        # Stop any active processing
        if self.processor and self.processor.is_processing: self.processor.stop_processing()
        if self.stage_processor.full_analysis_active: self.stage_processor.abort_stage_processing()  # Signals thread

        self.file_manager.close_video_action(clear_funscript_unconditionally=True)
        self.funscript_processor.reset_state_for_new_project()
        self.funscript_processor.update_funscript_stats_for_timeline(1, "Project Reset")
        self.funscript_processor.update_funscript_stats_for_timeline(2, "Project Reset")

        # Reset waveform data
        self.audio_waveform_data = None
        self.app_state_ui.show_audio_waveform = False

        # Reset waveform data
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
        self.app_state_ui.show_lr_dial_graph = self.app_settings.get("show_lr_dial_graph", app_settings_defaults.get("show_lr_dial_graph", True))
        self.app_state_ui.show_heatmap = self.app_settings.get("show_heatmap", app_settings_defaults.get("show_heatmap", True))
        self.app_state_ui.show_gauge_window = self.app_settings.get("show_gauge_window", app_settings_defaults.get("show_gauge_window", True))
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
        if self.undo_manager_t1: self.undo_manager_t1.clear_history()
        if self.undo_manager_t2: self.undo_manager_t2.clear_history()
        # Ensure they are re-linked to (now empty) actions lists
        self.funscript_processor._ensure_undo_managers_linked()
        self.app_state_ui.heatmap_dirty = True
        self.app_state_ui.funscript_preview_dirty = True
        self.app_state_ui.force_timeline_pan_to_current_frame = True

        if for_new_project:
            self.logger.info("New project state initialized.", extra={'status_message': True})
        self.energy_saver.reset_activity_timer()

    def _map_shortcut_to_glfw_key(self, shortcut_string_to_parse: str) -> Optional[Tuple[int, dict]]:
        """
        Parses a shortcut string (e.g., "CTRL+SHIFT+A") into a GLFW key code
        and a dictionary of modifiers.
        This method now correctly uses self.shortcut_manager.name_to_glfw_key
        as indicated by the original code.
        """
        if not shortcut_string_to_parse:
            self.logger.warning("Received an empty string for shortcut mapping.")
            return None

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
                    self.logger.warning(
                        f"Invalid shortcut string '{shortcut_string_to_parse}'. Multiple main keys identified: '{main_key_str}' and '{part_cleaned}'.")
                    return None
                main_key_str = part_cleaned

        if main_key_str is None:
            self.logger.warning(f"Invalid shortcut string '{shortcut_string_to_parse}'. No main key found.")
            return None

        if not self.shortcut_manager:
            self.logger.warning("Shortcut manager not available for mapping key name.")
            return None
        glfw_key_code = self.shortcut_manager.name_to_glfw_key(main_key_str)
        if glfw_key_code is None:
            return None
        return glfw_key_code, modifiers

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
            except:
                duration_s = 0.0
        return duration_s, total_frames, fps_val

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

        # Any other cleanup (e.g. closing files, releasing resources)
        # self.app_settings.save_settings() # Settings usually saved explicitly by user or before critical changes

        self.logger.info("Application logic shutdown complete.")

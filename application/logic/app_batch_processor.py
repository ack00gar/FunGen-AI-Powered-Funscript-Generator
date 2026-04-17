"""Batch processing functionality for ApplicationLogic."""
import os
import time
import threading
from datetime import datetime
from typing import Optional, Dict, Tuple, List, Any

from config.constants import FUNSCRIPT_METADATA_VERSION
from config.tracker_discovery import get_tracker_discovery


class AdaptiveTuningState:
    """Tracks progressive pipeline tuning state across batch videos."""
    __slots__ = (
        'current_producers', 'current_consumers',
        'best_producers', 'best_consumers', 'best_fps',
        'history', 'consumer_ceiling',
        'consecutive_no_improvement', 'is_converged',
        'status_message',
    )

    def __init__(self, producers: int, consumers: int):
        self.current_producers = producers
        self.current_consumers = consumers
        self.best_producers = producers
        self.best_consumers = consumers
        self.best_fps = 0.0
        self.history: List[Tuple[int, int, float, bool]] = []  # (p, c, avg_fps, success)
        self.consumer_ceiling: Optional[int] = None  # hard cap set when a config stalls
        self.consecutive_no_improvement = 0
        self.is_converged = False
        self.status_message = f"Starting with P={producers}/C={consumers}"


class AppBatchProcessor:
    """Handles batch processing operations."""

    def __init__(self, app_logic):
        self.app = app_logic

    def start_batch_processing(self, video_paths: List[str]):
        """
        Prepares for batch processing by creating a confirmation message and showing a dialog.
        """
        if not self.app._check_model_paths():
            return
        if self.app.is_batch_processing_active or self.app.stage_processor.full_analysis_active:
            self.app.logger.warning("Cannot start batch processing: A process is already active.",
                                extra={'status_message': True})
            return

        if not video_paths:
            self.app.logger.info("No videos provided for batch processing.", extra={'status_message': True})
            return

        # --- Prepare the confirmation message ---
        num_videos = len(video_paths)
        message_lines = [
            f"Found {num_videos} video{'s' if num_videos > 1 else ''} to script.",
            "Do you want to run batch processing?",
            ""  # Visual separator
        ]

        # Set the state to trigger the GUI dialog
        self.app.batch_confirmation_message = "\n".join(message_lines)
        self.app.batch_confirmation_videos = video_paths
        self.app.show_batch_confirmation_dialog = True
        self.app.energy_saver.reset_activity_timer()  # Ensure UI is responsive

    def _initiate_batch_processing_from_confirmation(self):
        """
        [Private] Called from the GUI. Reads the configured list of videos and
        settings from the GUI and starts the batch processing thread.
        """
        if not self.app._check_model_paths(): return
        if self.app.is_batch_processing_active: return
        gui = self.app.gui_instance
        if not gui or not gui.batch_videos_data:
            self.app.logger.error("Batch start requested, but GUI data is missing.")
            self._cancel_batch_processing_from_confirmation()
            return

        videos_to_process = []
        video_format_options = ["Auto (Heuristic)", "2D", "VR (he_sbs)", "VR (he_tb)", "VR (fisheye_sbs)", "VR (fisheye_tb)"]

        for video_data in gui.batch_videos_data:
            if video_data.get("selected", False):
                override_idx = video_data.get("override_format_idx", 0)
                override_format = video_format_options[override_idx] if 0 <= override_idx < len(video_format_options) else "Auto (Heuristic)"
                videos_to_process.append({"path": video_data["path"], "override_format": override_format})

        if not videos_to_process:
            self.app.logger.info("No videos selected for batch processing.", extra={'status_message': True})
            self._cancel_batch_processing_from_confirmation()
            return

        # Use the dynamically selected tracker name
        self.app.batch_tracker_name = gui.selected_batch_tracker_name
        self.app.batch_apply_ultimate_autotune = gui.batch_apply_ultimate_autotune_ui
        self.app.batch_copy_funscript_to_video_location = gui.batch_copy_funscript_to_video_location_ui
        self.app.batch_overwrite_mode = gui.batch_overwrite_mode_ui
        self.app.batch_generate_roll_file = gui.batch_generate_roll_file_ui
        self.app.batch_adaptive_tuning_enabled = getattr(gui, 'batch_adaptive_tuning_ui', False)
        self.app.batch_save_preprocessed_video = getattr(gui, 'batch_save_preprocessed_video_ui', False)

        self.app.logger.info(f"User confirmed. Starting batch with {len(videos_to_process)} videos.")
        self.app.batch_video_paths = videos_to_process # Now a list of dicts
        self.app.is_batch_processing_active = True
        self.app.current_batch_video_index = -1
        self.app.stop_batch_event.clear()

        self.app.batch_processing_thread = threading.Thread(target=self._run_batch_processing_thread, daemon=True, name="BatchProcessingThread")
        self.app.batch_processing_thread.start()

        self.app.show_batch_confirmation_dialog = False
        gui.batch_videos_data.clear()

    def _cancel_batch_processing_from_confirmation(self):
        """[Private] Called from the GUI when the user clicks 'Cancel'."""
        self.app.logger.info("Batch processing cancelled by user.", extra={'status_message': True})
        # Clear the confirmation dialog state
        self.app.show_batch_confirmation_dialog = False
        if self.app.gui_instance:
            self.app.gui_instance.batch_videos_data.clear()

    def abort_batch_processing(self):
        if not self.app.is_batch_processing_active:
            return

        self.app.logger.info("Aborting batch processing...", extra={'status_message': True})
        self.app.stop_batch_event.set()
        self.app.pause_batch_event.clear()  # Unblock any pause wait
        # Also signal the currently running stage analysis (if any) to stop
        self.app.stage_processor.abort_stage_processing()
        self.app.single_video_analysis_complete_event.set()  # Release the wait lock

    def pause_batch_processing(self):
        if self.app.is_batch_processing_active and not self.app.pause_batch_event.is_set():
            self.app.pause_batch_event.set()
            self.app.logger.info("Batch processing paused.", extra={'status_message': True})

    def resume_batch_processing(self):
        if self.app.is_batch_processing_active and self.app.pause_batch_event.is_set():
            self.app.pause_batch_event.clear()
            self.app.logger.info("Batch processing resumed.", extra={'status_message': True})

    def is_batch_paused(self) -> bool:
        return self.app.is_batch_processing_active and self.app.pause_batch_event.is_set()

    # --- Adaptive Batch Tuning Methods ---

    def _init_adaptive_tuning(self) -> AdaptiveTuningState:
        """Initialize adaptive tuning state from current settings."""
        p = self.app.stage_processor.num_producers_stage1
        c = self.app.stage_processor.num_consumers_stage1
        self.app.logger.info(f"Adaptive tuning initialized: P={p}, C={c}")
        return AdaptiveTuningState(producers=p, consumers=c)

    def _adaptive_tuning_record_result(self, state: AdaptiveTuningState, avg_fps: float, success: bool):
        """Record result from a video and hill-climb toward optimal settings."""
        p, c = state.current_producers, state.current_consumers
        state.history.append((p, c, avg_fps, success))

        if not success:
            # Config stalled/crashed — set ceiling and revert to best
            state.consumer_ceiling = c - 1
            state.current_producers = state.best_producers
            state.current_consumers = state.best_consumers
            state.status_message = f"Config P={p}/C={c} failed. Ceiling set to C={state.consumer_ceiling}. Reverted to P={state.best_producers}/C={state.best_consumers}"
            self.app.logger.warning(state.status_message)
            return

        if state.best_fps <= 0 or avg_fps > state.best_fps * 1.02:
            # Meaningful improvement (>2%)
            state.best_producers = p
            state.best_consumers = c
            state.best_fps = avg_fps
            state.consecutive_no_improvement = 0
            state.status_message = f"New best: P={p}/C={c} @ {avg_fps:.1f} FPS"
            self.app.logger.info(f"Adaptive tuning: {state.status_message}")
        else:
            state.consecutive_no_improvement += 1
            state.status_message = f"No improvement at P={p}/C={c} ({avg_fps:.1f} FPS vs best {state.best_fps:.1f}). Streak: {state.consecutive_no_improvement}"
            self.app.logger.info(f"Adaptive tuning: {state.status_message}")

        # Check convergence
        if state.consecutive_no_improvement >= 2:
            state.is_converged = True
            state.current_producers = state.best_producers
            state.current_consumers = state.best_consumers
            state.status_message = f"Converged at P={state.best_producers}/C={state.best_consumers} ({state.best_fps:.1f} FPS)"
            self.app.logger.info(f"Adaptive tuning: {state.status_message}")
            return

        # Hill-climb: try consumers +2
        next_c = c + 2
        # Respect ceiling
        if state.consumer_ceiling is not None and next_c > state.consumer_ceiling:
            next_c = c - 2
        # Bounds check
        if next_c < 1:
            next_c = 1
        # Skip already-tested configs
        tested_configs = {(h[0], h[1]) for h in state.history}
        if (p, next_c) in tested_configs:
            # Try the other direction
            alt_c = c - 2 if next_c == c + 2 else c + 2
            if state.consumer_ceiling is not None and alt_c > state.consumer_ceiling:
                alt_c = c  # Can't go either way — stay put and converge
            if alt_c < 1:
                alt_c = 1
            if (p, alt_c) in tested_configs or alt_c == c:
                # Nowhere new to go — converge
                state.is_converged = True
                state.current_producers = state.best_producers
                state.current_consumers = state.best_consumers
                state.status_message = f"Converged (exhausted options) at P={state.best_producers}/C={state.best_consumers} ({state.best_fps:.1f} FPS)"
                self.app.logger.info(f"Adaptive tuning: {state.status_message}")
                return
            next_c = alt_c

        state.current_consumers = next_c
        state.status_message = f"Next test: P={state.current_producers}/C={state.current_consumers}"
        self.app.logger.info(f"Adaptive tuning: {state.status_message}")

    def _adaptive_tuning_apply_best(self, state: AdaptiveTuningState):
        """Persist best discovered P/C to settings and stage processor."""
        if state is None or state.best_fps <= 0:
            return
        self.app.app_settings.set("num_producers_stage1", state.best_producers)
        self.app.app_settings.set("num_consumers_stage1", state.best_consumers)
        self.app.stage_processor.num_producers_stage1 = state.best_producers
        self.app.stage_processor.num_consumers_stage1 = state.best_consumers
        self.app.logger.info(f"Adaptive tuning: Persisted best settings P={state.best_producers}/C={state.best_consumers} ({state.best_fps:.1f} FPS)")

    def _run_batch_processing_thread(self):
        # Import the callback from app_cli_runner (where it's defined)
        from .app_cli_runner import cli_live_video_progress_callback

        # Initialize adaptive tuning if enabled
        if self.app.batch_adaptive_tuning_enabled:
            self.app.adaptive_tuning_state = self._init_adaptive_tuning()

        batch_failures = []  # (video_path, reason, timestamp)

        try:
            for i, video_data in enumerate(self.app.batch_video_paths):
                if self.app.stop_batch_event.is_set():
                    self.app.logger.info("Batch processing was aborted by user."); break

                # Pause checkpoint — wait between videos (event-driven, no busy loop)
                while self.app.pause_batch_event.is_set() and not self.app.stop_batch_event.is_set():
                    self.app.stop_batch_event.wait(0.5)
                if self.app.stop_batch_event.is_set():
                    self.app.logger.info("Batch processing was aborted while paused."); break

                self.app.current_batch_video_index = i
                video_path = video_data["path"]
                override_format = video_data["override_format"]
                video_basename = os.path.basename(video_path)

                # --- Temporarily Apply Format Override ---
                original_video_type_setting = self.app.processor.video_type_setting
                original_vr_format_setting = self.app.processor.vr_input_format
                if override_format != "Auto (Heuristic)":
                    self.app.logger.info(f"Applying format override: '{override_format}' for '{video_basename}'")
                    if override_format == "2D":
                        self.app.processor.set_active_video_type_setting("2D")
                    elif override_format.startswith("VR"):
                        try:
                            vr_format = override_format.split('(')[1].split(')')[0]
                            self.app.processor.set_active_vr_parameters(input_format=vr_format)
                        except IndexError:
                            self.app.logger.error(f"Could not parse VR format from override: '{override_format}'")

                print(f"\n--- Processing Video {i + 1} of {len(self.app.batch_video_paths)}: {video_basename} ---")

                batch_total = len(self.app.batch_video_paths)
                self.app.logger.info(f"Batch processing video {i + 1}/{batch_total}: {video_basename}",
                                     extra={'status_message': True})
                self.app.notify(f"Batch {i + 1}/{batch_total}: {video_basename}", "info", 3.0)

                # --- Pre-flight checks for overwrite strategy ---
                # This is now the very first step for each video in the loop.
                path_next_to_video = os.path.splitext(video_path)[0] + ".funscript"

                funscript_to_check = None
                if os.path.exists(path_next_to_video):
                    funscript_to_check = path_next_to_video

                if funscript_to_check:
                    if self.app.batch_overwrite_mode == 1:
                        # Mode 1: Process only if funscript is missing (skip any existing funscript)
                        self.app.logger.info(
                            f"Skipping '{video_basename}': Funscript already exists at '{funscript_to_check}'. (Mode: Only if Missing)")
                        continue

                    if self.app.batch_overwrite_mode == 0:
                        # Mode 0: Process all except own matching version (skip if up-to-date FunGen funscript exists)
                        funscript_data = self.app.file_manager._get_funscript_data(funscript_to_check)
                        if funscript_data:
                            author = funscript_data.get('author', '')
                            metadata = funscript_data.get('metadata', {})
                            # Ensure metadata is a dict before calling .get() on it
                            version = metadata.get('version', '') if isinstance(metadata, dict) else ''
                            if author.startswith("FunGen") and version == FUNSCRIPT_METADATA_VERSION:
                                self.app.logger.info(
                                    f"Skipping '{video_basename}': Up-to-date funscript from this program version already exists. (Mode: All except own matching version)")
                                continue

                    if self.app.batch_overwrite_mode == 2:
                        # Mode 2: Process ALL videos, including up-to-date FunGen funscript. Do not skip for any reason.
                        self.app.logger.info(
                            f"Processing '{video_basename}': Mode 2 selected, will process regardless of funscript existence or version.")
                # --- End of pre-flight checks ---

                open_success = self.app.file_manager.open_video_from_path(video_path)
                if not open_success:
                    self.app.logger.error(f"Failed to open video, skipping: {video_path}")
                    batch_failures.append((video_path, "Failed to open video", datetime.now().isoformat()))
                    continue

                time.sleep(1.0)
                if self.app.stop_batch_event.is_set(): break

                # Use the dynamically selected tracker name
                discovery = get_tracker_discovery()

                # Get the selected tracker using the name stored from GUI
                if hasattr(self.app, 'batch_tracker_name') and self.app.batch_tracker_name:
                    selected_tracker = discovery.get_tracker_info(self.app.batch_tracker_name)
                    if not selected_tracker:
                        self.app.logger.error(f"Invalid tracker name: {self.app.batch_tracker_name}. Skipping video.")
                        batch_failures.append((video_path, f"Invalid tracker: {self.app.batch_tracker_name}", datetime.now().isoformat()))
                        continue
                    if selected_tracker.requires_intervention or not selected_tracker.supports_batch:
                        self.app.logger.error(f"Tracker '{selected_tracker.display_name}' requires user intervention and cannot run in batch mode. Aborting.")
                        batch_failures.append((video_path, f"Tracker requires intervention: {selected_tracker.display_name}", datetime.now().isoformat()))
                        break
                    selected_mode = selected_tracker.internal_name
                else:
                    self.app.logger.error("No tracker selected for batch processing. Skipping video.")
                    batch_failures.append((video_path, "No tracker selected", datetime.now().isoformat()))
                    continue

                # Check tracker category to determine processing mode.
                # TOOL trackers (Oscillation, Chapter Maker, etc.) get resolved
                # to their runtime dispatch category based on the base class
                # they inherit; the declared category is UI grouping only and
                # would skip them here otherwise.
                from config.tracker_discovery import TrackerCategory
                runtime_category = discovery.get_runtime_category(self.app.batch_tracker_name)

                # --- OFFLINE MODES (Stage-based processing) ---
                if runtime_category == TrackerCategory.OFFLINE:
                    # Set processing speed to MAX_SPEED for batch/CLI offline processing
                    from config.constants import ProcessingSpeedMode
                    original_speed_mode = self.app.app_state_ui.selected_processing_speed_mode
                    self.app.app_state_ui.selected_processing_speed_mode = ProcessingSpeedMode.MAX_SPEED
                    self.app.logger.info("Set processing speed to MAX_SPEED for batch offline processing")

                    # Override preprocessed video setting for batch (default off to save disk)
                    original_save_preprocessed = getattr(self.app.stage_processor, 'save_preprocessed_video', False)
                    self.app.stage_processor.save_preprocessed_video = getattr(
                        self.app, 'batch_save_preprocessed_video', False)

                    # Apply adaptive tuning P/C if active and not converged
                    if self.app.adaptive_tuning_state and not self.app.adaptive_tuning_state.is_converged:
                        self.app.stage_processor.num_producers_stage1 = self.app.adaptive_tuning_state.current_producers
                        self.app.stage_processor.num_consumers_stage1 = self.app.adaptive_tuning_state.current_consumers
                        self.app.logger.info(f"Adaptive tuning: Using P={self.app.adaptive_tuning_state.current_producers}/C={self.app.adaptive_tuning_state.current_consumers} for {video_basename}")

                    self.app.single_video_analysis_complete_event.clear()
                    self.app.save_and_reset_complete_event.clear()
                    self.app.stage_processor.start_full_analysis(processing_mode=selected_mode)

                    # Block until the analysis for this single video is done
                    self.app.single_video_analysis_complete_event.wait()
                    if self.app.stop_batch_event.is_set(): break

                    # Record adaptive tuning result from completed video
                    if self.app.adaptive_tuning_state and not self.app.adaptive_tuning_state.is_converged:
                        fps_val = 0.0
                        success = True
                        try:
                            fps_str_raw = self.app.stage_processor.stage1_final_fps_str
                            if fps_str_raw and "FPS" in fps_str_raw:
                                fps_val = float(fps_str_raw.replace("FPS", "").strip())
                            else:
                                success = False
                        except (ValueError, TypeError, AttributeError):
                            success = False
                        self._adaptive_tuning_record_result(self.app.adaptive_tuning_state, fps_val, success)

                    # Pause checkpoint — wait after video analysis (event-driven)
                    while self.app.pause_batch_event.is_set() and not self.app.stop_batch_event.is_set():
                        self.app.stop_batch_event.wait(0.5)
                    if self.app.stop_batch_event.is_set(): break

                    # --- LOAD RESULTS IN CLI ---
                    if not self.app.gui_instance:
                        self.app.logger.info("CLI Mode: Loading analysis results into funscript processor.")
                        results_package = self.app.stage_processor.last_analysis_result
                        if results_package and "results_dict" in results_package:
                            results_dict = results_package["results_dict"]
                            result_script = results_dict.get("funscript")
                            if result_script:
                                primary = result_script.primary_actions
                                secondary = result_script.secondary_actions
                                self.app.logger.info(f"CLI Mode: Found funscript with {len(primary)} primary, {len(secondary)} secondary actions")
                                self.app.funscript_processor.clear_timeline_history_and_set_new_baseline(1, primary, "Analysis (CLI)")
                                self.app.funscript_processor.clear_timeline_history_and_set_new_baseline(2, secondary, "Analysis (CLI)")
                            else:
                                self.app.logger.warning(f"CLI Mode: results_dict has no 'funscript' key. Keys: {list(results_dict.keys())}")
                        else:
                            self.app.logger.error("CLI Mode: Analysis finished but no results were found to load.")
                    # --- END OF ADDED BLOCK ---

                    if not self.app.gui_instance:
                        self.app.on_offline_analysis_completed({"video_path": video_path})

                    # Block until saving and resetting are confirmed complete
                    self.app.logger.debug("Batch loop: Waiting for save/reset signal...")
                    self.app.save_and_reset_complete_event.wait(timeout=120)
                    self.app.logger.debug("Batch loop: Save/reset signal received. Proceeding.")

                    # Restore original settings
                    self.app.app_state_ui.selected_processing_speed_mode = original_speed_mode
                    self.app.stage_processor.save_preprocessed_video = original_save_preprocessed

                # --- LIVE MODES (Real-time tracking) ---
                elif runtime_category == TrackerCategory.LIVE:
                    self.app.logger.info(f"Running live mode: {selected_tracker.display_name} for {os.path.basename(video_path)}")

                    # Set processing speed to MAX_SPEED for batch/CLI live tracking
                    from config.constants import ProcessingSpeedMode
                    original_speed_mode = self.app.app_state_ui.selected_processing_speed_mode
                    self.app.app_state_ui.selected_processing_speed_mode = ProcessingSpeedMode.MAX_SPEED
                    self.app.logger.info("Set processing speed to MAX_SPEED for batch live tracking")

                    self.app.tracker.set_tracking_mode(selected_mode)

                    # Auto-set axis for axis projection trackers in CLI/batch mode
                    if "axis_projection" in selected_mode:
                        current_tracker = self.app.tracker.get_current_tracker()
                        if current_tracker and hasattr(current_tracker, 'set_axis'):
                            # Set default horizontal axis across middle of frame for VR SBS videos
                            margin = 50
                            width, height = 640, 640  # Processing frame size
                            axis_A = (margin, height // 2)  # Left side
                            axis_B = (width - margin, height // 2)  # Right side
                            result = current_tracker.set_axis(axis_A, axis_B)
                            self.app.logger.info(f"Auto-set axis for {selected_mode}: A={axis_A}, B={axis_B}, result={result}")

                    self.app.tracker.start_tracking()
                    self.app.processor.set_tracker_processing_enabled(True)

                    # Process the entire video from start to finish
                    self.app.processor.start_processing(
                        start_frame=0,
                        end_frame=-1,
                        cli_progress_callback=cli_live_video_progress_callback
                    )

                    # Block until the live processing thread finishes
                    if self.app.processor.processing_thread and self.app.processor.processing_thread.is_alive():
                        self.app.processor.processing_thread.join()

                    # Restore original processing speed mode
                    self.app.app_state_ui.selected_processing_speed_mode = original_speed_mode
                    self.app.logger.info("Restored original processing speed mode")

                    # This call now handles all post-processing AND saving/copying
                    self.app.on_processing_stopped(was_scripting_session=True)

                self.app.processor.video_type_setting = original_video_type_setting
                self.app.processor.vr_input_format = original_vr_format_setting
                self.app.logger.debug("Restored original video format settings for next iteration.")

                if self.app.stop_batch_event.is_set():
                    break

        except Exception as e:
            self.app.logger.error(f"An error occurred during the batch process: {e}", exc_info=True)
            batch_failures.append(("(batch process)", str(e), datetime.now().isoformat()))
        finally:
            # Write batch error report if any failures occurred
            if batch_failures:
                self._write_batch_error_report(batch_failures)

            # Persist best adaptive tuning settings
            if self.app.adaptive_tuning_state:
                self._adaptive_tuning_apply_best(self.app.adaptive_tuning_state)
                self.app.adaptive_tuning_state = None
            self.app.is_batch_processing_active = False
            self.app.current_batch_video_index = -1
            self.app.batch_video_paths = []
            self.app.stop_batch_event.clear()
            self.app.pause_batch_event.clear()
            self.app.logger.info("Batch processing finished.", extra={'status_message': True})

    def _write_batch_error_report(self, failures):
        """Write a batch error report file listing all failed videos."""
        try:
            output_dir = self.app.app_settings.get("output_folder_path", "output")
            os.makedirs(output_dir, exist_ok=True)
            date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{date_str}_FunGen_batch_errors.txt"
            filepath = os.path.join(output_dir, filename)

            with open(filepath, "w", encoding="utf-8") as f:
                f.write(f"FunGen Batch Error Report\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Total failures: {len(failures)}\n")
                f.write("=" * 60 + "\n\n")
                for video_path, reason, ts in failures:
                    f.write(f"[{ts}] {os.path.basename(video_path)}\n")
                    f.write(f"  Path:   {video_path}\n")
                    f.write(f"  Reason: {reason}\n\n")

            self.app.logger.warning(
                f"Batch errors ({len(failures)} failures) saved to: {filepath}",
                extra={"status_message": True})
        except Exception as e:
            self.app.logger.error(f"Failed to write batch error report: {e}")

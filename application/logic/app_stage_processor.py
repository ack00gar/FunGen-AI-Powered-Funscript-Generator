import os
import threading
import math
import time
from queue import Queue
from typing import Optional, List, Dict, Any, Tuple
import msgpack
import numpy as np
from bisect import bisect_left, bisect_right
import multiprocessing
import gc

import detection.cd.stage_1_cd as stage1_module
import detection.cd.stage_2_cd as stage2_module
import detection.cd.stage_3_of_processor as stage3_module

from config import constants
from config.constants import TrackerMode, SCENE_DETECTION_DEFAULT_THRESHOLD
from application.utils.video_segment import VideoSegment


class AppStageProcessor:
    def __init__(self, app_logic_instance):
        self.app = app_logic_instance
        self.logger = self.app.logger
        self.app_settings = self.app.app_settings

        # --- Threading Configuration ---
        self.save_preprocessed_video: bool = False
        self.update_settings_from_app()

        self.stage_completion_event: Optional[threading.Event] = None

        # --- Analysis State ---
        self.full_analysis_active: bool = False
        self.current_analysis_stage: int = 0
        self.stage_thread: Optional[threading.Thread] = None
        self.stop_stage_event = multiprocessing.Event()
        self.gui_event_queue = Queue()

        # --- Status and Progress Tracking ---
        self.reset_stage_status(stages=("stage1", "stage2", "stage3"))

        # --- State for Scene Detection ---
        self.scene_detection_active: bool = False
        self.scene_detection_status: str = "Idle"
        self.scene_detection_thread: Optional[threading.Thread] = None

        self.scene_detection_time_elapsed_str: str = "00:00:00"
        self.scene_detection_processing_fps_str: str = "0 FPS"
        self.scene_detection_eta_str: str = "N/A"

        # --- Rerun Flags ---
        self.force_rerun_stage1: bool = False
        self.force_rerun_stage2_segmentation: bool = False

        # --- Stage 2 Overlay Data ---
        self.stage2_overlay_data: Optional[List[Dict]] = None
        self.stage2_overlay_data_map: Optional[Dict[int, Dict]] = None

        # --- Fallback Constants ---
        self.S2_TOTAL_MAIN_STEPS_FALLBACK = getattr(stage2_module, 'ATR_PASS_COUNT', 6)

        self.refinement_analysis_active: bool = False
        self.refinement_thread: Optional[threading.Thread] = None

        self.frame_range_override: Optional[Tuple[int, int]] = None

    def start_interactive_refinement_analysis(self, chapter, track_id):
        if self.full_analysis_active or self.refinement_analysis_active:
            self.logger.warning("Another analysis is already running.", extra={'status_message': True})
            return
        # Check for the correct data map that is always available after Stage 2.
        if not self.stage2_overlay_data_map:
            self.logger.error("Cannot start refinement: Stage 2 overlay data map is not available.",
                              extra={'status_message': True})
            return

        self.refinement_analysis_active = True
        self.stop_stage_event.clear()

        self.refinement_thread = threading.Thread(
            target=self._run_interactive_refinement_thread,
            args=(chapter, track_id),
            daemon=True
        )
        self.refinement_thread.start()

    def _run_interactive_refinement_thread(self, chapter, track_id):
        try:
            # 1. PRE-SCAN: Use the corrected data source.
            track_id_positions = {}
            for frame_id in range(chapter.start_frame_id, chapter.end_frame_id + 1):
                # Read from stage2_overlay_data_map.
                frame_data = self.stage2_overlay_data_map.get(frame_id)
                if not frame_data: continue
                # The data is now a dictionary, not a FrameObject.
                for box_dict in frame_data.get("yolo_boxes", []):
                    if box_dict.get("track_id") == track_id:
                        track_id_positions[frame_id] = box_dict
                        break

            if not track_id_positions:
                self.logger.warning(f"Track ID {track_id} not found in chapter. Aborting refinement.")
                return

            # 2. BUILD REFINED TRACK (with interpolation).
            refined_track = {}
            sorted_known_frames = sorted(track_id_positions.keys())

            for frame_id in range(chapter.start_frame_id, chapter.end_frame_id + 1):
                if frame_id in track_id_positions:
                    refined_track[frame_id] = track_id_positions[frame_id]
                else:
                    prev_frames = [f for f in sorted_known_frames if f < frame_id]
                    next_frames = [f for f in sorted_known_frames if f > frame_id]
                    prev_known = prev_frames[-1] if prev_frames else None
                    next_known = next_frames[0] if next_frames else None

                    if prev_known and next_known:
                        t = (frame_id - prev_known) / float(next_known - prev_known)
                        prev_box_dict = track_id_positions[prev_known]
                        next_box_dict = track_id_positions[next_known]

                        # Interpolate using numpy arrays for vectorization.
                        interp_bbox = np.array(prev_box_dict['bbox']) + t * (
                                    np.array(next_box_dict['bbox']) - np.array(prev_box_dict['bbox']))

                        # Create a new dictionary for the interpolated box.
                        refined_track[frame_id] = {
                            "bbox": interp_bbox.tolist(),
                            "track_id": track_id,
                            "class_name": prev_box_dict.get('class_name'),
                            "status": "Interpolated"
                        }

            # 3. RE-CALCULATE FUNSCRIPT
            raw_actions = []
            fps = self.app.processor.video_info.get('fps', 30.0)
            if fps > 0:
                for frame_id, box_dict in refined_track.items():
                    if box := box_dict.get('bbox'):
                        distance = 100 - (box[3] / self.app.yolo_input_size) * 100
                        timestamp_ms = int(round((frame_id / fps) * 1000))
                        raw_actions.append({"at": timestamp_ms, "pos": int(np.clip(distance, 0, 100))})

            # --- 4. DYNAMIC AMPLIFICATION (Rolling Window with Percentiles) ---
            if not raw_actions: return

            amplified_actions = []
            window_ms = 4000  # Analyze a 4-second window around each point.

            # Create a sorted list of timestamps for efficient searching
            action_timestamps = [a['at'] for a in raw_actions]

            for i, action in enumerate(raw_actions):
                current_time = action['at']

                # Define the local window for analysis
                start_window_time = current_time - (window_ms / 2)
                end_window_time = current_time + (window_ms / 2)

                # Efficiently find the indices of actions within this time window
                start_idx = bisect_left(action_timestamps, start_window_time)
                end_idx = bisect_right(action_timestamps, end_window_time)

                local_actions = raw_actions[start_idx:end_idx]

                if not local_actions:
                    amplified_actions.append(action)  # Keep original if no neighbors
                    continue

                local_positions = [a['pos'] for a in local_actions]

                # Use percentiles to find the effective min/max, ignoring outliers.
                # This is similar to the robust logic in `scale_points_to_range`.
                effective_min = np.percentile(local_positions, 10)
                effective_max = np.percentile(local_positions, 90)
                effective_range = effective_max - effective_min

                if effective_range < 5:  # If local motion is negligible, don't amplify.
                    new_pos = action['pos']
                else:
                    # Normalize the current point's position within its local effective range
                    normalized_pos = (action['pos'] - effective_min) / effective_range
                    # Clip the value to handle points outside the percentile range (the outliers)
                    clipped_normalized_pos = np.clip(normalized_pos, 0.0, 1.0)
                    # Scale the normalized position to the full 0-100 range
                    new_pos = int(round(clipped_normalized_pos * 100))

                amplified_actions.append({"at": action['at'], "pos": new_pos})

            # 5. SEND AMPLIFIED RESULT TO MAIN THREAD
            if amplified_actions:
                payload = {"chapter": chapter, "new_actions": amplified_actions}
                self.gui_event_queue.put(("refinement_completed", payload, None))


        finally:
            self.refinement_analysis_active = False

    # REFACTORED for maintainability
    # Create as many stages you want without having to make a new function
    # Simply pass in a tuple of the stage name(s) you want to reset. stage
    def reset_stage_status(self, stages=("stage1", "stage2", "stage3")):
        if "stage1" in stages:
            self.stage1_status_text = "Not run."
            self.stage1_progress_value = 0.0
            self.stage1_progress_label = ""
            self.stage1_time_elapsed_str = "00:00:00"
            self.stage1_processing_fps_str = "0 FPS"
            self.stage1_instant_fps_str = "0 FPS"
            self.stage1_eta_str = "N/A"
            self.stage1_frame_queue_size = 0
            self.stage1_result_queue_size = 0
            self.stage1_final_elapsed_time_str = ""
            self.stage1_final_fps_str = ""
            # self.app.file_manager.stage1_output_msgpack_path = None
        if "stage2" in stages:
            self.stage2_status_text = "Not run."
            self.stage2_progress_value = 0.0
            self.stage2_progress_label = ""
            self.stage2_main_progress_value = 0.0
            self.stage2_main_progress_label = ""
            self.stage2_sub_progress_value = 0.0
            self.stage2_sub_progress_label = ""
            self.stage2_final_elapsed_time_str = ""
        if "stage3" in stages:
            self.stage3_status_text = "Not run."
            self.stage3_current_segment_label = ""
            self.stage3_segment_progress_value = 0.0
            self.stage3_overall_progress_label = ""
            self.stage3_overall_progress_value = 0.0
            self.stage3_time_elapsed_str = "00:00:00"
            self.stage3_processing_fps_str = "0 FPS"
            self.stage3_eta_str = "N/A"
            self.stage3_final_elapsed_time_str = ""
            self.stage3_final_fps_str = ""

    # --- Thread target for running scene detection ---
    def _run_scene_detection_thread(self, threshold=SCENE_DETECTION_DEFAULT_THRESHOLD):
        try:
            # 1. Run detection in the background
            scene_list = self.app.processor.detect_scenes(threshold=threshold)  # Use the threshold from the UI

            # 2. Check if the task was aborted
            if self.stop_stage_event.is_set() or not scene_list:
                return

            # 3. Create chapters from the results on the main thread via the queue
            self.gui_event_queue.put(("scene_detection_finished", scene_list, "Creating chapters..."))

        except Exception as e:
            self.logger.error(f"Scene detection thread failed: {e}", exc_info=True)
        finally:
            self.scene_detection_active = False

    # --- Public method to start the process from the UI ---
    def start_scene_detection_analysis(self, threshold=SCENE_DETECTION_DEFAULT_THRESHOLD):
        if self.full_analysis_active or self.scene_detection_active:
            self.logger.warning("Another analysis is already running.", extra={'status_message': True})
            return
        if not self.app.processor or not self.app.processor.is_video_open():
            self.logger.error("Cannot start scene detection: No video loaded.")
            return

        self.scene_detection_active = True
        self.scene_detection_status = "Starting..."
        self.stop_stage_event.clear()

        self.scene_detection_thread = threading.Thread(target=self._run_scene_detection_thread, args=(threshold,), daemon=True)
        self.scene_detection_thread.start()

    def _stage1_progress_callback(self, current, total, message="Processing...", time_elapsed=0.0, avg_fps=0.0, instant_fps=0.0, eta_seconds=0.0):
        progress = float(current) / total if total > 0 else -1.0
        progress_data = {
            "message": message, "current": current, "total": total,
            "time_elapsed": time_elapsed, "avg_fps": avg_fps, "instant_fps": instant_fps, "eta": eta_seconds
        }
        self.gui_event_queue.put(("stage1_progress_update", progress, progress_data))

    def _stage2_progress_callback(self, main_info_from_module, sub_info_from_module, force_update=False):
        actual_main_step_tuple_for_queue: tuple
        if isinstance(main_info_from_module, str):
            actual_main_step_tuple_for_queue = (-1, 0, main_info_from_module)
        elif isinstance(main_info_from_module, tuple) and len(main_info_from_module) == 3:
            actual_main_step_tuple_for_queue = main_info_from_module
        else:
            self.logger.warning(
                f"Malformed main_info_from_module in _stage2_progress_callback: {main_info_from_module}. Using placeholder.")
            actual_main_step_tuple_for_queue = (-1, 0, "Error: Invalid Main Step Data")

        actual_sub_step_tuple_for_queue: tuple
        if isinstance(sub_info_from_module, tuple) and len(sub_info_from_module) == 3:
            actual_sub_step_tuple_for_queue = sub_info_from_module
        elif isinstance(sub_info_from_module, (int, float)):
            self.logger.warning(
                f"Received numerical sub_info_from_module in _stage2_progress_callback: {sub_info_from_module}. Interpreting as sub_task_name with placeholder progress.")
            actual_sub_step_tuple_for_queue = (0, 0, f"Sub-status: {sub_info_from_module}")
        else:
            self.logger.warning(
                f"Malformed sub_info_from_module in _stage2_progress_callback: {sub_info_from_module}. Using placeholder.")
            actual_sub_step_tuple_for_queue = (0, 0, "Error: Invalid Sub Step Data")

        current_mode = self.app.app_state_ui.selected_tracker_mode
        if current_mode == TrackerMode.OFFLINE_3_STAGE:
            num_segmentation_main_steps = 3
            if actual_main_step_tuple_for_queue[1] > 0:
                original_current, original_total, name = actual_main_step_tuple_for_queue
                # REFACTORED for readability and maintainability
                if original_total > 0:
                    scaled_current = min(num_segmentation_main_steps, int(
                        (original_current / original_total) * num_segmentation_main_steps))
                else:
                    scaled_current = original_current
                actual_main_step_tuple_for_queue = (scaled_current, num_segmentation_main_steps, name)
        self.gui_event_queue.put(("stage2_dual_progress", actual_main_step_tuple_for_queue, actual_sub_step_tuple_for_queue))

    def _stage3_progress_callback(self, current_chapter_idx: int, total_chapters: int, chapter_name: str, current_chunk_idx: int, total_chunks: int, total_frames_processed_overall, total_frames_to_process_overall, processing_fps = 0.0, time_elapsed = 0.0, eta_seconds = 0.0):
        # REFACTORED for readability and maintainability
        if total_frames_to_process_overall > 0:
            overall_progress = float(total_frames_processed_overall) / total_frames_to_process_overall
        else:
            overall_progress = 0.0

        progress_data = {
            "current_chapter_idx": current_chapter_idx,
            "total_chapters": total_chapters,
            "chapter_name": chapter_name,
            "current_chunk_idx": current_chunk_idx,
            "total_chunks": total_chunks,
            "overall_progress": overall_progress,
            "total_frames_processed_overall": total_frames_processed_overall,
            "total_frames_to_process_overall": total_frames_to_process_overall,
            "fps": processing_fps,
            "time_elapsed": time_elapsed,
            "eta": eta_seconds
        }
        self.gui_event_queue.put(("stage3_progress_update", progress_data, None))

    def start_full_analysis(self, override_producers: Optional[int] = None,
                            override_consumers: Optional[int] = None,
                            completion_event: Optional[threading.Event] = None,
                            frame_range_override: Optional[Tuple[int, int]] = None,
                            is_autotune_run: bool = False):
        fm = self.app.file_manager
        fs_proc = self.app.funscript_processor

        if not fm.video_path:
            self.logger.info("Please load a video first.", extra={'status_message': True})
            return
        if self.full_analysis_active or (self.app.processor and self.app.processor.is_processing):
            self.logger.info("A process is already running.", extra={'status_message': True})
            return
        if not stage1_module or not stage2_module or not stage3_module:
            self.logger.error("Stage 1, Stage 2, or Stage 3 processing module not available.", extra={'status_message': True})
            return
        if not self.app.yolo_det_model_path or not os.path.exists(self.app.yolo_det_model_path):
            self.logger.error(f"Stage 1 Model not found: {self.app.yolo_det_model_path}", extra={'status_message': True})
            return

        self.full_analysis_active = True
        self.current_analysis_stage = 0
        self.stop_stage_event.clear()
        self.stage_completion_event = completion_event
        self.frame_range_override = frame_range_override

        # Store the flag for the thread to use it
        self.is_autotune_run_for_thread = is_autotune_run

        # Store the overrides to be used by the thread
        self.override_producers = override_producers
        self.override_consumers = override_consumers

        selected_mode = self.app.app_state_ui.selected_tracker_mode

        range_is_active, range_start_frame, range_end_frame = fs_proc.get_effective_scripting_range()

        # Determine the target msgpack path using the centralized file manager
        full_msgpack_path = fm.get_output_path_for_file(fm.video_path, ".msgpack")

        # If the file manager fails to return a path (e.g., in a stateless batch context),
        # construct a default path manually.
        if not full_msgpack_path:
            self.logger.warning(
                "get_output_path_for_file returned None. Manually constructing a fallback path for msgpack.")
            # Get the base output folder from settings.
            base_output_dir = self.app.app_settings.get("output_folder_path", constants.DEFAULT_OUTPUT_FOLDER)

            # Create a subdirectory for the video to keep files organized.
            video_filename_no_ext = os.path.splitext(os.path.basename(fm.video_path))[0]
            video_specific_output_dir = os.path.join(base_output_dir, video_filename_no_ext)

            # Create the directory if it doesn't exist.
            os.makedirs(video_specific_output_dir, exist_ok=True)

            # Construct the full path for the msgpack file.
            full_msgpack_path = os.path.join(video_specific_output_dir, video_filename_no_ext + ".msgpack")
            self.logger.info(f"Fallback msgpack path set to: {full_msgpack_path}")

        full_msgpack_exists = os.path.exists(full_msgpack_path)

        # Use the override if it exists, otherwise check for active scripting range
        if self.frame_range_override:
            start_f_name, end_f_name = self.frame_range_override
            range_specific_path = fm.get_output_path_for_file(fm.video_path, f"_range_{start_f_name}-{end_f_name}.msgpack")
            fm.stage1_output_msgpack_path = range_specific_path
            # For autotuner, we always want to rerun
            should_run_s1 = True
        elif range_is_active:
            if full_msgpack_exists and not self.force_rerun_stage1:
                fm.stage1_output_msgpack_path = full_msgpack_path
                should_run_s1 = False
            else:
                start_f_name = range_start_frame if range_start_frame is not None else 0
                end_f_name = range_end_frame if range_end_frame is not None else 'end'
                range_specific_path = fm.get_output_path_for_file(fm.video_path, f"_range_{start_f_name}-{end_f_name}.msgpack")
                fm.stage1_output_msgpack_path = range_specific_path
                should_run_s1 = self.force_rerun_stage1 or not os.path.exists(range_specific_path)
        else:
            fm.stage1_output_msgpack_path = full_msgpack_path
            should_run_s1 = self.force_rerun_stage1 or not full_msgpack_exists

        if not should_run_s1:
            self.stage1_status_text = f"Using existing: {os.path.basename(fm.stage1_output_msgpack_path or '')}"
            self.stage1_progress_value = 1.0
        else:
            self.reset_stage_status(stages=("stage1",)) # Reset all S1 state including final time
            self.stage1_status_text = "Queued..."

        self.reset_stage_status(stages=("stage2", "stage3"))
        self.stage2_status_text = "Queued..."
        if selected_mode == TrackerMode.OFFLINE_3_STAGE:
            self.stage3_status_text = "Queued..."

        self.logger.info("Starting Full Analysis sequence...", extra={'status_message': True})
        self.stage_thread = threading.Thread(target=self._run_full_analysis_thread_target, daemon=True)
        self.stage_thread.start()
        self.app.energy_saver.reset_activity_timer()

    def _run_full_analysis_thread_target(self):
        fm = self.app.file_manager
        fs_proc = self.app.funscript_processor
        stage1_success = False

        if self.app.is_batch_processing_active:
            # Batch processing uses an index, so we map it back to our enum key
            batch_mode_map = {
                #0: TrackerMode.LIVE_YOLO_ROI, 1: TrackerMode.LIVE_USER_ROI,
                1: TrackerMode.OFFLINE_2_STAGE, 0: TrackerMode.OFFLINE_3_STAGE
            }
            selected_mode = batch_mode_map.get(self.app.batch_processing_method_idx, TrackerMode.OFFLINE_2_STAGE)
            self.logger.info(f"[Thread] Using batch processing mode: {selected_mode.name}")
        else:
            selected_mode = self.app.app_state_ui.selected_tracker_mode

        try:
            # --- Stage 1 ---
            self.current_analysis_stage = 1
            range_is_active, range_start_frame, range_end_frame = fs_proc.get_effective_scripting_range()

            # Use the override if it exists, otherwise determine range normally
            frame_range_for_s1 = self.frame_range_override if self.frame_range_override else \
                ((range_start_frame, range_end_frame) if range_is_active else None)

            target_s1_path = fm.stage1_output_msgpack_path

            # Determine if this is an autotuner run
            is_autotune_context = self.frame_range_override is not None

            is_s1_data_source_ranged = (frame_range_for_s1 is not None) and target_s1_path and "_range_" in os.path.basename(target_s1_path)
            should_skip_stage1 = not self.force_rerun_stage1 and target_s1_path and os.path.exists(target_s1_path)

            if should_skip_stage1 and not self.frame_range_override:  # Never skip for autotuner
                stage1_success = True
                self.logger.info(f"[Thread] Stage 1 skipped, using: {target_s1_path}")
                self.gui_event_queue.put(("stage1_completed", "00:00:00 (Cached)", "Cached"))
            else:
                stage1_results = self._execute_stage1_logic(
                    frame_range=frame_range_for_s1,
                    output_path=target_s1_path,
                    num_producers_override=getattr(self, 'override_producers', None),
                    num_consumers_override=getattr(self, 'override_consumers', None),
                    is_autotune_run=is_autotune_context
                )
                stage1_success = stage1_results.get("success", False)
                if stage1_success:
                    max_fps_str = f"{stage1_results.get('max_fps', 0.0):.2f} FPS"
                    # Directly set the final FPS string to avoid the race condition.
                    # The autotuner reads this value immediately after the completion event is set.
                    self.stage1_final_fps_str = max_fps_str
                    self.gui_event_queue.put(("stage1_completed", self.stage1_time_elapsed_str, max_fps_str))

            if self.stop_stage_event.is_set() or not stage1_success:
                self.logger.info("[Thread] Exiting after Stage 1 due to stop event or failure.")
                if "Queued" in self.stage2_status_text:
                    self.gui_event_queue.put(("stage2_status_update", "Skipped", "S1 Failed/Aborted"))
                if "Queued" in self.stage3_status_text:
                    self.gui_event_queue.put(("stage3_status_update", "Skipped", "S1 Failed/Aborted"))
                return

            # If this is an autotuner run (indicated by frame_range_override),
            # our job is done after Stage 1. The 'finally' block will handle cleanup.
            if self.frame_range_override is not None:
                self.logger.info("[Thread] Autotuner context detected. Finishing after Stage 1.")
                return

            # --- Stage 2 ---
            self.current_analysis_stage = 2

            s2_overlay_path = None
            if fm.video_path:
                try:
                    s2_overlay_path = fm.get_output_path_for_file(fm.video_path, "_stage2_overlay.msgpack")
                except Exception as e:
                    self.logger.error(f"Error determining S2 overlay path: {e}")

            generate_s2_funscript_actions = selected_mode == TrackerMode.OFFLINE_2_STAGE

            s2_start_time = time.time()
            stage2_run_results = self._execute_stage2_logic(
                s2_overlay_output_path=s2_overlay_path,
                generate_funscript_actions=generate_s2_funscript_actions,
                is_ranged_data_source=is_s1_data_source_ranged
            )
            s2_end_time = time.time()
            stage2_success = stage2_run_results.get("success", False)

            if stage2_success:
                s2_elapsed_s = s2_end_time - s2_start_time
                s2_elapsed_str = f"{int(s2_elapsed_s // 3600):02d}:{int((s2_elapsed_s % 3600) // 60):02d}:{int(s2_elapsed_s % 60):02d}"
                self.gui_event_queue.put(("stage2_completed", s2_elapsed_str, None))

            if stage2_success and s2_overlay_path and os.path.exists(s2_overlay_path):
                self.gui_event_queue.put(("load_s2_overlay", s2_overlay_path, None))

            if stage2_success:
                video_segments_for_funscript = stage2_run_results["data"].get("video_segments", [])
                s2_output_data = stage2_run_results.get("data", {})

            if self.stop_stage_event.is_set() or not stage2_success:
                self.logger.info("[Thread] Exiting after Stage 2 due to stop event or failure.")
                if selected_mode == TrackerMode.OFFLINE_3_STAGE and "Queued" in self.stage3_status_text:
                     self.gui_event_queue.put(("stage3_status_update", "Skipped", "S2 Failed/Aborted"))
                return

            # --- Stage 3 (or Finish) ---
            if selected_mode == TrackerMode.OFFLINE_2_STAGE:
                if stage2_success:
                    packaged_data = {
                        "results_dict": s2_output_data,
                        "was_ranged": is_s1_data_source_ranged,
                        "range_frames": frame_range_for_s1 or (range_start_frame, range_end_frame)
                    }
                    self.gui_event_queue.put(("stage2_results_success", packaged_data, s2_overlay_path))

                completion_payload = {
                    "message": "AI CV (2-Stage) analysis completed successfully.",
                    "status": "Completed",
                    "video_path": fm.video_path,
                    "video_segments": video_segments_for_funscript
                }
                self.gui_event_queue.put(("analysis_message", completion_payload, None))
            elif selected_mode == TrackerMode.OFFLINE_3_STAGE:
                self.current_analysis_stage = 3
                atr_segments_objects = s2_output_data.get("atr_segments_objects", [])
                video_segments_for_gui = s2_output_data.get("video_segments", [])

                if video_segments_for_gui:
                    self.gui_event_queue.put(("stage2_results_success_segments_only", video_segments_for_gui, None))

                effective_range_is_active = frame_range_for_s1 is not None
                effective_start_frame = frame_range_for_s1[0] if effective_range_is_active else range_start_frame
                effective_end_frame = frame_range_for_s1[1] if effective_range_is_active else range_end_frame

                segments_for_s3 = self._filter_segments_for_range(atr_segments_objects, effective_range_is_active,
                                                                  effective_start_frame, effective_end_frame)

                if not segments_for_s3:
                    self.gui_event_queue.put(("analysis_message", "No relevant segments in range for Stage 3.", "Info"))
                    return

                self.app.s2_frame_objects_map_for_s3 = {fo.frame_id: fo for fo in s2_output_data.get("all_s2_frame_objects_list", [])}
                preprocessed_path_for_s3 = self.app.file_manager.preprocessed_video_path

                self.logger.info(f"Starting Stage 3 with {preprocessed_path_for_s3}.")

                stage3_success = self._execute_stage3_optical_flow_module(segments_for_s3, preprocessed_path_for_s3)

                if stage3_success:
                    self.gui_event_queue.put(("stage3_completed", self.stage3_time_elapsed_str, self.stage3_processing_fps_str))

                if self.stop_stage_event.is_set():
                    return

                if stage3_success and self.app.s2_frame_objects_map_for_s3:
                    if s2_overlay_path:
                        self.logger.info(f"Stage 3 complete. Rewriting augmented overlay data to {os.path.basename(s2_overlay_path)}")
                        try:
                            # The map was modified in-place by Stage 3
                            all_frames_data = [fo.to_overlay_dict() for fo in self.app.s2_frame_objects_map_for_s3.values()]

                            def numpy_default_handler(obj):
                                if isinstance(obj, np.integer):
                                    return int(obj)
                                elif isinstance(obj, np.floating):
                                    return float(obj)
                                elif isinstance(obj, np.ndarray):
                                    return obj.tolist()
                                raise TypeError(f"Object of type {obj.__class__.__name__} is not serializable for msgpack")

                            if all_frames_data is not None:
                                packed_data = msgpack.packb(all_frames_data, use_bin_type=True, default=numpy_default_handler)
                                if packed_data is not None:
                                    with open(s2_overlay_path, 'wb') as f:
                                        f.write(packed_data)
                                    self.logger.info("Successfully rewrote Stage 2 overlay file with Stage 3 data.")
                                else:
                                    self.logger.warning("msgpack.packb returned None, not writing overlay file.")
                            else:
                                self.logger.warning("all_frames_data is None, not writing overlay file.")

                            # Send event to GUI to (re)load the updated data
                            self.gui_event_queue.put(("load_s2_overlay", s2_overlay_path, None))

                        except Exception as e:
                            self.logger.error(f"Failed to save augmented Stage 3 overlay data: {e}", exc_info=True)
                    else:
                        self.logger.warning("Stage 3 completed, but no S2 overlay path was available to overwrite.")

                if stage3_success:
                    completion_payload = {
                        "message": "AI CV (3-Stage) analysis completed successfully.",
                        "status": "Completed",
                        "video_path": fm.video_path
                    }
                    self.gui_event_queue.put(("analysis_message", completion_payload, None))

        finally:
            self.full_analysis_active = False
            self.current_analysis_stage = 0
            self.frame_range_override = None
            if self.stage_completion_event:
                self.stage_completion_event.set()

            # Clear the large data map from memory
            if hasattr(self.app, 's2_frame_objects_map_for_s3'):
                self.logger.info("[Thread] Clearing Stage 2 data map from memory.")
                self.app.s2_frame_objects_map_for_s3 = None
                gc.collect() # Encourage garbage collection

            self.logger.info("[Thread] Full analysis thread finished or exited.")
            if hasattr(self.app, 'single_video_analysis_complete_event'):
                self.app.single_video_analysis_complete_event.set()

    def _filter_segments_for_range(self, all_segments: List[Any], range_is_active: bool, start_frame: Optional[int], end_frame: Optional[int]) -> List[Any]:
        if not range_is_active:
            return all_segments
        if start_frame is None:
            self.logger.warning(
                "Segment filtering called for active range but start_frame is None. Returning all segments.")
            return all_segments

        effective_end_frame = end_frame
        if effective_end_frame is None or effective_end_frame == -1:
            if self.app.processor and self.app.processor.total_frames > 0:
                effective_end_frame = self.app.processor.total_frames - 1
            else:
                return [seg for seg in all_segments if seg.end_frame_id >= start_frame]

        filtered_segments = [
            seg for seg in all_segments
            if max(seg.start_frame_id, start_frame) <= min(seg.end_frame_id, effective_end_frame)
        ]
        self.logger.info(f"Found {len(filtered_segments)} segments overlapping with the selected range.")
        return filtered_segments

    def _execute_stage1_logic(self, frame_range: Optional[Tuple[Optional[int], Optional[int]]] = None,
                                  output_path: Optional[str] = None,
                                  num_producers_override: Optional[int] = None,
                                  num_consumers_override: Optional[int] = None,
                                  is_autotune_run: bool = False) -> Dict[str, Any]:
        self.gui_event_queue.put(("stage1_status_update", "Running S1...", "Initializing S1..."))
        fm = self.app.file_manager
        self.stage1_frame_queue_size = 0
        self.stage1_result_queue_size = 0

        logger_config_for_stage1 = {
            'main_logger': self.logger,
            'log_file': self.app.app_log_file_path,
            'log_level': self.logger.level
        }
        try:
            if not stage1_module:
                self.gui_event_queue.put(("stage1_status_update", "Error - S1 Module not loaded.", "Error"))
                return {"success": False, "max_fps": 0.0}

            preprocessed_video_path = None
            if self.save_preprocessed_video:
                preprocessed_video_path = fm.get_output_path_for_file(fm.video_path, "_preprocessed.mkv")

            num_producers = num_producers_override if num_producers_override is not None else self.num_producers_stage1
            num_consumers = num_consumers_override if num_consumers_override is not None else self.num_consumers_stage1

            result_path, max_fps = stage1_module.perform_yolo_analysis(
                video_path_arg=fm.video_path,
                yolo_model_path_arg=self.app.yolo_det_model_path,
                yolo_pose_model_path_arg=self.app.yolo_pose_model_path,
                confidence_threshold=self.app.tracker.confidence_threshold,
                progress_callback=self._stage1_progress_callback,
                stop_event_external=self.stop_stage_event,
                num_producers_arg=num_producers,
                num_consumers_arg=num_consumers,
                hwaccel_method_arg=self.app.hardware_acceleration_method,
                hwaccel_avail_list_arg=self.app.available_ffmpeg_hwaccels,
                video_type_arg=self.app.processor.video_type_setting if self.app.processor else "auto",
                vr_input_format_arg=self.app.processor.vr_input_format if self.app.processor else "he",
                vr_fov_arg=self.app.processor.vr_fov if self.app.processor else 190,
                vr_pitch_arg=self.app.processor.vr_pitch if self.app.processor else 0,
                yolo_input_size_arg=self.app.yolo_input_size,
                app_logger_config_arg=logger_config_for_stage1,
                gui_event_queue_arg=self.gui_event_queue,
                frame_range_arg=frame_range,
                output_filename_override=output_path,
                save_preprocessed_video_arg=self.save_preprocessed_video,
                preprocessed_video_path_arg=preprocessed_video_path,
                is_autotune_run_arg=is_autotune_run
            )
            if self.stop_stage_event.is_set():
                self.gui_event_queue.put(("stage1_status_update", "S1 Aborted by user.", "Aborted"))
                self.gui_event_queue.put(
                    ("stage1_progress_update", 0.0, {"message": "Aborted", "current": 0, "total": 1}))
                return {"success": False, "max_fps": 0.0}
            if result_path and os.path.exists(result_path):
                fm.stage1_output_msgpack_path = result_path
                final_msg = f"S1 Completed. Output: {os.path.basename(result_path)}"
                self.gui_event_queue.put(("stage1_status_update", final_msg, "Done"))
                self.gui_event_queue.put(("stage1_progress_update", 1.0, {"message": "Done", "current": 1, "total": 1}))
                self.app.project_manager.project_dirty = True
                return {"success": True, "max_fps": max_fps}
            self.gui_event_queue.put(("stage1_status_update", "S1 Failed (no output file).", "Failed"))
            return {"success": False, "max_fps": 0.0}
        except Exception as e:
            self.logger.error(f"Stage 1 execution error in AppLogic: {e}", exc_info=True,
                              extra={'status_message': True})
            self.gui_event_queue.put(("stage1_status_update", f"S1 Error - {str(e)}", "Error"))
            return {"success": False, "max_fps": 0.0}

    def _execute_stage2_logic(self, s2_overlay_output_path: Optional[str], generate_funscript_actions: bool = True, is_ranged_data_source: bool = False) -> Dict[str, Any]:
        self.gui_event_queue.put(("stage2_status_update", "Running S2...", "Initializing S2..."))
        initial_total_main_steps = getattr(stage2_module, 'ATR_PASS_COUNT', self.S2_TOTAL_MAIN_STEPS_FALLBACK)
        if not generate_funscript_actions:
            initial_total_main_steps = getattr(stage2_module, 'ATR_PASS_COUNT_SEGMENTATION_ONLY', 3)  # Assume S2 module defines this
            self.gui_event_queue.put(("stage2_status_update", "Running S2 (Segmentation)...", "Initializing S2 Seg..."))

        self.gui_event_queue.put(("stage2_dual_progress", (1, initial_total_main_steps, "Initializing..."), (0, 1, "Starting")))
        fm = self.app.file_manager
        try:
            if not stage2_module:
                msg = "Error - S2 Module not loaded."
                self.gui_event_queue.put(("stage2_status_update", msg, "Error"))
                return {"success": False, "error": msg}
            if not fm.stage1_output_msgpack_path:
                msg = "Error - S1 output missing for S2."
                self.gui_event_queue.put(("stage2_status_update", msg, "Error"))
                return {"success": False, "error": msg}

            range_is_active, range_start_frame, range_end_frame = self.app.funscript_processor.get_effective_scripting_range()

            stage2_results = stage2_module.perform_contact_analysis(
                video_path_arg=fm.video_path,
                msgpack_file_path_arg=fm.stage1_output_msgpack_path,
                progress_callback=self._stage2_progress_callback,
                stop_event=self.stop_stage_event,
                ml_model_dir_path_arg=self.app.pose_model_artifacts_dir,
                output_overlay_msgpack_path=s2_overlay_output_path,
                parent_logger_arg=self.logger,
                yolo_input_size_arg=self.app.yolo_input_size,
                video_type_arg=self.app.processor.video_type_setting if self.app.processor else "auto",
                vr_input_format_arg=self.app.processor.vr_input_format if self.app.processor else "he",
                vr_fov_arg=self.app.processor.vr_fov if self.app.processor else 190,
                vr_pitch_arg=self.app.processor.vr_pitch if self.app.processor else 0,
                vr_vertical_third_filter_arg=self.app_settings.get("vr_filter_stage2", True),
                enable_of_debug_prints=self.app_settings.get("debug_prints_stage2", False),
                discarded_classes_runtime_arg=self.app.discarded_tracking_classes,
                scripting_range_active_arg=range_is_active,
                scripting_range_start_frame_arg=range_start_frame,
                scripting_range_end_frame_arg=range_end_frame,
                is_ranged_data_source=is_ranged_data_source,
                generate_funscript_actions_arg=generate_funscript_actions
            )
            if self.stop_stage_event.is_set():
                msg = "S2 Aborted by user."
                self.gui_event_queue.put(("stage2_status_update", msg, "Aborted"))
                current_main_step = int(self.stage2_main_progress_value * initial_total_main_steps)
                self.gui_event_queue.put(("stage2_dual_progress", (current_main_step, initial_total_main_steps, "Aborted"), (0, 1, "Aborted")))
                return {"success": False, "error": msg}

            if stage2_results and "error" not in stage2_results:
                if generate_funscript_actions:
                    packaged_data = {
                        "results_dict": stage2_results,
                        "was_ranged": range_is_active,
                        "range_frames": (range_start_frame, range_end_frame)
                    }
                    self.gui_event_queue.put(("stage2_results_success", packaged_data, s2_overlay_output_path))
                    status_msg = "S2 Completed. Results Processed."
                else:
                    status_msg = "S2 Segmentation Completed."
                self.gui_event_queue.put(("stage2_status_update", status_msg, "Done"))
                self.gui_event_queue.put(("stage2_dual_progress", (initial_total_main_steps, initial_total_main_steps, "Completed" if generate_funscript_actions else "Segmentation Done"), (1, 1, "Done")))
                self.app.project_manager.project_dirty = True
                return {"success": True, "data": stage2_results}
            error_msg = stage2_results.get("error", "Unknown S2 failure") if stage2_results else "S2 returned None."
            self.gui_event_queue.put(("stage2_status_update", f"S2 Failed: {error_msg}", "Failed"))
            return {"success": False, "error": error_msg}
        except Exception as e:
            self.logger.error(f"Stage 2 execution error in AppLogic: {e}", exc_info=True, extra={'status_message': True})
            error_msg = f"S2 Exception: {str(e)}"
            self.gui_event_queue.put(("stage2_status_update", error_msg, "Error"))
            return {"success": False, "error": error_msg}

    def _execute_stage3_optical_flow_module(self, atr_segments_objects: List[Any], preprocessed_video_path: Optional[str]) -> bool:
        """ Wrapper to call the new Stage 3 OF module. """
        fs_proc = self.app.funscript_processor

        if not self.app.file_manager.video_path:
            self.logger.error("Stage 3: Video path not available.")
            self.gui_event_queue.put(("stage3_status_update", "Error: Video path missing", "Error"))
            return False

        if not stage3_module:  # Check if the imported module is valid
            self.logger.error("Stage 3: Optical Flow processing module (stage3_module) not loaded.")
            self.gui_event_queue.put(("stage3_status_update", "Error: S3 Module missing", "Error"))
            return False

        tracker_config_s3 = {
            "confidence_threshold": self.app_settings.get('tracker_confidence_threshold', 0.4),  # Example name
            "roi_padding": self.app_settings.get('tracker_roi_padding', 20),
            "roi_update_interval": self.app_settings.get('s3_roi_update_interval', constants.DEFAULT_ROI_UPDATE_INTERVAL),
            "roi_smoothing_factor": self.app_settings.get('tracker_roi_smoothing_factor', constants.DEFAULT_ROI_SMOOTHING_FACTOR),
            "dis_flow_preset": self.app_settings.get('tracker_dis_flow_preset', "ULTRAFAST"),
            "target_size_preprocess": self.app.tracker.target_size_preprocess if self.app.tracker else (640, 640),
            "flow_history_window_smooth": self.app_settings.get('tracker_flow_history_window_smooth', 3),
            "adaptive_flow_scale": self.app_settings.get('tracker_adaptive_flow_scale', True),
            "use_sparse_flow": self.app_settings.get('tracker_use_sparse_flow', False),
            "base_amplification_factor": self.app_settings.get('tracker_base_amplification', constants.DEFAULT_LIVE_TRACKER_BASE_AMPLIFICATION),
            "class_specific_amplification_multipliers": self.app_settings.get('tracker_class_specific_multipliers', constants.DEFAULT_CLASS_AMP_MULTIPLIERS),
            "y_offset": self.app_settings.get('tracker_y_offset', constants.DEFAULT_LIVE_TRACKER_Y_OFFSET),
            "x_offset": self.app_settings.get('tracker_x_offset', constants.DEFAULT_LIVE_TRACKER_X_OFFSET),
            "sensitivity": self.app_settings.get('tracker_sensitivity', constants.DEFAULT_LIVE_TRACKER_SENSITIVITY)
        }

        video_fps_s3 = 30.0
        if self.app.processor and self.app.processor.video_info:
            video_fps_s3 = self.app.processor.video_info.get('fps', 30.0)
            if video_fps_s3 <= 0: video_fps_s3 = 30.0
        elif self.app.project_manager.current_project_data and \
                self.app.project_manager.current_project_data.get('video_info'):
            video_fps_s3 = self.app.project_manager.current_project_data['video_info'].get('fps', 30.0)
            if video_fps_s3 <= 0: video_fps_s3 = 30.0

        common_app_config_s3 = {
            "yolo_det_model_path": self.app.yolo_det_model_path,  # Path to actual model file
            "yolo_pose_model_path": self.app.yolo_pose_model_path,
            "yolo_input_size": self.app.yolo_input_size,
            "video_fps": video_fps_s3,
            "output_delay_frames": self.app.tracker.output_delay_frames if self.app.tracker else 0,
            "num_warmup_frames_s3": self.app_settings.get('s3_num_warmup_frames', 10 + (self.app.tracker.output_delay_frames if self.app.tracker else 0)),
            "roi_narrow_factor_hjbj": self.app_settings.get("roi_narrow_factor_hjbj", constants.DEFAULT_ROI_NARROW_FACTOR_HJBJ),
            "min_roi_dim_hjbj": self.app_settings.get("min_roi_dim_hjbj", constants.DEFAULT_MIN_ROI_DIM_HJBJ),
            "tracking_axis_mode": self.app.tracking_axis_mode,
            "single_axis_output_target": self.app.single_axis_output_target,
            "s3_show_roi_debug": self.app_settings.get("s3_show_roi_debug", False),
            "hardware_acceleration_method": self.app.hardware_acceleration_method,
            "available_ffmpeg_hwaccels": self.app.available_ffmpeg_hwaccels,
            "video_type": self.app.processor.video_type_setting if self.app.processor else "auto",
            "vr_input_format": self.app.processor.vr_input_format if self.app.processor else "he",
            "vr_fov": self.app.processor.vr_fov if self.app.processor else 190,
            "vr_pitch": self.app.processor.vr_pitch if self.app.processor else 0,
            "s3_chunk_size": self.app.app_settings.get("s3_chunk_size", 1000),
            "s3_overlap_size": self.app.app_settings.get("s3_overlap_size", 30)

        }

        s3_results = stage3_module.perform_stage3_analysis(
            video_path=self.app.file_manager.video_path,
            preprocessed_video_path_arg=preprocessed_video_path,
            atr_segments_list=atr_segments_objects,
            s2_frame_objects_map=self.app.s2_frame_objects_map_for_s3,
            tracker_config=tracker_config_s3,
            common_app_config=common_app_config_s3,
            progress_callback=self._stage3_progress_callback,
            stop_event=self.stop_stage_event,
            parent_logger=self.logger
        )

        if self.stop_stage_event.is_set(): return False

        if s3_results and "error" not in s3_results:
            final_s3_primary_actions = s3_results.get("primary_actions", [])
            final_s3_secondary_actions = s3_results.get("secondary_actions", [])
            self.logger.info(f"Stage 3 Optical Flow generated {len(final_s3_primary_actions)} primary and {len(final_s3_secondary_actions)} secondary actions.")

            range_is_active, range_start_f, range_end_f_effective = fs_proc.get_effective_scripting_range()
            op_desc_s3 = "Stage 3 Opt.Flow"
            video_total_frames_s3 = self.app.processor.total_frames if self.app.processor else 0
            video_duration_ms_s3 = fs_proc.frame_to_ms(video_total_frames_s3 - 1) if video_total_frames_s3 > 0 else 0

            if range_is_active:
                start_ms = fs_proc.frame_to_ms(range_start_f if range_start_f is not None else 0)
                end_ms = fs_proc.frame_to_ms(range_end_f_effective) if range_end_f_effective is not None else video_duration_ms_s3
                op_desc_s3_range = f"{op_desc_s3} (Range F{range_start_f or 'Start'}-{range_end_f_effective if range_end_f_effective is not None else 'End'})"
                fs_proc.clear_actions_in_range_and_inject_new(1, final_s3_primary_actions, start_ms, end_ms, op_desc_s3_range + " (T1)")
                fs_proc.clear_actions_in_range_and_inject_new(2, final_s3_secondary_actions, start_ms, end_ms, op_desc_s3_range + " (T2)")
            else:
                fs_proc.clear_timeline_history_and_set_new_baseline(1, final_s3_primary_actions, op_desc_s3 + " (T1)")
                fs_proc.clear_timeline_history_and_set_new_baseline(2, final_s3_secondary_actions, op_desc_s3 + " (T2)")

            self.gui_event_queue.put(("stage3_status_update", "Stage 3 Completed.", "Done"))
            self.app.project_manager.project_dirty = True

            # Update chapters for GUI if video_segments are present (3-stage fix)
            if "video_segments" in s3_results:
                fs_proc.video_chapters.clear()
                for seg_data in s3_results["video_segments"]:
                    fs_proc.video_chapters.append(VideoSegment.from_dict(seg_data))
                self.app.app_state_ui.heatmap_dirty = True
                self.app.app_state_ui.funscript_preview_dirty = True
            return True
        else:
            error_msg = s3_results.get("error", "Unknown S3 failure") if s3_results else "S3 returned None."
            self.logger.error(f"Stage 3 execution failed: {error_msg}")
            self.gui_event_queue.put(("stage3_status_update", f"S3 Failed: {error_msg}", "Failed"))
            return False

    def abort_stage_processing(self):
        if self.full_analysis_active and self.stage_thread and self.stage_thread.is_alive():
            self.logger.info("Aborting current analysis stage(s)...", extra={'status_message': True})
            self.stop_stage_event.set()
            self.current_analysis_stage = -1  # Mark as aborting
        elif self.scene_detection_active and self.scene_detection_thread and self.scene_detection_thread.is_alive():
            self.logger.info("Aborting scene detection...", extra={'status_message': True})
            self.stop_stage_event.set()
            # Call scene_manager.stop() if available for fast abort
            if hasattr(self.app.processor, '_active_scene_manager') and self.app.processor._active_scene_manager is not None:
                try:
                    self.app.processor._active_scene_manager.stop()
                    self.logger.info("Called scene_manager.stop() to abort scene detection.")
                except Exception as e:
                    self.logger.warning(f"Failed to call scene_manager.stop(): {e}")
        else:
            self.logger.info("No analysis pipeline running to abort.", extra={'status_message': False})
        self.app.energy_saver.reset_activity_timer()

    def process_gui_events(self):
        if self.full_analysis_active or self.scene_detection_active or self.refinement_analysis_active:
            if hasattr(self.app, 'energy_saver'):
                self.app.energy_saver.reset_activity_timer()

        fm = self.app.file_manager
        fs_proc = self.app.funscript_processor
        while not self.gui_event_queue.empty():
            try:
                queue_item = self.gui_event_queue.get_nowait()
                if not isinstance(queue_item, tuple) or len(queue_item) < 2:
                    continue

                event_type, data1, data2 = queue_item[0], queue_item[1], queue_item[2] if len(queue_item) > 2 else None

                if event_type == "scene_detection_finished":
                    scene_list, status_text = data1, data2
                    if status_text is None:
                        status_text = "Chapters created."
                    self.scene_detection_status = status_text
                    fs_proc.video_chapters.clear()
                    default_pos_key = next(iter(constants.POSITION_INFO_MAPPING), "Default")
                    for start_frame, end_frame in scene_list:
                        fs_proc.create_new_chapter_from_data({
                            "start_frame_str": str(start_frame),
                            "end_frame_str": str(end_frame - 1),
                            "position_short_name_key": default_pos_key,
                            "segment_type": "Scene",
                            "source": "scene_detection"})
                    self.app.logger.info(f"Created {len(scene_list)} chapters from detected scenes.")
                    self.app.project_manager.project_dirty = True
                    self.app.app_state_ui.heatmap_dirty = True
                    self.app.app_state_ui.funscript_preview_dirty = True
                    self.scene_detection_status = "Completed"
                elif event_type == "stage1_progress_update":
                    prog_val, prog_data = data1, data2
                    if isinstance(prog_data, dict):
                        self.stage1_progress_value = prog_val if prog_val != -1.0 else self.stage1_progress_value
                        self.stage1_progress_label = str(prog_data.get("message", ""))
                        t_el, avg_fps, instant_fps, eta = prog_data.get("time_elapsed", 0.0), prog_data.get("avg_fps", 0.0), prog_data.get("instant_fps", 0.0), prog_data.get("eta", 0.0)
                        self.stage1_time_elapsed_str = f"{int(t_el // 3600):02d}:{int((t_el % 3600) // 60):02d}:{int(t_el % 60):02d}"
                        self.stage1_processing_fps_str = f"{int(avg_fps)} FPS"
                        self.stage1_instant_fps_str = f"{int(instant_fps)} FPS"
                        if math.isnan(eta) or math.isinf(eta):
                            self.stage1_eta_str = "Calculating..."
                        elif eta > 0:
                            self.stage1_eta_str = f"{int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d}"
                        else:
                            self.stage1_eta_str = "Done"
                elif event_type == "stage1_status_update":
                    self.stage1_status_text = str(data1)
                    if data2 is not None: self.stage1_progress_label = str(data2)
                elif event_type == "stage1_completed":
                    self.stage1_final_elapsed_time_str = str(data1)
                    self.stage1_final_fps_str = str(data2)
                    self.stage1_status_text = "Completed"
                    self.stage1_progress_value = 1.0
                elif event_type == "stage1_queue_update":
                    queue_data = data1
                    if isinstance(queue_data, dict):
                        self.stage1_frame_queue_size = queue_data.get("frame_q_size", self.stage1_frame_queue_size)
                        self.stage1_result_queue_size = queue_data.get("result_q_size", self.stage1_result_queue_size)
                elif event_type == "stage2_dual_progress":
                    main_step_info, sub_step_info = data1, data2
                    if isinstance(main_step_info, tuple) and len(main_step_info) == 3:
                        main_current, total_main, main_name = main_step_info
                        self.stage2_main_progress_value = float(main_current) / total_main if total_main > 0 else 0.0
                        self.stage2_main_progress_label = f"{main_name} ({int(main_current)}/{int(total_main)})"
                    if isinstance(sub_step_info, tuple) and len(sub_step_info) == 3:
                        sub_current, sub_total, sub_name = sub_step_info
                        self.stage2_sub_progress_value = float(sub_current) / sub_total if sub_total > 0 else 0.0
                        self.stage2_sub_progress_label = f"{sub_name} ({int(sub_current)}/{int(sub_total)})"
                elif event_type == "stage2_status_update":
                    self.stage2_status_text = str(data1)
                    if data2 is not None:
                        self.stage2_progress_label = str(data2)
                elif event_type == "stage2_completed":
                    self.stage2_final_elapsed_time_str = str(data1)
                    self.stage2_status_text = "Completed"
                    self.stage2_main_progress_value = 1.0
                    self.stage2_sub_progress_value = 1.0
                elif event_type == "stage2_results_success":
                    packaged_data, s2_overlay_path_written = data1, data2
                    results_dict = packaged_data.get("results_dict", {})
                    video_segments_data = results_dict.get("video_segments", [])
                    # Use the same flag to protect chapters during a 2-Stage run.
                    if self.force_rerun_stage2_segmentation:
                        self.logger.info("Overwriting chapters with new 2-Stage analysis results as requested.")
                        fs_proc.video_chapters.clear()
                        if isinstance(video_segments_data, list):
                            for seg_data in video_segments_data:
                                if isinstance(seg_data, dict):
                                    fs_proc.video_chapters.append(VideoSegment.from_dict(seg_data))
                    else:
                        self.app.logger.info("Preserving existing chapters. Stage 2 funscript generated without altering chapters.")

                    # Get the generated actions from Stage 2
                    primary_actions = results_dict.get("primary_actions", [])
                    secondary_actions = results_dict.get("secondary_actions", [])

                    # Get the application's current axis settings
                    axis_mode = self.app.tracking_axis_mode
                    target_timeline = self.app.single_axis_output_target

                    self.app.logger.info(f"Applying 2-Stage results with axis mode: {axis_mode} and target: {target_timeline}.")

                    if axis_mode == "both":
                        # Overwrite both timelines with the new results.
                        fs_proc.clear_timeline_history_and_set_new_baseline(1, primary_actions, "Stage 2 (Primary)")
                        fs_proc.clear_timeline_history_and_set_new_baseline(2, secondary_actions, "Stage 2 (Secondary)")

                    elif axis_mode == "vertical":
                        # Overwrite ONLY the target timeline, leaving the other one completely untouched.
                        if target_timeline == "primary":
                            self.app.logger.info("Writing to Timeline 1, Timeline 2 is untouched.")
                            fs_proc.clear_timeline_history_and_set_new_baseline(1, primary_actions, "Stage 2 (Vertical)")
                        else:  # Target is secondary
                            self.app.logger.info("Writing to Timeline 2, Timeline 1 is untouched.")
                            fs_proc.clear_timeline_history_and_set_new_baseline(2, primary_actions, "Stage 2 (Vertical)")

                    elif axis_mode == "horizontal":
                        # Do nothing, as Stage 2 does not produce horizontal data. Both timelines remain untouched.
                        self.app.logger.info("Horizontal axis mode selected, but 2-Stage analysis only produces vertical data. No timelines were modified.")

                    self.stage2_status_text = "S2 Completed. Results Processed."
                    self.app.project_manager.project_dirty = True
                    self.logger.info("Processed Stage 2 results.")

                elif event_type == "stage2_results_success_segments_only":
                    video_segments_data = data1

                    # Only modify chapters if the user forced a re-run of segmentation.
                    if self.force_rerun_stage2_segmentation:
                        self.logger.info("Overwriting chapters with new segmentation results as requested.")
                        fs_proc.video_chapters.clear()  # Now safe inside the check
                        if isinstance(video_segments_data, list):
                            for seg_data in video_segments_data:
                                if isinstance(seg_data, dict):
                                    fs_proc.video_chapters.append(VideoSegment.from_dict(seg_data))
                    else:
                        self.logger.info("Preserving existing chapters. S2 segmentation was not re-run.")

                    self.stage2_status_text = "S2 Segmentation Processed."
                    self.app.project_manager.project_dirty = True
                elif event_type == "load_s2_overlay":
                    overlay_path = data1
                    if overlay_path and os.path.exists(overlay_path):
                        self.logger.info(f"Loading generated Stage 2 overlay data from: {overlay_path}")
                        fm.load_stage2_overlay_data(overlay_path)
                elif event_type == "stage3_progress_update":
                    prog_data = data1
                    if isinstance(prog_data, dict):
                        chap_idx = prog_data.get('current_chapter_idx', 0)
                        total_chaps = prog_data.get('total_chapters', 0)
                        chap_name = prog_data.get('chapter_name', '')
                        chunk_idx = prog_data.get('current_chunk_idx', 0)
                        total_chunks = prog_data.get('total_chunks', 0)

                        self.stage3_current_segment_label = f"Chapter: {chap_idx}/{total_chaps} ({chap_name})"
                        self.stage3_overall_progress_label = f"Overall Task: Chunk {chunk_idx}/{total_chunks}"

                        self.stage3_segment_progress_value = prog_data.get('segment_progress', 0.0)
                        self.stage3_overall_progress_value = prog_data.get('overall_progress', 0.0)
                        processed_overall = prog_data.get('total_frames_processed_overall', 0)
                        to_process_overall = prog_data.get('total_frames_to_process_overall', 0)
                        if to_process_overall > 0:
                            self.stage3_overall_progress_label = f"Overall S3: {processed_overall}/{to_process_overall}"
                        else:
                            self.stage3_overall_progress_label = f"Overall S3: {self.stage3_overall_progress_value * 100:.0f}%"
                        self.stage3_status_text = "Running Stage 3 (Optical Flow)..."
                        t_el_s3, fps_s3, eta_s3 = prog_data.get("time_elapsed", 0.0), prog_data.get("fps", 0.0), prog_data.get("eta", 0.0)
                        self.stage3_time_elapsed_str = f"{int(t_el_s3 // 3600):02d}:{int((t_el_s3 % 3600) // 60):02d}:{int(t_el_s3 % 60):02d}" if not math.isnan(t_el_s3) else "Calculating..."
                        self.stage3_processing_fps_str = f"{fps_s3:.1f} FPS" if not math.isnan(fps_s3) else "N/A FPS"

                        is_s3_done = (chunk_idx >= total_chunks and total_chunks > 0)

                        if math.isnan(eta_s3) or math.isinf(eta_s3):
                            self.stage3_eta_str = "Calculating..."
                        elif eta_s3 > 1.0 and not is_s3_done:
                            self.stage3_eta_str = f"{int(eta_s3 // 3600):02d}:{int((eta_s3 % 3600) // 60):02d}:{int(eta_s3 % 60):02d}"
                        else:
                            self.stage3_eta_str = "Done"
                elif event_type == "stage3_status_update":
                    self.stage3_status_text = str(data1)
                    if data2 is not None: self.stage3_overall_progress_label = str(data2)
                elif event_type == "stage3_completed":
                    self.stage3_final_elapsed_time_str = str(data1)
                    self.stage3_final_fps_str = str(data2)
                    self.stage3_status_text = "Completed"
                    self.stage3_overall_progress_value = 1.0
                elif event_type == "analysis_message":
                    payload = data1 if isinstance(data1, dict) else {}
                    log_msg = payload.get("message", str(data1))
                    status_override = payload.get("status", data2)
                    video_path_from_event = payload.get("video_path")
                    segments_from_event = payload.get("video_segments")

                    if log_msg:
                        self.logger.info(log_msg, extra={'status_message': True})

                    if status_override == "Completed":
                        if not video_path_from_event:
                            self.logger.warning("Completion event is missing its video path. Cannot save funscripts.")
                        else:
                            # --- Update the central chapter list from the event data BEFORE saving ---
                            if segments_from_event is not None:
                                self.logger.info(f"Updating app state with {len(segments_from_event)} chapters for saving.")
                                self.app.funscript_processor.video_chapters = [
                                    VideoSegment.from_dict(chap_data) for chap_data in segments_from_event if isinstance(chap_data, dict)
                                ]

                            self.app.file_manager.save_raw_funscripts_after_generation(video_path_from_event)
                            post_processing_enabled = self.app.app_settings.get("enable_auto_post_processing", False)
                            if self.app.is_batch_processing_active:
                                # If in batch mode, use the choice made in the batch dialog
                                post_processing_enabled = self.app.batch_apply_post_processing

                            chapters_for_save = segments_from_event

                            if post_processing_enabled:
                                self.logger.info("Triggering auto post-processing after completed analysis.")
                                self.app.funscript_processor.apply_automatic_post_processing()
                                # After post-processing, the funscript_processor's chapter list is the most current.
                                chapters_for_save = self.app.funscript_processor.video_chapters
                            else:
                                self.logger.info("Auto post-processing disabled for this run, skipping.")

                            self.logger.info("Saving final funscripts...")

                            saved_funscript_paths = self.app.file_manager.save_final_funscripts(video_path_from_event, chapters=chapters_for_save)

                            # Check if we are in batch mode and if the user requested a copy
                            if self.app.is_batch_processing_active and self.app.batch_copy_funscript_to_video_location:
                                if saved_funscript_paths and isinstance(saved_funscript_paths, list):
                                    video_dir = os.path.dirname(video_path_from_event)
                                    for source_path in saved_funscript_paths:
                                        if not source_path or not os.path.exists(source_path):
                                            continue
                                        try:
                                            file_basename = os.path.basename(source_path)
                                            destination_path = os.path.join(video_dir, file_basename)
                                            # Manually copy the file
                                            with open(source_path, 'rb') as src_file:
                                                content = src_file.read()
                                            with open(destination_path, 'wb') as dest_file:
                                                dest_file.write(content)
                                            self.logger.info(f"Saved copy of {file_basename} next to video.")
                                        except Exception as e:
                                            self.logger.error(f"Failed to save copy of {os.path.basename(source_path)} next to video: {e}")
                                else:
                                    self.logger.warning("save_final_funscripts did not return file paths. Cannot save copy next to video.")

                            # --- Save the project file for the completed video ---
                            self.logger.info("Saving project file for completed video...")
                            project_filepath = self.app.file_manager.get_output_path_for_file(video_path_from_event, constants.PROJECT_FILE_EXTENSION)
                            self.app.project_manager.save_project(project_filepath)

                        # Check if we are in simple mode and should auto-run post-processing
                        is_simple_mode = getattr(self.app.app_state_ui, 'ui_view_mode', 'expert') == 'simple'
                        is_offline_analysis = self.app.app_state_ui.selected_tracker_mode in [TrackerMode.OFFLINE_2_STAGE, TrackerMode.OFFLINE_3_STAGE]

                        if is_simple_mode and is_offline_analysis:
                            self.logger.info("Simple Mode: Automatically applying Ultimate Autotune with defaults...")
                            self.app.set_status_message("Analysis complete! Applying auto-enhancements...")
                            # Trigger the autotune on the primary timeline (timeline 1)
                            if hasattr(self.app, 'trigger_ultimate_autotune_with_defaults'):
                                self.app.trigger_ultimate_autotune_with_defaults(timeline_num=1)

                    elif status_override == "Aborted":
                        if self.current_analysis_stage == 1 or self.stage1_status_text.startswith(
                            "Running"): self.stage1_status_text = "S1 Aborted."
                        if self.current_analysis_stage == 2 or self.stage2_status_text.startswith(
                            "Running"): self.stage2_status_text = "S2 Aborted."
                        if self.current_analysis_stage == 3 or self.stage3_status_text.startswith(
                            "Running"): self.stage3_status_text = "S3 Aborted."
                    elif status_override == "Failed":
                        if self.current_analysis_stage == 1 or self.stage1_status_text.startswith(
                            "Running"): self.stage1_status_text = "S1 Failed."
                        if self.current_analysis_stage == 2 or self.stage2_status_text.startswith(
                            "Running"): self.stage2_status_text = "S2 Failed."
                        if self.current_analysis_stage == 3 or self.stage3_status_text.startswith(
                            "Running"): self.stage3_status_text = "S3 Failed."

                    # --- Signal the batch loop to continue, regardless of outcome ---
                    if self.app.is_batch_processing_active and hasattr(self.app, 'save_and_reset_complete_event'):
                        self.logger.debug(f"Signaling batch loop to continue after handling '{status_override}' status.")
                        self.app.save_and_reset_complete_event.set()

                elif event_type == "refinement_completed":
                    payload = data1
                    chapter = payload.get('chapter')
                    new_actions = payload.get('new_actions')

                    if chapter and new_actions:
                        self.app.funscript_processor.apply_interactive_refinement(chapter, new_actions)

                else:
                    self.logger.warning(f"Unknown GUI event type received: {event_type}")
            except Exception as e:
                self.logger.error(f"Error processing GUI event in AppLogic's StageProcessor: {e}", exc_info=True)

    def shutdown_app_threads(self):
        self.stop_stage_event.set()
        if self.stage_thread and self.stage_thread.is_alive():
            self.logger.info("Waiting for app stage processing thread to finish...", extra={'status_message': False})
            self.stage_thread.join(timeout=5.0)
            if self.stage_thread.is_alive():
                self.logger.warning("App stage processing thread did not finish cleanly.", extra={'status_message': False})
            else:
                self.logger.info("App stage processing thread finished.", extra={'status_message': False})
        self.stage_thread = None

    # REFACTORED replaces duplicate code in __init__ and deals with edge cases (ie 'None' values)
    def update_settings_from_app(self):
        prod_usr = self.app_settings.get("num_producers_stage1")
        cons_usr = self.app_settings.get("num_consumers_stage1")
        self.save_preprocessed_video = self.app_settings.get("save_preprocessed_video", False)

        if not prod_usr or not cons_usr:
            cpu_cores = os.cpu_count() or 4
            self.num_producers_stage1 = max(1, min(5, cpu_cores // 2 - 2) if cpu_cores > 4 else 1)
            self.num_consumers_stage1 = max(1, min(9, cpu_cores // 2 + 2) if cpu_cores > 4 else 1)
        else:
            self.num_producers_stage1 = prod_usr
            self.num_consumers_stage1 = cons_usr

    def save_settings_to_app(self):
        self.app_settings.set("num_producers_stage1", self.num_producers_stage1)
        self.app_settings.set("num_consumers_stage1", self.num_consumers_stage1)
        self.app_settings.set("save_preprocessed_video", self.save_preprocessed_video)

    def get_project_save_data(self) -> Dict:
        return {
            "stage1_output_msgpack_path": self.app.file_manager.stage1_output_msgpack_path,
            "stage2_overlay_msgpack_path": self.app.file_manager.stage2_output_msgpack_path,
            "stage2_status_text": self.stage2_status_text,
            "stage3_status_text": self.stage3_status_text,
        }

    def update_project_specific_settings(self, project_data: Dict):
        self.stage2_status_text = project_data.get("stage2_status_text", "Not run.")
        self.stage3_status_text = project_data.get("stage3_status_text", "Not run.")
        self.stage2_progress_value, self.stage2_progress_label = 0.0, ""
        self.stage2_main_progress_value, self.stage2_main_progress_label = 0.0, ""
        self.stage2_sub_progress_value, self.stage2_sub_progress_label = 0.0, ""
        self.stage3_current_segment_label, self.stage3_segment_progress_value = "", 0.0
        self.stage3_overall_progress_label, self.stage3_overall_progress_value = "", 0.0
        self.stage3_time_elapsed_str, self.stage3_processing_fps_str, self.stage3_eta_str = "00:00:00", "0 FPS", "N/A"

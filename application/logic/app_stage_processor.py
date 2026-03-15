import os
import threading
import time
from queue import Queue
from typing import Optional, List, Dict, Any, Tuple
import msgpack
import numpy as np
from bisect import bisect_left, bisect_right
import multiprocessing
import gc

from application.utils.checkpoint_manager import (
    ProcessingStage, CheckpointData,
    get_checkpoint_manager,
)
from application.gui_components.dynamic_tracker_ui import get_dynamic_tracker_ui

import detection.cd.stage_1_cd as stage1_module
import detection.cd.stage_2_cd as stage2_module
import detection.cd.stage_3_of_processor as stage3_module

from config import constants
from .app_stage_gui_events import StageGuiEventsMixin
from .app_stage_executor import StageExecutorMixin
from .app_stage_checkpoints import StageCheckpointMixin


class AppStageProcessor(StageGuiEventsMixin, StageExecutorMixin, StageCheckpointMixin):
    def __init__(self, app_logic_instance):
        self.app = app_logic_instance
        self.logger = self.app.logger
        self.app_settings = self.app.app_settings

        # --- Threading Configuration ---
        self.update_settings_from_app()

        self.stage_completion_event: Optional[threading.Event] = None
        
        # --- Checkpoint Management ---
        self.checkpoint_manager = get_checkpoint_manager()
        self.current_checkpoint_id: Optional[str] = None
        self.resume_data: Optional[CheckpointData] = None

        # --- Analysis State ---
        self.full_analysis_active: bool = False
        self.current_analysis_stage: int = 0
        self.stage_thread: Optional[threading.Thread] = None
        self.stop_stage_event = multiprocessing.Event()
        self.gui_event_queue = Queue()

        # --- Status and Progress Tracking ---
        self.reset_stage_status(stages=("stage1", "stage2", "stage3"))



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
        self.last_analysis_result: Optional[Dict] = None

        self.on_stage1_progress = self._stage1_progress_callback
        self.on_stage2_progress = self._stage2_progress_callback
        self.on_stage3_progress = self._stage3_progress_callback

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
            daemon=True,
            name="InteractiveRefinementThread",
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
            self.stage1_decode_ms = 0.0
            self.stage1_unwarp_ms = 0.0
            self.stage1_yolo_det_ms = 0.0
            self.stage1_yolo_pose_ms = 0.0
            # self.app.file_manager.stage1_output_msgpack_path = None
        if "stage2" in stages:
            self.stage2_status_text = "Not run."
            self.stage2_progress_value = 0.0
            self.stage2_progress_label = ""
            self.stage2_main_progress_value = 0.0
            self.stage2_main_progress_label = ""
            self.stage2_sub_progress_value = 0.0
            self.stage2_sub_progress_label = ""
            self.stage2_sub_time_elapsed_str = ""
            self.stage2_sub_processing_fps_str = ""
            self.stage2_sub_eta_str = ""
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



    def _stage1_progress_callback(self, current, total, message="Processing...", time_elapsed=0.0, avg_fps=0.0, instant_fps=0.0, eta_seconds=0.0, timing=None):
        progress = float(current) / total if total > 0 else -1.0
        progress_data = {
            "message": message, "current": current, "total": total,
            "time_elapsed": time_elapsed, "avg_fps": avg_fps, "instant_fps": instant_fps, "eta": eta_seconds
        }
        if timing:
            progress_data["timing"] = timing
        self.gui_event_queue.put(("stage1_progress_update", progress, progress_data))
        
        # Create checkpoint if needed
        stage_data = {
            "current_frame": current,
            "message": message,
            "avg_fps": avg_fps,
            "time_elapsed": time_elapsed
        }
        self._create_checkpoint_if_needed(ProcessingStage.STAGE_1_OBJECT_DETECTION, current, total, stage_data)

    def _stage2_progress_callback(self, main_info_from_module, sub_info_from_module, force_update=False):
        """A simplified callback to directly pass progress data to the GUI queue."""
        if not self.gui_event_queue:
            return

        # Basic validation
        if not isinstance(main_info_from_module, tuple) or len(main_info_from_module) != 3:
            self.logger.warning(f"Malformed main_info in S2 callback: {main_info_from_module}")
            main_info_from_module = (-1, 0, "Invalid Main Step")

        if not isinstance(sub_info_from_module, (dict, tuple)):
            self.logger.warning(f"Malformed sub_info in S2 callback: {sub_info_from_module}")
            sub_info_from_module = (0, 1, "Invalid Sub Step")

        # Directly put the validated/corrected data onto the queue.
        self.gui_event_queue.put(("stage2_dual_progress", main_info_from_module, sub_info_from_module))
        
        # Create checkpoint if needed (use main progress for frame tracking) - throttle
        try:
            now = time.time()
            if not hasattr(self, "_last_s2_checkpoint_ts"):
                self._last_s2_checkpoint_ts = 0.0
            if (now - self._last_s2_checkpoint_ts) >= 2.0:
                main_current, main_total, main_name = main_info_from_module
                if isinstance(sub_info_from_module, dict):
                    sub_current = sub_info_from_module.get("current", 0)
                    stage_data = {
                        "main_step": main_current,
                        "main_total": main_total,
                        "main_name": main_name,
                        "sub_current": sub_current,
                        "sub_info": sub_info_from_module
                    }
                else:
                    sub_current, sub_total, sub_name = sub_info_from_module
                    stage_data = {
                        "main_step": main_current,
                        "main_total": main_total,
                        "main_name": main_name,
                        "sub_current": sub_current,
                        "sub_total": sub_total,
                        "sub_name": sub_name
                    }
                
                composite_frame = main_current * 1000 + (sub_current if isinstance(sub_current, int) else 0)
                composite_total = main_total * 1000
                self._create_checkpoint_if_needed(ProcessingStage.STAGE_2_OPTICAL_FLOW, composite_frame, composite_total, stage_data)
                self._last_s2_checkpoint_ts = now
        except Exception:
            # Don't let checkpoint errors interrupt processing
            pass

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
        
        # Create checkpoint if needed
        stage_data = {
            "current_chapter": current_chapter_idx,
            "total_chapters": total_chapters,
            "chapter_name": chapter_name,
            "current_chunk": current_chunk_idx,
            "total_chunks": total_chunks,
            "processing_fps": processing_fps,
            "time_elapsed": time_elapsed
        }
        self._create_checkpoint_if_needed(ProcessingStage.STAGE_3_FUNSCRIPT_GENERATION,  total_frames_processed_overall, total_frames_to_process_overall, stage_data)

    def start_full_analysis(self, processing_mode: str,
                            override_producers: Optional[int] = None,
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
        is_hybrid = self._is_hybrid_tracker(processing_mode)

        if not is_hybrid:
            if not stage1_module or not stage2_module or not stage3_module:
                self.logger.error("Stage 1, Stage 2, or Stage 3 processing module not available.", extra={'status_message': True})
                return
            if not self.app.yolo_det_model_path or not os.path.exists(self.app.yolo_det_model_path):
                self.logger.error(f"Stage 1 Model not found: {self.app.yolo_det_model_path}", extra={'status_message': True})
                return
        else:
            # Hybrid trackers still need YOLO model for sparse detection
            if not self.app.yolo_det_model_path or not os.path.exists(self.app.yolo_det_model_path):
                self.logger.error(f"YOLO Model not found: {self.app.yolo_det_model_path}", extra={'status_message': True})
                return

        self.full_analysis_active = True
        self.current_analysis_stage = 0
        self.stop_stage_event.clear()
        self.stage_completion_event = completion_event
        self.frame_range_override = frame_range_override

        # Store the explicitly passed mode for the thread to use
        self.processing_mode_for_thread = processing_mode

        # Store the flag for the thread to use it
        self.is_autotune_run_for_thread = is_autotune_run

        # Store the overrides to be used by the thread
        self.override_producers = override_producers
        self.override_consumers = override_consumers

        selected_mode = self.app.app_state_ui.selected_tracker_name
        range_is_active, range_start_frame, range_end_frame = fs_proc.get_effective_scripting_range()

        # Hybrid trackers handle everything internally — skip Stage 1 artifact checks
        if is_hybrid:
            tracker_display = self._get_tracker_display_name(processing_mode)
            should_run_s1 = False
            self.reset_stage_status(stages=("stage1", "stage2", "stage3"))
            self.stage1_status_text = f"N/A ({tracker_display})"
            self.stage1_progress_value = 1.0
            self.stage2_status_text = "Queued..."
            self.logger.info(f"Starting {tracker_display} analysis sequence...", extra={'status_message': True})
            self.stage_thread = threading.Thread(target=self._run_full_analysis_thread_target, daemon=True, name="StagePipelineThread")
            self.stage_thread.start()
            self.app.energy_saver.reset_activity_timer()
            return

        # --- MODIFIED LOGIC TO CHECK FOR BOTH FILES ---
        full_msgpack_path = fm.get_output_path_for_file(fm.video_path, ".msgpack")
        preprocessed_video_path = fm.get_output_path_for_file(fm.video_path, "_preprocessed.mp4")

        # Stage 1 can be skipped only if BOTH the msgpack and preprocessed video exist and are valid
        msgpack_valid = os.path.exists(full_msgpack_path) and self._validate_preprocessed_artifacts(full_msgpack_path, preprocessed_video_path)
        full_run_artifacts_exist = msgpack_valid and os.path.exists(preprocessed_video_path)

        if self.frame_range_override:
            start_f_name, end_f_name = self.frame_range_override
            range_specific_path = fm.get_output_path_for_file(fm.video_path, f"_range_{start_f_name}-{end_f_name}.msgpack")
            fm.stage1_output_msgpack_path = range_specific_path
            should_run_s1 = True # Always rerun for autotuner
            self.logger.info("Autotuner mode: Forcing Stage 1 run for performance testing.")
        elif range_is_active:
            # Ranged analysis is more complex and usually for specific reprocessing,
            # so we assume it relies on the full preprocessed/msgpack files.
            fm.stage1_output_msgpack_path = full_msgpack_path
            if not full_run_artifacts_exist:
                should_run_s1 = True
                self.logger.info("Ranged analysis requested, but full Stage 1 artifacts are missing. Running Stage 1.")
            else:
                should_run_s1 = self.force_rerun_stage1
                if should_run_s1:
                    self.logger.info("Ranged analysis with force rerun: Running Stage 1.")
                else:
                    self.logger.info("Ranged analysis: Using existing full Stage 1 artifacts.")
        else: # Full analysis
            fm.stage1_output_msgpack_path = full_msgpack_path
            should_run_s1 = self.force_rerun_stage1 or not full_run_artifacts_exist
            if not should_run_s1:
                self.logger.info("All necessary Stage 1 artifacts exist. Skipping Stage 1 run.")
            elif self.force_rerun_stage1:
                self.logger.info("Forcing Stage 1 re-run as requested.")
            else:
                self.logger.info("One or more Stage 1 artifacts missing. Running Stage 1.")


        if not should_run_s1:
            self.stage1_status_text = f"Using existing: {os.path.basename(fm.stage1_output_msgpack_path or '')}"
            self.stage1_progress_value = 1.0
        else:
            self.reset_stage_status(stages=("stage1",)) # Reset all S1 state including final time
            self.stage1_status_text = "Queued..."

        self.reset_stage_status(stages=("stage2", "stage3"))
        self.stage2_status_text = "Queued..."
        if self._is_stage3_tracker(selected_mode) or self._is_mixed_stage3_tracker(selected_mode):
            self.stage3_status_text = "Queued..."

        self.logger.info("Starting Full Analysis sequence...", extra={'status_message': True})
        self.stage_thread = threading.Thread(target=self._run_full_analysis_thread_target, daemon=True, name="StagePipelineThread")
        self.stage_thread.start()
        self.app.energy_saver.reset_activity_timer()

    def _run_full_analysis_thread_target(self):
        fm = self.app.file_manager
        fs_proc = self.app.funscript_processor
        stage1_success = False
        stage2_success = False
        stage3_success = False
        preprocessed_path_for_s3 = None  # Initialize to prevent UnboundLocalError

        # Always use the tracker mode from the UI state, which is the single source of truth.
        selected_mode = self.processing_mode_for_thread
        # Handle both string (new dynamic system) and enum (legacy) modes
        mode_name = selected_mode if isinstance(selected_mode, str) else selected_mode.name
        self.logger.info(f"[Thread] Using processing mode: {mode_name}")

        try:
            # --- Hybrid Tracker: bypass standard Stage 1 + Stage 2 pipeline ---
            if self._is_hybrid_tracker(selected_mode):
                tracker_display = self._get_tracker_display_name(selected_mode)
                self.current_analysis_stage = 2  # Show as Stage 2 in UI (it does its own internal stages)
                self.logger.info(f"[Thread] Running {tracker_display}: {mode_name}")

                s2_start_time = time.time()
                hybrid_results = self._execute_hybrid_tracker(selected_mode)
                s2_end_time = time.time()

                hybrid_success = hybrid_results.get("success", False)
                stage1_success = hybrid_success  # For finally block cleanup
                stage2_success = hybrid_success

                if hybrid_success:
                    s2_elapsed_s = s2_end_time - s2_start_time
                    s2_elapsed_str = f"{int(s2_elapsed_s // 3600):02d}:{int((s2_elapsed_s % 3600) // 60):02d}:{int(s2_elapsed_s % 60):02d}"
                    self.gui_event_queue.put(("stage2_completed", s2_elapsed_str, None))

                    # Package results in the same format as standard Stage 2
                    output_data = hybrid_results.get("data", {})
                    packaged_data = {
                        "results_dict": output_data,
                        "was_ranged": False,
                        "range_frames": (0, -1)
                    }
                    self.last_analysis_result = packaged_data
                    self.gui_event_queue.put(("stage2_results_success", packaged_data, None))

                    completion_payload = {
                        "message": f"{tracker_display} analysis completed successfully.",
                        "status": "Completed",
                        "video_path": fm.video_path
                    }
                    self.gui_event_queue.put(("analysis_message", completion_payload, None))
                else:
                    error_msg = hybrid_results.get("error", "Unknown tracker failure")
                    self.gui_event_queue.put(("stage2_status_update", f"Failed: {error_msg}", "Failed"))
                    self.gui_event_queue.put(("analysis_message", {
                        "message": f"{tracker_display} analysis failed: {error_msg}",
                        "status": "Failed",
                        "video_path": fm.video_path
                    }, None))

                return  # Skip the standard Stage 1/2/3 pipeline

            # --- Stage 1 ---
            self.current_analysis_stage = 1
            range_is_active, range_start_frame, range_end_frame = fs_proc.get_effective_scripting_range()

            # Use the override if it exists, otherwise determine range normally
            frame_range_for_s1 = self.frame_range_override if self.frame_range_override else \
                ((range_start_frame, range_end_frame) if range_is_active else None)

            target_s1_path = fm.stage1_output_msgpack_path
            preprocessed_video_path = fm.get_output_path_for_file(fm.video_path, "_preprocessed.mp4")


            # Determine if this is an autotuner run
            is_autotune_context = self.frame_range_override is not None

            msgpack_exists = os.path.exists(target_s1_path) if target_s1_path else False
            preprocessed_video_exists = os.path.exists(preprocessed_video_path) if preprocessed_video_path else False

            if self.save_preprocessed_video:
                # If we want a preprocessed video, both must exist to skip Stage 1.
                full_run_artifacts_exist = msgpack_exists and preprocessed_video_exists
            else:
                # If we don't care about a preprocessed video, only the msgpack matters.
                full_run_artifacts_exist = msgpack_exists

            should_skip_stage1 = (not self.force_rerun_stage1 and full_run_artifacts_exist)

            if should_skip_stage1 and not self.frame_range_override:  # Never skip for autotuner
                stage1_success = True
                self.logger.info(f"[Thread] Stage 1 skipped, using existing artifacts.")
                self.gui_event_queue.put(("stage1_completed", "00:00:00 (Cached)", "Cached"))
                # Since we skipped, the preprocessed path is the one that already exists.
                preprocessed_path_for_s3 = preprocessed_video_path if preprocessed_video_exists else None
                
                # IMPORTANT: Load preprocessed video when Stage 1 is skipped too
                if preprocessed_path_for_s3 and os.path.exists(preprocessed_path_for_s3) and getattr(self, 'save_preprocessed_video', False):
                    fm.preprocessed_video_path = preprocessed_path_for_s3
                    
                    # CRITICAL: Update video processor to use preprocessed video for display and processing
                    if self.app.processor:
                        self.app.processor.set_active_video_source(preprocessed_path_for_s3)
                    
                    self.logger.info(f"Stage 1 skipped: Using existing preprocessed video for subsequent stages: {os.path.basename(preprocessed_path_for_s3)}")
                    # Notify GUI that we're working with preprocessed video
                    self.gui_event_queue.put(("preprocessed_video_loaded", {
                        "path": preprocessed_path_for_s3,
                        "message": f"Using cached preprocessed video: {os.path.basename(preprocessed_path_for_s3)}"
                    }, None))
            else:
                stage1_results = self._execute_stage1_logic(
                    frame_range=frame_range_for_s1,
                    output_path=target_s1_path,
                    num_producers_override=getattr(self, 'override_producers', None),
                    num_consumers_override=getattr(self, 'override_consumers', None),
                    is_autotune_run=is_autotune_context
                )
                stage1_success = stage1_results.get("success", False)
                preprocessed_path_for_s3 = stage1_results.get("preprocessed_video_path")

                if stage1_success:
                    max_fps_str = f"{stage1_results.get('max_fps', 0.0):.2f} FPS"
                    # Directly set the final FPS string to avoid the race condition.
                    # The autotuner reads this value immediately after the completion event is set.
                    self.stage1_final_fps_str = max_fps_str
                    self.gui_event_queue.put(("stage1_completed", self.stage1_time_elapsed_str, max_fps_str))
                    
                    # IMPORTANT: Update file manager to use preprocessed video if it was created
                    # This ensures subsequent stages (Stage 2 and 3) process the preprocessed file, not the original
                    if preprocessed_path_for_s3 and os.path.exists(preprocessed_path_for_s3) and getattr(self, 'save_preprocessed_video', False):
                        # Validate the preprocessed video before loading it
                        try:
                            from detection.cd.stage_1_cd import _validate_preprocessed_video_completeness
                            expected_frames = len(fm.stage1_output_msgpack_path) if hasattr(fm, 'stage1_output_msgpack_path') else 0
                            if self.app.processor and self.app.processor.video_info:
                                expected_frames = self.app.processor.video_info.get('total_frames', 0)
                                fps = self.app.processor.video_info.get('fps', 30.0)
                                
                                if _validate_preprocessed_video_completeness(preprocessed_path_for_s3, expected_frames, fps, self.logger):
                                    # Successfully validated - update file manager to use preprocessed video
                                    fm.preprocessed_video_path = preprocessed_path_for_s3
                                    
                                    # CRITICAL: Update video processor to use preprocessed video for display and processing
                                    if self.app.processor:
                                        self.app.processor.set_active_video_source(preprocessed_path_for_s3)
                                    
                                    self.logger.info(f"Stage 1 completed: Now using preprocessed video for subsequent stages: {os.path.basename(preprocessed_path_for_s3)}")
                                    
                                    # Notify GUI that we're now working with preprocessed video
                                    self.gui_event_queue.put(("preprocessed_video_loaded", {
                                        "path": preprocessed_path_for_s3,
                                        "message": f"Now using preprocessed video: {os.path.basename(preprocessed_path_for_s3)}"
                                    }, None))
                                else:
                                    self.logger.warning(f"Preprocessed video validation failed after Stage 1: {preprocessed_path_for_s3}")
                        except Exception as e:
                            self.logger.error(f"Error updating file manager with preprocessed video after Stage 1: {e}")
                    elif getattr(self, 'save_preprocessed_video', False):
                        self.logger.warning("Save/Reuse Preprocessed Video is enabled but no valid preprocessed video was created after Stage 1")

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

            generate_s2_funscript_actions = self._is_stage2_tracker(selected_mode) or self._is_mixed_stage3_tracker(selected_mode)
            is_s1_data_source_ranged = (frame_range_for_s1 is not None)

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
                if (self._is_stage3_tracker(selected_mode) or self._is_mixed_stage3_tracker(selected_mode)) and "Queued" in self.stage3_status_text:
                     self.gui_event_queue.put(("stage3_status_update", "Skipped", "S2 Failed/Aborted"))
                return

            # --- Stage 3 (or Finish) ---
            self.logger.info(f"[DEBUG] Determining stage progression for mode: {selected_mode}")
            self.logger.info(f"[DEBUG] is_stage2_tracker: {self._is_stage2_tracker(selected_mode)}")
            self.logger.info(f"[DEBUG] is_stage3_tracker: {self._is_stage3_tracker(selected_mode)}")
            self.logger.info(f"[DEBUG] is_mixed_stage3_tracker: {self._is_mixed_stage3_tracker(selected_mode)}")
            
            if self._is_stage2_tracker(selected_mode):
                if stage2_success:
                    packaged_data = {
                        "results_dict": s2_output_data,
                        "was_ranged": is_s1_data_source_ranged,
                        "range_frames": frame_range_for_s1 or (range_start_frame, range_end_frame)
                    }
                    self.last_analysis_result = packaged_data

                    self.gui_event_queue.put(("stage2_results_success", packaged_data, s2_overlay_path))

                completion_payload = {
                    "message": "AI CV (2-Stage) analysis completed successfully.",
                    "status": "Completed",
                    "video_path": fm.video_path
                }
                self.gui_event_queue.put(("analysis_message", completion_payload, None))
            elif self._is_mixed_stage3_tracker(selected_mode):
                self.logger.info(f"[DEBUG] Starting Mixed Stage 3 processing for mode: {selected_mode}")
                self.current_analysis_stage = 3
                segments_objects = s2_output_data.get("segments_objects", [])

                # Send complete Stage 2 results to properly update UI chapters
                if s2_output_data:
                    packaged_data = {
                        "results_dict": s2_output_data,
                        "was_ranged": is_s1_data_source_ranged,
                        "range_frames": frame_range_for_s1 or (range_start_frame, range_end_frame)
                    }
                    self.gui_event_queue.put(("stage2_results_success", packaged_data, s2_overlay_path))

                effective_range_is_active = frame_range_for_s1 is not None
                effective_start_frame = frame_range_for_s1[0] if effective_range_is_active else range_start_frame
                effective_end_frame = frame_range_for_s1[1] if effective_range_is_active else range_end_frame

                segments_for_s3 = self._filter_segments_for_range(segments_objects, effective_range_is_active,
                                                                  effective_start_frame, effective_end_frame)

                if not segments_for_s3:
                    self.gui_event_queue.put(("analysis_message", "No relevant segments in range for Mixed Stage 3.", "Info"))
                    return

                frame_objects_list = s2_output_data.get("all_s2_frame_objects_list", [])

                # Store SQLite database path for Mixed Stage 3
                self.app.s2_sqlite_db_path = s2_output_data.get("sqlite_db_path")

                # If frame objects were cleared from memory (SQLite mode), reload from database
                if not frame_objects_list and self.app.s2_sqlite_db_path:
                    try:
                        from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
                        storage = Stage2SQLiteStorage(self.app.s2_sqlite_db_path, self.logger)
                        frame_range = storage.get_frame_range()
                        if frame_range and frame_range[0] is not None:
                            frame_objects_map = storage.get_frame_objects_range(frame_range[0], frame_range[1])
                            self.app.s2_frame_objects_map_for_s3 = frame_objects_map
                            self.logger.info(f"Mixed Stage 3 data preparation: {len(frame_objects_map)} frame objects reloaded from SQLite database")
                        else:
                            self.app.s2_frame_objects_map_for_s3 = {}
                            self.logger.warning("Mixed Stage 3 data preparation: SQLite database has no frame objects")
                        storage.close()
                    except Exception as e:
                        self.logger.error(f"Failed to reload frame objects from SQLite: {e}", exc_info=True)
                        self.app.s2_frame_objects_map_for_s3 = {}
                else:
                    self.app.s2_frame_objects_map_for_s3 = {fo.frame_id: fo for fo in frame_objects_list}
                    self.logger.info(f"Mixed Stage 3 data preparation: {len(frame_objects_list)} frame objects loaded from cached Stage 2 data")

                self.logger.info(f"Starting Mixed Stage 3 with {preprocessed_path_for_s3}.")

                s3_results_dict = self._execute_stage3_mixed_module(segments_for_s3, preprocessed_path_for_s3, s2_output_data)
                stage3_success = s3_results_dict is not None

                if stage3_success:
                    self.gui_event_queue.put(("stage3_completed", self.stage3_time_elapsed_str, self.stage3_processing_fps_str))

                    packaged_data = {
                        "results_dict": s3_results_dict,
                        "was_ranged": effective_range_is_active,
                        "range_frames": (effective_start_frame, effective_end_frame)
                    }
                    self.last_analysis_result = packaged_data
                    
                    # Process Stage 3 mixed results immediately
                    self.gui_event_queue.put(("stage3_results_success", packaged_data, None))

                if stage3_success:
                    completion_payload = {
                        "message": "AI CV (3-Stage Mixed) analysis completed successfully.",
                        "status": "Completed",
                        "video_path": fm.video_path
                    }
                    self.gui_event_queue.put(("analysis_message", completion_payload, None))
            elif self._is_stage3_tracker(selected_mode):
                self.current_analysis_stage = 3
                segments_objects = s2_output_data.get("segments_objects", [])
                video_segments_for_gui = s2_output_data.get("video_segments", [])

                # Send complete Stage 2 results to properly update UI chapters
                if s2_output_data:
                    packaged_data = {
                        "results_dict": s2_output_data,
                        "was_ranged": is_s1_data_source_ranged,
                        "range_frames": frame_range_for_s1 or (range_start_frame, range_end_frame)
                    }
                    self.gui_event_queue.put(("stage2_results_success", packaged_data, s2_overlay_path))

                effective_range_is_active = frame_range_for_s1 is not None
                effective_start_frame = frame_range_for_s1[0] if effective_range_is_active else range_start_frame
                effective_end_frame = frame_range_for_s1[1] if effective_range_is_active else range_end_frame

                segments_for_s3 = self._filter_segments_for_range(segments_objects, effective_range_is_active,
                                                                  effective_start_frame, effective_end_frame)

                if not segments_for_s3:
                    self.gui_event_queue.put(("analysis_message", "No relevant segments in range for Stage 3.", "Info"))
                    return

                frame_objects_list = s2_output_data.get("all_s2_frame_objects_list", [])

                # Store SQLite database path for Stage 3
                self.app.s2_sqlite_db_path = s2_output_data.get("sqlite_db_path")

                # If frame objects were cleared from memory (SQLite mode), reload from database
                if not frame_objects_list and self.app.s2_sqlite_db_path:
                    try:
                        from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
                        storage = Stage2SQLiteStorage(self.app.s2_sqlite_db_path, self.logger)
                        frame_range = storage.get_frame_range()
                        if frame_range and frame_range[0] is not None:
                            frame_objects_map = storage.get_frame_objects_range(frame_range[0], frame_range[1])
                            self.app.s2_frame_objects_map_for_s3 = frame_objects_map
                            self.logger.info(f"Stage 3 data preparation: {len(frame_objects_map)} frame objects reloaded from SQLite database")
                        else:
                            self.app.s2_frame_objects_map_for_s3 = {}
                            self.logger.warning("Stage 3 data preparation: SQLite database has no frame objects")
                        storage.close()
                    except Exception as e:
                        self.logger.error(f"Failed to reload frame objects from SQLite: {e}", exc_info=True)
                        self.app.s2_frame_objects_map_for_s3 = {}
                else:
                    self.app.s2_frame_objects_map_for_s3 = {fo.frame_id: fo for fo in frame_objects_list}
                    self.logger.info(f"Stage 3 data preparation: {len(frame_objects_list)} frame objects loaded from cached Stage 2 data")

                self.logger.info(f"Starting Stage 3 with {preprocessed_path_for_s3}.")

                if self._is_mixed_stage3_tracker(selected_mode):
                    s3_results_dict = self._execute_stage3_mixed_module(segments_for_s3, preprocessed_path_for_s3, s2_output_data)
                elif self._has_modular_stage3(mode_name):
                    s3_results_dict = self._execute_stage3_modular_tracker(mode_name, segments_for_s3, preprocessed_path_for_s3)
                else:
                    s3_results_dict = self._execute_stage3_optical_flow_module(segments_for_s3, preprocessed_path_for_s3)
                stage3_success = s3_results_dict is not None

                if stage3_success:
                    self.gui_event_queue.put(("stage3_completed", self.stage3_time_elapsed_str, self.stage3_processing_fps_str))

                    packaged_data = {
                        "results_dict": s3_results_dict,
                        "was_ranged": effective_range_is_active,
                        "range_frames": (effective_start_frame, effective_end_frame)
                    }
                    self.last_analysis_result = packaged_data

                    # Debug: log what's in the results
                    _fs = s3_results_dict.get("funscript")
                    if _fs:
                        self.logger.info(f"[DEBUG] Stage 3 last_analysis_result set. funscript primary={len(_fs.primary_actions)}, secondary={len(_fs.secondary_actions)}")
                    else:
                        self.logger.info(f"[DEBUG] Stage 3 last_analysis_result set. Keys: {list(s3_results_dict.keys())}")

                    # Process Stage 3 results immediately
                    self.gui_event_queue.put(("stage3_results_success", packaged_data, None))

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
                                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable for msgpack")

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

            # CRITICAL FIX: Ensure tracker is stopped and disabled after offline analysis
            # This prevents the play button from triggering live tracking that overrides the offline signal
            if self.app.tracker:
                self.logger.info("Stopping tracker after offline analysis completion")
                self.app.tracker.stop_tracking()
            if self.app.processor:
                self.logger.info("Disabling tracker processing after offline analysis completion")
                self.app.processor.enable_tracker_processing = False

            # Clean up checkpoints on successful completion
            if stage1_success and stage2_success and (self._is_stage2_tracker(selected_mode) or self._is_hybrid_tracker(selected_mode) or stage3_success):
                self._cleanup_checkpoints_on_completion()

            # Clear the large data map and SQLite path from memory (if not already cleared)
            if hasattr(self.app, 's2_frame_objects_map_for_s3') and self.app.s2_frame_objects_map_for_s3 is not None:
                self.logger.info("[Thread] Clearing remaining Stage 2 data map from memory.")
                self.app.s2_frame_objects_map_for_s3 = None

            if hasattr(self.app, 's2_sqlite_db_path') and self.app.s2_sqlite_db_path:
                # Check if we should retain the database
                retain_database = self.app_settings.get("retain_stage2_database", True)
                
                # CRITICAL: Never delete database during 3-stage pipeline until Stage 3 completes
                # Stage 3 depends on the Stage 2 database for processing
                is_3_stage_pipeline = self._is_stage3_tracker(selected_mode) or self._is_mixed_stage3_tracker(selected_mode)
                stage3_completed = stage3_success if is_3_stage_pipeline else True
                
                if not retain_database and stage3_completed:
                    # Only clean up the database file if:
                    # 1. User has disabled database retention, AND
                    # 2. We're not in a 3-stage pipeline OR Stage 3 has completed successfully
                    try:
                        from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
                        temp_storage = Stage2SQLiteStorage(self.app.s2_sqlite_db_path, self.logger)
                        temp_storage.cleanup_database(remove_main_db=True)
                        self.logger.info("Stage 2 database file removed (retain_stage2_database=False)")
                    except Exception as e:
                        self.logger.warning(f"Failed to clean up Stage 2 database: {e}")
                elif not retain_database and is_3_stage_pipeline and not stage3_completed:
                    # In 3-stage pipeline, keep database until Stage 3 completes
                    self.logger.info(f"Stage 2 database retained for Stage 3 processing: {self.app.s2_sqlite_db_path}")
                else:
                    self.logger.info(f"Stage 2 database retained at: {self.app.s2_sqlite_db_path}")
                
                # Only clear the path reference if we're not in a 3-stage pipeline or Stage 3 completed
                if stage3_completed:
                    self.app.s2_sqlite_db_path = None

            # Clean up Stage 2 overlay file using same retention logic as database
            if fm.video_path:
                try:
                    s2_overlay_path = fm.get_output_path_for_file(fm.video_path, "_stage2_overlay.msgpack")
                    if os.path.exists(s2_overlay_path):
                        retain_database = self.app_settings.get("retain_stage2_database", True)
                        
                        # Use same logic as database cleanup
                        is_3_stage_pipeline = self._is_stage3_tracker(selected_mode) or self._is_mixed_stage3_tracker(selected_mode)
                        stage3_completed = stage3_success if is_3_stage_pipeline else True
                        
                        if not retain_database and stage3_completed:
                            # Clean up overlay file when database retention is disabled and safe to do so
                            try:
                                os.unlink(s2_overlay_path)
                                self.logger.info("Stage 2 overlay file removed (retain_stage2_database=False)")
                            except Exception as e:
                                self.logger.warning(f"Failed to remove Stage 2 overlay file: {e}")
                        elif not retain_database and is_3_stage_pipeline and not stage3_completed:
                            self.logger.info(f"Stage 2 overlay file retained for Stage 3 processing: {s2_overlay_path}")
                        else:
                            self.logger.info(f"Stage 2 overlay file retained at: {s2_overlay_path}")
                except Exception as e:
                    self.logger.warning(f"Error handling Stage 2 overlay file cleanup: {e}")

            gc.collect() # Encourage garbage collection

            self.logger.info("[Thread] Full analysis thread finished or exited.")
            if hasattr(self.app, 'single_video_analysis_complete_event'):
                self.app.single_video_analysis_complete_event.set()

    def abort_stage_processing(self):
        if self.full_analysis_active and self.stage_thread and self.stage_thread.is_alive():
            self.logger.info("Aborting current analysis stage(s)...", extra={'status_message': True})
            self.stop_stage_event.set()
            self.current_analysis_stage = -1  # Mark as aborting

        else:
            self.logger.info("No analysis pipeline running to abort.", extra={'status_message': False})
        self.app.energy_saver.reset_activity_timer()

    # process_gui_events() and all _handle_*() helpers live in StageGuiEventsMixin
    # (app_stage_gui_events.py) — inherited via the class declaration above.

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
    def _is_stage3_tracker(self, tracker_name):
        """Check if tracker is a 3-stage offline tracker."""
        tracker_ui = get_dynamic_tracker_ui()
        return tracker_ui.is_stage3_tracker(tracker_name)
    
    def _is_mixed_stage3_tracker(self, tracker_name):
        """Check if tracker is a mixed 3-stage offline tracker."""
        tracker_ui = get_dynamic_tracker_ui()
        return tracker_ui.is_mixed_stage3_tracker(tracker_name)
    
    def _is_stage2_tracker(self, tracker_name):
        """Check if tracker is a 2-stage offline tracker."""
        tracker_ui = get_dynamic_tracker_ui()
        return tracker_ui.is_stage2_tracker(tracker_name)
    
    def _is_hybrid_tracker(self, tracker_name):
        """Check if tracker handles Stage 1 internally (uses is_hybrid_tracker property)."""
        tracker_ui = get_dynamic_tracker_ui()
        return tracker_ui.is_hybrid_tracker(tracker_name)

    def _is_offline_tracker(self, tracker_name):
        """Check if tracker is any offline tracker."""
        tracker_ui = get_dynamic_tracker_ui()
        return tracker_ui.is_offline_tracker(tracker_name)

    def _get_tracker_display_name(self, tracker_name):
        """Get human-readable display name for a tracker."""
        tracker_ui = get_dynamic_tracker_ui()
        return tracker_ui.get_tracker_display_name(tracker_name)

    def update_settings_from_app(self):
        prod_usr = self.app_settings.get("num_producers_stage1")
        cons_usr = self.app_settings.get("num_consumers_stage1")
        # Always save preprocessed video for optical flow recovery in Stage 2
        self.save_preprocessed_video = self.app_settings.get("save_preprocessed_video", True)

        if not prod_usr or not cons_usr:
            cpu_cores = os.cpu_count() or 4
            self.num_producers_stage1 = max(1, min(5, cpu_cores // 2 - 2) if cpu_cores > 4 else 1)
            self.num_consumers_stage1 = max(1, min(9, cpu_cores // 2 + 2) if cpu_cores > 4 else 1)
            # MPS (Apple Silicon) shares unified memory with the system.
            # Scale consumers by available memory so low-RAM Macs don't OOM
            # while beefy Mac Studios keep their full throughput.
            if constants.DEVICE == 'mps':
                try:
                    import psutil
                    total_gb = psutil.virtual_memory().total / (1024 ** 3)
                except ImportError:
                    total_gb = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES') / (1024 ** 3)
                usable_gb = total_gb - constants.MPS_MEMORY_HEADROOM_GB
                mps_max = max(1, int(usable_gb / constants.MPS_MEMORY_PER_CONSUMER_GB))
                if self.num_consumers_stage1 > mps_max:
                    logger.info(
                        f"MPS backend: {total_gb:.0f}GB unified memory → "
                        f"scaling consumers from {self.num_consumers_stage1} to {mps_max} "
                        f"({constants.MPS_MEMORY_PER_CONSUMER_GB}GB/consumer + "
                        f"{constants.MPS_MEMORY_HEADROOM_GB}GB headroom)")
                    self.num_consumers_stage1 = mps_max
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
            "stage2_database_path": getattr(self.app, 's2_sqlite_db_path', None),
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

import os
import pickle
import threading
import math
import time
from queue import Queue
from typing import Optional, List, Dict, Any, Tuple, Union
import msgpack
import numpy as np
from bisect import bisect_left, bisect_right
import multiprocessing
import gc

from application.utils.checkpoint_manager import (
    CheckpointManager, ProcessingStage, CheckpointData,
    get_checkpoint_manager, initialize_checkpoint_manager
)
from application.gui_components.dynamic_tracker_ui import get_dynamic_tracker_ui

from detection.cd.data_structures.segments import Segment
import detection.cd.stage_1_cd as stage1_module
import detection.cd.stage_2_cd as stage2_module
import detection.cd.stage_3_of_processor as stage3_module
import detection.cd.stage_3_mixed_processor as stage3_mixed_module

from config import constants
from config.constants import ChapterSource, ChapterSegmentType
# TrackerMode removed - using dynamic discovery
from application.utils.stage_output_validator import can_skip_stage2_for_stage3
from application.utils import VideoSegment


class AppStageProcessor:
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
    
    def check_resumable_tasks(self) -> List[Tuple[str, CheckpointData]]:
        """Check for tasks that can be resumed from checkpoints."""
        return self.checkpoint_manager.get_resumable_tasks()
    
    def can_resume_video(self, video_path: str) -> Optional[CheckpointData]:
        """Check if a specific video has resumable progress."""
        return self.checkpoint_manager.find_latest_checkpoint(video_path)
    
    def start_resume_from_checkpoint(self, checkpoint_data: CheckpointData) -> bool:
        """Resume processing from a checkpoint."""
        try:
            if self.full_analysis_active:
                self.logger.warning("Cannot resume: Analysis already running.", extra={'status_message': True})
                return False
                
            if not os.path.exists(checkpoint_data.video_path):
                self.logger.error(f"Cannot resume: Video file not found: {checkpoint_data.video_path}", 
                                extra={'status_message': True})
                return False
            
            self.resume_data = checkpoint_data
            self.logger.info(f"Resuming {checkpoint_data.processing_stage.value} from {checkpoint_data.progress_percentage:.1f}%", 
                           extra={'status_message': True})
            
            # Load the video first
            if self.app.file_manager.video_path != checkpoint_data.video_path:
                # Would need to trigger video loading through the app
                # For now, assume video is already loaded
                pass
            
            # Resume based on the stage
            if checkpoint_data.processing_stage == ProcessingStage.STAGE_1_OBJECT_DETECTION:
                return self._resume_stage1(checkpoint_data)
            elif checkpoint_data.processing_stage == ProcessingStage.STAGE_2_OPTICAL_FLOW:
                return self._resume_stage2(checkpoint_data)
            elif checkpoint_data.processing_stage == ProcessingStage.STAGE_3_FUNSCRIPT_GENERATION:
                return self._resume_stage3(checkpoint_data)
            else:
                self.logger.error(f"Cannot resume: Unknown stage {checkpoint_data.processing_stage}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to resume from checkpoint: {e}", exc_info=True)
            return False
    
    def _resume_stage1(self, checkpoint_data: CheckpointData) -> bool:
        """Resume Stage 1 processing from checkpoint."""
        # For Stage 1, we typically need to restart as it's difficult to resume YOLO processing
        # But we can use the checkpoint to restore settings and show progress
        settings = checkpoint_data.processing_settings
        self.logger.info("Stage 1 resume: Restarting with original settings")
        
        # Restore settings and start fresh
        # Use tracker name instead of enum
        processing_mode = settings.get('processing_mode', 'stage3_optical_flow')
        return self._start_full_analysis_with_settings(processing_mode, settings)
    
    def _resume_stage2(self, checkpoint_data: CheckpointData) -> bool:
        """Resume Stage 2 processing from checkpoint."""
        # Stage 2 can potentially resume from intermediate data
        settings = checkpoint_data.processing_settings
        stage_data = checkpoint_data.stage_data
        
        self.logger.info(f"Stage 2 resume: Continuing from {checkpoint_data.progress_percentage:.1f}%")
        # Use tracker name instead of enum
        processing_mode = settings.get('processing_mode', 'stage3_optical_flow')
        return self._start_full_analysis_with_settings(processing_mode, settings)
    
    def _resume_stage3(self, checkpoint_data: CheckpointData) -> bool:
        """Resume Stage 3 processing from checkpoint."""
        # Stage 3 can resume from segment data
        settings = checkpoint_data.processing_settings
        stage_data = checkpoint_data.stage_data
        
        self.logger.info(f"Stage 3 resume: Continuing from segment {stage_data.get('current_segment', 0)}")
        # Use tracker name instead of enum
        processing_mode = settings.get('processing_mode', 'stage3_optical_flow')
        return self._start_full_analysis_with_settings(processing_mode, settings)
    
    def _start_full_analysis_with_settings(self, processing_mode: str, settings: Dict[str, Any]) -> bool:
        """Start full analysis with restored settings from checkpoint."""
        try:
            # Restore settings
            override_producers = settings.get('num_producers_override')
            override_consumers = settings.get('num_consumers_override')
            frame_range_override = settings.get('frame_range_override')
            
            # Start the analysis
            self.start_full_analysis(
                processing_mode=processing_mode,
                override_producers=override_producers,
                override_consumers=override_consumers,
                frame_range_override=frame_range_override,
                is_autotune_run=settings.get('is_autotune_run', False)
            )
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start analysis with restored settings: {e}")
            return False
    
    def delete_checkpoint_for_video(self, video_path: str) -> bool:
        """Delete all checkpoints for a specific video."""
        try:
            count = self.checkpoint_manager.delete_video_checkpoints(video_path)
            if count > 0:
                self.logger.info(f"Deleted {count} checkpoints for video", extra={'status_message': True})
            return count > 0
        except Exception as e:
            self.logger.error(f"Failed to delete checkpoints: {e}")
            return False
    
    def _create_checkpoint_if_needed(self, stage: ProcessingStage, frame_index: int, 
                                   total_frames: int, stage_data: Dict[str, Any]) -> None:
        """Create a checkpoint if enough time has passed."""
        if not self.app.file_manager.video_path:
            return
            
        if not self.checkpoint_manager.should_create_checkpoint(self.app.file_manager.video_path):
            return
        
        try:
            progress_percentage = (frame_index / total_frames * 100) if total_frames > 0 else 0
            
            processing_settings = {
                'processing_mode': getattr(self, 'processing_mode_for_thread', 'stage3_optical_flow'),
                'num_producers_override': getattr(self, 'override_producers', None),
                'num_consumers_override': getattr(self, 'override_consumers', None),
                'frame_range_override': getattr(self, 'frame_range_override', None),
                'is_autotune_run': getattr(self, 'is_autotune_run_for_thread', False),
                'yolo_det_model_path': self.app.yolo_det_model_path,
                'yolo_pose_model_path': self.app.yolo_pose_model_path,
                'confidence_threshold': self.app.tracker.confidence_threshold if self.app.tracker else 0.4,
                'yolo_input_size': self.app.yolo_input_size
            }
            
            checkpoint_id = self.checkpoint_manager.create_checkpoint(
                video_path=self.app.file_manager.video_path,
                stage=stage,
                progress_percentage=progress_percentage,
                frame_index=frame_index,
                total_frames=total_frames,
                stage_data=stage_data,
                processing_settings=processing_settings
            )
            
            if checkpoint_id:
                self.current_checkpoint_id = checkpoint_id
                
        except Exception as e:
            self.logger.error(f"Failed to create checkpoint: {e}")
    
    def _cleanup_checkpoints_on_completion(self):
        """Clean up checkpoints when processing completes successfully."""
        if self.app.file_manager.video_path:
            try:
                # Keep the final checkpoint but remove intermediate ones
                self.checkpoint_manager.delete_video_checkpoints(self.app.file_manager.video_path)
                self.current_checkpoint_id = None
            except Exception as e:
                self.logger.error(f"Failed to cleanup checkpoints: {e}")

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



    def _stage1_progress_callback(self, current, total, message="Processing...", time_elapsed=0.0, avg_fps=0.0, instant_fps=0.0, eta_seconds=0.0):
        progress = float(current) / total if total > 0 else -1.0
        progress_data = {
            "message": message, "current": current, "total": total,
            "time_elapsed": time_elapsed, "avg_fps": avg_fps, "instant_fps": instant_fps, "eta": eta_seconds
        }
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

        # Store the explicitly passed mode for the thread to use
        self.processing_mode_for_thread = processing_mode

        # Store the flag for the thread to use it
        self.is_autotune_run_for_thread = is_autotune_run

        # Store the overrides to be used by the thread
        self.override_producers = override_producers
        self.override_consumers = override_consumers

        selected_mode = self.app.app_state_ui.selected_tracker_name
        range_is_active, range_start_frame, range_end_frame = fs_proc.get_effective_scripting_range()

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
                self.app.s2_frame_objects_map_for_s3 = {fo.frame_id: fo for fo in frame_objects_list}
                self.logger.info(f"Mixed Stage 3 data preparation: {len(frame_objects_list)} frame objects loaded from cached Stage 2 data")

                # Store SQLite database path for Mixed Stage 3
                self.app.s2_sqlite_db_path = s2_output_data.get("sqlite_db_path")

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
                self.app.s2_frame_objects_map_for_s3 = {fo.frame_id: fo for fo in frame_objects_list}
                self.logger.info(f"Stage 3 data preparation: {len(frame_objects_list)} frame objects loaded from cached Stage 2 data")

                # Store SQLite database path for Stage 3
                self.app.s2_sqlite_db_path = s2_output_data.get("sqlite_db_path")

                self.logger.info(f"Starting Stage 3 with {preprocessed_path_for_s3}.")

                if self._is_mixed_stage3_tracker(selected_mode):
                    s3_results_dict = self._execute_stage3_mixed_module(segments_for_s3, preprocessed_path_for_s3, s2_output_data)
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
            if stage1_success and stage2_success and (self._is_stage2_tracker(selected_mode) or stage3_success):
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

            #preprocessed_video_path = None
            #if self.save_preprocessed_video:
            #    preprocessed_video_path = fm.get_output_path_for_file(fm.video_path, "_preprocessed.mp4")

            # Preprocessed video is now optional.
            preprocessed_video_path = fm.get_output_path_for_file(fm.video_path, "_preprocessed.mp4")

            num_producers = num_producers_override if num_producers_override is not None else self.num_producers_stage1
            num_consumers = num_consumers_override if num_consumers_override is not None else self.num_consumers_stage1

            result_path, max_fps = stage1_module.perform_yolo_analysis(
                video_path_arg=fm.video_path,
                yolo_model_path_arg=self.app.yolo_det_model_path,
                yolo_pose_model_path_arg=self.app.yolo_pose_model_path,
                confidence_threshold=self.app.tracker.confidence_threshold,
                progress_callback=self.on_stage1_progress, # Use the public attribute
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
                preprocessed_video_path_arg=preprocessed_video_path if self.save_preprocessed_video else None,
                is_autotune_run_arg=is_autotune_run
            )
            if self.stop_stage_event.is_set():
                self.gui_event_queue.put(("stage1_status_update", "S1 Aborted by user.", "Aborted"))
                self.gui_event_queue.put(
                    ("stage1_progress_update", 0.0, {"message": "Aborted", "current": 0, "total": 1}))
                return {"success": False, "max_fps": 0.0}
            if result_path and os.path.exists(result_path):
                fm.stage1_output_msgpack_path = result_path
                # Store preprocessed video path if it was created
                if self.save_preprocessed_video and os.path.exists(preprocessed_video_path):
                    fm.preprocessed_video_path = preprocessed_video_path
                    self.logger.info(f"Preprocessed video saved: {os.path.basename(preprocessed_video_path)}")
                final_msg = f"S1 Completed. Output: {os.path.basename(result_path)}"
                self.gui_event_queue.put(("stage1_status_update", final_msg, "Done"))
                self.gui_event_queue.put(("stage1_progress_update", 1.0, {"message": "Done", "current": 1, "total": 1}))
                self.app.project_manager.project_dirty = True
                return {"success": True, "max_fps": max_fps, "preprocessed_video_path": preprocessed_video_path if self.save_preprocessed_video else None}
            self.gui_event_queue.put(("stage1_status_update", "S1 Failed (no output file).", "Failed"))
            return {"success": False, "max_fps": 0.0, "preprocessed_video_path": None}
        except Exception as e:
            self.logger.error(f"Stage 1 execution error in AppLogic: {e}", exc_info=True,
                              extra={'status_message': True})
            self.gui_event_queue.put(("stage1_status_update", f"S1 Error - {str(e)}", "Error"))
            return {"success": False, "max_fps": 0.0, "preprocessed_video_path": None}
    
    def _load_existing_stage2_data(self, stage2_data_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Load existing Stage 2 data from database and overlay files.
        
        Args:
            stage2_data_info: Information about Stage 2 assets from validation
            
        Returns:
            Dictionary with loaded Stage 2 data or None if loading fails
        """
        try:
            file_paths = stage2_data_info.get('file_paths', {})
            db_path = file_paths.get('database')
            overlay_path = file_paths.get('overlay_msgpack')
            
            # DEBUG: Log initial data
            self.logger.info(f"DEBUG _load_existing_stage2_data: db_path={db_path}, overlay_path={overlay_path}")

            loaded_data = {
                "video_segments": [],
                "segments_objects": [],
                "overlay_data": None,
                "frame_objects_map": {},
                "all_s2_frame_objects_list": []
            }
            
            # Load segments and frame data from database
            if db_path and os.path.exists(db_path):
                try:
                    import sqlite3
                    with sqlite3.connect(db_path) as conn:
                        conn.row_factory = sqlite3.Row
                        cursor = conn.cursor()
                        
                        # Load segments data
                        segment_tables = ['atr_segments', 'segments']
                        for table_name in segment_tables:
                            try:
                                from application.utils.video_segment import VideoSegment

                                cursor.execute(f"SELECT * FROM {table_name}")
                                for segment_row in cursor:
                                    _seg = pickle.loads(segment_row['segment_data'])

                                    # Basic segment reconstruction - adjust based on actual DB schema
                                    segment = VideoSegment(
                                        start_frame_id=_seg.start_frame_id,
                                        end_frame_id=_seg.end_frame_id,
                                        class_id=getattr(_seg, 'class_id', None),
                                        class_name=getattr(_seg, 'class_name', 'unknown'),
                                        segment_type=ChapterSegmentType.POSITION.value,
                                        position_short_name=_seg.position_short_name,
                                        position_long_name=_seg.position_long_name,
                                        source=ChapterSource.STAGE2.value
                                    )
                                    loaded_data["video_segments"].append(segment)
                                    loaded_data["segments_objects"].append(segment)
                                break  # Use first successful table
                            except sqlite3.Error:
                                continue
                                
                        # Load frame objects from database for Stage 3
                        try:
                            from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
                            storage = Stage2SQLiteStorage(db_path, self.logger)
                            
                            # Get frame range to load all frame objects
                            min_frame, max_frame = storage.get_frame_range()
                            if min_frame is not None and max_frame is not None:
                                frame_objects_dict = storage.get_frame_objects_range(min_frame, max_frame)
                                
                                # Populate both data structures that Stage 3 expects
                                loaded_data["frame_objects_map"] = frame_objects_dict
                                loaded_data["all_s2_frame_objects_list"] = list(frame_objects_dict.values())
                                
                                self.logger.info(f"Loaded {len(frame_objects_dict)} frame objects from database")
                            
                            storage.close()
                        except Exception as fe:
                            self.logger.warning(f"Failed to load frame objects from database: {fe}")
                        
                        # Store reference to database for Stage 3
                        self.app.s2_sqlite_db_path = db_path
                        loaded_data["sqlite_db_path"] = db_path
                        
                except Exception as e:
                    self.logger.warning(f"Failed to load segments from database: {e}")
            
            # Load overlay data if available
            if overlay_path and os.path.exists(overlay_path):
                try:
                    import msgpack
                    with open(overlay_path, 'rb') as f:
                        overlay_data = msgpack.unpack(f, raw=False)
                        loaded_data["overlay_data"] = overlay_data
                except Exception as e:
                    self.logger.warning(f"Failed to load overlay data: {e}")
            
            # Ensure we have some data
            has_segments = bool(loaded_data["video_segments"])
            has_frame_objects = stage2_data_info.get('frame_objects_available', False)
            
            if has_segments or has_frame_objects:
                self.logger.info(f"Loaded existing Stage 2 data: {len(loaded_data['video_segments'])} segments")
                
                # If we have frame objects but no segments, recreate segments from overlay data
                # This uses the original Stage 2 logic to properly reconstruct segments
                if not has_segments and has_frame_objects and loaded_data.get("overlay_data"):
                    self.logger.info("Reconstructing video segments from Stage 2 overlay data")
                    
                    try:
                        # Reconstruct frame objects from overlay data
                        frame_objects = self._reconstruct_frame_objects_from_overlay(loaded_data["overlay_data"])
                        
                        if frame_objects:
                            # Use the original Stage 2 logic to create segments
                            from detection.cd.stage_2_cd import _aggregate_segments
                            
                            # Get FPS from app processor if available
                            fps = 30.0  # Default fallback
                            if self.app and hasattr(self.app, 'processor') and self.app.processor:
                                video_info = getattr(self.app.processor, 'video_info', {})
                                fps = video_info.get('fps', 30.0)
                            
                            # Recreate segments using Stage 2 logic
                            # Use default min_segment_duration (1 second = fps frames)
                            min_segment_duration_frames = int(fps * 1.0)
                            segments = _aggregate_segments(frame_objects, fps, min_segment_duration_frames, self.logger)
                            
                            # Convert segments to video segments format
                            from application.utils.video_segment import VideoSegment
                            for segment in segments:
                                # Get segment data from segment
                                segment_dict = segment.to_dict()
                                
                                # Create VideoSegment using the data from Segment
                                video_segment = VideoSegment(
                                    start_frame_id=segment_dict['start_frame_id'],
                                    end_frame_id=segment_dict['end_frame_id'],
                                    class_id=1,  # Default class ID
                                    class_name=segment_dict['class_name'],
                                    segment_type=ChapterSegmentType.POSITION.value,
                                    position_short_name=segment_dict['position_short_name'],
                                    position_long_name=segment_dict['position_long_name'],
                                    duration=segment_dict['duration'],
                                    source=ChapterSource.STAGE3.value
                                )
                                loaded_data["video_segments"].append(video_segment)
                                loaded_data["segments_objects"].append(segment)
                            
                            # Also add frame objects for Stage 3
                            frame_objects_map = {fo.frame_id: fo for fo in frame_objects}
                            loaded_data["frame_objects_map"] = frame_objects_map
                            loaded_data["all_s2_frame_objects_list"] = frame_objects
                            
                            self.logger.info(f"Reconstructed {len(atr_segments)} segments and {len(frame_objects)} frame objects from overlay data")
                        else:
                            self.logger.warning("Failed to reconstruct frame objects from overlay data")
                            
                    except Exception as e:
                        self.logger.warning(f"Failed to reconstruct segments from overlay: {e}")
                        # Fall back to single segment if reconstruction fails
                        self.logger.info("Falling back to single full-video segment")
                        from application.utils.video_segment import VideoSegment
                        estimated_frame_count = stage2_data_info.get('estimated_frame_count', 17982)
                        fallback_segment = VideoSegment(
                            start_frame_id=0, end_frame_id=estimated_frame_count - 1,
                            class_id=1, class_name="mixed", segment_type="SexAct",
                            position_short_name="Mixed", position_long_name="Mixed Content"
                        )
                        loaded_data["video_segments"].append(fallback_segment)
                
                # Create funscript object from loaded segments for consistency with unified architecture
                try:
                    from funscript.dual_axis_funscript import DualAxisFunscript
                    funscript_obj = DualAxisFunscript()
                    
                    # Get FPS from app processor
                    fps = 30.0  # Default fallback
                    if self.app and hasattr(self.app, 'processor') and self.app.processor:
                        video_info = getattr(self.app.processor, 'video_info', {})
                        fps = video_info.get('fps', 30.0)
                    
                    # Add chapters from video segments
                    if loaded_data["video_segments"]:
                        funscript_obj.set_chapters_from_segments(loaded_data["video_segments"], fps)
                        self.logger.info(f"Created funscript with {len(funscript_obj.chapters)} chapters from loaded segments")
                    
                    # Regenerate actions from frame objects for mixed mode compatibility
                    if loaded_data.get("all_s2_frame_objects_list"):
                        frame_objects_list = loaded_data["all_s2_frame_objects_list"]
                        actions_generated = 0
                        
                        for frame_obj in frame_objects_list:
                            if hasattr(frame_obj, 'atr_funscript_distance') and hasattr(frame_obj, 'frame_id'):
                                try:
                                    timestamp_ms = int((frame_obj.frame_id / fps) * 1000)
                                    pos_0_100 = int(frame_obj.atr_funscript_distance)
                                    pos_0_100 = max(0, min(100, pos_0_100))  # Clamp to valid range
                                    funscript_obj.add_action(timestamp_ms, pos_0_100)
                                    actions_generated += 1
                                except (ValueError, TypeError, AttributeError):
                                    continue
                        
                        if actions_generated > 0:
                            self.logger.info(f"Regenerated {actions_generated} funscript actions from cached frame objects for mixed mode")
                        else:
                            self.logger.warning("No valid funscript actions could be regenerated from cached frame objects")
                    
                    loaded_data["funscript"] = funscript_obj
                    
                except Exception as e:
                    self.logger.warning(f"Failed to create funscript from loaded data: {e}")
                
                return loaded_data
            else:
                self.logger.warning("No usable Stage 2 data found in existing assets")
                return None
                
        except Exception as e:
            self.logger.error(f"Error loading existing Stage 2 data: {e}")
            return None
    
    def _process_stage2_results_direct(self, packaged_data: Dict[str, Any], s2_overlay_path: Optional[str] = None):
        """
        Process Stage 2 results directly without going through GUI event queue.
        This is used for CLI mode and when loading existing Stage 2 data.
        """
        fs_proc = self.app.funscript_processor
        results_dict = packaged_data.get("results_dict", {})
        
        # Get the funscript object first
        funscript_obj = results_dict.get("funscript")
        
        # Process the funscript object or fall back to raw actions
        if funscript_obj:
            # Use funscript object (preferred)
            primary_actions = funscript_obj.primary_actions
            secondary_actions = funscript_obj.secondary_actions

            # Get the application's current axis settings
            axis_mode = self.app.tracking_axis_mode
            target_timeline = self.app.single_axis_output_target

            self.app.logger.info(f"Applying 2-Stage results with axis mode: {axis_mode} and target: {target_timeline}.")

            # Only clear and update timelines if we have actions to write
            if axis_mode == "both":
                # Overwrite both timelines only if there are actions
                if primary_actions:
                    fs_proc.clear_timeline_history_and_set_new_baseline(1, primary_actions, "Stage 2 (Primary)")
                else:
                    self.app.logger.warning("No primary actions - Timeline 1 unchanged")
                
                if secondary_actions:
                    fs_proc.clear_timeline_history_and_set_new_baseline(2, secondary_actions, "Stage 2 (Secondary)")
                else:
                    self.app.logger.warning("No secondary actions - Timeline 2 unchanged")

            elif axis_mode == "vertical":
                # Overwrite ONLY the target timeline if there are actions
                if primary_actions:
                    if target_timeline == "primary":
                        self.app.logger.info("Writing to Timeline 1, Timeline 2 is untouched.")
                        fs_proc.clear_timeline_history_and_set_new_baseline(1, primary_actions, "Stage 2 (Vertical)")
                    else:  # Target is secondary
                        self.app.logger.info("Writing to Timeline 2, Timeline 1 is untouched.")
                        fs_proc.clear_timeline_history_and_set_new_baseline(2, primary_actions, "Stage 2 (Vertical)")
                else:
                    self.app.logger.warning(f"No vertical actions - Timeline {1 if target_timeline == 'primary' else 2} unchanged")

            elif axis_mode == "horizontal":
                # Overwrite ONLY the target timeline if there are actions
                if secondary_actions:
                    if target_timeline == "primary":
                        self.app.logger.info("Writing horizontal data to Timeline 1, Timeline 2 is untouched.")
                        fs_proc.clear_timeline_history_and_set_new_baseline(1, secondary_actions, "Stage 2 (Horizontal)")
                    else:  # Target is secondary
                        self.app.logger.info("Writing horizontal data to Timeline 2, Timeline 1 is untouched.")
                        fs_proc.clear_timeline_history_and_set_new_baseline(2, secondary_actions, "Stage 2 (Horizontal)")
                else:
                    self.app.logger.warning(f"No horizontal actions - Timeline {1 if target_timeline == 'primary' else 2} unchanged")

                self.logger.info("Updating chapters with Stage 2 analysis results.")
                fs_proc.video_chapters.clear()
            
            # Extract chapters from the funscript object instead of separate video_segments_data
            if  hasattr(funscript_obj, 'chapters') and funscript_obj.chapters:
                fps = self.app.processor.video_info.get('fps', 30.0) if self.app.processor and self.app.processor.video_info else 30.0
                for chapter in funscript_obj.chapters:
                    # Convert funscript chapter back to VideoSegment
                    start_frame = int((chapter.get('start', 0) / 1000.0) * fps)
                    self.logger.info(f"Line 1378, start_frame: {start_frame}")
                    end_frame = int((chapter.get('end', 0) / 1000.0) * fps)
                    self.logger.info(f"Line 1380, end_frame: {end_frame}")
                    
                    from application.utils.video_segment import VideoSegment
                    video_segment = VideoSegment(
                        start_frame_id=start_frame,
                        end_frame_id=end_frame,
                        class_id=chapter.get('class_id'),  # Preserve class_id for corruption recovery
                        class_name=chapter.get('name', 'Unknown'),
                        segment_type="SexAct",
                        position_short_name=chapter.get('position_short', chapter.get('name', '')),
                        position_long_name=chapter.get('position_long', chapter.get('description', chapter.get('name', 'Unknown'))),
                        source="stage2_funscript"
                    )
                    fs_proc.video_chapters.append(video_segment)
                self.logger.info(f"Extracted {len(funscript_obj.chapters)} chapters from funscript object")

        self.stage2_status_text = "S2 Completed. Results Processed."
        self.app.project_manager.project_dirty = True
        self.logger.info("Processed Stage 2 results directly.")
    
    def _reconstruct_frame_objects_from_overlay(self, overlay_data):
        """
        Reconstruct minimal frame objects from Stage 2 overlay data for segment creation.
        
        Args:
            overlay_data: List of frame overlay dictionaries from msgpack
            
        Returns:
            List of minimal frame objects with position data needed for segmentation
        """
        try:
            from detection.cd.data_structures import FrameObject
            
            frame_objects = []
            for frame_dict in overlay_data:
                if isinstance(frame_dict, dict) and 'frame_id' in frame_dict:
                    # Create minimal frame object with just the data needed for segmentation
                    frame_obj = FrameObject(
                        frame_id=frame_dict.get('frame_id', 0),
                        yolo_input_size=640  # Standard size used in validation
                    )
                    
                    # Set the position data which is essential for segment creation
                    frame_obj.atr_assigned_position = frame_dict.get('atr_assigned_position', 'unknown')
                    
                    # Add any other fields that might be needed for segment logic
                    frame_obj.motion_mode = frame_dict.get('motion_mode', 'unknown')
                    frame_obj.active_interaction_track_id = frame_dict.get('active_interaction_track_id', 0)
                    
                    frame_objects.append(frame_obj)
            
            self.logger.debug(f"Reconstructed {len(frame_objects)} frame objects from overlay data")
            return frame_objects
            
        except Exception as e:
            self.logger.warning(f"Error reconstructing frame objects from overlay: {e}")
            return []

    def _execute_stage2_logic(self, s2_overlay_output_path: Optional[str], generate_funscript_actions: bool = True, is_ranged_data_source: bool = False) -> Dict[str, Any]:
        self.gui_event_queue.put(("stage2_status_update", "Checking existing S2...", "Validating"))
        
        fm = self.app.file_manager
        
        # Check if we can skip Stage 2 by reusing existing assets
        if not self.force_rerun_stage2_segmentation:
            from application.utils.stage_output_validator import can_skip_stage2_for_stage3
            
            # Get the correct output folder where Stage 2 files are stored
            output_folder = os.path.dirname(fm.get_output_path_for_file(fm.video_path, "_dummy.tmp"))
            
            # Pass project-saved database path if available for priority checking
            project_db_path = getattr(self.app, 's2_sqlite_db_path', None)
            can_skip, stage2_data = can_skip_stage2_for_stage3(fm.video_path, False, output_folder, self.logger, project_db_path)
            
            if can_skip:
                self.logger.info("Stage 2 assets found and validated - skipping Stage 2 processing")
                self.gui_event_queue.put(("stage2_status_update", "Reusing existing S2...", "Loading cached results"))
                
                # Load existing Stage 2 data
                existing_data = self._load_existing_stage2_data(stage2_data)
                if existing_data:
                    # Update progress to show completion
                    self.gui_event_queue.put(("stage2_dual_progress", (6, 6, "Loaded from cache"), (1, 1, "Complete")))
                    self.gui_event_queue.put(("stage2_status_update", "S2 Complete (Cached)", "Loaded from cache"))
                    
                    if s2_overlay_output_path and os.path.exists(s2_overlay_output_path):
                        self.gui_event_queue.put(("load_s2_overlay", s2_overlay_output_path, None))
                    
                    # Process the existing Stage 2 data through results processing
                    # In CLI mode, directly process instead of using GUI event queue
                    packaged_data = {
                        "results_dict": existing_data,
                        "was_ranged": False,
                        "range_frames": (0, -1)
                    }
                    
                    # Process Stage 2 results directly (for CLI mode compatibility)
                    try:
                        self._process_stage2_results_direct(packaged_data, s2_overlay_output_path)
                    except Exception as e:
                        self.logger.warning(f"Failed to process Stage 2 results directly: {e}")
                        # Fallback to GUI event queue
                        self.gui_event_queue.put(("stage2_results_success", packaged_data, s2_overlay_output_path))
                    
                    self.logger.info("DEBUG: Returning early with cached data")
                    return {
                        "success": True,
                        "data": existing_data,
                        "skipped": True,
                        "skip_reason": "Existing Stage 2 assets validated and reused"
                    }
                else:
                    self.logger.warning("Failed to load existing Stage 2 data - will reprocess")
            else:
                self.logger.debug("Stage 2 assets not suitable for reuse - processing from scratch")
        else:
            self.logger.debug("Stage 2 force rerun enabled - processing from scratch")
        
        self.gui_event_queue.put(("stage2_status_update", "Running S2...", "Initializing S2..."))
        initial_total_main_steps = getattr(stage2_module, 'ATR_PASS_COUNT', self.S2_TOTAL_MAIN_STEPS_FALLBACK)
        if not generate_funscript_actions:
            initial_total_main_steps = getattr(stage2_module, 'ATR_PASS_COUNT_SEGMENTATION_ONLY', 3)  # Assume S2 module defines this
            self.gui_event_queue.put(("stage2_status_update", "Running S2 (Segmentation)...", "Initializing S2 Seg..."))

        self.gui_event_queue.put(("stage2_dual_progress", (1, initial_total_main_steps, "Initializing..."), (0, 1, "Starting")))
        
        try:
            if not stage2_module:
                msg = "Error - S2 Module not loaded."
                self.gui_event_queue.put(("stage2_status_update", msg, "Error"))
                return {"success": False, "error": msg}
            if not fm.stage1_output_msgpack_path:
                msg = "Error - S1 output missing for S2."
                self.gui_event_queue.put(("stage2_status_update", msg, "Error"))
                return {"success": False, "error": msg}

            # Stage 2 OF recovery requires the preprocessed video from Stage 1.
            preprocessed_video_path_for_s2 = None
            if self.app.file_manager.preprocessed_video_path and os.path.exists(self.app.file_manager.preprocessed_video_path):
                preprocessed_video_path_for_s2 = self.app.file_manager.preprocessed_video_path
            else:
                # Fallback: try to guess the path if the direct reference is missing.
                preprocessed_video_path_for_s2 = fm.get_output_path_for_file(fm.video_path, "_preprocessed.mp4")
                if not os.path.exists(preprocessed_video_path_for_s2):
                    self.logger.warning("Optical flow recovery may fail: Preprocessed video from Stage 1 not found.")
                    preprocessed_video_path_for_s2 = None

            range_is_active, range_start_frame, range_end_frame = self.app.funscript_processor.get_effective_scripting_range()

            self.logger.info("Using Stage 2 implementation")
            stage2_results = stage2_module.perform_contact_analysis(
                video_path_arg=fm.video_path,
                msgpack_file_path_arg=fm.stage1_output_msgpack_path,
                preprocessed_video_path_arg=preprocessed_video_path_for_s2,
                progress_callback=self.on_stage2_progress,
                stop_event=self.stop_stage_event,
                app=self.app,
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
                generate_funscript_actions_arg=generate_funscript_actions,
                output_folder_path=os.path.dirname(fm.get_output_path_for_file(fm.video_path, "_dummy.tmp"))
            )
            if self.stop_stage_event.is_set():
                msg = "S2 Aborted by user."
                self.gui_event_queue.put(("stage2_status_update", msg, "Aborted"))
                current_main_step = int(self.stage2_main_progress_value * initial_total_main_steps)
                self.gui_event_queue.put(("stage2_dual_progress", (current_main_step, initial_total_main_steps, "Aborted"), (0, 1, "Aborted")))
                return {"success": False, "error": msg}

            if stage2_results and "error" not in stage2_results:
                # Capture and save the database path from Stage 2 results
                sqlite_db_path = stage2_results.get("sqlite_db_path")
                if sqlite_db_path:
                    self.app.s2_sqlite_db_path = sqlite_db_path
                    self.logger.info(f"Stage 2 database path saved: {sqlite_db_path}")
                
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

    def _execute_stage3_optical_flow_module(self, segments_objects: List[Any], preprocessed_video_path: Optional[str]) -> bool:
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
            "target_size_preprocess": getattr(self.app.tracker, 'target_size_preprocess', (640, 640)) if self.app.tracker else (640, 640),
            "flow_history_window_smooth": self.app_settings.get('tracker_flow_history_window_smooth', 3),
            "adaptive_flow_scale": self.app_settings.get('tracker_adaptive_flow_scale', True),
            "use_sparse_flow": self.app_settings.get('tracker_use_sparse_flow', False),
            "base_amplification_factor": self.app_settings.get('tracker_base_amplification', constants.DEFAULT_LIVE_TRACKER_BASE_AMPLIFICATION),
            "class_specific_amplification_multipliers": self.app_settings.get('tracker_class_specific_multipliers', constants.DEFAULT_CLASS_AMP_MULTIPLIERS),
            "y_offset": self.app_settings.get('tracker_y_offset', constants.DEFAULT_LIVE_TRACKER_Y_OFFSET),
            "x_offset": self.app_settings.get('tracker_x_offset', constants.DEFAULT_LIVE_TRACKER_X_OFFSET),
            "sensitivity": self.app_settings.get('tracker_sensitivity', constants.DEFAULT_LIVE_TRACKER_SENSITIVITY),
            "oscillation_grid_size": self.app_settings.get('oscillation_detector_grid_size', 20),
            "oscillation_sensitivity": self.app_settings.get('oscillation_detector_sensitivity', 1.0)
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
            "output_delay_frames": getattr(self.app.tracker, 'output_delay_frames', 0) if self.app.tracker else 0,
            "num_warmup_frames_s3": self.app_settings.get('s3_num_warmup_frames', 10 + (getattr(self.app.tracker, 'output_delay_frames', 0) if self.app.tracker else 0)),
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

        # Get SQLite database path from app instance
        sqlite_db_path = getattr(self.app, 's2_sqlite_db_path', None)

        s3_results = stage3_module.perform_stage3_analysis(
            video_path=self.app.file_manager.video_path,
            preprocessed_video_path_arg=preprocessed_video_path,
            atr_segments_list=segments_objects,
            s2_frame_objects_map=self.app.s2_frame_objects_map_for_s3,
            tracker_config=tracker_config_s3,
            common_app_config=common_app_config_s3,
            progress_callback=self.on_stage3_progress,  # Use the public attribute
            stop_event=self.stop_stage_event,
            parent_logger=self.logger,
            sqlite_db_path=sqlite_db_path
        )
        
        # Clear Stage 2 memory map immediately after Stage 3 starts processing
        # Stage 3 has already consumed the data it needs
        if hasattr(self.app, 's2_frame_objects_map_for_s3') and self.app.s2_frame_objects_map_for_s3:
            map_size = len(self.app.s2_frame_objects_map_for_s3)
            self.app.s2_frame_objects_map_for_s3 = None
            self.logger.info(f"[Memory] Cleared Stage 2 data map ({map_size} frames) early to reduce memory pressure")
            
            # Force garbage collection to ensure immediate memory release
            import gc
            gc.collect()

        if self.stop_stage_event.is_set(): return False

        if s3_results and "error" not in s3_results:
            # Get funscript object or fall back to raw actions
            funscript_obj = s3_results.get("funscript")
            if funscript_obj:
                final_s3_primary_actions = funscript_obj.primary_actions
                final_s3_secondary_actions = funscript_obj.secondary_actions
            else:
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
                if final_s3_primary_actions:
                    fs_proc.clear_actions_in_range_and_inject_new(1, final_s3_primary_actions, start_ms, end_ms, op_desc_s3_range + " (T1)")
                else:
                    self.logger.warning("No primary actions from Stage 3 - Timeline 1 range unchanged")
                
                if final_s3_secondary_actions:
                    fs_proc.clear_actions_in_range_and_inject_new(2, final_s3_secondary_actions, start_ms, end_ms, op_desc_s3_range + " (T2)")
                else:
                    self.logger.info("No secondary actions from Stage 3 - Timeline 2 range unchanged")
            else:
                # Only update timelines if there are actions
                if final_s3_primary_actions:
                    fs_proc.clear_timeline_history_and_set_new_baseline(1, final_s3_primary_actions, op_desc_s3 + " (T1)")
                else:
                    self.logger.warning("No primary actions from Stage 3 - Timeline 1 unchanged")
                
                if final_s3_secondary_actions:
                    fs_proc.clear_timeline_history_and_set_new_baseline(2, final_s3_secondary_actions, op_desc_s3 + " (T2)")
                else:
                    self.logger.info("No secondary actions from Stage 3 - Timeline 2 unchanged")

            self.gui_event_queue.put(("stage3_status_update", "Stage 3 Completed.", "Done"))
            self.app.project_manager.project_dirty = True

            # Update chapters for GUI if video_segments are present (3-stage fix)
            if "video_segments" in s3_results:
                fs_proc.video_chapters.clear()
                for seg_data in s3_results["video_segments"]:
                    fs_proc.video_chapters.append(VideoSegment.from_dict(seg_data))
                self.app.app_state_ui.heatmap_dirty = True
                self.app.app_state_ui.funscript_preview_dirty = True
            return s3_results
        else:
            error_msg = s3_results.get("error", "Unknown S3 failure") if s3_results else "S3 returned None."
            self.logger.error(f"Stage 3 execution failed: {error_msg}")
            self.gui_event_queue.put(("stage3_status_update", f"S3 Failed: {error_msg}", "Failed"))
            return None

    def _execute_stage3_mixed_module(self, segments_objects: List[Any], preprocessed_video_path: Optional[str], s2_output_data: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
        """Execute Mixed Stage 3 processing using stage_3_mixed_processor if available."""
        if stage3_mixed_module is None:
            self.logger.error("Stage 3 Mixed module not available.")
            return None
        fs_proc = self.app.funscript_processor
        fm = self.app.file_manager
        if not fm or not fm.video_path:
            self.logger.error("Stage 3 Mixed: Video path not available.")
            return None
        common_app_config = {
            "yolo_det_model_path": self.app.yolo_det_model_path,
            "yolo_pose_model_path": self.app.yolo_pose_model_path,
            "yolo_input_size": self.app.yolo_input_size,
            "video_fps": (self.app.processor.video_info.get('fps', 30.0) if self.app.processor and self.app.processor.video_info else 30.0),
        }
        try:
            # Get Stage 2 funscript from the output data
            self.logger.info(f"Stage 2 data available for mixed mode: {s2_output_data is not None}")
            stage2_funscript = s2_output_data.get("funscript") if s2_output_data else None
            self.logger.info(f"Stage 2 funscript available: {stage2_funscript is not None}")
            if stage2_funscript:
                self.logger.info(f"Stage 2 funscript has primary_actions: {hasattr(stage2_funscript, 'primary_actions')}")
                if hasattr(stage2_funscript, 'primary_actions'):
                    self.logger.info(f"Stage 2 funscript primary_actions count: {len(stage2_funscript.primary_actions)}")
            
            results = stage3_mixed_module.perform_mixed_stage_analysis(
                video_path=fm.video_path,
                preprocessed_video_path_arg=preprocessed_video_path,
                atr_segments_list=segments_objects,
                s2_frame_objects_map=self.app.s2_frame_objects_map_for_s3 or {},  # Legacy fallback
                tracker_config={},
                common_app_config=common_app_config,
                progress_callback=self.on_stage3_progress,
                stop_event=self.stop_stage_event,
                parent_logger=self.logger,
                sqlite_db_path=getattr(self.app, 's2_sqlite_db_path', None),
                stage2_funscript=stage2_funscript,  # New: Pass Stage 2 funscript directly
            )
            return results
        except Exception as e:
            self.logger.error(f"Stage 3 Mixed execution failed: {e}", exc_info=True)
            return None

    def abort_stage_processing(self):
        if self.full_analysis_active and self.stage_thread and self.stage_thread.is_alive():
            self.logger.info("Aborting current analysis stage(s)...", extra={'status_message': True})
            self.stop_stage_event.set()
            self.current_analysis_stage = -1  # Mark as aborting

        else:
            self.logger.info("No analysis pipeline running to abort.", extra={'status_message': False})
        self.app.energy_saver.reset_activity_timer()

    def process_gui_events(self):
        if self.full_analysis_active or self.refinement_analysis_active:
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

                if event_type == "stage1_progress_update":
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
                elif event_type == "preprocessed_video_loaded":
                    # Handle the event when preprocessed video is loaded after Stage 1
                    if isinstance(data1, dict):
                        preprocessed_info = data1
                        message = preprocessed_info.get("message", "Preprocessed video loaded")
                        # Update status message and log info
                        self.app.logger.info(message, extra={'status_message': True})
                        # Update UI info panel if available
                        if hasattr(self.app, 'set_status_message'):
                            self.app.set_status_message(message)
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

                    # --- Handle both tuple (for simple steps) and dict (for complex steps) ---
                    if isinstance(sub_step_info, dict):
                        # This is our new, detailed progress payload
                        sub_current = sub_step_info.get("current", 0)
                        sub_total = sub_step_info.get("total", 0)
                        self.stage2_sub_progress_value = float(sub_current) / sub_total if sub_total > 0 else 0.0
                        self.stage2_sub_progress_label = f"{sub_step_info.get('message', '')} ({sub_current}/{sub_total})"

                        t_el = sub_step_info.get("time_elapsed", 0.0)
                        fps = sub_step_info.get("fps", 0.0)
                        eta = sub_step_info.get("eta", 0.0)

                        self.stage2_sub_time_elapsed_str = f"{int(t_el // 3600):02d}:{int((t_el % 3600) // 60):02d}:{int(t_el % 60):02d}"
                        self.stage2_sub_processing_fps_str = f"{int(fps)} FPS"
                        if math.isnan(eta) or math.isinf(eta) or eta <= 0:
                            self.stage2_sub_eta_str = "N/A"
                        else:
                            self.stage2_sub_eta_str = f"{int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d}"

                    elif isinstance(sub_step_info, tuple) and len(sub_step_info) == 3:
                        # Fallback for simple steps that don't provide timing info
                        sub_current, sub_total, sub_name = sub_step_info
                        self.stage2_sub_progress_value = float(sub_current) / sub_total if sub_total > 0 else 0.0
                        self.stage2_sub_progress_label = f"{sub_name} ({int(sub_current)}/{int(sub_total)})"
                        self.stage2_sub_time_elapsed_str, self.stage2_sub_processing_fps_str, self.stage2_sub_eta_str = "", "", ""
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
                    # Get the funscript object first
                    funscript_obj = results_dict.get("funscript")
                    
                    # For Stage 2 results, always update chapters with the new analysis results
                    # This is the primary output of Stage 2 analysis
                    #should_update_chapters = True

                    # Process the funscript object or fall back to raw actions
                    if funscript_obj:
                        # Use funscript object (preferred)
                        primary_actions = funscript_obj.primary_actions
                        secondary_actions = funscript_obj.secondary_actions
                        
                    #if should_update_chapters:
                        self.logger.info("Updating chapters with Stage 2 analysis results.")
                        fs_proc.video_chapters.clear()
                        
                        # Extract chapters from the funscript object instead of separate video_segments_data
                        if hasattr(funscript_obj, 'chapters') and funscript_obj.chapters:
                            fps = self.app.processor.video_info.get('fps', 30.0) if self.app.processor and self.app.processor.video_info else 30.0
                            # Clear existing chapters and add new ones from Stage 3 funscript
                            fs_proc.video_chapters.clear()
                            for chapter in funscript_obj.chapters:
                                start_frame = int((chapter.get('start', 0) / 1000.0) * fps)
                                end_frame = int((chapter.get('end', 0) / 1000.0) * fps)
                                
                                video_segment = VideoSegment(
                                    start_frame_id=start_frame,
                                    end_frame_id=end_frame,
                                    class_id=None,
                                    class_name=chapter.get('name', 'Unknown'),
                                    segment_type="SexAct",
                                    position_short_name=chapter.get('name', ''),
                                    position_long_name=chapter.get('description', chapter.get('name', 'Unknown')),
                                    source="stage3_funscript"
                                )
                                fs_proc.video_chapters.append(video_segment)
                            self.logger.info(f"Updated {len(funscript_obj.chapters)} chapters from Stage 2 funscript")
                    else:
                        # No funscript object available - initialize with empty actions
                        primary_actions = []
                        secondary_actions = []
                        self.app.logger.warning("No funscript object available from Stage 2 - using empty action lists")

                    # Get the application's current axis settings
                    axis_mode = self.app.tracking_axis_mode
                    target_timeline = self.app.single_axis_output_target

                    self.app.logger.info(f"Applying 2-Stage results with axis mode: {axis_mode} and target: {target_timeline}.")

                    # Only clear and update timelines if we have actions to write
                    if axis_mode == "both":
                        # Overwrite both timelines with the new results only if there are actions
                        if primary_actions:
                            fs_proc.clear_timeline_history_and_set_new_baseline(1, primary_actions, "Stage 2 (Primary)")
                            self.app.logger.info(f"Applied {len(primary_actions)} primary actions to Timeline 1")
                        else:
                            self.app.logger.warning("No primary actions from Stage 2 - Timeline 1 unchanged")
                        
                        if secondary_actions:
                            fs_proc.clear_timeline_history_and_set_new_baseline(2, secondary_actions, "Stage 2 (Secondary)")
                            self.app.logger.info(f"Applied {len(secondary_actions)} secondary actions to Timeline 2")
                        else:
                            self.app.logger.warning("No secondary actions from Stage 2 - Timeline 2 unchanged")

                    elif axis_mode == "vertical":
                        # Overwrite ONLY the target timeline if there are actions
                        if primary_actions:
                            if target_timeline == "primary":
                                self.app.logger.info("Writing to Timeline 1, Timeline 2 is untouched.")
                                fs_proc.clear_timeline_history_and_set_new_baseline(1, primary_actions, "Stage 2 (Vertical)")
                            else:  # Target is secondary
                                self.app.logger.info("Writing to Timeline 2, Timeline 1 is untouched.")
                                fs_proc.clear_timeline_history_and_set_new_baseline(2, primary_actions, "Stage 2 (Vertical)")
                        else:
                            self.app.logger.warning(f"No vertical actions from Stage 2 - Timeline {1 if target_timeline == 'primary' else 2} unchanged")

                    elif axis_mode == "horizontal":
                        # Overwrite ONLY the target timeline with the secondary (horizontal) actions if available
                        if secondary_actions:
                            if target_timeline == "primary":
                                self.app.logger.info("Writing horizontal data to Timeline 1, Timeline 2 is untouched.")
                                fs_proc.clear_timeline_history_and_set_new_baseline(1, secondary_actions, "Stage 2 (Horizontal)")
                            else:  # Target is secondary
                                self.app.logger.info("Writing horizontal data to Timeline 2, Timeline 1 is untouched.")
                                fs_proc.clear_timeline_history_and_set_new_baseline(2, secondary_actions, "Stage 2 (Horizontal)")
                        else:
                            self.app.logger.warning(f"No horizontal actions from Stage 2 - Timeline {1 if target_timeline == 'primary' else 2} unchanged")

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
                elif event_type == "stage3_results_success":
                    packaged_data = data1
                    if not isinstance(packaged_data, dict):
                        self.logger.warning(f"stage3_results_success received non-dict data: {type(packaged_data)}")
                        continue
                    results_dict = packaged_data.get("results_dict", {})
                    
                    # Validate results_dict is actually a dictionary
                    if not isinstance(results_dict, dict):
                        self.logger.error(f"Stage 3 results_dict is not a dictionary: {type(results_dict)} = {results_dict}")
                        continue
                    
                    # Extract funscript object from Stage 3 results
                    funscript_obj = results_dict.get("funscript")
                    if funscript_obj:
                        self.logger.info("Processing Stage 3 results with funscript object")
                        
                        # Extract actions from funscript
                        primary_actions = funscript_obj.primary_actions
                        secondary_actions = funscript_obj.secondary_actions
                        
                        # Update chapters from funscript if available
                        if hasattr(funscript_obj, 'chapters') and funscript_obj.chapters:
                            fps = self.app.processor.video_info.get('fps', 30.0) if self.app.processor and self.app.processor.video_info else 30.0
                            # Clear existing chapters and add new ones from Stage 3 funscript
                            fs_proc.video_chapters.clear()
                            for chapter in funscript_obj.chapters:
                                start_frame = int((chapter.get('start', 0) / 1000.0) * fps)
                                end_frame = int((chapter.get('end', 0) / 1000.0) * fps)
                                
                                video_segment = VideoSegment(
                                    start_frame_id=start_frame,
                                    end_frame_id=end_frame,
                                    class_id=None,
                                    class_name=chapter.get('name', 'Unknown'),
                                    segment_type="SexAct",
                                    position_short_name=chapter.get('name', ''),
                                    position_long_name=chapter.get('description', chapter.get('name', 'Unknown')),
                                    source="stage3_funscript"
                                )
                                fs_proc.video_chapters.append(video_segment)
                            self.logger.info(f"Updated {len(funscript_obj.chapters)} chapters from Stage 3 funscript")
                        
                        # Apply actions to timeline (Stage 3 typically writes to primary timeline)
                        axis_mode = self.app.tracking_axis_mode
                        target_timeline = self.app.single_axis_output_target
                        
                        self.app.logger.info(f"Applying Stage 3 results with axis mode: {axis_mode} and target: {target_timeline}")
                        
                        # Only clear and update timelines if we have actions to write
                        if axis_mode == "both":
                            # Write to both timelines only if there are actions
                            if primary_actions:
                                fs_proc.clear_timeline_history_and_set_new_baseline(1, primary_actions, "Stage 3 (Primary)")
                                self.logger.info(f"Applied {len(primary_actions)} primary actions to Timeline 1")
                            else:
                                self.logger.warning("No primary actions from Stage 3 - Timeline 1 unchanged")
                            
                            if secondary_actions:
                                fs_proc.clear_timeline_history_and_set_new_baseline(2, secondary_actions, "Stage 3 (Secondary)")
                                self.logger.info(f"Applied {len(secondary_actions)} secondary actions to Timeline 2")
                            else:
                                self.logger.info("No secondary actions from Stage 3 - Timeline 2 unchanged")
                                
                        elif axis_mode in ["vertical", "horizontal"]:
                            # Write to target timeline only if there are actions
                            actions_to_use = primary_actions  # Stage 3 typically produces primary actions
                            if actions_to_use:
                                if target_timeline == "primary":
                                    fs_proc.clear_timeline_history_and_set_new_baseline(1, actions_to_use, "Stage 3")
                                    self.logger.info(f"Applied {len(actions_to_use)} actions to Timeline 1")
                                else:  # secondary
                                    fs_proc.clear_timeline_history_and_set_new_baseline(2, actions_to_use, "Stage 3")
                                    self.logger.info(f"Applied {len(actions_to_use)} actions to Timeline 2")
                            else:
                                self.logger.warning(f"No actions from Stage 3 - Timeline {1 if target_timeline == 'primary' else 2} unchanged")
                        
                        self.stage3_status_text = "S3 Completed. Results Processed."
                        self.app.project_manager.project_dirty = True
                        self.logger.info(f"Applied {len(primary_actions)} Stage 3 actions to funscript processor")
                    else:
                        self.logger.warning("Stage 3 results missing funscript object - no actions applied")
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
                    status_override = payload.get("status", data2)
                    log_msg = payload.get("message", str(data1))

                    if status_override == "Completed":
                        if log_msg:
                            self.logger.info(log_msg, extra={'status_message': True})
                        # Delegate all finalization logic to the main app logic controller
                        self.app.on_offline_analysis_completed(payload)

                    elif status_override == "Aborted":
                        if self.current_analysis_stage == 1 or self.stage1_status_text.startswith(
                            "Running"): self.stage1_status_text = "S1 Aborted."
                        if self.current_analysis_stage == 2 or self.stage2_status_text.startswith(
                            "Running"): self.stage2_status_text = "S2 Aborted."
                        if self.current_analysis_stage == 3 or self.stage3_status_text.startswith(
                            "Running"): self.stage3_status_text = "S3 Aborted."
                        # Signal batch loop to continue on abort
                        if self.app.is_batch_processing_active and hasattr(self.app, 'save_and_reset_complete_event'):
                            self.logger.debug(f"Signaling batch loop to continue after handling '{status_override}' status.")
                            self.app.save_and_reset_complete_event.set()

                    elif status_override == "Failed":
                        if self.current_analysis_stage == 1 or self.stage1_status_text.startswith("Running"): self.stage1_status_text = "S1 Failed."
                        if self.current_analysis_stage == 2 or self.stage2_status_text.startswith("Running"): self.stage2_status_text = "S2 Failed."
                        if self.current_analysis_stage == 3 or self.stage3_status_text.startswith("Running"): self.stage3_status_text = "S3 Failed."
                        # Signal batch loop to continue on failure
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
    
    def _is_offline_tracker(self, tracker_name):
        """Check if tracker is any offline tracker."""
        tracker_ui = get_dynamic_tracker_ui()
        return tracker_ui.is_offline_tracker(tracker_name)

    def update_settings_from_app(self):
        prod_usr = self.app_settings.get("num_producers_stage1")
        cons_usr = self.app_settings.get("num_consumers_stage1")
        # Always save preprocessed video for optical flow recovery in Stage 2
        self.save_preprocessed_video = self.app_settings.get("save_preprocessed_video", True)

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
            "stage2_database_path": getattr(self.app, 's2_sqlite_db_path', None),
            "stage2_status_text": self.stage2_status_text,
            "stage3_status_text": self.stage3_status_text,
        }

    def _validate_preprocessed_artifacts(self, msgpack_path: str, video_path: str) -> bool:
        """
        Validates that preprocessed artifacts are complete and consistent.

        Args:
            msgpack_path: Path to the msgpack file
            video_path: Path to the preprocessed video file

        Returns:
            True if artifacts are valid and consistent, False otherwise
        """
        try:
            # Import validation functions from stage_1_cd
            from detection.cd.stage_1_cd import _validate_preprocessed_file_completeness, _validate_preprocessed_video_completeness

            if not self.app.processor or not self.app.processor.video_info:
                self.logger.warning("Cannot validate preprocessed artifacts: video info not available")
                return False

            expected_frames = self.app.processor.video_info.get('total_frames', 0)
            expected_fps = self.app.processor.video_info.get('fps', 30.0)

            if expected_frames <= 0:
                self.logger.warning("Cannot validate preprocessed artifacts: invalid frame count")
                return False

            # Validate msgpack file
            if not _validate_preprocessed_file_completeness(msgpack_path, expected_frames, self.logger):
                self.logger.warning(f"Preprocessed msgpack validation failed: {os.path.basename(msgpack_path)}")
                return False

            # Validate video file if it exists
            if os.path.exists(video_path):
                if not _validate_preprocessed_video_completeness(video_path, expected_frames, expected_fps, self.logger):
                    self.logger.warning(f"Preprocessed video validation failed: {os.path.basename(video_path)}")
                    return False

            self.logger.info("Preprocessed artifacts validation passed")
            return True

        except Exception as e:
            self.logger.error(f"Error validating preprocessed artifacts: {e}")
            return False

    def update_project_specific_settings(self, project_data: Dict):
        self.stage2_status_text = project_data.get("stage2_status_text", "Not run.")
        self.stage3_status_text = project_data.get("stage3_status_text", "Not run.")
        self.stage2_progress_value, self.stage2_progress_label = 0.0, ""
        self.stage2_main_progress_value, self.stage2_main_progress_label = 0.0, ""
        self.stage2_sub_progress_value, self.stage2_sub_progress_label = 0.0, ""
        self.stage3_current_segment_label, self.stage3_segment_progress_value = "", 0.0
        self.stage3_overall_progress_label, self.stage3_overall_progress_value = "", 0.0
        self.stage3_time_elapsed_str, self.stage3_processing_fps_str, self.stage3_eta_str = "00:00:00", "0 FPS", "N/A"

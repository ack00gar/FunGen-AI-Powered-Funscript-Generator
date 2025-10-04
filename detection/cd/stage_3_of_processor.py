import time
import logging
import cv2
import os
import numpy as np
from typing import Optional, List, Dict, Any, Tuple
from multiprocessing import Process, Queue, Event, Value
from queue import Empty

from funscript import DualAxisFunscript
from video import VideoProcessor
from detection.cd.data_structures import Segment, FrameObject
from config import constants



def stage3_worker_proc(
        worker_id: int,
        task_queue: Queue,
        result_queue: Queue,
        stop_event: Event,
        total_frames_processed_counter: Value,
        video_path: str,
        preprocessed_video_path: Optional[str],
        tracker_config: Dict[str, Any],
        common_app_config: Dict[str, Any],
        logger_config: Dict[str, Any],
        use_sqlite: bool = False
):
    """
    A worker process that pulls a chunk definition from the task queue,
    performs optical flow analysis on it, and puts the resulting
    funscript actions for its unique portion into the result queue.
    """
    worker_logger = logging.getLogger(f"S3_Worker-{worker_id}_{os.getpid()}")
    if not worker_logger.hasHandlers():
        log_level = logger_config.get('log_level', logging.INFO)
        worker_logger.setLevel(log_level)
        log_file = logger_config.get('log_file')
        handler = logging.FileHandler(log_file) if log_file else logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')
        handler.setFormatter(formatter)
        worker_logger.addHandler(handler)

    worker_logger.info(f"Worker {worker_id} started.")

    class MockFileManager:
        def __init__(self, path: Optional[str]):
            self.preprocessed_video_path = path

        def get_output_path_for_file(self, video_path: str, extension: str) -> str:
            """
            A mock method to satisfy the VideoProcessor's internal check.
            It returns the pre-determined preprocessed path or an empty string,
            preventing a TypeError with os.path.exists(None).
            """
            if extension == "_preprocessed.mp4":
                return self.preprocessed_video_path if self.preprocessed_video_path is not None else ""
            # Return an empty string for any other unexpected request.
            return ""


    class VPAppProxy:
        pass

    vp_app_proxy = VPAppProxy()
    vp_app_proxy.logger = worker_logger.getChild("VideoProcessor")
    vp_app_proxy.hardware_acceleration_method = common_app_config.get("hardware_acceleration_method", "none")
    vp_app_proxy.available_ffmpeg_hwaccels = common_app_config.get("available_ffmpeg_hwaccels", [])
    vp_app_proxy.file_manager = MockFileManager(preprocessed_video_path)

    # --- Use the preprocessed path if it exists and is valid for VideoProcessor initialization ---
    video_path_to_use = video_path  # Default to original
    video_type_for_vp = common_app_config.get('video_type', 'auto')

    if preprocessed_video_path and os.path.exists(preprocessed_video_path):
        # Validate preprocessed video before using it
        try:
            from detection.cd.stage_1_cd import _validate_preprocessed_video_completeness

            # Get frame count from original video for validation
            fps = common_app_config.get('video_fps', 30.0)

            # Try to get accurate frame count from original video
            try:
                import subprocess
                import json
                cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                       '-show_entries', 'stream=nb_frames,duration',
                       '-show_entries', 'format=duration',
                       '-of', 'json', video_path]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    stream_info = data.get('streams', [{}])[0]
                    format_info = data.get('format', {})
                    nb_frames_str = stream_info.get('nb_frames')
                    dur_str = stream_info.get('duration', format_info.get('duration', '0'))
                    duration = float(dur_str) if dur_str and dur_str != 'N/A' else 0.0
                    expected_frames = int(nb_frames_str) if nb_frames_str and nb_frames_str != 'N/A' else round(duration * fps)
                else:
                    expected_frames = 10000  # Fallback
            except Exception:
                expected_frames = 10000  # Fallback if ffprobe fails

            # Use a reasonable tolerance for Stage 3 (typically works with chunks)
            tolerance = max(100, expected_frames // 100)  # 1% tolerance, minimum 100 frames

            if _validate_preprocessed_video_completeness(preprocessed_video_path, expected_frames, fps, worker_logger, tolerance_frames=tolerance):
                video_path_to_use = preprocessed_video_path
                video_type_for_vp = 'flat'
                worker_logger.info(f"Worker {worker_id} using validated preprocessed video: {os.path.basename(video_path_to_use)} ({expected_frames} frames)")
            else:
                worker_logger.warning(f"Worker {worker_id} preprocessed video validation failed, using original: {os.path.basename(video_path)}")
        except Exception as e:
            worker_logger.error(f"Worker {worker_id} error validating preprocessed video: {e}")
    else:
        worker_logger.info(f"Worker {worker_id} will use original video source: {os.path.basename(video_path_to_use)} with type '{video_type_for_vp}'")


    video_processor = VideoProcessor(app_instance=vp_app_proxy,
                                     yolo_input_size=common_app_config.get('yolo_input_size', 640),
                                     video_type=video_type_for_vp) # --- Use the determined video type

    if not video_processor.open_video(video_path): # Pass original path for metadata, open_video handles the switch internally
        worker_logger.error(f"VideoProcessor could not open video: {video_path}")
        return

    determined_video_type = video_processor.determined_video_type

    try:
        # Lazy import to avoid circular dependency
        from tracker import ROITracker
        # ROITracker is now TrackerManager, use correct constructor
        roi_tracker_instance = ROITracker(
            app_logic_instance=None,
            tracker_model_path=common_app_config.get('yolo_det_model_path', '')
        )
        # Set tracker properties for oscillation detector
        roi_tracker_instance.y_offset = tracker_config.get('y_offset', constants.DEFAULT_LIVE_TRACKER_Y_OFFSET)
        roi_tracker_instance.x_offset = tracker_config.get('x_offset', constants.DEFAULT_LIVE_TRACKER_X_OFFSET)
        roi_tracker_instance.sensitivity = tracker_config.get('sensitivity', constants.DEFAULT_LIVE_TRACKER_SENSITIVITY)
        roi_tracker_instance.output_delay_frames = common_app_config.get('output_delay_frames', 0)
        roi_tracker_instance.current_video_fps_for_delay = common_app_config.get('video_fps', 30.0)
        # Configure oscillation detector mode based on settings
        od_mode = common_app_config.get('stage3_oscillation_detector_mode', 'current')
        if od_mode == "legacy":
            if not roi_tracker_instance.set_tracking_mode("oscillation_legacy"):
                worker_logger.error("Failed to set legacy oscillation detector mode")
                return  # Exit worker process
        else:
            # Try experimental modes first, fallback to legacy
            if not roi_tracker_instance.set_tracking_mode("oscillation_experimental_2"):
                if not roi_tracker_instance.set_tracking_mode("oscillation_experimental"):
                    if not roi_tracker_instance.set_tracking_mode("oscillation_legacy"):
                        worker_logger.error("Failed to set any oscillation detector mode")
                        return  # Exit worker process
        roi_tracker_instance.oscillation_grid_size = tracker_config.get('oscillation_grid_size', 20)
        roi_tracker_instance.oscillation_sensitivity = tracker_config.get('oscillation_sensitivity', 1.0)
        
        # Create a mock app instance for oscillation detector settings
        class MockApp:
            def __init__(self):
                self.tracking_axis_mode = common_app_config.get("tracking_axis_mode", "both")
                self.single_axis_output_target = common_app_config.get("single_axis_output_target", "primary")
                # Mock app_settings
                self.app_settings = {
                    "oscillation_detector_grid_size": tracker_config.get('oscillation_grid_size', 20),
                    "oscillation_detector_sensitivity": tracker_config.get('oscillation_sensitivity', 1.0),
                    "live_oscillation_dynamic_amp_enabled": True,
                    "live_oscillation_amp_window_ms": 4000
                }
                # Mock processor for VR detection
                class MockProcessor:
                    def __init__(self):
                        self.determined_video_type = determined_video_type
                        self.fps = common_app_config.get('video_fps', 30.0)
                    
                    def get_preprocessed_video_status(self):
                        return {
                            'preprocessed_exists': False,
                            'preprocessing_active': False,
                            'preprocessing_progress': 0.0
                        }
                
                self.processor = MockProcessor()
                self.discarded_tracking_classes = []
        
        mock_app = MockApp()
        roi_tracker_instance.app = mock_app
    except Exception as e:
        worker_logger.error(f"Failed to initialize OscillationDetector: {e}", exc_info=True)
        return

    # Initialize SQLite storage for worker if needed
    worker_sqlite_storage = None

    while not stop_event.is_set():
        try:
            task = task_queue.get(timeout=0.5)
            if task is None:
                break

            # Handle both SQLite and memory-based tasks
            if use_sqlite and len(task) == 6:
                # SQLite-based task
                segment_obj, chunk_start, chunk_end, output_start, output_end, sqlite_db_path = task

                # Initialize SQLite storage for this worker if not already done
                if worker_sqlite_storage is None:
                    try:
                        from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
                        worker_sqlite_storage = Stage2SQLiteStorage(sqlite_db_path, worker_logger)
                        worker_logger.info(f"Worker {worker_id} initialized SQLite storage")
                    except Exception as e:
                        worker_logger.error(f"Worker {worker_id} failed to initialize SQLite: {e}")
                        continue

                # Load chunk data from SQLite on-demand
                chunk_data_map = worker_sqlite_storage.get_frame_objects_range(chunk_start, chunk_end)
                worker_logger.info(
                    f"Processing SQLite chunk F{chunk_start}-{chunk_end} ({len(chunk_data_map)} frames) for Chapter '{getattr(segment_obj, 'position_short_name', None) or getattr(segment_obj, 'major_position', None) or getattr(segment_obj, 'position_long_name', 'Unknown')}'"
                )
            else:
                # Memory-based task (fallback)
                segment_obj, chunk_data_map, chunk_start, chunk_end, output_start, output_end = task
                worker_logger.info(
                    f"Processing memory chunk F{chunk_start}-{chunk_end} for Chapter '{getattr(segment_obj, 'position_short_name', None) or getattr(segment_obj, 'major_position', None) or getattr(segment_obj, 'position_long_name', 'Unknown')}'"
                )

            # Initialize funscript for oscillation detector
            roi_tracker_instance.funscript = DualAxisFunscript(logger=worker_logger)
            roi_tracker_instance.start_tracking()
            roi_tracker_instance.main_interaction_class = getattr(segment_obj, 'position_short_name', None) or getattr(segment_obj, 'major_position', None) or getattr(segment_obj, 'position_long_name', 'Unknown')

            frame_stream = video_processor.stream_frames_for_segment(
                start_frame_abs_idx=chunk_start,
                num_frames_to_read=(chunk_end - chunk_start + 1),
                stop_event=stop_event
            )

            for frame_id, frame_image in frame_stream:
                if stop_event.is_set(): break
                if frame_image is None: continue

                frame_time_ms = int(round((frame_id / common_app_config.get('video_fps', 30.0)) * 1000.0))

                # Process frame using oscillation detector (full-frame processing)
                processed_frame, action_log = roi_tracker_instance.process_frame_for_oscillation(frame_image, frame_time_ms, frame_id)
                
                # Only count frames in the output range for processing counter
                if output_start <= frame_id <= output_end:
                    with total_frames_processed_counter.get_lock():
                        total_frames_processed_counter.value += 1

                roi_tracker_instance.internal_frame_counter += 1
            
            # Extract actions from the oscillation detector's funscript, filtering to output range
            chunk_funscript = roi_tracker_instance.funscript
            output_start_ms = int(round((output_start / common_app_config.get('video_fps', 30.0)) * 1000.0))
            output_end_ms = int(round((output_end / common_app_config.get('video_fps', 30.0)) * 1000.0))
            
            # Filter actions to only include those in the output time range
            filtered_primary_actions = [action for action in chunk_funscript.primary_actions 
                                      if output_start_ms <= action['at'] <= output_end_ms]
            filtered_secondary_actions = [action for action in chunk_funscript.secondary_actions 
                                        if output_start_ms <= action['at'] <= output_end_ms]

            result_queue.put({
                "primary_actions": filtered_primary_actions,
                "secondary_actions": filtered_secondary_actions
            })

        except Empty:
            worker_logger.debug("Task queue is empty.")
            continue
        except Exception as e:
            worker_logger.error(f"Error processing a chunk: {e}", exc_info=True)
            if not stop_event.is_set():
                stop_event.set()
            break

    # Clean up resources before worker termination
    try:
        # Clean up ROI tracker resources (ModelPool removed)
        if 'roi_tracker_instance' in locals() and roi_tracker_instance is not None:
            roi_tracker_instance.prev_gray_main_roi = None
            roi_tracker_instance.prev_gray_user_roi_patch = None
            roi_tracker_instance.prev_gray_oscillation_area_patch = None
            worker_logger.debug(f"Worker {worker_id}: ROI tracker GPU resources cleared")
        
        # Clean up video processor
        video_processor.reset(close_video=True)
        
        # Force garbage collection to ensure cleanup
        import gc
        gc.collect()
        
        # Clear GPU cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                worker_logger.debug(f"Worker {worker_id}: GPU cache cleared")
        except ImportError:
            pass  # torch not available
            
    except Exception as cleanup_error:
        worker_logger.warning(f"Worker {worker_id}: Error during cleanup: {cleanup_error}")
    
    worker_logger.info(f"Worker {worker_id} finished with resource cleanup.")


def perform_stage3_analysis(
        video_path: str,
        preprocessed_video_path_arg: Optional[str],
        atr_segments_list: List[Segment],
        s2_frame_objects_map: Optional[Dict[int, FrameObject]] = None,
        tracker_config: Dict[str, Any] = None,
        common_app_config: Dict[str, Any] = None,
        progress_callback: callable = None,
        stop_event: Event = None,
        parent_logger: logging.Logger = None,
        num_workers: int = 4,
        sqlite_db_path: Optional[str] = None
) -> Dict[str, Any]:
    logger = parent_logger.getChild("S3_Orchestrator")
    logger.info(f"--- Starting Stage 3 Analysis with {num_workers} Workers (Overlapping Chunk Model) ---")
    s3_start_time = time.time()

    # Initialize SQLite storage if available
    sqlite_storage = None
    use_sqlite = False

    logger.info(f"Stage 3 received SQLite path: {sqlite_db_path}")
    logger.info(f"Frame objects map available: {s2_frame_objects_map is not None}")

    if sqlite_db_path is not None:
        logger.info(f"Checking if SQLite file exists: {sqlite_db_path}")
        if os.path.exists(sqlite_db_path):
            logger.info(f"SQLite file found, attempting to initialize storage")
            try:
                from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
                sqlite_storage = Stage2SQLiteStorage(sqlite_db_path, logger)
                logger.info(f"Using SQLite storage for Stage 3: {sqlite_db_path}")

                # Get frame count and range for memory optimization
                frame_count = sqlite_storage.get_frame_count()
                frame_range = sqlite_storage.get_frame_range()
                logger.info(f"SQLite database contains {frame_count} frames, range: {frame_range}")
                use_sqlite = True

            except Exception as e:
                logger.warning(f"Failed to initialize SQLite storage, falling back to memory: {e}")
                use_sqlite = False
                sqlite_storage = None
        else:
            logger.warning(f"SQLite database file not found: {sqlite_db_path}")
    else:
        logger.info("No SQLite database path provided")

    if not use_sqlite and not s2_frame_objects_map:
        logger.error("No data source available: neither SQLite nor in-memory frame objects map")
        # Create empty funscript with chapters for consistency
        empty_funscript = DualAxisFunscript()
        video_fps = common_app_config.get('video_fps', 30.0) if common_app_config else 30.0
        empty_funscript.set_chapters_from_segments(atr_segments_list, video_fps)
        return {"success": False, "funscript": empty_funscript, "error": "No data source available", "video_segments": [seg.to_dict() if hasattr(seg, 'to_dict') else seg.__dict__ for seg in atr_segments_list]}

    # Get chunking parameters from the configuration
    CHUNK_SIZE = common_app_config.get("s3_chunk_size", 1000)
    OVERLAP_SIZE = common_app_config.get("s3_overlap_size", 30)
    logger.info(f"Using chunk size: {CHUNK_SIZE}, overlap: {OVERLAP_SIZE}")

    # Handle both Segment and VideoSegment objects
    relevant_segments = []
    for seg in atr_segments_list:
        # Prefer short codes when available
        position_name = getattr(seg, 'position_short_name', None) or getattr(seg, 'major_position', None) or getattr(seg, 'position_long_name', '')
        if position_name not in ["NR", "C-Up", "Not Relevant", "Close Up"]:
            relevant_segments.append(seg)

    logger.info(f"Found {len(relevant_segments)} relevant segments to process in Stage 3.")
    logger.info(f"Details: {[seg.to_dict() for seg in relevant_segments]}")

    if not relevant_segments:
        logger.info("No relevant segments to process in Stage 3.")
        # Create empty funscript with chapters for consistency
        empty_funscript = DualAxisFunscript()
        video_fps = common_app_config.get('video_fps', 30.0) if common_app_config else 30.0
        empty_funscript.set_chapters_from_segments(atr_segments_list, video_fps)
        return {"success": True, "funscript": empty_funscript, "total_frames_processed": 0, "processing_method": "optical_flow", "video_segments": [seg.to_dict() if hasattr(seg, 'to_dict') else seg.__dict__ for seg in atr_segments_list]}

    total_frames_to_process = sum(seg.end_frame_id - seg.start_frame_id + 1 for seg in relevant_segments)

    logger_config = {'log_file': None, 'log_level': parent_logger.level}
    for handler in parent_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger_config['log_file'] = handler.baseFilename
            break

    task_queue = Queue()
    result_queue = Queue()
    total_frames_processed_counter = Value('i', 0)

    # Prepare chunked data for workers before they start
    logger.info("Preparing optimized data chunks for Stage 3 workers...")
    all_tasks = []

    if use_sqlite:
        # SQLite-based chunking - no need to preload data
        for segment in relevant_segments:
            step_size = CHUNK_SIZE - OVERLAP_SIZE
            for i, start_frame in enumerate(range(segment.start_frame_id, segment.end_frame_id + 1, step_size)):
                chunk_start = start_frame
                chunk_end = min(chunk_start + CHUNK_SIZE - 1, segment.end_frame_id)

                output_start = chunk_start if i == 0 else chunk_start + OVERLAP_SIZE
                output_end = chunk_end
                if output_start > output_end: continue

                # Task includes frame range, SQLite will load data on-demand
                task = (segment, chunk_start, chunk_end, output_start, output_end, sqlite_db_path)
                all_tasks.append(task)

        logger.info(f"Prepared {len(all_tasks)} SQLite-based data chunks.")
    else:
        # Memory-based chunking (fallback)
        for segment in relevant_segments:
            step_size = CHUNK_SIZE - OVERLAP_SIZE
            for i, start_frame in enumerate(range(segment.start_frame_id, segment.end_frame_id + 1, step_size)):
                chunk_start = start_frame
                chunk_end = min(chunk_start + CHUNK_SIZE - 1, segment.end_frame_id)

                # Create the small, targeted data map for this chunk
                chunk_data_map = {
                    frame_id: s2_frame_objects_map[frame_id]
                    for frame_id in range(chunk_start, chunk_end + 1)
                    if frame_id in s2_frame_objects_map
                }

                output_start = chunk_start if i == 0 else chunk_start + OVERLAP_SIZE
                output_end = chunk_end
                if output_start > output_end: continue

                # The new task payload includes the pre-filtered data
                task = (segment, chunk_data_map, chunk_start, chunk_end, output_start, output_end)
                all_tasks.append(task)

        # The large map is no longer needed in this scope and can be garbage collected
        if s2_frame_objects_map:
            del s2_frame_objects_map
        logger.info(f"Prepared {len(all_tasks)} memory-based data chunks. Main S2 data map released from memory.")

    for task in all_tasks:
        task_queue.put(task)

    for _ in range(num_workers):
        task_queue.put(None)

    processes: List[Process] = []
    for i in range(num_workers):
        p = Process(target=stage3_worker_proc,
                    args=(i, task_queue, result_queue, stop_event, total_frames_processed_counter, video_path,
                          preprocessed_video_path_arg, tracker_config, common_app_config,
                          logger_config, use_sqlite))
        processes.append(p)
        p.start()

    all_primary_actions, all_secondary_actions = [], []
    processed_task_count = 0
    total_tasks = len(all_tasks)

    # Pre-calculate frames per segment for progress reporting
    frames_per_segment = [(s.end_frame_id - s.start_frame_id + 1) for s in relevant_segments]
    cumulative_frames = np.cumsum(frames_per_segment)

    while any(p.is_alive() for p in processes) and not stop_event.is_set():
        # Check for completed results without blocking for a long time
        try:
            result = result_queue.get(timeout=0.05) # Use a very short timeout
            all_primary_actions.extend(result["primary_actions"])
            all_secondary_actions.extend(result["secondary_actions"])
            processed_task_count += 1
        except Empty:
            pass

        time_elapsed_s3 = time.time() - s3_start_time
        current_frames_done = total_frames_processed_counter.value
        # Avoid division by zero at the very start
        true_fps = current_frames_done / time_elapsed_s3 if time_elapsed_s3 > 0.1 else 0.0
        eta_s3 = (total_frames_to_process - current_frames_done) / true_fps if true_fps > 0 else float('inf')

        # --- Determine current chapter based on total frames processed ---
        current_chapter_idx_for_progress = 1
        chapter_name_for_progress = "Starting..."
        if relevant_segments:
            # Find the index of the first cumulative total that is >= current frames done
            chapter_index = np.searchsorted(cumulative_frames, current_frames_done, side='left')
            if chapter_index < len(relevant_segments):
                current_chapter_idx_for_progress = chapter_index + 1
                chapter_name_for_progress = getattr(relevant_segments[chapter_index], 'position_short_name', None) or getattr(relevant_segments[chapter_index], 'major_position', None) or getattr(relevant_segments[chapter_index], 'position_long_name', 'Unknown')
            else: # If done, lock to the last chapter
                current_chapter_idx_for_progress = len(relevant_segments)
                chapter_name_for_progress = getattr(relevant_segments[-1], 'position_short_name', None) or getattr(relevant_segments[-1], 'major_position', None) or getattr(relevant_segments[-1], 'position_long_name', 'Unknown')

        # --- Call progress_callback with both chapter and chunk info ---
        progress_callback(
            # Chapter Info
            current_chapter_idx=current_chapter_idx_for_progress,
            total_chapters=len(relevant_segments),
            chapter_name=chapter_name_for_progress,
            # Chunk Info (previously was current_segment_idx/total_segments)
            current_chunk_idx=processed_task_count,
            total_chunks=total_tasks,
            # Overall Progress Info
            total_frames_processed_overall=current_frames_done,
            total_frames_to_process_overall=total_frames_to_process,
            processing_fps=true_fps,
            time_elapsed=time_elapsed_s3,
            eta_seconds=eta_s3
        )

        # Sleep briefly to prevent this loop from consuming too much CPU
        time.sleep(0.1)

    if stop_event.is_set():
        logger.warning("Stop event detected. Terminating workers.")
        for p in processes:
            if p.is_alive(): p.terminate()

    # --- Cleanup phase to ensure all results are collected after workers finish ---
    while not result_queue.empty():
        try:
            result = result_queue.get_nowait()
            all_primary_actions.extend(result["primary_actions"])
            all_secondary_actions.extend(result["secondary_actions"])
            processed_task_count += 1
        except Empty:
            break

    for p in processes:
        p.join()

    all_primary_actions.sort(key=lambda x: x['at'])
    all_secondary_actions.sort(key=lambda x: x['at'])

    logger.info(
        f"Stage 3 complete. Aggregated {len(all_primary_actions)} primary actions from {processed_task_count} chunks.")

    # Clean up SQLite database file if we used it
    if use_sqlite and sqlite_storage and sqlite_db_path:
        try:
            sqlite_storage.close()
            sqlite_storage.cleanup_temp_files()

            # Respect user's database retention setting
            retain_database = common_app_config.get("retain_stage2_database", True)
            cleanup_db_file = not retain_database  # Only cleanup if retention is disabled
            
            if cleanup_db_file and os.path.exists(sqlite_db_path):
                os.remove(sqlite_db_path)
                logger.info(f"Cleaned up SQLite database file (retain_stage2_database=False): {sqlite_db_path}")
            elif os.path.exists(sqlite_db_path):
                logger.info(f"Preserved SQLite database file (retain_stage2_database=True): {sqlite_db_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up SQLite database: {e}")

    # Create funscript object
    funscript_obj = DualAxisFunscript(logger=logger)
    
    # Add actions to funscript object
    for action in all_primary_actions:
        funscript_obj.add_action(action['at'], action['pos'], None)
    for action in all_secondary_actions:
        if action['at'] in [a['at'] for a in funscript_obj.primary_actions]:
            # Update existing action with secondary position
            for existing_action in funscript_obj.secondary_actions:
                if existing_action['at'] == action['at']:
                    existing_action['pos'] = action['pos']
                    break
        else:
            funscript_obj.add_action(action['at'], None, action['pos'])
    
    # Set chapters from segments
    if atr_segments_list:
        video_fps = common_app_config.get('video_fps', 30.0) if common_app_config else 30.0
        funscript_obj.set_chapters_from_segments(atr_segments_list, video_fps)

    return {
        "success": True,
        "funscript": funscript_obj,
        "total_frames_processed": total_frames_to_process,
        "processing_method": "optical_flow",
        "video_segments": [seg.to_dict() if hasattr(seg, 'to_dict') else seg.__dict__ for seg in atr_segments_list]
    }

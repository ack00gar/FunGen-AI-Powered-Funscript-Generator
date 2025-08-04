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
from tracker import ROITracker
from detection.cd.stage_2_cd import ATRSegment, FrameObject
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
            if extension == "_preprocessed.mkv":
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
                       '-show_entries', 'stream=nb_frames', '-of', 'json', video_path]
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=5)
                if result.returncode == 0:
                    data = json.loads(result.stdout)
                    stream_info = data.get('streams', [{}])[0]
                    nb_frames_str = stream_info.get('nb_frames')
                    expected_frames = int(nb_frames_str) if nb_frames_str and nb_frames_str != 'N/A' else 10000
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
        roi_tracker_instance = ROITracker(
            app_logic_instance=None,
            tracker_model_path=common_app_config.get('yolo_det_model_path', ''),
            pose_model_path=common_app_config.get('yolo_pose_model_path', ''),
            load_models_on_init=False,
            confidence_threshold=tracker_config.get('confidence_threshold', 0.4),
            roi_padding=tracker_config.get('roi_padding', 20),
            roi_update_interval=tracker_config.get('roi_update_interval', constants.DEFAULT_ROI_UPDATE_INTERVAL),
            roi_smoothing_factor=tracker_config.get('roi_smoothing_factor', constants.DEFAULT_ROI_SMOOTHING_FACTOR),
            base_amplification_factor=tracker_config.get('base_amplification_factor', constants.DEFAULT_LIVE_TRACKER_BASE_AMPLIFICATION),
            dis_flow_preset=tracker_config.get('dis_flow_preset', "ULTRAFAST"),
            adaptive_flow_scale=tracker_config.get('adaptive_flow_scale', True),
            use_sparse_flow=tracker_config.get('use_sparse_flow', False),
            logger=worker_logger.getChild("ROITracker"),
            video_type_override=determined_video_type
        )
        # Manually set tracker properties from the config dictionaries
        roi_tracker_instance.y_offset = tracker_config.get('y_offset', constants.DEFAULT_LIVE_TRACKER_Y_OFFSET)
        roi_tracker_instance.x_offset = tracker_config.get('x_offset', constants.DEFAULT_LIVE_TRACKER_X_OFFSET)
        roi_tracker_instance.sensitivity = tracker_config.get('sensitivity', constants.DEFAULT_LIVE_TRACKER_SENSITIVITY)
        roi_tracker_instance.output_delay_frames = common_app_config.get('output_delay_frames', 0)
        roi_tracker_instance.current_video_fps_for_delay = common_app_config.get('video_fps', 30.0)
        roi_tracker_instance.tracking_mode = "YOLO_ROI"
    except Exception as e:
        worker_logger.error(f"Failed to initialize ROITracker: {e}", exc_info=True)
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
                    f"Processing SQLite chunk F{chunk_start}-{chunk_end} ({len(chunk_data_map)} frames) for Chapter '{segment_obj.major_position}'"
                )
            else:
                # Memory-based task (fallback)
                segment_obj, chunk_data_map, chunk_start, chunk_end, output_start, output_end = task
                worker_logger.info(
                    f"Processing memory chunk F{chunk_start}-{chunk_end} for Chapter '{segment_obj.major_position}'"
                )

            chunk_funscript = DualAxisFunscript(logger=worker_logger)
            roi_tracker_instance.start_tracking()
            roi_tracker_instance.main_interaction_class = segment_obj.major_position

            frame_stream = video_processor.stream_frames_for_segment(
                start_frame_abs_idx=chunk_start,
                num_frames_to_read=(chunk_end - chunk_start + 1),
                stop_event=stop_event
            )

            for frame_id, frame_image in frame_stream:
                if stop_event.is_set(): break
                if frame_image is None: continue

                processed_frame = roi_tracker_instance.preprocess_frame(frame_image)
                current_frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                frame_time_ms = int(round((frame_id / common_app_config.get('video_fps', 30.0)) * 1000.0))

                # Use the small, pre-filtered chunk_data_map
                frame_obj_s2 = chunk_data_map.get(frame_id, FrameObject(frame_id, common_app_config.get('yolo_input_size', 640)))

                # ROI Definition Logic (uses segment_obj for context)
                run_roi_definition = (roi_tracker_instance.roi is None) or (
                            roi_tracker_instance.roi_update_interval > 0 and roi_tracker_instance.internal_frame_counter % roi_tracker_instance.roi_update_interval == 0)
                if run_roi_definition:
                    candidate_roi_xywh: Optional[Tuple[int, int, int, int]] = None
                    if frame_obj_s2.atr_locked_penis_state.active and frame_obj_s2.atr_locked_penis_state.box:
                        lp_box_coords_xyxy = frame_obj_s2.atr_locked_penis_state.box
                        lp_x1, lp_y1, lp_x2, lp_y2 = lp_box_coords_xyxy
                        current_penis_box_for_roi_calc = (lp_x1, lp_y1, lp_x2 - lp_x1, lp_y2 - lp_y1)
                        interacting_objects_for_roi_calc = []
                        relevant_classes_for_pos = []

                        if segment_obj.major_position == "Cowgirl / Missionary":
                            relevant_classes_for_pos = ["pussy"]
                        elif segment_obj.major_position == "Rev. Cowgirl / Doggy":
                            relevant_classes_for_pos = ["butt"]
                        elif segment_obj.major_position == "Handjob / Blowjob":
                            relevant_classes_for_pos = ["face", "hand"]
                        elif segment_obj.major_position == "Blowjob":
                            relevant_classes_for_pos = ["face"]
                        elif segment_obj.major_position == "Handjob":
                            relevant_classes_for_pos = ["hand"]
                        elif segment_obj.major_position == "Boobjob":
                            relevant_classes_for_pos = ["breast", "hand"]
                        elif segment_obj.major_position == "Footjob":
                            relevant_classes_for_pos = ["foot"]

                        for contact_dict in frame_obj_s2.atr_detected_contact_boxes:
                            box_rec = contact_dict.get("box_rec")
                            if box_rec and contact_dict.get("class_name") in relevant_classes_for_pos:
                                interacting_objects_for_roi_calc.append({
                                    "box": (box_rec.x1, box_rec.y1, box_rec.width, box_rec.height),
                                    "class_name": box_rec.class_name
                                })

                        if current_penis_box_for_roi_calc[2] > 0 and current_penis_box_for_roi_calc[3] > 0:
                            candidate_roi_xywh = roi_tracker_instance._calculate_combined_roi(processed_frame.shape[:2],
                                                                                              current_penis_box_for_roi_calc,
                                                                                              interacting_objects_for_roi_calc)
                            if determined_video_type == 'VR' and candidate_roi_xywh:
                                penis_w = current_penis_box_for_roi_calc[2]
                                rx, ry, rw, rh = candidate_roi_xywh
                                new_rw = min(rw, penis_w * 2) if segment_obj.major_position not in ["Handjob", "Blowjob", "Handjob / Blowjob"] else penis_w
                                if new_rw > 0:
                                    original_center_x = rx + rw / 2
                                    new_rx = int(original_center_x - new_rw / 2)
                                    frame_width = processed_frame.shape[1]
                                    final_rw = int(min(new_rw, frame_width))
                                    final_rx = max(0, min(new_rx, frame_width - final_rw))
                                    candidate_roi_xywh = (final_rx, ry, final_rw, rh)

                    if candidate_roi_xywh:
                        roi_tracker_instance.roi = roi_tracker_instance._smooth_roi_transition(candidate_roi_xywh)

                # Optical Flow Processing
                primary_pos, secondary_pos = 50, 50
                if roi_tracker_instance.roi and roi_tracker_instance.roi[2] > 0 and roi_tracker_instance.roi[3] > 0:
                    rx, ry, rw, rh = roi_tracker_instance.roi
                    patch = current_frame_gray[ry:ry + rh, rx:rx + rw]
                    if patch.size > 0:
                        primary_pos, secondary_pos, _, _, features_out = roi_tracker_instance.process_main_roi_content(
                            processed_frame, patch, roi_tracker_instance.prev_gray_main_roi,
                            roi_tracker_instance.prev_features_main_roi)
                        roi_tracker_instance.prev_gray_main_roi = patch.copy()
                        roi_tracker_instance.prev_features_main_roi = features_out

                # Conditional Output for the unique part of the chunk
                if output_start <= frame_id <= output_end:
                    smoothing_window = roi_tracker_instance.flow_history_window_smooth
                    auto_delay = (smoothing_window - 1) / 2.0 if smoothing_window > 1 else 0.0
                    total_delay = roi_tracker_instance.output_delay_frames + auto_delay
                    delay_ms = (
                                           total_delay / roi_tracker_instance.current_video_fps_for_delay) * 1000.0 if roi_tracker_instance.current_video_fps_for_delay > 0 else 0
                    adj_time_ms = max(0, int(round(frame_time_ms - delay_ms)))

                    # (Axis mapping logic as in the original file)
                    primary_to_write, secondary_to_write = None, None
                    tracking_axis_mode = common_app_config.get("tracking_axis_mode", "both")
                    single_axis_target = common_app_config.get("single_axis_output_target", "primary")
                    if tracking_axis_mode == "both":
                        primary_to_write, secondary_to_write = primary_pos, secondary_pos
                    elif tracking_axis_mode == "vertical":
                        if single_axis_target == "primary":
                            primary_to_write = primary_pos
                        else:
                            secondary_to_write = primary_pos
                    elif tracking_axis_mode == "horizontal":
                        if single_axis_target == "primary":
                            primary_to_write = secondary_pos
                        else:
                            secondary_to_write = secondary_pos

                    chunk_funscript.add_action(adj_time_ms, primary_to_write, secondary_to_write, False)
                    with total_frames_processed_counter.get_lock():
                        total_frames_processed_counter.value += 1

                roi_tracker_instance.internal_frame_counter += 1

            result_queue.put({
                "primary_actions": chunk_funscript.primary_actions,
                "secondary_actions": chunk_funscript.secondary_actions
            })

        except Empty:
            worker_logger.info("Task queue is empty.")
            continue
        except Exception as e:
            worker_logger.error(f"Error processing a chunk: {e}", exc_info=True)
            if not stop_event.is_set():
                stop_event.set()
            break

    video_processor.reset(close_video=True)
    worker_logger.info(f"Worker {worker_id} finished.")


def perform_stage3_analysis(
        video_path: str,
        preprocessed_video_path_arg: Optional[str],
        atr_segments_list: List[ATRSegment],
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
        return {"primary_actions": [], "secondary_actions": [], "video_segments": [s.to_dict() for s in atr_segments_list]}

    # Get chunking parameters from the configuration
    CHUNK_SIZE = common_app_config.get("s3_chunk_size", 1000)
    OVERLAP_SIZE = common_app_config.get("s3_overlap_size", 30)
    logger.info(f"Using chunk size: {CHUNK_SIZE}, overlap: {OVERLAP_SIZE}")

    relevant_segments = [seg for seg in atr_segments_list if seg.major_position not in ["Not Relevant", "Close Up"]]

    logger.info(f"Found {len(relevant_segments)} relevant segments to process in Stage 3.")
    logger.info(f"Details: {[seg.to_dict() for seg in relevant_segments]}")

    if not relevant_segments:
        logger.info("No relevant segments to process in Stage 3.")
        return {"primary_actions": [], "secondary_actions": [],
                "video_segments": [s.to_dict() for s in atr_segments_list]}

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
                chapter_name_for_progress = relevant_segments[chapter_index].major_position
            else: # If done, lock to the last chapter
                current_chapter_idx_for_progress = len(relevant_segments)
                chapter_name_for_progress = relevant_segments[-1].major_position

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

            # Optionally keep the database file for debugging
            cleanup_db_file = common_app_config.get("cleanup_stage2_db", True)
            if cleanup_db_file and os.path.exists(sqlite_db_path):
                os.remove(sqlite_db_path)
                logger.info(f"Cleaned up SQLite database file: {sqlite_db_path}")
            elif os.path.exists(sqlite_db_path):
                logger.info(f"Preserved SQLite database file for debugging: {sqlite_db_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up SQLite database: {e}")

    return {
        "primary_actions": all_primary_actions,
        "secondary_actions": all_secondary_actions,
        "video_segments": [s.to_dict() for s in atr_segments_list]
    }

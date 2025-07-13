import time
import logging
import cv2
import os
from typing import Optional, List, Dict, Any, Tuple
from multiprocessing import Process, Queue, Event, Value
from queue import Empty

from funscript.dual_axis_funscript import DualAxisFunscript
from video.video_processor import VideoProcessor
from tracker.tracker import ROITracker
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
    s2_frame_objects_map: Dict[int, FrameObject],
    tracker_config: Dict[str, Any],
    common_app_config: Dict[str, Any],
    logger_config: Dict[str, Any]
):
    """
    A worker process that pulls a video segment from the task queue,
    performs optical flow analysis on it, and puts the resulting
    funscript actions into the result queue.
    """
    # --- Logger setup inside the worker process ---
    # This prevents the pickling error by creating a new logger instance
    # in each child process instead of passing the parent's logger.
    worker_logger = logging.getLogger(f"S3_Worker-{worker_id}_{os.getpid()}")
    if not worker_logger.hasHandlers():
        log_level = logger_config.get('log_level', logging.INFO)
        worker_logger.setLevel(log_level)
        log_file = logger_config.get('log_file')
        if log_file:
            handler = logging.FileHandler(log_file)
        else:
            handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')
        handler.setFormatter(formatter)
        worker_logger.addHandler(handler)

    worker_logger.info(f"Worker {worker_id} started.")

    # --- Initialize worker-specific instances of VideoProcessor and ROITracker ---
    # These cannot be shared between processes due to their internal state.
    class MockFileManager:
        def __init__(self, path):
            self.preprocessed_video_path = path

    class VPAppProxy:
        pass

    vp_app_proxy = VPAppProxy()
    vp_app_proxy.logger = worker_logger.getChild("VideoProcessor")
    vp_app_proxy.hardware_acceleration_method = common_app_config.get("hardware_acceleration_method", "none")
    vp_app_proxy.available_ffmpeg_hwaccels = common_app_config.get("available_ffmpeg_hwaccels", [])
    vp_app_proxy.file_manager = MockFileManager(preprocessed_video_path)

    video_processor = VideoProcessor(
        app_instance=vp_app_proxy,
        yolo_input_size=common_app_config.get('yolo_input_size', 640),
        video_type=common_app_config.get('video_type', 'auto'),
        vr_input_format=common_app_config.get('vr_input_format', 'he'),
        vr_fov=common_app_config.get('vr_fov', 190),
        vr_pitch=common_app_config.get('vr_pitch', 0)
    )

    if not video_processor.open_video(video_path):
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
        roi_tracker_instance.y_offset = tracker_config.get('y_offset', constants.DEFAULT_LIVE_TRACKER_Y_OFFSET)
        roi_tracker_instance.x_offset = tracker_config.get('x_offset', constants.DEFAULT_LIVE_TRACKER_X_OFFSET)
        roi_tracker_instance.sensitivity = tracker_config.get('sensitivity', constants.DEFAULT_LIVE_TRACKER_SENSITIVITY)
        roi_tracker_instance.output_delay_frames = common_app_config.get('output_delay_frames', 0)
        roi_tracker_instance.current_video_fps_for_delay = common_app_config.get('video_fps', 30.0)
        roi_tracker_instance.tracking_mode = "YOLO_ROI"
    except Exception as e:
        worker_logger.error(f"Failed to initialize ROITracker: {e}", exc_info=True)
        return

    while not stop_event.is_set():
        try:
            # Get a segment to process from the queue
            segment_obj = task_queue.get(timeout=0.5)
            if segment_obj is None: # Sentinel value
                break

            worker_logger.info(f"Processing segment: {segment_obj.major_position} (F{segment_obj.start_frame_id}-{segment_obj.end_frame_id})")

            # Create a fresh funscript for this segment's results
            segment_funscript = DualAxisFunscript(logger=worker_logger)

            # Reset tracker state for the new segment
            roi_tracker_instance.internal_frame_counter = 0
            roi_tracker_instance.prev_gray_main_roi = None
            roi_tracker_instance.prev_features_main_roi = None
            roi_tracker_instance.roi = None
            roi_tracker_instance.primary_flow_history_smooth.clear()
            roi_tracker_instance.secondary_flow_history_smooth.clear()
            roi_tracker_instance.main_interaction_class = segment_obj.major_position
            roi_tracker_instance.start_tracking()

            num_warmup_frames = common_app_config.get('num_warmup_frames_s3', 10)
            start_frame = max(0, segment_obj.start_frame_id - num_warmup_frames)
            end_frame = segment_obj.end_frame_id
            num_frames_to_read = end_frame - start_frame + 1

            if num_frames_to_read <= 0:
                continue

            frame_stream = video_processor.stream_frames_for_segment(
                start_frame_abs_idx=start_frame,
                num_frames_to_read=num_frames_to_read,
                stop_event=stop_event
            )

            for frame_id, frame_image in frame_stream:
                if stop_event.is_set():
                    break

                if frame_image is None:
                    continue

                processed_frame = roi_tracker_instance.preprocess_frame(frame_image)
                current_frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                frame_time_ms = int(round((frame_id / common_app_config.get('video_fps', 30.0)) * 1000.0))
                frame_obj_s2 = s2_frame_objects_map.get(frame_id)

                if not frame_obj_s2: # Create a minimal placeholder if missing
                    frame_obj_s2 = FrameObject(frame_id, common_app_config.get('yolo_input_size', 640))

                # --- ROI Definition Logic ---
                run_roi_definition = (roi_tracker_instance.roi is None) or \
                                     (roi_tracker_instance.roi_update_interval > 0 and \
                                      roi_tracker_instance.internal_frame_counter % roi_tracker_instance.roi_update_interval == 0)

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
                            candidate_roi_xywh = roi_tracker_instance._calculate_combined_roi(
                                processed_frame.shape[:2],
                                current_penis_box_for_roi_calc,
                                interacting_objects_for_roi_calc
                            )
                            if determined_video_type == 'VR' and candidate_roi_xywh:
                                penis_w = current_penis_box_for_roi_calc[2]
                                rx, ry, rw, rh = candidate_roi_xywh
                                new_rw = 0

                                if segment_obj.major_position in ["Handjob", "Blowjob", "Handjob / Blowjob"]:
                                    new_rw = penis_w
                                else:
                                    new_rw = min(rw, penis_w * 2)

                                if new_rw > 0:
                                    original_center_x = rx + rw / 2
                                    new_rx = int(original_center_x - new_rw / 2)
                                    frame_width = processed_frame.shape[1]
                                    final_rw = int(min(new_rw, frame_width))
                                    final_rx = max(0, min(new_rx, frame_width - final_rw))
                                    candidate_roi_xywh = (final_rx, ry, final_rw, rh)

                    if candidate_roi_xywh:
                        roi_tracker_instance.roi = roi_tracker_instance._smooth_roi_transition(
                            candidate_roi_xywh)

                # --- Optical Flow Processing ---
                primary_pos, secondary_pos = 50, 50
                if roi_tracker_instance.roi and roi_tracker_instance.roi[2] > 0 and roi_tracker_instance.roi[3] > 0:
                    rx, ry, rw, rh = roi_tracker_instance.roi
                    main_roi_patch_gray = current_frame_gray[ry:ry + rh, rx:rx + rw]
                    if main_roi_patch_gray.size > 0:
                        primary_pos, secondary_pos, _, _, _ = roi_tracker_instance.process_main_roi_content(
                            processed_frame, main_roi_patch_gray,
                            roi_tracker_instance.prev_gray_main_roi,
                            roi_tracker_instance.prev_features_main_roi
                        )
                        roi_tracker_instance.prev_gray_main_roi = main_roi_patch_gray.copy()

                # --- Funscript Writing (to the segment-specific funscript) ---
                if segment_obj.start_frame_id <= frame_id <= segment_obj.end_frame_id:
                    smoothing_window = roi_tracker_instance.flow_history_window_smooth
                    auto_delay = (smoothing_window - 1) / 2.0 if smoothing_window > 1 else 0.0
                    total_delay = roi_tracker_instance.output_delay_frames + auto_delay
                    delay_ms = (total_delay / roi_tracker_instance.current_video_fps_for_delay) * 1000.0
                    adj_time_ms = max(0, int(round(frame_time_ms - delay_ms)))

                    primary_to_write, secondary_to_write = None, None
                    tracking_axis_mode = common_app_config.get("tracking_axis_mode", "both")
                    single_axis_target = common_app_config.get("single_axis_output_target", "primary")
                    if tracking_axis_mode == "both":
                        primary_to_write, secondary_to_write = primary_pos, secondary_pos
                    elif tracking_axis_mode == "vertical":
                        if single_axis_target == "primary": primary_to_write = primary_pos
                        else: secondary_to_write = primary_pos
                    elif tracking_axis_mode == "horizontal":
                        if single_axis_target == "primary": primary_to_write = secondary_pos
                        else: secondary_to_write = secondary_pos

                    segment_funscript.add_action(adj_time_ms, primary_to_write, secondary_to_write, False)

                roi_tracker_instance.internal_frame_counter += 1

                with total_frames_processed_counter.get_lock():
                    total_frames_processed_counter.value += 1

            # --- Segment processing is complete, put results on the queue ---
            result_queue.put({
                "original_segment_idx": segment_obj.id,
                "primary_actions": segment_funscript.primary_actions,
                "secondary_actions": segment_funscript.secondary_actions
            })

        except Empty:
            worker_logger.info("Task queue is empty.")
            break
        except Exception as e:
            worker_logger.error(f"Error processing a segment: {e}", exc_info=True)
            if not stop_event.is_set():
                stop_event.set()
            break

    video_processor.reset(close_video=True)
    worker_logger.info(f"Worker {worker_id} finished.")


def perform_stage3_analysis(
    video_path: str,
    preprocessed_video_path_arg: Optional[str],
    atr_segments_list: List[ATRSegment],
    s2_frame_objects_map: Dict[int, FrameObject],
    tracker_config: Dict[str, Any],
    common_app_config: Dict[str, Any],
    progress_callback: callable,
    stop_event: Event,
    parent_logger: logging.Logger,
    num_workers: int = 4
) -> Dict[str, Any]:
    """
    Main entry point for Stage 3. Now orchestrates parallel processing of segments.
    """
    logger = parent_logger.getChild("S3_Orchestrator")
    logger.info(f"--- Starting Stage 3 Analysis with {num_workers} Workers ---")
    s3_start_time = time.time()

    relevant_segments = [seg for seg in atr_segments_list if seg.major_position not in ["Not Relevant", "Close Up"]]
    if not relevant_segments:
        logger.info("No relevant segments to process in Stage 3.")
        return {"primary_actions": [], "secondary_actions": [], "video_segments": [s.to_dict() for s in atr_segments_list]}

    # Estimate total frames for a more accurate ETA
    total_frames_to_process = sum(seg.end_frame_id - seg.start_frame_id + 1 for seg in relevant_segments)

    # --- Create picklable logger configuration ---
    logger_config = {
        'log_file': None,
        'log_level': parent_logger.level
    }
    for handler in parent_logger.handlers:
        if isinstance(handler, logging.FileHandler):
            logger_config['log_file'] = handler.baseFilename
            break

    task_queue = Queue()
    result_queue = Queue()
    # Create the shared counter for true FPS
    total_frames_processed_counter = Value('i', 0)

    for seg in relevant_segments:
        task_queue.put(seg)

    for _ in range(num_workers):
        task_queue.put(None)

    processes: List[Process] = []
    for i in range(num_workers):
        p = Process(
            target=stage3_worker_proc,
            args=(i, task_queue, result_queue, stop_event, total_frames_processed_counter,
                  video_path, preprocessed_video_path_arg,
                  s2_frame_objects_map, tracker_config, common_app_config, logger_config)
        )
        processes.append(p)
        p.start()

    all_primary_actions = []
    all_secondary_actions = []
    processed_segment_count = 0
    total_segments = len(relevant_segments)

    while processed_segment_count < total_segments and not stop_event.is_set():
        try:
            result = result_queue.get(timeout=1.0)
            all_primary_actions.extend(result["primary_actions"])
            all_secondary_actions.extend(result["secondary_actions"])
            processed_segment_count += 1

            time_elapsed_s3 = time.time() - s3_start_time
            current_frames_done = total_frames_processed_counter.value
            true_fps = current_frames_done / time_elapsed_s3 if time_elapsed_s3 > 0 else 0
            eta_s3 = (total_frames_to_process - current_frames_done) / true_fps if true_fps > 0 else 0

            progress_callback(
                current_segment_idx=processed_segment_count,
                total_segments=total_segments,
                current_segment_name=f"Aggregating segment {processed_segment_count}",
                frame_in_segment=current_frames_done,
                total_frames_in_segment=total_frames_to_process,
                total_frames_processed_overall=current_frames_done,
                total_frames_to_process_overall=total_frames_to_process,
                processing_fps=true_fps,
                time_elapsed=time_elapsed_s3,
                eta_seconds=eta_s3,
                original_segment_idx=result.get("original_segment_idx")
            )
        except Empty:
            if any(p.is_alive() for p in processes):
                continue
            else:
                logger.warning("Result queue is empty and all workers have exited. Some results may be missing.")
                break

    if stop_event.is_set():
        logger.warning("Stop event detected. Terminating workers.")
        for p in processes:
            if p.is_alive():
                p.terminate()
                p.join(timeout=1)

    for p in processes:
        p.join()

    all_primary_actions.sort(key=lambda x: x['at'])
    all_secondary_actions.sort(key=lambda x: x['at'])

    logger.info(f"Stage 3 complete. Aggregated {len(all_primary_actions)} primary actions from {processed_segment_count} segments.")

    return {
        "primary_actions": all_primary_actions,
        "secondary_actions": all_secondary_actions,
        "video_segments": [s.to_dict() for s in atr_segments_list]
    }

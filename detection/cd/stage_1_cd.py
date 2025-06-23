import numpy as np
import msgpack
import time
from multiprocessing import Process, Queue, Event, Value, freeze_support
from threading import Thread as PyThread
from queue import Empty, Full
from ultralytics import YOLO
import os
import logging
from typing import Optional, Tuple, List
import subprocess
from queue import Queue as StdLibQueue

from video.video_processor import VideoProcessor
from config import constants


class Stage1QueueMonitor:
    def __init__(self):
        self.frame_queue_puts = Value('i', 0)
        self.frame_queue_gets = Value('i', 0)
        self.result_queue_puts = Value('i', 0)
        self.result_queue_gets = Value('i', 0)

    def frame_queue_put(self, queue, item, block=True, timeout=None):
        with self.frame_queue_puts.get_lock():
            self.frame_queue_puts.value += 1
            queue.put(item, block=block, timeout=timeout)

    def frame_queue_get(self, queue, block=True, timeout=None):
        with self.frame_queue_gets.get_lock():
            item = queue.get(block=block, timeout=timeout)
            self.frame_queue_gets.value += 1
            return item

    def result_queue_put(self, queue, item):
        with self.result_queue_puts.get_lock():
            self.result_queue_puts.value += 1
            queue.put(item)

    def result_queue_get(self, queue, block=True, timeout=None):
        with self.result_queue_gets.get_lock():
            item = queue.get(block=block, timeout=timeout)
            self.result_queue_gets.value += 1
            return item

    def get_frame_queue_size(self, queue: Queue):
        """Returns the approximate size of the queue."""
        try:
            return queue.qsize()
        except NotImplementedError: # Some platforms might not implement qsize()
            # Fallback to the original, less accurate method if qsize() is not available
            with self.frame_queue_puts.get_lock(), self.frame_queue_gets.get_lock():
                return self.frame_queue_puts.value - self.frame_queue_gets.value


    def get_result_queue_size(self):
        with self.result_queue_puts.get_lock(), self.result_queue_gets.get_lock():
            return self.result_queue_puts.value - self.result_queue_gets.value


def video_processor_producer_proc(
        producer_idx: int,
        video_path_producer: str,
        yolo_input_size_producer: int,
        video_type_setting_producer: str,
        vr_input_format_setting_producer: str,
        vr_fov_setting_producer: int,
        vr_pitch_setting_producer: int,
        start_frame_abs_num: int,
        num_frames_in_segment: int,
        frame_queue: Queue,
        queue_monitor_local: Stage1QueueMonitor,
        stop_event_local: Event,
        hwaccel_method_producer: Optional[str],
        hwaccel_avail_list_producer: Optional[List[str]],
        logger_config_for_vp_in_producer: Optional[dict] = None
):
    frames_put_to_queue_this_producer = 0
    vp_instance = None
    producer_logger = None

    try:
        # Create a proxy object for the VideoProcessor instance
        class VPAppProxy:
            pass
        vp_app_proxy = VPAppProxy()
        vp_app_proxy.hardware_acceleration_method = hwaccel_method_producer
        vp_app_proxy.available_ffmpeg_hwaccels = hwaccel_avail_list_producer if hwaccel_avail_list_producer is not None else []

        # Instantiate the VideoProcessor using the proxy object.
        vp_instance = VideoProcessor(
            app_instance=vp_app_proxy,
            tracker=None,
            yolo_input_size=yolo_input_size_producer,
            video_type=video_type_setting_producer,
            vr_input_format=vr_input_format_setting_producer,
            vr_fov=vr_fov_setting_producer,
            vr_pitch=vr_pitch_setting_producer,
            fallback_logger_config=logger_config_for_vp_in_producer
        )
        producer_logger = vp_instance.logger
        if not producer_logger:
            producer_logger = logging.getLogger(f"S1_VP_Producer_{producer_idx}_{os.getpid()}_Fallback")
            if not producer_logger.hasHandlers():
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')
                handler.setFormatter(formatter)
                producer_logger.addHandler(handler)
                producer_logger.setLevel(logging.INFO)
                producer_logger.warning("Used fallback logger for VP Producer.")

        if not vp_instance.open_video(video_path_producer):
            producer_logger.critical(
                f"[S1 VP Producer-{producer_idx}] VideoProcessor could not open video '{video_path_producer}'.")
            return

        producer_logger.info(
            f"[S1 VP Producer-{producer_idx}] Streaming segment: Video='{os.path.basename(vp_instance.video_path)}', StartFrameAbs={start_frame_abs_num}, NumFrames={num_frames_in_segment}, YOLOSize={vp_instance.yolo_input_size}")

        for frame_id, frame in vp_instance.stream_frames_for_segment(start_frame_abs_num, num_frames_in_segment, stop_event=stop_event_local):
            if stop_event_local.is_set():
                producer_logger.info(
                    f"[S1 VP Producer-{producer_idx}] Stop event detected. Processed {frames_put_to_queue_this_producer} frames.")
                break

            try:
                # Attempt to put the frame on the queue with a 0.5 second timeout
                queue_monitor_local.frame_queue_put(frame_queue, (frame_id, np.copy(frame)), block=True, timeout=0.5)
            except Full:
                # If the queue is full, check the stop event and continue the loop to check again.
                if stop_event_local.is_set():
                    producer_logger.info(f"[S1 VP Producer-{producer_idx}] Stop event detected while frame queue was full. Frame {frame_id} not added.")
                    break
                continue # Go to the next iteration of the main for loop to re-check stop event



            if stop_event_local.is_set(): # Check if stop was set during the put_success loop
                producer_logger.info(
                    f"[S1 VP Producer-{producer_idx}] Stop event detected after attempting to queue frame {frame_id}. Loop terminating.")
                break
            frames_put_to_queue_this_producer += 1

        if not stop_event_local.is_set() and frames_put_to_queue_this_producer < num_frames_in_segment:
            producer_logger.warning(
                f"[S1 VP Producer-{producer_idx}] Streamed {frames_put_to_queue_this_producer} frames, but expected {num_frames_in_segment}. Video might be shorter or stream ended early.")

        producer_logger.info(
            f"[S1 VP Producer-{producer_idx}] Segment streaming loop ended. Put {frames_put_to_queue_this_producer} frames to queue (Target: {num_frames_in_segment}). Stop event: {stop_event_local.is_set()}")

    except Exception as e:
        # Use producer_logger if available, otherwise print as a last resort
        effective_logger = producer_logger if producer_logger else logging.getLogger(f"S1_VP_Producer_{producer_idx}_{os.getpid()}_ExceptionFallback")
        if not effective_logger.hasHandlers() and not producer_logger : # Configure fallback if it's the exception fallback
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')
            handler.setFormatter(formatter)
            effective_logger.addHandler(handler)
            effective_logger.setLevel(logging.INFO)
        effective_logger.critical(f"[S1 VP Producer-{producer_idx}] Error in producer {producer_idx}: {e}", exc_info=True)
        if not stop_event_local.is_set() : stop_event_local.set() # Signal main process about the error
    finally:
        effective_logger_final = producer_logger if producer_logger else logging.getLogger(f"S1_VP_Producer_{producer_idx}_{os.getpid()}_FinallyFallback")
        if not effective_logger_final.hasHandlers() and not producer_logger: # Configure fallback if it's the finally fallback
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')
            handler.setFormatter(formatter)
            effective_logger_final.addHandler(handler)
            effective_logger_final.setLevel(logging.INFO)

        # Ensure FFmpeg process managed by VideoProcessor is cleaned up
        if vp_instance and hasattr(vp_instance, 'ffmpeg_process') and vp_instance.ffmpeg_process and vp_instance.ffmpeg_process.poll() is None:
            effective_logger_final.info(
                f"[S1 VP Producer-{producer_idx}] Ensuring FFmpeg process from vp_instance is stopped in producer cleanup (PID: {vp_instance.ffmpeg_process.pid}).")
            if vp_instance.ffmpeg_process.stdout: vp_instance.ffmpeg_process.stdout.close()
            if vp_instance.ffmpeg_process.stderr: vp_instance.ffmpeg_process.stderr.close()
            vp_instance.ffmpeg_process.terminate() # Send SIGTERM
            try:
                vp_instance.ffmpeg_process.wait(timeout=1.0) # Wait for graceful termination
                if vp_instance.ffmpeg_process.poll() is None: # Check if still running
                    effective_logger_final.warning(f"[S1 VP Producer-{producer_idx}] FFmpeg process did not terminate after 1s, killing.")
                    vp_instance.ffmpeg_process.kill() # Send SIGKILL
                    vp_instance.ffmpeg_process.wait(timeout=0.5) # Wait for kill
            except subprocess.TimeoutExpired: # Should be part of Python's subprocess module
                 effective_logger_final.warning(f"[S1 VP Producer-{producer_idx}] FFmpeg process termination timed out. Killing if still alive.")
                 if vp_instance.ffmpeg_process.poll() is None:
                     vp_instance.ffmpeg_process.kill()
                     vp_instance.ffmpeg_process.wait(timeout=0.5) # Wait for kill
            except Exception as e_ffmpeg_cleanup:
                 effective_logger_final.error(f"[S1 VP Producer-{producer_idx}] Exception during FFmpeg cleanup: {e_ffmpeg_cleanup}")
                 if vp_instance.ffmpeg_process.poll() is None:
                     vp_instance.ffmpeg_process.kill() # Final attempt

        frames_count_final = frames_put_to_queue_this_producer if 'frames_put_to_queue_this_producer' in locals() else 'unknown'
        effective_logger_final.info(
            f"[S1 VP Producer-{producer_idx}] Fully Exited. Final count of frames put to queue: {frames_count_final}. Stop event: {stop_event_local.is_set()}")
        # Ensure VideoProcessor's own reset/cleanup if it has one that's relevant
        #if vp_instance and hasattr(vp_instance, 'stop_processing'):
        #    vp_instance.stop_processing() # This should handle its internal ffmpeg if any was for main loop, not segment


def consumer_proc(frame_queue, result_queue, consumer_idx, yolo_det_model_path, yolo_pose_model_path,
                  confidence_threshold, yolo_input_size_consumer, queue_monitor_local, stop_event_local,
                  logger_config_for_consumer: Optional[dict] = None, video_fps: float = 30.0):
    # --- Logger setup ---
    consumer_logger = logging.getLogger(f"S1_Consumer_{consumer_idx}_{os.getpid()}")

    det_model, pose_model = None, None
    try:
        # Load BOTH models in the same worker
        consumer_logger.info(f"[S1 Consumer-{consumer_idx}] Loading models...")
        det_model = YOLO(yolo_det_model_path, task='detect')

        # Force CPU for pose model on Apple MPS to avoid known bugs ?
        pose_device = constants.DEVICE
        pose_model = YOLO(yolo_pose_model_path, task='pose')
        consumer_logger.info(
            f"[S1 Consumer-{consumer_idx}] Models loaded. Detection on '{constants.DEVICE}', Pose on '{pose_device}'.")

        while not stop_event_local.is_set():
            try:
                item = queue_monitor_local.frame_queue_get(frame_queue, block=True, timeout=0.5)
                if item is None:
                    consumer_logger.info(f"[S1 Consumer-{consumer_idx}] Received sentinel. Exiting loop.")
                    break
                frame_id, frame = item

                # --- Step 1: Perform Detection (on every frame) ---
                det_results = det_model(frame, device=constants.DEVICE, verbose=False, imgsz=yolo_input_size_consumer,
                                        conf=confidence_threshold)
                detections = []
                for r in det_results:
                    if r.boxes:
                        for i in range(len(r.boxes)):
                            box_data = r.boxes[i]
                            detections.append({
                                'bbox': box_data.xyxy[0].tolist(),
                                'confidence': float(box_data.conf[0]),
                                'class': int(box_data.cls[0]),
                                'name': det_model.names[int(box_data.cls[0])]
                            })

                # --- Step 2: Conditionally perform Pose Estimation ---
                # Run only on the first frame of every second (approximately)
                poses = []  # Default to empty
                if frame_id % int(video_fps) == 0:
                    pose_results = pose_model(frame, ...)
                    for r in pose_results:
                        if r.keypoints and r.boxes:
                            for i in range(len(r.boxes)):
                                poses.append({
                                    'bbox': r.boxes.xyxy[i].tolist(),
                                    'keypoints': r.keypoints.data[i].tolist()
                                })

                # --- Step 3: Package results ---
                # The 'poses' list will either have data or be empty.
                result_payload = {
                    "detections": detections,
                    "poses": poses
                }
                queue_monitor_local.result_queue_put(result_queue, (frame_id, result_payload))

            except Empty:
                continue
            except Exception as e:
                consumer_logger.error(f"[S1 Consumer-{consumer_idx}] Error processing frame: {e}", exc_info=True)

    except Exception as e:
        consumer_logger.critical(f"[S1 Consumer-{consumer_idx}] Critical setup error: {e}", exc_info=True)
        stop_event_local.set()
    finally:
        del det_model, pose_model
        consumer_logger.info(f"[S1 Consumer-{consumer_idx}] Exiting.")


def logger_proc(frame_processing_queue, result_queue, output_file_local, expected_frames,
                progress_callback_local, queue_monitor_local, stop_event_local,
                s1_start_time_param, parent_logger: logging.Logger,
                gui_event_queue_arg: Optional[StdLibQueue] = None):
    results_dict = {}
    written_count = 0
    last_progress_update_time = time.time()
    first_result_received_time = None
    parent_logger.info(f"[S1 Logger] Expecting {expected_frames} frames. Writing to {output_file_local}")

    if progress_callback_local:
        progress_callback_local(0, expected_frames, "Logger starting...", 0.0, 0.0, -1)

    while (written_count < expected_frames if expected_frames > 0 else True) and not stop_event_local.is_set():
        try:
            item = queue_monitor_local.result_queue_get(result_queue, block=True, timeout=0.5)

            # Check for the sentinel value (None) to signal the end of results.
            if item is None or item[0] is None:
                parent_logger.info("[S1 Logger] Received end-of-stream sentinel. Finalizing results.")
                break

            # Simple get from the single result queue
            frame_id, payload = item
            if frame_id not in results_dict:
                results_dict[frame_id] = payload
                written_count += 1
                if first_result_received_time is None:
                    first_result_received_time = time.time()

            current_time = time.time()
            if progress_callback_local and (
                    current_time - last_progress_update_time > 0.2 or written_count == expected_frames):
                # Send queue updates to GUI
                if gui_event_queue_arg:
                    try:
                        frame_q_size = queue_monitor_local.get_frame_queue_size(frame_processing_queue)
                        gui_event_queue_arg.put(("stage1_queue_update",
                                                 {"frame_q_size": frame_q_size,
                                                  "result_q_size": queue_monitor_local.get_result_queue_size(),
                                                  "pose_q_size": 0}, None))

                    except Exception:
                        pass

                # Calculate and send main progress
                time_elapsed = 0.0
                if first_result_received_time is not None:
                    time_elapsed = current_time - first_result_received_time

                # Guard to prevent division by zero
                if time_elapsed > 0:
                    fps = written_count / time_elapsed
                else:
                    fps = 0.0  # Default to 0 FPS if no time has passed

                eta = (expected_frames - written_count) / fps if fps > 0 else 0
                progress_callback_local(written_count, expected_frames, "Detecting objects & poses...", time_elapsed,
                                        fps, eta)
                last_progress_update_time = current_time
        except Empty:
            continue
        except Exception as e:
            parent_logger.error(f"[S1 Logger] Error processing result: {e}", exc_info=True)

    parent_logger.info(f"[S1 Logger] Result gathering loop ended. Written count: {written_count}.")

    if stop_event_local.is_set():
        parent_logger.warning(f"[S1 Logger] Abort signal received. Skipping save of partial file: {output_file_local}")
        return

    # Save the results
    ordered_results = [results_dict.get(i, {"detections": [], "poses": []}) for i in range(expected_frames)]

    # --- POSE MEMORIZATION LOGIC ---
    parent_logger.info("[S1 Logger] Assembling final results and filling pose gaps...")
    ordered_results = [results_dict.get(i) for i in range(expected_frames)]

    last_known_poses = []
    for i in range(expected_frames):
        # The result for the current frame might be None if it was missed
        frame_result = ordered_results[i]
        if frame_result is None:
            # If a frame is missing entirely, create a default structure
            ordered_results[i] = {"detections": [], "poses": last_known_poses}
            continue

        # Check if this frame has newly calculated poses
        if frame_result.get("poses"):
            # If yes, update our cache
            last_known_poses = frame_result["poses"]
        else:
            # If no, fill it with the last known poses from the cache
            frame_result["poses"] = last_known_poses

    try:
        with open(output_file_local, 'wb') as f:
            f.write(msgpack.packb(ordered_results, use_bin_type=True))
        parent_logger.info(f"Save complete. Wrote {len(ordered_results)} entries to {output_file_local}.")
    except Exception as e:
        parent_logger.error(f"Error writing output file '{output_file_local}': {e}", exc_info=True)


def perform_yolo_analysis(
        video_path_arg: str,
        yolo_model_path_arg: str,
        yolo_pose_model_path_arg: str,
        confidence_threshold: float,
        progress_callback: callable,
        stop_event_external: Event,
        num_producers_arg: int,
        num_consumers_arg: int,
        video_type_arg: str = 'auto',
        vr_input_format_arg: str = 'he',
        vr_fov_arg: int = 190,
        vr_pitch_arg: int = 0,
        yolo_input_size_arg: int = 640,
        app_logger_config_arg: Optional[dict] = None,
        gui_event_queue_arg: Optional[StdLibQueue] = None,
        hwaccel_method_arg: Optional[str] = 'auto',
        hwaccel_avail_list_arg: Optional[List[str]] = None,
        frame_range_arg: Optional[Tuple[Optional[int], Optional[int]]] = None,
        output_filename_override: Optional[str] = None
):
    process_logger = None
    fallback_config_for_subprocesses = None

    if output_filename_override:
        result_file_local = output_filename_override
        os.makedirs(os.path.dirname(result_file_local), exist_ok=True)
    else:
        result_file_local = os.path.splitext(video_path_arg)[0] + '.msgpack'

    if app_logger_config_arg and app_logger_config_arg.get('main_logger'):
        process_logger = app_logger_config_arg['main_logger']
    else:
        process_logger = logging.getLogger("S1_Lib_Orchestrator")
        if not process_logger.hasHandlers():
            handler = logging.StreamHandler()
            log_level_orch = logging.INFO
            if app_logger_config_arg and app_logger_config_arg.get('log_level') is not None:
                log_level_orch = app_logger_config_arg['log_level']
            if app_logger_config_arg and app_logger_config_arg.get('log_file'):
                try:
                    handler = logging.FileHandler(app_logger_config_arg['log_file'])
                except Exception:
                    pass
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s')
            handler.setFormatter(formatter)
            process_logger.addHandler(handler)
            process_logger.setLevel(log_level_orch)

    if app_logger_config_arg and app_logger_config_arg.get('log_file') and app_logger_config_arg.get(
            'log_level') is not None:
        fallback_config_for_subprocesses = {
            'log_file': app_logger_config_arg['log_file'],
            'log_level': app_logger_config_arg['log_level']
        }
    elif process_logger.handlers:
        for h in process_logger.handlers:
            if isinstance(h, logging.FileHandler):
                fallback_config_for_subprocesses = {
                    'log_file': h.baseFilename,
                    'log_level': process_logger.level
                }
                break
    if not fallback_config_for_subprocesses:
        fallback_config_for_subprocesses = {'log_file': None, 'log_level': logging.INFO}

    process_logger.info(f"Stage 1 YOLO Analysis started with orchestrator logger: {process_logger.name}")

    if not os.path.exists(yolo_pose_model_path_arg):
        msg = "Error: YOLO Pose model not found"
        process_logger.error(msg)
        if progress_callback: progress_callback(0, 0, msg, 0, 0, 0)
        return None

    s1_start_time = time.time()
    stop_event_internal = Event()

    def monitor_external_stop():
        stop_event_external.wait()
        if not stop_event_internal.is_set():
            process_logger.info("[S1 Lib Monitor] External stop received. Relaying to internal stop event.")
            stop_event_internal.set()

    stop_monitor_thread = PyThread(target=monitor_external_stop, daemon=True)
    stop_monitor_thread.start()

    queue_monitor = Stage1QueueMonitor()

    class VPAppProxy:
        pass

    vp_app_proxy = VPAppProxy()
    vp_app_proxy.hardware_acceleration_method = hwaccel_method_arg
    vp_app_proxy.available_ffmpeg_hwaccels = hwaccel_avail_list_arg if hwaccel_avail_list_arg is not None else []

    main_vp_for_info = VideoProcessor(app_instance=vp_app_proxy,
                                      fallback_logger_config={'logger_instance': process_logger})

    if not main_vp_for_info.open_video(video_path_arg):
        return None
    full_video_total_frames = main_vp_for_info.video_info.get('total_frames', 0)
    video_fps = main_vp_for_info.video_info.get('fps', 30.0)
    main_vp_for_info.reset(close_video=True)
    del main_vp_for_info

    processing_start_frame = 0
    processing_end_frame = full_video_total_frames - 1
    if frame_range_arg:
        start, end = frame_range_arg
        if start is not None: processing_start_frame = start
        if end is not None and end != -1: processing_end_frame = end
    total_frames_to_process = processing_end_frame - processing_start_frame + 1
    if total_frames_to_process <= 0:
        return None

    frame_processing_queue = Queue(maxsize=constants.STAGE1_FRAME_QUEUE_MAXSIZE)
    yolo_result_queue = Queue()
    producers_list, consumers_list = [], []
    logger_p_thread = None

    try:
        # --- PROCESS CREATION ---
        frames_per_producer = total_frames_to_process // num_producers_arg
        extra_frames = total_frames_to_process % num_producers_arg
        current_frame = processing_start_frame
        for i in range(num_producers_arg):
            num_frames = frames_per_producer + (1 if i < extra_frames else 0)
            if num_frames > 0:
                p_args = (i, video_path_arg, yolo_input_size_arg, video_type_arg, vr_input_format_arg, vr_fov_arg,
                          vr_pitch_arg, current_frame, num_frames, frame_processing_queue, queue_monitor,
                          stop_event_internal, hwaccel_method_arg, hwaccel_avail_list_arg,
                          fallback_config_for_subprocesses)
                producers_list.append(Process(target=video_processor_producer_proc, args=p_args, daemon=True))
                current_frame += num_frames

        for i in range(num_consumers_arg):
            c_args = (frame_processing_queue, yolo_result_queue, i, yolo_model_path_arg, yolo_pose_model_path_arg,
                      confidence_threshold, yolo_input_size_arg, queue_monitor, stop_event_internal,
                      fallback_config_for_subprocesses, video_fps)
            consumers_list.append(Process(target=consumer_proc, args=c_args, daemon=True))

        logger_thread_args = (frame_processing_queue, yolo_result_queue, result_file_local, total_frames_to_process,
                              progress_callback, queue_monitor, stop_event_internal,
                              s1_start_time, process_logger, gui_event_queue_arg)

        logger_p_thread = PyThread(target=logger_proc, args=logger_thread_args, daemon=True)

        # --- PROCESS STARTUP ---
        for p in producers_list:
            p.start()
        for p in consumers_list:
            p.start()
        logger_p_thread.start()

        # --- PROCESS JOINING AND SENTINEL LOGIC ---
        for p in producers_list:
            p.join()

        if not stop_event_internal.is_set():
            process_logger.info("[S1 Lib] All producers finished. Sending sentinels to consumers.")
            for _ in range(len(consumers_list)):
                queue_monitor.frame_queue_put(frame_processing_queue, None)

        for p in consumers_list:
            p.join()

        # After all consumers are finished, send a final sentinel to the logger queue.
        if not stop_event_internal.is_set():
            process_logger.info("[S1 Lib] All consumers finished. Sending end-of-stream sentinel to logger.")
            try:
                # Use a tuple to match the expected format (frame_id, payload)
                yolo_result_queue.put((None, None), timeout=1.0)
            except Full:
                process_logger.warning("[S1 Lib] Timed out sending sentinel to logger queue.")

        logger_p_thread.join()

        if stop_event_external.is_set(): return None
        return result_file_local if os.path.exists(result_file_local) else None


    except Exception as e:
        process_logger.critical(f"[S1 Lib] CRITICAL EXCEPTION in perform_yolo_analysis: {e}", exc_info=True)
        if not stop_event_internal.is_set():
            stop_event_internal.set()
        return None
    finally:
        # This 'finally' block ensures cleanup happens no matter what.
        process_logger.info("[S1 Lib] Entering cleanup block.")
        all_processes = producers_list + consumers_list
        for p in all_processes:
            if p.is_alive():
                process_logger.info(f"[S1 Lib] Terminating hanging process: {p.pid}")
                p.terminate()  # Forcefully terminate
                p.join(timeout=1.0)  # Wait for OS to clean up

        # Also ensure the logger thread is joined.
        if logger_p_thread and logger_p_thread.is_alive():
            logger_p_thread.join(timeout=1.0)
        process_logger.info("[S1 Lib] Cleanup complete.")


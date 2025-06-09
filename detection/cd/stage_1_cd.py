import numpy as np
import msgpack
import time
from multiprocessing import Process, Queue, Event, Value, freeze_support
from threading import Thread as PyThread
from queue import Empty, Full
from ultralytics import YOLO
import platform
import os
import argparse
import logging
from typing import Optional, Tuple, List
import subprocess
from queue import Queue as StdLibQueue
import torch

from video.video_processor import VideoProcessor

# --- Default CONFIG ---
QUEUE_MAXSIZE = 100
DEVICE_TO_USE = 'cpu'
try:
    if platform.processor() == 'arm' and platform.system() == 'Darwin':
        DEVICE_TO_USE = 'mps'
    elif torch.cuda.is_available():
        DEVICE_TO_USE = 'cuda'
    else:
        DEVICE_TO_USE = 'cpu'
except Exception as e:
    print(f"Device detection error for tracker: {e}")


class Stage1QueueMonitor:
    def __init__(self):
        self.frame_queue_puts = Value('i', 0)
        self.frame_queue_gets = Value('i', 0)
        self.result_queue_puts = Value('i', 0)
        self.result_queue_gets = Value('i', 0)

    def frame_queue_put(self, queue, item):
        with self.frame_queue_puts.get_lock(): self.frame_queue_puts.value += 1
        queue.put(item)

    def frame_queue_get(self, queue, block=True, timeout=None):
        item = queue.get(block=block, timeout=timeout)
        with self.frame_queue_gets.get_lock(): self.frame_queue_gets.value += 1
        return item

    def result_queue_put(self, queue, item):
        with self.result_queue_puts.get_lock(): self.result_queue_puts.value += 1
        queue.put(item)

    def result_queue_get(self, queue, block=True, timeout=None):
        item = queue.get(block=block, timeout=timeout)
        with self.result_queue_gets.get_lock(): self.result_queue_gets.value += 1
        return item

    def get_frame_queue_size(self):
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

        for frame_id, frame in vp_instance.stream_frames_for_segment(start_frame_abs_num, num_frames_in_segment):
            if stop_event_local.is_set():
                producer_logger.info(
                    f"[S1 VP Producer-{producer_idx}] Stop event detected. Processed {frames_put_to_queue_this_producer} frames.")
                break

            put_success = False
            while not put_success and not stop_event_local.is_set():
                try:
                    queue_monitor_local.frame_queue_put(frame_queue, (frame_id, np.copy(frame)))
                    put_success = True
                except Full:
                    if stop_event_local.is_set(): # Check stop event again if queue is full
                        producer_logger.info(f"[S1 VP Producer-{producer_idx}] Stop event detected while frame queue full. Frame {frame_id} not added.")
                        break
                    time.sleep(0.01) # Brief pause if queue is full
                except Exception as e_put: # Catch other potential errors during put
                    producer_logger.error(f"[S1 VP Producer-{producer_idx}] Error putting frame {frame_id} to queue: {e_put}", exc_info=True)
                    # Decide if this is a fatal error for the producer
                    stop_event_local.set() # Signal a problem
                    break


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


def consumer_proc(frame_queue, result_queue, consumer_idx, yolo_model_path_local,
                  confidence_threshold, yolo_input_size_consumer, queue_monitor_local, stop_event_local,
                  logger_config_for_consumer: Optional[dict] = None):

    consumer_logger = logging.getLogger(f"S1_Consumer_{consumer_idx}_{os.getpid()}")
    if not consumer_logger.hasHandlers():
        log_level_cons = logging.INFO
        handler_cons = logging.StreamHandler() # Default to StreamHandler
        if logger_config_for_consumer:
            if logger_config_for_consumer.get('log_level') is not None:
                log_level_cons = logger_config_for_consumer['log_level']
            if logger_config_for_consumer.get('log_file'):
                try:
                    handler_cons = logging.FileHandler(logger_config_for_consumer['log_file'])
                except Exception as e_fh:
                    print(f"[S1 Consumer-{consumer_idx}] Error setting up FileHandler: {e_fh}. Defaulting to StreamHandler.")
                    handler_cons = logging.StreamHandler() # Fallback if FileHandler fails

        formatter_cons = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')
        handler_cons.setFormatter(formatter_cons)
        consumer_logger.addHandler(handler_cons)
        consumer_logger.setLevel(log_level_cons)
    consumer_logger.info(f"[S1 Consumer-{consumer_idx}] Logger initialized.")

    processed_count = 0
    model = None
    try:
        model = YOLO(yolo_model_path_local, task='detect')  # Changed task to 'detect'
        consumer_logger.info(f"[S1 Consumer-{consumer_idx}] Model loaded: {yolo_model_path_local} for detection.")
        while not stop_event_local.is_set():
            try:
                item = queue_monitor_local.frame_queue_get(frame_queue, block=True, timeout=0.5) # Timeout to allow stop_event check
                if item is None: # Sentinel for graceful shutdown
                    consumer_logger.info(
                        f"[S1 Consumer-{consumer_idx}] Received sentinel. Processed {processed_count} frames. Exiting loop.")
                    break # Exit while loop
                frame_id, frame = item

                # Perform pure detection
                results = model(frame, device=DEVICE_TO_USE, verbose=False, imgsz=yolo_input_size_consumer, conf=confidence_threshold)

                detections = []
                for r_idx, r in enumerate(results): # Process results
                    boxes = r.boxes
                    if boxes is None or len(boxes) == 0: continue # No boxes in this result
                    for box_idx in range(len(boxes)):
                        box_data_xyxy = None
                        if boxes.xyxy is not None and len(boxes.xyxy) > box_idx:
                            box_data_xyxy = boxes.xyxy[box_idx].tolist()
                        elif boxes.xywh is not None and len(boxes.xywh) > box_idx: # Fallback if xyxy not present
                            x_center, y_center, w_box, h_box = boxes.xywh[box_idx].tolist()
                            x1 = x_center - w_box / 2
                            y1 = y_center - h_box / 2
                            x2 = x_center + w_box / 2
                            y2 = y_center + h_box / 2
                            box_data_xyxy = [x1,y1,x2,y2]
                        else: box_data_xyxy = [0,0,0,0] # Should not happen if boxes is valid

                        conf = float(boxes.conf[box_idx]) if boxes.conf is not None and len(boxes.conf) > box_idx else 0.0
                        cls = int(boxes.cls[box_idx]) if boxes.cls is not None and len(boxes.cls) > box_idx else -1

                        detections.append({'bbox': box_data_xyxy,
                                           'confidence': conf,
                                           'class': cls,
                                           'name': model.names[cls] if cls >=0 and cls in model.names else 'unknown'
                                           })

                # Put results to queue
                put_success = False
                while not put_success and not stop_event_local.is_set():
                    try:
                        queue_monitor_local.result_queue_put(result_queue, (frame_id, detections))
                        put_success = True
                    except Full:
                        if stop_event_local.is_set():
                            consumer_logger.info(f"[S1 Consumer-{consumer_idx}] Stop event while result queue full for frame {frame_id}.")
                            break
                        time.sleep(0.01)
                if stop_event_local.is_set() and not put_success: # Check if loop exited due to stop
                     consumer_logger.info(f"[S1 Consumer-{consumer_idx}] Failed to put result for frame {frame_id} due to stop signal.")
                     break # Exit main while loop

                processed_count += 1
            except Empty: # Timeout from frame_queue.get()
                if stop_event_local.is_set():
                    consumer_logger.info(
                        f"[S1 Consumer-{consumer_idx}] Frame queue empty and stop event set. Processed {processed_count} frames. Exiting loop.")
                    break # Exit while loop
                continue # Continue to check stop_event and try getting from queue again
            except Exception as e_consumer_loop:
                fid_str = frame_id if 'frame_id' in locals() else 'N/A'
                consumer_logger.error(
                    f"[S1 Consumer-{consumer_idx}] Error processing frame {fid_str}: {e_consumer_loop}", exc_info=True)
                # Optionally, set stop_event_local here if error is considered fatal for all consumers
                # stop_event_local.set()
                # break

            if stop_event_local.is_set(): # Final check at end of loop iteration
                consumer_logger.info(f"[S1 Consumer-{consumer_idx}] Stop event detected at end of loop. Processed {processed_count} frames.")
                break

    except Exception as e_setup: # Error during model loading or initial setup
        consumer_logger.critical(f"[S1 Consumer-{consumer_idx}] Critical error during setup: {e_setup}", exc_info=True)
        if not stop_event_local.is_set(): stop_event_local.set() # Signal main process about the error
    finally:
        if model is not None:
            del model # Release model resources
            consumer_logger.info(f"[S1 Consumer-{consumer_idx}] YOLO model deleted.")
        final_processed_count = processed_count if 'processed_count' in locals() else 'unknown'
        consumer_logger.info(
            f"[S1 Consumer-{consumer_idx}] Exiting. Total frames processed: {final_processed_count}. Stop event: {stop_event_local.is_set()}")


def logger_proc(result_queue, output_file_local, expected_frames,
                progress_callback_local, queue_monitor_local, stop_event_local,
                s1_start_time_param, parent_logger: logging.Logger):
    results_dict = {}
    written_count = 0 # Frames for which results have been received and stored in results_dict
    last_progress_update_time = time.time()
    first_result_received_time = None
    parent_logger.info(f"[S1 Logger] Expecting {expected_frames} frames. Writing to {output_file_local}")

    if progress_callback_local:
        initial_eta = 0.0 if expected_frames > 0 else -1 # -1 for N/A if no expected frames
        progress_callback_local(0, expected_frames, "Logger starting...", 0.0, 0.0, initial_eta)

    # Main loop to gather results
    # Loop continues as long as not all expected frames are written OR stop event is not set
    # If expected_frames is 0 or negative, it implies indefinite processing until stop or sentinel.
    # For this use case, expected_frames should always be positive.
    while (written_count < expected_frames if expected_frames > 0 else True) and not stop_event_local.is_set():
        try:
            # Timeout allows periodic check of stop_event_local
            frame_id, detections = queue_monitor_local.result_queue_get(result_queue, block=True, timeout=0.5)
            if frame_id not in results_dict: # Avoid reprocessing if a duplicate ID somehow appears
                results_dict[frame_id] = detections
                if first_result_received_time is None:
                    first_result_received_time = time.time()
                written_count += 1

            current_time = time.time()
            # Progress update logic (every 0.2s, at specific counts, or completion)
            if progress_callback_local and (
                    current_time - last_progress_update_time > 0.2 or \
                    (expected_frames > 0 and written_count == expected_frames) or \
                    (expected_frames > 0 and written_count % (max(1, expected_frames // 20)) == 0) or \
                    (expected_frames <=0 and written_count % 20 == 0) # Update every 20 frames if total unknown
                ):
                time_elapsed_seconds = current_time - s1_start_time_param
                fps_for_callback = 0.0
                eta_for_callback = 0.0 # Default to 0 (calculating or done)

                if first_result_received_time is not None and written_count > 0:
                    time_since_first_result = current_time - first_result_received_time
                    if time_since_first_result > 0.001:
                        fps_for_callback = written_count / time_since_first_result

                    if expected_frames > 0 and fps_for_callback > 0.001 and written_count < expected_frames:
                        eta_for_callback = (expected_frames - written_count) / fps_for_callback
                    elif expected_frames > 0 and written_count < expected_frames: # Still expecting frames, but FPS is zero
                        if time_elapsed_seconds > 5: eta_for_callback = float('inf') # Stalled
                    # If expected_frames is 0, ETA is not applicable
                    elif expected_frames <= 0: eta_for_callback = -1 # N/A

                progress_callback_local(written_count, expected_frames, "Detecting objects...",
                                        time_elapsed_seconds, fps_for_callback, eta_for_callback)
                last_progress_update_time = current_time
        except Empty: # Timeout from result_queue.get()
            # This is normal if consumers are slower or if all results are processed
            if expected_frames > 0 and written_count >= expected_frames:
                parent_logger.info(f"[S1 Logger] All expected frames ({written_count}/{expected_frames}) received. Exiting processing loop.")
                break
            if stop_event_local.is_set(): # Check stop event if queue was empty
                parent_logger.info(f"[S1 Logger] Result queue empty and stop event set. Processed {written_count} frames. Exiting loop.")
                break
            continue # Continue to check stop_event and try getting from queue again
        except Exception as e:
            parent_logger.error(f"[S1 Logger] Error processing result: {e}", exc_info=True)

    # After loop: either stop_event_local is set, or all expected frames are gathered (if expected_frames > 0)
    parent_logger.info(f"[S1 Logger] Result gathering loop ended. Written count: {written_count}. Stop event: {stop_event_local.is_set()}")

    # If stop event was set, try to drain any remaining items in the queue quickly
    if stop_event_local.is_set():
        parent_logger.info(f"[S1 Logger] Stop event active. Draining remaining results from queue...")
        drained_count_on_stop = 0
        # Safety limit for draining to prevent infinite loop if producers keep adding (should not happen if they also stop)
        # Max items to drain = current queue size + a small buffer
        max_drain_items = queue_monitor_local.get_result_queue_size() + 50
        while drained_count_on_stop < max_drain_items :
            try:
                frame_id, detections = queue_monitor_local.result_queue_get(result_queue, block=False) # Non-blocking
                if frame_id not in results_dict:
                    results_dict[frame_id] = detections
                    # Do not increment written_count here as it's for "normally processed" items for ETA calc
                drained_count_on_stop += 1
            except Empty:
                parent_logger.info(f"[S1 Logger] Result queue empty during stop-drain after {drained_count_on_stop} items.")
                break # Queue is empty
            except Exception as e_drain:
                parent_logger.error(f"[S1 Logger] Error draining result queue on stop: {e_drain}", exc_info=True)
                break
        parent_logger.info(f"[S1 Logger] Drained {drained_count_on_stop} additional items on stop. Total unique items in dict: {len(results_dict)}.")
        written_count = len(results_dict) # Update written_count to actual items for saving

    parent_logger.debug(
        f"[S1 Logger] Finalizing save. Items in dict: {len(results_dict)}. Expected originally: {expected_frames}.")

    ordered_results = []
    missing_frames_count = 0

    # Determine the starting frame ID from the collected results
    start_frame_offset = 0
    if results_dict:
        start_frame_offset = min(results_dict.keys())
        parent_logger.info(f"[S1 Logger] Detected start frame offset for saving: {start_frame_offset}")

    if expected_frames > 0:
        for i in range(expected_frames):
            absolute_frame_id = start_frame_offset + i
            if absolute_frame_id in results_dict:
                ordered_results.append(results_dict[absolute_frame_id])
            else:
                ordered_results.append([])  # Pad with empty list for missing frames
                missing_frames_count += 1
    elif results_dict:
        parent_logger.info(
            f"[S1 Logger] expected_frames was not positive. Saving all {len(results_dict)} collected results, ordered by frame ID.")
        sorted_frame_ids = sorted(results_dict.keys())
        for fid_key in sorted_frame_ids:
            ordered_results.append(results_dict[fid_key])
    else: # No results_dict or expected_frames was not positive
        parent_logger.warning(f"[S1 Logger] No results in dict to save, or expected_frames was {expected_frames}.")


    if missing_frames_count > 0:
        parent_logger.warning(
            f"[S1 Logger] {missing_frames_count} frames were missing from results and have been padded with empty detections in the output.")
    parent_logger.debug(f"[S1 Logger] Final number of entries for saving: {len(ordered_results)}")

    # Save the results to file
    final_save_succeeded = False
    if ordered_results or (expected_frames > 0 and stop_event_local.is_set()): # Save even if results are empty if processing was aborted but file is expected
        try:
            with open(output_file_local, 'wb') as f:
                packed = msgpack.packb(ordered_results, use_bin_type=True)
                f.write(packed)
            final_save_succeeded = True

            file_size_kb = os.path.getsize(output_file_local) / 1024 if os.path.exists(output_file_local) else 0
            parent_logger.info(
                f"[S1 Logger] Save complete. Wrote {len(ordered_results)} frame entries to {output_file_local}. Size: {file_size_kb:.2f} KB")
        except Exception as e:
            parent_logger.error(f"[S1 Logger] Error writing output file '{output_file_local}': {e}", exc_info=True)
    else:
        parent_logger.info(f"[S1 Logger] No results to write to {output_file_local} (ordered_results is empty and not an aborted run expecting a file).")


    # Final progress update
    if progress_callback_local:
        final_time_elapsed_total = time.time() - s1_start_time_param
        final_fps_for_callback = 0.0
        # Use len(results_dict) for FPS calc as it reflects actual items processed by consumers
        if first_result_received_time is not None and len(results_dict) > 0:
            duration_since_first_result_for_final_fps = time.time() - first_result_received_time
            if duration_since_first_result_for_final_fps > 0.001:
                final_fps_for_callback = len(results_dict) / duration_since_first_result_for_final_fps

        # Progress current value should be based on what was actually put into the output file
        progress_current_val_final = len(ordered_results)

        if stop_event_local.is_set():
            status_message = "Detection aborted & saved." if final_save_succeeded else "Detection aborted (save failed)."
        elif not final_save_succeeded and (ordered_results or expected_frames > 0):
             status_message = "Error: Output file not saved correctly."
        else:
            status_message = "Detection complete & saved."

        progress_callback_local(progress_current_val_final, expected_frames, status_message,
                                final_time_elapsed_total, final_fps_for_callback, 0.0) # ETA is 0 for complete/aborted

    parent_logger.info(f"[S1 Logger] Logger process fully finished. Stop event: {stop_event_local.is_set()}")


def perform_yolo_analysis(
        video_path_arg: str,
        yolo_model_path_arg: str,
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
    process_logger = None # Logger for this main function/orchestrator
    fallback_config_for_subprocesses = None # Config to pass to subprocess loggers

    # Set output file path
    if output_filename_override:
        result_file_local = output_filename_override
        output_dir = os.path.dirname(result_file_local)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        # Fallback if no override is provided (should not happen in normal app flow)
        result_file_local = os.path.splitext(video_path_arg)[0] + '.msgpack'



    # Setup logger for this orchestrator function
    if app_logger_config_arg and app_logger_config_arg.get('main_logger'):
        process_logger = app_logger_config_arg['main_logger']
    else: # Fallback logger for perform_yolo_analysis itself
        process_logger = logging.getLogger("S1_Lib_Orchestrator")
        if not process_logger.hasHandlers():
            handler = logging.StreamHandler()
            log_level_orch = logging.INFO
            if app_logger_config_arg and app_logger_config_arg.get('log_level') is not None:
                log_level_orch = app_logger_config_arg['log_level']
            if app_logger_config_arg and app_logger_config_arg.get('log_file'):
                 try: handler = logging.FileHandler(app_logger_config_arg['log_file'])
                 except Exception: pass # Stick to stream handler if file fails
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(threadName)s - %(message)s')
            handler.setFormatter(formatter)
            process_logger.addHandler(handler)
            process_logger.setLevel(log_level_orch)
            process_logger.info("S1_Lib_Orchestrator using its own fallback/configured logger.")

    # Prepare logger config that will be passed to subprocesses for their own loggers
    if app_logger_config_arg and app_logger_config_arg.get('log_file') and app_logger_config_arg.get('log_level') is not None:
        fallback_config_for_subprocesses = {
            'log_file': app_logger_config_arg['log_file'],
            'log_level': app_logger_config_arg['log_level']
        }
    elif process_logger.handlers: # If orchestrator has a file handler, subprocesses can use it too
        for h in process_logger.handlers:
            if isinstance(h, logging.FileHandler):
                fallback_config_for_subprocesses = {
                    'log_file': h.baseFilename,
                    'log_level': process_logger.level
                }
                break
    if not fallback_config_for_subprocesses: # Ultimate fallback: subprocesses will use StreamHandler
         fallback_config_for_subprocesses = {'log_file': None, 'log_level': logging.INFO}


    process_logger.info(f"Stage 1 YOLO Analysis started with orchestrator logger: {process_logger.name}")

    if not os.path.exists(video_path_arg):
        msg = f"Error: Video not found {os.path.basename(video_path_arg)}"
        process_logger.error(msg)
        if progress_callback:
            progress_callback(0, 0, msg, 0, 0, 0)
            return None
    if not os.path.exists(yolo_model_path_arg):
        msg = "Error: YOLO model not found"
        process_logger.error(msg)
        if progress_callback:
            progress_callback(0, 0, msg, 0, 0, 0)
            return None

    s1_start_time = time.time()
    # Internal stop event for this function to manage its subprocesses
    # This can be triggered by the external stop_event_external or by internal errors.
    stop_event_internal = Event()

    # In perform_yolo_analysis, around line 552
    def monitor_external_stop():
        process_logger.info("[S1 Lib Monitor] External stop monitor thread started.")  # Added log
        while not stop_event_internal.is_set():  # Check internal stop as a way to break loop
            # Wait for external_stop_event with a timeout
            if stop_event_external.wait(timeout=0.5):  # Returns True if event is set, False on timeout
                if not stop_event_internal.is_set():
                    process_logger.info("[S1 Lib Monitor] External stop received. Relaying to internal stop event.")
                    stop_event_internal.set()
                break  # Exit loop once external event is processed
        # If loop exited, log reason
        if stop_event_internal.is_set() and not stop_event_external.is_set():
            process_logger.info("[S1 Lib Monitor] Internal stop detected. Monitor thread exiting.")
        elif stop_event_external.is_set():
            process_logger.info("[S1 Lib Monitor] External stop processed. Monitor thread exiting.")
        else:  # Should only happen if loop condition broken unexpectedly
            process_logger.info("[S1 Lib Monitor] Loop exited. Monitor thread exiting.")

    stop_monitor_thread = PyThread(target=monitor_external_stop, daemon=True)
    stop_monitor_thread.start()

    queue_monitor = Stage1QueueMonitor()

    process_logger.info(
        f"[S1 Lib] YOLO device: {DEVICE_TO_USE}. P: {num_producers_arg}, C: {num_consumers_arg} for '{os.path.basename(video_path_arg)}'.")
    if progress_callback: progress_callback(0, 1, "Initializing Stage 1...", 0, 0, 0)

    last_queue_log_time = time.time()
    queue_log_interval = 5  # Log queue sizes every 5 seconds (adjustable)

    last_queue_update_time = time.time()
    queue_update_interval_gui = 2  # e.g., update GUI every 2 seconds

    # Get video info using a local VideoProcessor instance
    # This VP instance needs a logger. We can pass the orchestrator's logger config.
    class AppInstanceProxyForVPInfo: pass # Simple proxy
    app_proxy_vp_info = AppInstanceProxyForVPInfo()
    app_proxy_vp_info.logger = process_logger # VP will use this logger
    app_proxy_vp_info.hardware_acceleration_method = "none" # For info gathering, HW accel not critical

    main_vp_for_info = VideoProcessor(
        app_instance=app_proxy_vp_info, # Provides logger
        tracker=None, yolo_input_size=yolo_input_size_arg, video_type=video_type_arg,
        vr_input_format=vr_input_format_arg, vr_fov=vr_fov_arg, vr_pitch=vr_pitch_arg
        # fallback_logger_config is not needed if app_instance.logger is provided and used by VP
    )

    if not main_vp_for_info.open_video(video_path_arg):
        msg = f"Error: VideoProcessor could not open video for info: {os.path.basename(video_path_arg)}"
        process_logger.error(msg)
        if progress_callback: progress_callback(0, 0, msg, time.time() - s1_start_time, 0, 0)
        if not stop_event_internal.is_set(): stop_event_internal.set() # Signal error
        stop_monitor_thread.join(timeout=0.5) # ensure monitor exits
        return None

    full_video_total_frames = main_vp_for_info.video_info.get('total_frames', 0)
    vid_fps = main_vp_for_info.video_info.get('fps', 0)
    main_vp_for_info.reset(close_video=True) # Close video and release resources for this instance
    del main_vp_for_info # Delete the instance

    # Determine the actual frame range to process
    processing_start_frame = 0
    processing_end_frame = full_video_total_frames - 1

    if frame_range_arg:
        start_arg, end_arg = frame_range_arg
        if start_arg is not None:
            processing_start_frame = max(0, start_arg)
        if end_arg is not None and end_arg != -1:
            processing_end_frame = min(end_arg, full_video_total_frames - 1)

        if processing_start_frame > processing_end_frame:
            process_logger.error(
                f"Invalid frame range for Stage 1: Start {processing_start_frame} > End {processing_end_frame}. Aborting.")
            return None

        total_frames_to_process = processing_end_frame - processing_start_frame + 1
        process_logger.info(
            f"[S1 Lib] Processing specified range: Frames {processing_start_frame} to {processing_end_frame} ({total_frames_to_process} frames).")
    else:
        total_frames_to_process = full_video_total_frames
        process_logger.info(f"[S1 Lib] Processing full video ({total_frames_to_process} frames).")

    if total_frames_to_process <= 0:
        msg = f"Error: No frames to process in the specified range/video (calculated {total_frames_to_process} frames)."
        process_logger.error(msg)
        if progress_callback: progress_callback(0, 0, msg, time.time() - s1_start_time, 0, 0)
        return None

    frame_processing_queue = Queue(maxsize=QUEUE_MAXSIZE)
    yolo_result_queue = Queue() # Unbounded, logger proc will handle consumption
    producers_list = []
    consumers_list = []
    logger_p_thread = None # For the logger thread (not process)

    try:
        # Distribute the FRAME RANGE among producers
        frames_per_producer_dist = [
            total_frames_to_process // num_producers_arg + (1 if i < total_frames_to_process % num_producers_arg else 0)
            for i in
            range(num_producers_arg)]
        if total_frames_to_process > 0 and total_frames_to_process < num_producers_arg:
            frames_per_producer_dist = [0] * num_producers_arg
            for i in range(total_frames_to_process): frames_per_producer_dist[i] = 1
        process_logger.info(
            f"[S1 Lib] Frame distribution for range: {frames_per_producer_dist}. Sum: {sum(frames_per_producer_dist)}")

        current_absolute_frame_offset = processing_start_frame
        for i in range(num_producers_arg):
            num_frames_for_this_prod = frames_per_producer_dist[i]
            if num_frames_for_this_prod == 0:
                process_logger.info(f"[S1 Lib] Producer {i} assigned 0 frames, skipping.")
                continue
            p_args = (i, video_path_arg, yolo_input_size_arg, video_type_arg, vr_input_format_arg,
                      vr_fov_arg, vr_pitch_arg, current_absolute_frame_offset, num_frames_for_this_prod,
                      frame_processing_queue, queue_monitor, stop_event_internal, hwaccel_method_arg,
                      hwaccel_avail_list_arg, fallback_config_for_subprocesses)
            p = Process(target=video_processor_producer_proc, args=p_args, daemon=True)
            producers_list.append(p)
            current_absolute_frame_offset += num_frames_for_this_prod

        for p_idx, p_proc in enumerate(producers_list):
            p_proc.start()
            process_logger.info(f"[S1 Lib] Started producer {p_idx} (PID: {p_proc.pid})")
        if not producers_list: process_logger.warning("[S1 Lib] No producers started (e.g. 0 frames or 0 producers).")

        for i in range(num_consumers_arg):
            c_args = (frame_processing_queue, yolo_result_queue, i, yolo_model_path_arg, confidence_threshold, yolo_input_size_arg,
                      queue_monitor, stop_event_internal, fallback_config_for_subprocesses)
            c = Process(target=consumer_proc, args=c_args, daemon=True)
            consumers_list.append(c)
        for c_idx, c_proc in enumerate(consumers_list):
            c_proc.start()
            process_logger.info(f"[S1 Lib] Started consumer {c_idx} (PID: {c_proc.pid})")

        logger_thread_args = (yolo_result_queue, result_file_local, total_frames_to_process,
                              progress_callback, queue_monitor, stop_event_internal,
                              s1_start_time, process_logger)
        logger_p_thread = PyThread(target=logger_proc, args=logger_thread_args, daemon=True)
        logger_p_thread.start()
        process_logger.info("[S1 Lib] Started logger thread.")

        # --- Process Joining Logic ---
        process_logger.info("[S1 Lib] Orchestrator now waiting for producers.")
        producers_alive_flags = [True] * len(producers_list)
        producers_finished_normally = 0

        while any(producers_alive_flags):
            if stop_event_internal.is_set():
                process_logger.info(
                    "[S1 Lib] Stop event detected in producer join loop. Proceeding to aggressive termination.")
                break  # Exit this loop and go to final aggressive cleanup if needed

            # --- Queue Monitoring during producer phase ---
            current_time_monitor = time.time()
            if current_time_monitor - last_queue_log_time > queue_log_interval:
                frame_q_size = queue_monitor.get_frame_queue_size() #
                result_q_size = queue_monitor.get_result_queue_size() #
                process_logger.info(f"[S1 Lib Monitor] Frame Queue: ~{frame_q_size}, Result Queue: ~{result_q_size}")
                if gui_event_queue_arg:
                    # Send a specific event type for queue updates
                    gui_event_queue_arg.put(("stage1_queue_update", {"frame_q_size": frame_q_size, "result_q_size": result_q_size}, None))

                last_queue_log_time = current_time_monitor
            # --- End Queue Monitoring ---

            all_producers_joined_this_iteration = True
            for i_prod, p_proc in enumerate(producers_list):
                if not producers_alive_flags[i_prod]:
                    continue  # Already processed this producer

                p_proc.join(timeout=0.1)  # Always use a short timeout

                if not p_proc.is_alive():
                    producers_alive_flags[i_prod] = False
                    if not stop_event_internal.is_set():  # Check after join
                        producers_finished_normally += 1
                        process_logger.info(
                            f"[S1 Lib] P{i_prod} (PID: {p_proc.pid}) finished normally. ({producers_finished_normally}/{len(producers_list)})")
                    else:
                        process_logger.info(
                            f"[S1 Lib] P{i_prod} (PID: {p_proc.pid}) exited (stop was active during/after join).")
                else:
                    all_producers_joined_this_iteration = False  # At least one producer is still alive and running

            if all_producers_joined_this_iteration and not any(producers_alive_flags):  # All done
                break

            time.sleep(0.05)  # Brief pause to prevent busy-waiting if many procs are still alive

        # After the loop, if stop_event_internal is set, or if there are still live producers (e.g. loop exited by break)
        if stop_event_internal.is_set() or any(
                f for i, f in enumerate(producers_alive_flags) if producers_list[i].is_alive()):
            process_logger.info(
                "[S1 Lib] Post-producer-join-loop: Stop event is active or producers still alive. Aggressively terminating remaining producers.")
            for i_prod, p_proc in enumerate(producers_list):
                if p_proc.is_alive():  # Check again, even if flag was false, process might be zombie
                    process_logger.warning(f"[S1 Lib] Aggressively terminating P{i_prod} (PID: {p_proc.pid}).")
                    p_proc.terminate()
                    p_proc.join(timeout=0.2)
                    if p_proc.is_alive():
                        process_logger.error(f"[S1 Lib] P{i_prod} (PID: {p_proc.pid}) FAILED to terminate. Killing.")
                        p_proc.kill()
                        p_proc.join(timeout=0.1)
                        if p_proc.is_alive():
                            process_logger.critical(
                                f"[S1 Lib] P{i_prod} (PID: {p_proc.pid}) CRITICAL: FAILED to die after kill.")
                        else:
                            process_logger.info(f"[S1 Lib] P{i_prod} (PID: {p_proc.pid}) successfully killed.")
                    else:
                        process_logger.info(f"[S1 Lib] P{i_prod} (PID: {p_proc.pid}) successfully terminated.")

            # Keep original logging for normal/aborted exit status
            if not p_proc.is_alive():
                if not stop_event_internal.is_set():
                    producers_finished_normally += 1
                    process_logger.info(
                        f"[S1 Lib] P{i_prod} (PID: {p_proc.pid}) finished normally. ({producers_finished_normally}/{len(producers_list)})")
                else:
                    process_logger.info(f"[S1 Lib] P{i_prod} (PID: {p_proc.pid}) exited (stop was active).")


        if stop_event_internal.is_set():
            process_logger.info("[S1 Lib] Producer processing loop finished due to stop signal.")
        elif producers_finished_normally == len(producers_list):
            process_logger.info("[S1 Lib] All producers completed their work normally.")
        else:
            process_logger.warning(f"[S1 Lib] Producer loop finished. {producers_finished_normally}/{len(producers_list)} finished normally. Stop: {stop_event_internal.is_set()}")

        # Signal consumers to stop or finish, only if producers are done or we are stopping
        if not stop_event_internal.is_set() and producers_finished_normally == len(producers_list):
            process_logger.info("[S1 Lib] Producers done. Sending sentinels to consumers.")
            for i in range(len(consumers_list)): # Send one sentinel per consumer
                try: queue_monitor.frame_queue_put(frame_processing_queue, None)
                except Full:
                    process_logger.warning(f"[S1 Lib] Frame queue full when trying to put sentinel {i+1}/{len(consumers_list)}. Stop: {stop_event_internal.is_set()}")
                    if stop_event_internal.is_set(): break # Don't persist if stopping
                    time.sleep(0.1) # Wait a bit and retry if needed, or assume consumer will pick up
        elif stop_event_internal.is_set():
            process_logger.info("[S1 Lib] Stop event active. Consumers will be joined/terminated without all sentinels potentially.")
            # Consumers should also react to stop_event_internal directly.

        process_logger.info("[S1 Lib] Orchestrator now waiting for consumers.")
        consumers_alive_flags = [True] * len(consumers_list)
        consumers_finished_normally = 0
        while any(consumers_alive_flags):
            if stop_event_internal.is_set():
                process_logger.info(
                    "[S1 Lib] Stop event detected in consumer join loop. Proceeding to aggressive termination.")
                break

            # --- Queue Monitoring during consumer phase ---
            current_time_monitor = time.time()
            if current_time_monitor - last_queue_log_time > queue_log_interval:
                frame_q_size = queue_monitor.get_frame_queue_size() #
                result_q_size = queue_monitor.get_result_queue_size() #
                process_logger.info(f"[S1 Lib Monitor] Frame Queue: ~{frame_q_size}, Result Queue: ~{result_q_size}")
                if gui_event_queue_arg:
                    gui_event_queue_arg.put(("stage1_queue_update", {"frame_q_size": frame_q_size, "result_q_size": result_q_size}, None))

                last_queue_log_time = current_time_monitor
            # --- End Queue Monitoring ---

            all_consumers_joined_this_iteration = True
            for i_cons, c_proc in enumerate(consumers_list):
                if not consumers_alive_flags[i_cons]:
                    continue

                c_proc.join(timeout=0.1)  # Always use a short timeout

                if not c_proc.is_alive():
                    consumers_alive_flags[i_cons] = False
                    if not stop_event_internal.is_set():
                        consumers_finished_normally += 1
                        process_logger.info(
                            f"[S1 Lib] C{i_cons} (PID: {c_proc.pid}) finished normally. ({consumers_finished_normally}/{len(consumers_list)})")
                    else:
                        process_logger.info(
                            f"[S1 Lib] C{i_cons} (PID: {c_proc.pid}) exited (stop was active during/after join).")
                else:
                    all_consumers_joined_this_iteration = False

            if all_consumers_joined_this_iteration and not any(consumers_alive_flags):
                break

            time.sleep(0.05)

        if stop_event_internal.is_set() or any(
                f for i, f in enumerate(consumers_alive_flags) if consumers_list[i].is_alive()):
            process_logger.info(
                "[S1 Lib] Post-consumer-join-loop: Stop event is active or consumers still alive. Aggressively terminating remaining consumers.")
            for i_cons, c_proc in enumerate(consumers_list):
                if c_proc.is_alive():
                    process_logger.warning(f"[S1 Lib] Aggressively terminating C{i_cons} (PID: {c_proc.pid}).")
                    c_proc.terminate()
                    c_proc.join(timeout=0.2)
                    if c_proc.is_alive():
                        process_logger.error(f"[S1 Lib] C{i_cons} (PID: {c_proc.pid}) FAILED to terminate. Killing.")
                        c_proc.kill()
                        c_proc.join(timeout=0.1)
                        if c_proc.is_alive():
                            process_logger.critical(
                                f"[S1 Lib] C{i_cons} (PID: {c_proc.pid}) CRITICAL: FAILED to die after kill.")
                        else:
                            process_logger.info(f"[S1 Lib] C{i_cons} (PID: {c_proc.pid}) successfully killed.")
                    else:
                        process_logger.info(f"[S1 Lib] C{i_cons} (PID: {c_proc.pid}) successfully terminated.")

            # Keep original logging for normal/aborted exit status
            if not c_proc.is_alive():
                if not stop_event_internal.is_set():
                    consumers_finished_normally += 1
                    process_logger.info(
                        f"[S1 Lib] C{i_cons} (PID: {c_proc.pid}) finished normally. ({consumers_finished_normally}/{len(consumers_list)})")
                else:
                    process_logger.info(f"[S1 Lib] C{i_cons} (PID: {c_proc.pid}) exited (stop was active).")


        if stop_event_internal.is_set():
            process_logger.info("[S1 Lib] Consumer processing loop finished due to stop signal.")
        elif consumers_finished_normally == len(consumers_list):
            process_logger.info("[S1 Lib] All consumers completed their work normally.")
        else:
            process_logger.warning(f"[S1 Lib] Consumer loop finished. {consumers_finished_normally}/{len(consumers_list)} finished normally. Stop: {stop_event_internal.is_set()}")


        process_logger.info("[S1 Lib] Orchestrator now waiting for logger thread.")
        if logger_p_thread:
            logger_join_timeout = 15.0 if stop_event_internal.is_set() else max(30.0, total_frames_to_process * 0.05 if total_frames_to_process >0 else 30.0)
            logger_p_thread.join(timeout=logger_join_timeout)
            if logger_p_thread.is_alive():
                process_logger.warning("[S1 Lib] Logger thread timed out. It might be stuck writing. Signalling stop again.")
                if not stop_event_internal.is_set(): stop_event_internal.set() # Make sure it knows to stop
                logger_p_thread.join(timeout=10.0) # One last chance
                if logger_p_thread.is_alive():
                     process_logger.error("[S1 Lib] Logger thread FAILED to join. Output file might be incomplete.")
            else:
                process_logger.info("[S1 Lib] Logger thread finished.")


        # Final check of stop status and return value
        if stop_event_external.is_set(): # Prefer external stop reason if both are set
            process_logger.info("[S1 Lib] Processing aborted by external signal. Output may be incomplete.")
            if os.path.exists(result_file_local) and os.path.getsize(result_file_local) > 10:
                 process_logger.info(f"[S1 Lib] Partial results saved to: {result_file_local}")
                 return result_file_local # Return partial if available
            return None
        elif stop_event_internal.is_set(): # Internal error or propagated external stop
            process_logger.info("[S1 Lib] Processing stopped by internal signal or error. Output may be incomplete.")
            if os.path.exists(result_file_local) and os.path.getsize(result_file_local) > 10:
                 process_logger.info(f"[S1 Lib] Partial results saved to: {result_file_local}")
                 return result_file_local
            return None

        # If no stop event was set and processing seems to have completed
        if os.path.exists(result_file_local) and os.path.getsize(result_file_local) > 10:
            process_logger.info(f"[S1 Lib] Successfully created output: {result_file_local}")
            return result_file_local
        else:
            process_logger.error(f"[S1 Lib] Output file error or file is too small/missing: {result_file_local}")
            return None

    except Exception as e:
        process_logger.critical(f"[S1 Lib] CRITICAL EXCEPTION in perform_yolo_analysis: {e}", exc_info=True)
        if progress_callback:
            progress_callback(0, total_frames_to_process if total_frames_to_process > 0 else 1,
                              f"Runtime Error: {e}", time.time() - s1_start_time, 0, 0)
        if not stop_event_internal.is_set(): stop_event_internal.set() # Signal all components to stop due to error
        return None # Indicate failure
    finally:
        process_logger.info(
            f"[S1 Lib `finally`] Entering cleanup. Internal stop event: {stop_event_internal.is_set()}, External stop event: {stop_event_external.is_set()}")  # Enhanced logging

        if not stop_event_internal.is_set() and stop_event_external.is_set():  # Ensure internal is set if external is
            process_logger.info("[S1 Lib `finally`] External stop was set, ensuring internal stop is also set.")
            stop_event_internal.set()
        elif not stop_event_internal.is_set():  # If no stop was signaled at all (e.g. normal completion or other exception)
            process_logger.info(
                "[S1 Lib `finally`] No stop event was set prior to finally. Setting internal stop event as a final measure for cleanup.")
            stop_event_internal.set()  # Ensure all subprocesses know to wrap up if they somehow are still active

        # Attempt to join/terminate any remaining alive processes
        for p_list_idx, (p_list, p_name) in enumerate([(producers_list, "Producer"), (consumers_list, "Consumer")]):
            for i, p_proc in enumerate(p_list):
                if p_proc.is_alive():
                    process_logger.warning(f"[S1 Lib Cleanup `finally`] {p_name} {i} (PID: {p_proc.pid}) still alive.")
                    # If aborting (stop_event_internal is set), be more aggressive
                    process_logger.warning(
                        f"[S1 Lib Cleanup `finally`] Aggressively terminating {p_name} {i} (PID: {p_proc.pid}).")
                    p_proc.terminate()  # Try graceful SIGTERM first
                    p_proc.join(0.2)  # Very short wait (e.g., 200ms)
                    if p_proc.is_alive():
                        process_logger.warning(
                            f"[S1 Lib Cleanup `finally`] {p_name} {i} (PID: {p_proc.pid}) did not terminate. Killing.")
                        p_proc.kill()  # Force SIGKILL
                        p_proc.join(0.1)  # Even shorter wait for kill (e.g., 100ms)
                        if p_proc.is_alive():
                            process_logger.error(
                                f"[S1 Lib Cleanup `finally`] {p_name} {i} (PID: {p_proc.pid}) FAILED TO DIE after kill. May be orphaned.")

        if logger_p_thread and logger_p_thread.is_alive():
            process_logger.warning("[S1 Lib Cleanup `finally`] Logger thread still alive. Waiting briefly (max 1s)...")
            logger_p_thread.join(1.0)  # Reduced timeout
            if logger_p_thread.is_alive():
                process_logger.error("[S1 Lib Cleanup `finally`] Logger thread FAILED to join in finally.")

        if stop_monitor_thread and stop_monitor_thread.is_alive():
            process_logger.info("[S1 Lib Cleanup `finally`] Stop monitor thread joining (max 0.1s)...")
            stop_monitor_thread.join(0.1)  # Should exit quickly as its event.wait() would have passed
            if stop_monitor_thread.is_alive():
                process_logger.warning("[S1 Lib Cleanup `finally`] Stop monitor thread did not join.")

        process_logger.info("[S1 Lib] Cleanup in `finally` block finished. Orchestrator exiting.")


if __name__ == "__main__":
    freeze_support() # Important for Windows and sometimes macOS when using multiprocessing
    parser = argparse.ArgumentParser(description="Stage 1: YOLO Video Processing (VideoProcessor Streaming)")
    parser.add_argument("--video_path", type=str, required=True, help="Path to the video file")
    parser.add_argument("--yolo_model_path", type=str, required=True, help="Path to the YOLO .pt model file")
    parser.add_argument("--num_producers", type=int, default=2, help="Number of producer processes")
    parser.add_argument("--num_consumers", type=int, default=2, help="Number of consumer processes")
    parser.add_argument("--yolo_input_size", type=int, default=640, help="Input size for YOLO model (e.g., 640)")
    parser.add_argument("--video_type", type=str, default='auto', choices=['auto', '2D', 'VR'],
                        help="Video type for VideoProcessor")
    parser.add_argument("--vr_input_format", type=str, default='he',
                        help="VR input format for VideoProcessor (e.g., he, fisheye)")
    parser.add_argument("--vr_fov", type=int, default=190, help="VR FOV for VideoProcessor")
    parser.add_argument("--vr_pitch", type=int, default=0, help="VR pitch for VideoProcessor's v360 filter")
    parser.add_argument("--log_file", type=str, default=None, help="Optional log file path for CLI execution.")
    parser.add_argument("--log_level", type=str, default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"], help="Log level for CLI execution.")


    args = parser.parse_args()

    # Setup basic logger for CLI execution
    cli_main_logger = logging.getLogger("CLI_S1_YOLO_Main")
    log_level_cli = getattr(logging, args.log_level.upper(), logging.INFO)
    cli_main_logger.setLevel(log_level_cli)

    log_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(processName)s (%(process)d) - %(threadName)s (%(thread)d) - %(message)s')

    if args.log_file:
        fh = logging.FileHandler(args.log_file, mode='a') # Append mode
        fh.setFormatter(log_formatter)
        cli_main_logger.addHandler(fh)
    else:
        sh = logging.StreamHandler()
        sh.setFormatter(log_formatter)
        cli_main_logger.addHandler(sh)


    def cli_progress_callback(current_val, total_val, message_str, time_elapsed, fps, eta):
        elapsed_str = f"{int(time_elapsed // 3600):02d}:{int((time_elapsed % 3600) // 60):02d}:{int(time_elapsed % 60):02d}"
        eta_str = "N/A"
        percent = 0.0
        if total_val > 0:
            percent = (current_val / total_val) * 100 if current_val > 0 else 0.0

        if total_val > 0 and current_val >= total_val :
            eta_str = "Done"
        elif eta == float('inf'):
            eta_str = "Stalled"
        elif eta == -1: # Explicit N/A for ETA
            eta_str = "N/A"
        elif eta > 0:
            eta_str = f"{int(eta // 3600):02d}:{int((eta % 3600) // 60):02d}:{int(eta % 60):02d}"
        elif time_elapsed > 1 and fps < 0.01 and (total_val <=0 or current_val < total_val) :
            eta_str = "Stalled"
        elif total_val > 0 and current_val == 0 and message_str == "Logger starting...":
            eta_str = "Calculating..."


        cli_main_logger.info(
            f"Progress: {message_str} - {current_val}/{total_val if total_val > 0 else 'N/A'} ({percent:.2f}%) | FPS: {fps:.2f} | Elapsed: {elapsed_str} | ETA: {eta_str}")


    cli_stop_event = Event()
    cli_main_logger.info(f"Starting CLI Test: Video: {args.video_path}, Model: {args.yolo_model_path}")
    cli_start_time = time.time()

    # Prepare logger config for perform_yolo_analysis when called from CLI
    cli_logger_config = {
        'main_logger': cli_main_logger,
        'log_file': args.log_file, # Can be None
        'log_level': log_level_cli
    }

    try:
        output_file = perform_yolo_analysis(
            video_path_arg=args.video_path,
            yolo_model_path_arg=args.yolo_model_path,
            confidence_threshold=0.3, # Example, make it an arg if needed
            progress_callback=cli_progress_callback,
            stop_event_external=cli_stop_event,
            num_producers_arg=args.num_producers,
            num_consumers_arg=args.num_consumers,
            video_type_arg=args.video_type,
            vr_input_format_arg=args.vr_input_format,
            vr_fov_arg=args.vr_fov,
            vr_pitch_arg=args.vr_pitch,
            yolo_input_size_arg=args.yolo_input_size,
            app_logger_config_arg=cli_logger_config
        )
    except KeyboardInterrupt:
        cli_main_logger.info("CLI Test aborted by KeyboardInterrupt (Ctrl+C).")
        cli_stop_event.set() # Signal stop to the analysis function
        # perform_yolo_analysis should handle this via stop_event_external and its finally block
        output_file = None # No valid output
        # Wait a moment for cleanup if perform_yolo_analysis is running in the main thread
        time.sleep(2) # Give some time for perform_yolo_analysis to react to stop_event
    finally:
        cli_main_logger.info(f"CLI Test ended. Total Time: {time.time() - cli_start_time:.2f}s. Output: {output_file if 'output_file' in locals() else 'N/A'}")

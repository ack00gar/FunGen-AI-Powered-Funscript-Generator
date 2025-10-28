import numpy as np
import msgpack
import time
from multiprocessing import Process, Queue, Event, Value, freeze_support
import platform
import sys
from threading import Thread as PyThread
from queue import Empty, Full
from ultralytics import YOLO
import os
import logging
from typing import Optional, Tuple, List
import subprocess
from queue import Queue as StdLibQueue

from video import VideoProcessor
from config import constants

log_vid = logging.getLogger(__name__)

def _validate_preprocessed_file_completeness(file_path: str, expected_frames: int, logger: logging.Logger, tolerance_frames: int = 5) -> bool:
    """
    Validates that a preprocessed msgpack file contains the expected number of frames within tolerance.

    Args:
        file_path: Path to the msgpack file
        expected_frames: Expected number of frames
        logger: Logger instance
        tolerance_frames: Acceptable frame count difference

    Returns:
        True if file is complete and valid, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Preprocessed file does not exist: {file_path}")
            return False

        # Check file size - empty files are definitely invalid
        if os.path.getsize(file_path) < 100:  # Minimum reasonable size
            logger.warning(f"Preprocessed file is too small (likely empty): {file_path}")
            return False

        # Load and validate the msgpack content
        with open(file_path, 'rb') as f:
            data = msgpack.unpackb(f.read(), raw=False)

        if not isinstance(data, list):
            logger.warning(f"Preprocessed file has invalid format (not a list): {file_path}")
            return False

        actual_frames = len(data)
        frame_diff = abs(actual_frames - expected_frames)

        if frame_diff > tolerance_frames:
            logger.warning(f"Preprocessed file frame count mismatch: expected {expected_frames}, got {actual_frames} (diff: {frame_diff})")
            return False

        # Check that frames have valid structure
        valid_frames = 0
        for frame_data in data:
            if isinstance(frame_data, dict) and ('detections' in frame_data or 'poses' in frame_data):
                valid_frames += 1

        if valid_frames < (actual_frames * 0.8):  # At least 80% should be valid
            logger.warning(f"Preprocessed file has too many invalid frames: {valid_frames}/{actual_frames}")
            return False

        logger.info(f"Preprocessed file validation passed: {actual_frames} frames (expected {expected_frames})")
        return True

    except Exception as e:
        logger.error(f"Error validating preprocessed file {file_path}: {e}")
        return False

def _is_already_preprocessed_video(video_path: str, logger: logging.Logger) -> bool:
    """
    Checks if a video file is already a preprocessed video to prevent double preprocessing.

    Args:
        video_path: Path to the video file
        logger: Logger instance

    Returns:
        True if the video is already preprocessed, False otherwise
    """
    # Check by filename pattern
    if video_path.endswith("_preprocessed.mp4"):
        logger.warning(f"Video appears to be already preprocessed (by filename): {os.path.basename(video_path)}")
        return True
    
    return False

def _validate_preprocessed_video_completeness(video_path: str, expected_frames: int, expected_fps: float, logger: logging.Logger, tolerance_frames: int = 5) -> bool:
    """
    Validates that a preprocessed video has the expected duration and frame count.

    Args:
        video_path: Path to the video file
        expected_frames: Expected number of frames
        expected_fps: Expected FPS
        logger: Logger instance
        tolerance_frames: Acceptable frame count difference

    Returns:
        True if video is complete and valid, False otherwise
    """
    try:
        if not os.path.exists(video_path):
            logger.warning(f"Preprocessed video does not exist: {video_path}")
            return False

        # Check file size - very small files are likely corrupted
        file_size = os.path.getsize(video_path)
        if file_size < 500000:  # Less than 500kb is suspicious for any video
            logger.warning(f"Preprocessed video is suspiciously small: {file_size} bytes")
            return False

        # Use ffprobe to get video info
        import subprocess
        import json

        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries', 'stream=nb_frames,duration,r_frame_rate',
               '-of', 'json', video_path]

        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=10, creationflags=creation_flags)
        if result.returncode != 0:
            logger.warning(f"ffprobe failed for preprocessed video: {video_path}")
            return False

        data = json.loads(result.stdout)
        stream_info = data.get('streams', [{}])[0]

        # Check frame count
        nb_frames_str = stream_info.get('nb_frames')
        if nb_frames_str and nb_frames_str != 'N/A':
            actual_frames = int(nb_frames_str)
            frame_diff = abs(actual_frames - expected_frames)

            if frame_diff > tolerance_frames:
                logger.warning(f"Preprocessed video frame count mismatch: expected {expected_frames}, got {actual_frames}")
                return False
        else:
            # Fallback: check duration
            duration_str = stream_info.get('duration')
            if duration_str:
                duration = float(duration_str)
                expected_duration = expected_frames / expected_fps
                duration_diff = abs(duration - expected_duration)

                if duration_diff > (tolerance_frames / expected_fps):
                    logger.warning(f"Preprocessed video duration mismatch: expected {expected_duration:.2f}s, got {duration:.2f}s")
                    return False

        logger.info(f"Preprocessed video validation passed: {video_path}")
        return True

    except Exception as e:
        logger.error(f"Error validating preprocessed video {video_path}: {e}")
        return False

def _cleanup_incomplete_file(file_path: str, logger: logging.Logger) -> None:
    """
    Safely removes an incomplete/corrupted preprocessed file.

    Args:
        file_path: Path to the file to remove
        logger: Logger instance
    """
    try:
        if os.path.exists(file_path):
            # Create a backup with timestamp before deletion
            backup_path = f"{file_path}.incomplete.{int(time.time())}"
            os.rename(file_path, backup_path)
            logger.info(f"Moved incomplete file to: {os.path.basename(backup_path)}")
    except Exception as e:
        logger.error(f"Failed to cleanup incomplete file {file_path}: {e}")
        # Try direct deletion as fallback
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.info(f"Deleted incomplete file: {os.path.basename(file_path)}")
        except Exception as e2:
            logger.error(f"Failed to delete incomplete file {file_path}: {e2}")

def _determine_actual_hwaccel(selected_method: str, available_methods: List[str]) -> str:
    """Determines the best HW accel method based on user selection, platform, and availability."""
    system = platform.system().lower()

    if selected_method == "auto":
        if system == 'darwin' and 'videotoolbox' in available_methods:
            return 'videotoolbox'
        elif system in ['linux', 'windows']:
            if 'cuda' in available_methods or 'nvdec' in available_methods: return 'cuda'
            if 'qsv' in available_methods: return 'qsv'
            if system == 'linux' and 'vaapi' in available_methods: return 'vaapi'
        return 'none' # Fallback for auto
    elif selected_method in available_methods:
        return selected_method # Use user's valid selection
    else:
        return 'none' # Fallback if selection is invalid

# --- Integrated FFmpegEncoder Class ---
class FFmpegEncoder:
    def __init__(self, output_file: str, width: int, height: int, fps: float, ffmpeg_path: str, hwaccel_method: str):
        self.encoder_process = None
        self.output_file = output_file
        self.width = width
        self.height = height
        self.fps = fps
        self.ffmpeg_path = ffmpeg_path
        self.hwaccel_method = hwaccel_method
        self.stderr_thread = None

    def start(self):
        encoder_cmd = [
            self.ffmpeg_path, "-y", "-hide_banner",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{self.width}x{self.height}",
            "-r", str(self.fps), "-i", "pipe:0",
            *self._get_encoder_args(),
            self.output_file, "-loglevel", "error"
        ]
        log_vid.info(f"Encoder command: {' '.join(encoder_cmd)}")
        # Windows fix: prevent terminal windows from spawning
        creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
        self.encoder_process = subprocess.Popen(encoder_cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creation_flags)
        self.stderr_thread = PyThread(target=self._read_stderr, daemon=True)
        self.stderr_thread.start()

    def _read_stderr(self):
        if self.encoder_process and self.encoder_process.stderr:
            for line in iter(self.encoder_process.stderr.readline, b""):
                decoded = line.decode("utf-8", errors="replace").strip()
                if decoded: log_vid.error(f"FFmpeg: {decoded}")

    def encode_frame(self, frame_bytes):
        if self.encoder_process and self.encoder_process.stdin:
            try:
                self.encoder_process.stdin.write(frame_bytes)
            except (BrokenPipeError, IOError):
                log_vid.error("Encoder pipe broke. Stopping encoding for this frame.")
                self.stop()

    def stop(self):
        if not self.encoder_process: return
        log_vid.info("Stopping encoder process...")
        if self.encoder_process.stdin:
            try:
                self.encoder_process.stdin.close()
            except (BrokenPipeError, IOError):
                pass
        try:
            self.encoder_process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            log_vid.warning("FFmpeg did not terminate gracefully. Killing.")
            self.encoder_process.kill()
        if self.stderr_thread: self.stderr_thread.join(timeout=1.0)
        self.encoder_process = None
        log_vid.info("Encoder process stopped.")

    def _detect_encoding_devices(self) -> None:
        """
        Detects and logs all available hardware encoding devices with their device IDs.
        This function tests each encoder type and reports availability.
        """
        log_vid.info("=== ENCODING DEVICE DETECTION ===")
        
        # Define encoder types to test
        encoder_configs = [
            ("hevc_nvenc", "NVENC (NVIDIA)"),
            ("hevc_amf", "AMF (AMD)"),
            ("hevc_qsv", "QSV (Intel)"),
            ("hevc_vaapi", "VAAPI (Linux)")
        ]
        
        all_devices = []
        device_id = 0
        
        for encoder_codec, encoder_name in encoder_configs:
            try:
                subprocess.check_output([
                    self.ffmpeg_path, "-hide_banner", "-f", "lavfi", "-i", "testsrc", 
                    "-c:v", encoder_codec, "-f", "null", "-"
                ], stderr=subprocess.PIPE, text=True, timeout=5)
                # If we get here, encoder is available
                all_devices.append(f"Device {device_id}: {encoder_name}")
                device_id += 1
            except subprocess.CalledProcessError as e:
                if encoder_codec in e.stderr:
                    # Encoder exists but failed - might be device-specific
                    all_devices.append(f"Device {device_id}: {encoder_name} - failed")
                    device_id += 1
            except Exception:
                pass

        if all_devices:
            log_vid.info("Detected encoding devices:")
            for device in all_devices:
                log_vid.info(f"  {device}")
        else:
            log_vid.info("No hardware encoding devices detected")
            
        log_vid.info("=== END ENCODING DEVICE DETECTION ===")

    def _get_encoder_args(self) -> List[str]:
        """
        Chooses the best available hardware encoder with a more robust priority.
        Falls back to software libx265 if none is available.
        Respects the user's hardware acceleration setting.
        """

        self._detect_encoding_devices()

        if self.hwaccel_method == "none":
            log_vid.info("Using software libx265 encoder (CPU-only mode).")
            return ["-c:v", "libx265", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p"]

        try:
            output = subprocess.check_output([self.ffmpeg_path, "-hide_banner", "-encoders"], text=True)
        except Exception as e:
            log_vid.warning(f"Failed to query FFmpeg encoders: {e}")
            output = ""

        # macOS is a special case and usually works well.
        if "hevc_videotoolbox" in output:
            log_vid.info("Using H.265 Apple VideoToolbox for hardware encoding.")
            return ["-c:v", "hevc_videotoolbox", "-q:v", "20", "-pix_fmt", "yuv420p", "-b:v", "0"]

        if "hevc_nvenc" in output:
            log_vid.info("Using H.265 NVENC for hardware encoding.")
            return ["-c:v", "hevc_nvenc", "-preset", "fast", "-qp", "26", "-pix_fmt", "yuv420p"]

        elif "hevc_amf" in output:
            log_vid.info("Using H.265 AMD AMF for hardware encoding.")
            return ["-c:v", "hevc_amf", "-quality", "quality", "-qp_i", "26", "-qp_p", "26", "-pix_fmt", "yuv420p"]

        elif "hevc_qsv" in output:
            log_vid.info("Using H.265 Intel QSV for hardware encoding.")
            return ["-c:v", "hevc_qsv", "-preset", "fast", "-global_quality", "26", "-pix_fmt", "yuv420p"]

        elif "hevc_vaapi" in output: # Primarily for Linux
            log_vid.info("Using H.265 VAAPI for hardware encoding.")
            return ["-c:v", "hevc_vaapi", "-qp", "26", "-pix_fmt", "yuv420p"]
        
        # Fallback to software encoder if no hardware options are found
        log_vid.warning("No supported hardware encoder found in FFmpeg build. Falling back to software libx265.")
        return ["-c:v", "libx265", "-preset", "ultrafast", "-crf", "26", "-pix_fmt", "yuv420p"]

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
        except NotImplementedError:  # Some platforms might not implement qsize()
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
        logger_config_for_vp_in_producer: Optional[dict] = None,
        is_encoding_preprocessed_video: bool = False,
        output_path_for_encoding: Optional[str] = None
):
    frames_put_to_queue_this_producer = 0
    vp_instance = None
    producer_logger = None
    encoder = None

    try:
        # Create a proxy object for the VideoProcessor instance
        class VPAppProxy:
            pass

        vp_app_proxy = VPAppProxy()
        vp_app_proxy.hardware_acceleration_method = hwaccel_method_producer
        vp_app_proxy.available_ffmpeg_hwaccels = hwaccel_avail_list_producer if hwaccel_avail_list_producer is not None else []
        # Force v360 unwarp method for offline tracking to avoid low FPS with GPU unwarp (metal/opengl)
        vp_app_proxy.app_settings = {'vr_unwarp_method': 'v360'}

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

        if is_encoding_preprocessed_video and output_path_for_encoding:
            width = vp_instance.yolo_input_size
            height = vp_instance.yolo_input_size
            fps = vp_instance.video_info.get('fps', 30.0)
            ffmpeg_path = os.environ.get("FFMPEG_PATH", "ffmpeg")

            encoder = FFmpegEncoder(
                output_file=output_path_for_encoding,
                width=width,
                height=height,
                fps=fps,
                ffmpeg_path=ffmpeg_path,
                hwaccel_method=hwaccel_method_producer
            )
            producer_logger.info(f"Starting FFmpeg encoder for preprocessed video.")
            encoder.start()

        producer_logger.info(
            f"[S1 VP Producer-{producer_idx}] Streaming segment: Video='{os.path.basename(vp_instance.video_path)}', StartFrameAbs={start_frame_abs_num}, NumFrames={num_frames_in_segment}, YOLOSize={vp_instance.yolo_input_size}")

        for frame_id, frame in vp_instance.stream_frames_for_segment(start_frame_abs_num, num_frames_in_segment, stop_event=stop_event_local):
            if stop_event_local.is_set():
                producer_logger.info(
                    f"[S1 VP Producer-{producer_idx}] Stop event detected. Processed {frames_put_to_queue_this_producer} frames.")
                break

            if encoder:
                encoder.encode_frame(frame.tobytes())

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
        if encoder:
            producer_logger.info(f"[S1 VP Producer-{producer_idx}] Stopping video encoder...")
            encoder.stop()

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

def consumer_proc(frame_queue, result_queue, consumer_idx, yolo_det_model_path, yolo_pose_model_path,
                  confidence_threshold, yolo_input_size_consumer, queue_monitor_local, stop_event_local,
                  logger_config_for_consumer: Optional[dict] = None, video_fps: float = 30.0):
    # --- Logger setup ---
    consumer_logger = logging.getLogger(f"S1_Consumer_{consumer_idx}_{os.getpid()}")

    det_model, pose_model = None, None
    try:
        # Pre-import torchvision to prevent circular import errors during YOLO warmup
        # This ensures torchvision.extension is fully initialized before lazy imports
        try:
            import torchvision
            consumer_logger.debug(f"[S1 Consumer-{consumer_idx}] Pre-imported torchvision v{torchvision.__version__}")
        except Exception as e:
            consumer_logger.warning(f"[S1 Consumer-{consumer_idx}] Failed to pre-import torchvision: {e}")

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
                                'class_name': det_model.names[int(box_data.cls[0])]
                            })

                # --- Step 2: Conditionally perform Pose Estimation ---
                # Run roughly once per second; safe for low FPS values
                poses = []  # Default to empty
                if frame_id % max(1, int(round(video_fps))) == 0:
                    pose_results = pose_model(frame, device=pose_device, verbose=False, imgsz=yolo_input_size_consumer, conf=confidence_threshold)
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
                gui_event_queue_arg: Optional[StdLibQueue] = None,
                max_fps_container: Optional[list] = None):
    results_dict = {}
    written_count = 0
    last_progress_update_time = time.time()
    first_result_received_time = None
    # --- Variables for instant FPS ---
    last_instant_fps_update_time = time.time()
    frames_since_last_instant_update = 0
    instant_fps = 0.0
    max_instant_fps = 0.0
    parent_logger.info(f"[S1 Logger] Expecting {expected_frames} frames. Writing to {output_file_local}")

    if progress_callback_local:
        # New signature: current, total, message, time_elapsed, avg_fps, instant_fps, eta_seconds
        progress_callback_local(0, expected_frames, "Logger starting...", 0.0, 0.0, 0.0, -1)

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
                frames_since_last_instant_update += 1
                if first_result_received_time is None:
                    first_result_received_time = time.time()
                    last_instant_fps_update_time = first_result_received_time

            current_time = time.time()

            # --- Instant FPS calculation every 1 second ---
            time_since_last_instant_update = current_time - last_instant_fps_update_time
            if time_since_last_instant_update >= 1.0:
                instant_fps = frames_since_last_instant_update / time_since_last_instant_update
                max_instant_fps = max(max_instant_fps, instant_fps)
                frames_since_last_instant_update = 0
                last_instant_fps_update_time = current_time

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

                # --- Calculate and send main progress ---
                time_elapsed = 0.0
                avg_fps = 0.0
                if first_result_received_time is not None:
                    time_elapsed = current_time - first_result_received_time
                    if time_elapsed > 0:
                        avg_fps = written_count / time_elapsed

                eta = (expected_frames - written_count) / avg_fps if avg_fps > 0 else 0
                progress_callback_local(written_count, expected_frames, "Detecting objects & poses...", time_elapsed, avg_fps, instant_fps, eta)
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

        # Validate the saved file
        if not _validate_preprocessed_file_completeness(output_file_local, expected_frames, parent_logger):
            parent_logger.warning(f"Preprocessed file validation failed. File may be incomplete: {output_file_local}")
    except Exception as e:
        parent_logger.error(f"Error writing output file '{output_file_local}': {e}", exc_info=True)

    if max_fps_container is not None:
        max_fps_container[0] = max_instant_fps
    parent_logger.info(f"[S1 Logger] Final Max FPS recorded: {max_instant_fps:.2f}")


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
        output_filename_override: Optional[str] = None,
        save_preprocessed_video_arg: bool = True,
        preprocessed_video_path_arg: Optional[str] = None,
        is_autotune_run_arg: bool = False
):
    process_logger = None
    fallback_config_for_subprocesses = None

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

    # Now, validate the output path.
    if output_filename_override:
        result_file_local = output_filename_override
        # Ensure the directory exists.
        output_dir = os.path.dirname(result_file_local)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
    else:
        # If no output path is provided, abort the process.
        process_logger.critical("Stage 1 Critical Error: No output file path was specified for the msgpack file. Aborting analysis.")
        if progress_callback:
            progress_callback(0, 0, "Stage 1 Error: No output path specified.", 0, 0, 0)
        return None, 0.0

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
        return None, 0.0

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

    # --- Preprocessed Video Logic (Now Optional) ---
    video_path_to_use = video_path_arg
    video_type_to_use = video_type_arg
    is_encoding_preprocessed_video = False

    # Default to multiple producers. Only use 1 if we are actively encoding.
    num_producers_effective = num_producers_arg

    if save_preprocessed_video_arg:
        process_logger.info("Preprocessed video generation/reuse is ENABLED for Stage 1.")
        
        # CRITICAL: Check if the input video is already preprocessed to prevent double preprocessing
        if _is_already_preprocessed_video(video_path_arg, process_logger):
            process_logger.error("REFUSING TO PROCESS: Input video appears to be already preprocessed!")
            process_logger.error("Double preprocessing can corrupt the output and waste resources.")
            process_logger.error("Please use the original video file for processing.")
            return None, 0.0
        
        if preprocessed_video_path_arg and os.path.exists(preprocessed_video_path_arg):
            process_logger.info(f"Found existing preprocessed video. Using: {preprocessed_video_path_arg}")
            video_path_to_use = preprocessed_video_path_arg
            video_type_to_use = 'flat'
        elif preprocessed_video_path_arg:
            process_logger.info(f"No existing preprocessed video found. Will encode to: {preprocessed_video_path_arg}")
            is_encoding_preprocessed_video = True
            num_producers_effective = 1  # Force producers to 1 for safe encoding to a single file
            preprocessed_dir = os.path.dirname(preprocessed_video_path_arg)
            if preprocessed_dir:
                os.makedirs(preprocessed_dir, exist_ok=True)
        else:
            # This case is a safeguard; the calling function should prevent it.
            process_logger.warning("Save preprocessed video was enabled, but no output path was provided. Disabling feature for this run.")
    else:
        process_logger.info("Preprocessed video generation/reuse is DISABLED for Stage 1.")

    class VPAppProxy:
        pass

    vp_app_proxy = VPAppProxy()
    vp_app_proxy.hardware_acceleration_method = hwaccel_method_arg
    vp_app_proxy.available_ffmpeg_hwaccels = hwaccel_avail_list_arg if hwaccel_avail_list_arg is not None else []
    # Force v360 unwarp method for offline tracking to avoid low FPS with GPU unwarp (metal/opengl)
    vp_app_proxy.app_settings = {'vr_unwarp_method': 'v360'}

    main_vp_for_info = VideoProcessor(app_instance=vp_app_proxy,
                                      fallback_logger_config={'logger_instance': process_logger})

    if not main_vp_for_info.open_video(video_path_to_use):
        return None, 0.0
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
        return None, 0.0

    frame_processing_queue = Queue(maxsize=constants.STAGE1_FRAME_QUEUE_MAXSIZE)
    yolo_result_queue = Queue()
    producers_list, consumers_list = [], []
    logger_p_thread = None
    max_fps_container = [0.0]

    try:
        # --- PROCESS CREATION ---
        frames_per_producer = total_frames_to_process // num_producers_effective
        extra_frames = total_frames_to_process % num_producers_effective
        current_frame = processing_start_frame
        for i in range(num_producers_effective):
            num_frames = frames_per_producer + (1 if i < extra_frames else 0)
            if num_frames > 0:
                encoding_path_arg = preprocessed_video_path_arg if is_encoding_preprocessed_video else None
                p_args = (i, video_path_to_use, yolo_input_size_arg, video_type_to_use, vr_input_format_arg, vr_fov_arg,
                          vr_pitch_arg, current_frame, num_frames, frame_processing_queue, queue_monitor,
                          stop_event_internal, hwaccel_method_arg, hwaccel_avail_list_arg,
                          fallback_config_for_subprocesses, is_encoding_preprocessed_video, encoding_path_arg)
                producers_list.append(Process(target=video_processor_producer_proc, args=p_args, daemon=True))
                current_frame += num_frames

        for i in range(num_consumers_arg):
            c_args = (frame_processing_queue, yolo_result_queue, i, yolo_model_path_arg, yolo_pose_model_path_arg,
                      confidence_threshold, yolo_input_size_arg, queue_monitor, stop_event_internal,
                      fallback_config_for_subprocesses, video_fps)
            consumers_list.append(Process(target=consumer_proc, args=c_args, daemon=True))

        logger_thread_args = (frame_processing_queue, yolo_result_queue, result_file_local, total_frames_to_process,
                              progress_callback, queue_monitor, stop_event_internal,
                              s1_start_time, process_logger, gui_event_queue_arg, max_fps_container)

        logger_p_thread = PyThread(target=logger_proc, args=logger_thread_args, daemon=True)

        # --- PROCESS STARTUP ---
        for p in producers_list:
            p.start()
        for p in consumers_list:
            p.start()
        logger_p_thread.start()

        # --- PROCESS JOINING AND SENTINEL LOGIC ---
        producers_finished = False
        all_procs = producers_list + consumers_list
        loop_start_time = time.time()
        ANALYSIS_TIMEOUT_SECONDS = 60  # 1 minute

        while any(p.is_alive() for p in all_procs):
            # Conditionally check for timeout ONLY if it's an autotuner run
            if is_autotune_run_arg and (time.time() - loop_start_time > ANALYSIS_TIMEOUT_SECONDS):
                process_logger.critical(
                    f"[S1 Lib] Autotuner analysis timed out after {ANALYSIS_TIMEOUT_SECONDS} seconds. "
                    "A subprocess likely hung. Aborting this test."
                )
                if not stop_event_internal.is_set():
                    stop_event_internal.set()
                break  # Exit the monitoring loop to trigger cleanup

            # If an abort is requested, break the loop immediately to trigger cleanup.
            if stop_event_internal.is_set():
                process_logger.warning("[S1 Lib] Abort detected in main monitoring loop.")
                break

            # Check if all producers are finished.
            if not producers_finished and not any(p.is_alive() for p in producers_list):
                process_logger.info("[S1 Lib] All producers finished. Sending sentinels to consumers.")
                for _ in range(len(consumers_list)):
                    queue_monitor.frame_queue_put(frame_processing_queue, None)
                producers_finished = True

            time.sleep(0.5)  # Wait between checks to avoid busy-waiting.

        # After the main loop, check if it was a natural finish (not an abort).
        if not stop_event_internal.is_set():
            process_logger.info("[S1 Lib] All consumers finished. Sending end-of-stream sentinel to logger.")
            try:
                yolo_result_queue.put((None, None), timeout=1.0)
            except Full:
                process_logger.warning("[S1 Lib] Timed out sending sentinel to logger queue.")

        # Final wait for the logger thread to finish writing the file.
        if logger_p_thread.is_alive():
            logger_p_thread.join()

        if stop_event_external.is_set():
            return None, 0.0

        final_max_fps = max_fps_container[0]
        process_logger.info(f"Stage 1 analysis completed. Final Max FPS: {final_max_fps:.2f}")

        # Final validation of both preprocessed video and msgpack file
        result_success = True
        if os.path.exists(result_file_local):
            if not _validate_preprocessed_file_completeness(result_file_local, total_frames_to_process, process_logger):
                process_logger.error(f"Stage 1 msgpack file validation failed: {result_file_local}")
                _cleanup_incomplete_file(result_file_local, process_logger)
                result_success = False
        else:
            result_success = False

        # Validate preprocessed video if it was created
        if is_encoding_preprocessed_video and preprocessed_video_path_arg:
            if os.path.exists(preprocessed_video_path_arg):
                if not _validate_preprocessed_video_completeness(preprocessed_video_path_arg, total_frames_to_process, video_fps, process_logger):
                    process_logger.error(f"Preprocessed video validation failed: {preprocessed_video_path_arg}")
                    _cleanup_incomplete_file(preprocessed_video_path_arg, process_logger)
                    # Don't fail the entire stage if just the preprocessed video is bad
                    # result_success = False

        return (result_file_local, final_max_fps) if result_success else (None, 0.0)


    except Exception as e:
        process_logger.critical(f"[S1 Lib] CRITICAL EXCEPTION in perform_yolo_analysis: {e}", exc_info=True)
        if not stop_event_internal.is_set():
            stop_event_internal.set()
        return None, 0.0
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

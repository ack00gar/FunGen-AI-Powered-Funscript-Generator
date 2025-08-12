import time
import threading
import subprocess
import json
import shlex
import numpy as np
import cv2
import platform
import sys
from typing import Optional, Iterator, Tuple, List, Dict, Any
import logging
import os
from collections import OrderedDict

from config import constants


try:
    from scipy.io import wavfile
    SCIPY_AVAILABLE_FOR_AUDIO = True
except ImportError:
    SCIPY_AVAILABLE_FOR_AUDIO = False

class VideoProcessor:
    def __init__(self, app_instance, tracker: Optional[type] = None, yolo_input_size=640,
                 video_type='auto', vr_input_format='he_sbs',  # Default VR to SBS Equirectangular
                 vr_fov=190, vr_pitch=-21,
                 fallback_logger_config: Optional[dict] = None,
                 cache_size: int = 50):
        self.app = app_instance
        self.tracker = tracker
        logger_assigned_correctly = False

        if app_instance and hasattr(app_instance, 'logger'):
            self.logger = app_instance.logger
            logger_assigned_correctly = True
        elif fallback_logger_config and fallback_logger_config.get('logger_instance'):
            self.logger = fallback_logger_config['logger_instance']
            logger_assigned_correctly = True

        if not logger_assigned_correctly:
            logger_name = f"{self.__class__.__name__}_{os.getpid()}"
            self.logger = logging.getLogger(logger_name)

            if not self.logger.hasHandlers():
                log_level = logging.INFO
                if fallback_logger_config and fallback_logger_config.get('log_level') is not None:
                    log_level = fallback_logger_config['log_level']
                self.logger.setLevel(log_level)

                handler_to_add = None
                if fallback_logger_config and fallback_logger_config.get('log_file'):
                    handler_to_add = logging.FileHandler(fallback_logger_config['log_file'])
                else:
                    handler_to_add = logging.StreamHandler()

                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(process)d - %(message)s')
                handler_to_add.setFormatter(formatter)
                self.logger.addHandler(handler_to_add)

        self.logger.info(f"VideoProcessor logger '{self.logger.name}' initialized.")

        self.video_path = ""
        self._active_video_source_path: str = ""
        self.video_info = {}
        self.ffmpeg_process: Optional[subprocess.Popen] = None  # Main output process (pipe2 if active)
        self.ffmpeg_pipe1_process: Optional[subprocess.Popen] = None  # Pipe1 process, if active
        self.is_processing = False
        self.pause_event = threading.Event()
        self.processing_thread = None
        self.current_frame = None
        self.fps = 0.0
        self.target_fps = 30
        self.actual_fps = 0
        self.last_fps_update_time = time.time()
        self.frames_for_fps_calc = 0
        self.frame_lock = threading.Lock()
        self.seek_request_frame_index = None
        self.total_frames = 0
        self.current_frame_index = 0
        self.current_stream_start_frame_abs = 0
        self.frames_read_from_current_stream = 0

        self.yolo_input_size = yolo_input_size
        self.video_type_setting = video_type
        self.vr_input_format = vr_input_format
        self.vr_fov = vr_fov
        self.vr_pitch = vr_pitch

        self.determined_video_type = None
        self.ffmpeg_filter_string = ""
        self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3

        self.stop_event = threading.Event()
        self.processing_start_frame_limit = 0
        self.processing_end_frame_limit = -1

        # --- State for context-aware tracking ---
        self.last_processed_chapter_id: Optional[str] = None

        self.enable_tracker_processing = False
        if self.tracker is None:
            if self.logger:
                self.logger.info("No tracker provided. Tracker processing will be disabled.")
        else:
            self.logger.debug("Tracker is available, but processing is DISABLED by default. An explicit call is needed to enable it.")

        # Frame Caching
        self.frame_cache = OrderedDict()
        self.frame_cache_max_size = cache_size
        self.frame_cache_lock = threading.Lock()
        self.batch_fetch_size = 120

    def _clear_cache(self):
        with self.frame_cache_lock:
            if self.frame_cache is not None:
                try:
                    if self.frame_cache is not None:
                        cache_len = len(self.frame_cache)
                    else:
                        cache_len = 0
                except Exception:
                    cache_len = 0
                if cache_len > 0:
                    self.logger.debug(f"Clearing frame cache (had {cache_len} items).")
                    self.frame_cache.clear()

    def set_active_video_type_setting(self, video_type: str):
        if video_type not in ['auto', '2D', 'VR']:
            self.logger.warning(f"Invalid video_type: {video_type}.")
            return
        if self.video_type_setting != video_type:
            self.video_type_setting = video_type
            self.logger.info(f"Video type setting changed to: {self.video_type_setting}.")

    def set_active_yolo_input_size(self, size: int):
        if size <= 0:
            self.logger.warning(f"Invalid yolo_input_size: {size}.")
            return
        if self.yolo_input_size != size:
            self.yolo_input_size = size
            self.logger.info(f"YOLO input size changed to: {self.yolo_input_size}.")
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3

    def set_active_vr_parameters(self, fov: Optional[int] = None, pitch: Optional[int] = None, input_format: Optional[str] = None):
        changed = False
        if fov is not None and self.vr_fov != fov:
            self.vr_fov = fov
            changed = True
            self.logger.info(f"VR FOV changed to: {self.vr_fov}.")
        if pitch is not None and self.vr_pitch != pitch:
            self.vr_pitch = pitch
            changed = True
            self.logger.info(f"VR Pitch changed to: {self.vr_pitch}.")
        if input_format is not None and self.vr_input_format != input_format:
            valid_formats = ["he", "fisheye", "he_sbs", "fisheye_sbs", "he_tb", "fisheye_tb"]
            if input_format in valid_formats:
                self.vr_input_format = input_format
                self.video_type_setting = 'VR'
                changed = True
                self.logger.info(f"VR Input Format changed by UI to: {self.vr_input_format}.")
            else:
                self.logger.warning(f"Unknown VR input format '{input_format}'. Not changed. Valid: {valid_formats}")

    def set_tracker_processing_enabled(self, enable: bool):
        if enable and self.tracker is None:
            self.logger.warning("Cannot enable tracker processing because no tracker is available.")
            self.enable_tracker_processing = False
        else:
            self.enable_tracker_processing = enable
    
    def set_active_video_source(self, video_source_path: str):
        """
        Update the active video source path (e.g., to switch to preprocessed video).
        
        Args:
            video_source_path: Path to the video file to use as the active source
        """
        if not os.path.exists(video_source_path):
            self.logger.warning(f"Cannot set active video source: file does not exist: {video_source_path}")
            return
            
        old_source = self._active_video_source_path
        self._active_video_source_path = video_source_path
        
        # Update the FFmpeg filter string since preprocessed videos don't need filtering
        self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
        
        source_type = "preprocessed" if self._is_using_preprocessed_video() else "original"
        self.logger.info(f"Active video source updated: {os.path.basename(video_source_path)} ({source_type})")
        
        # Notify about the change
        if old_source != video_source_path:
            if self._is_using_preprocessed_video():
                self.logger.info("Now using preprocessed video - filters disabled for optimal performance")
            else:
                self.logger.info("Now using original video - filters will be applied on-the-fly")

    def open_video(self, video_path: str, from_project_load: bool = False) -> bool:
        self.stop_processing()
        self.video_path = video_path # This will always be the ORIGINAL video path
        self._clear_cache()
        self.video_info = self._get_video_info(video_path)
        if not self.video_info or self.video_info.get("total_frames", 0) == 0:
            self.logger.warning(f"Failed to get valid video info for {video_path}")
            self.video_path = ""
            self.video_info = {}
            return False

        # --- Set the active source path ---
        self._active_video_source_path = self.video_path  # Default to original
        preprocessed_path = None
        # Proactively search for the preprocessed file for the *current* video
        if self.app and hasattr(self.app, 'file_manager'):
            potential_preprocessed_path = self.app.file_manager.get_output_path_for_file(self.video_path, "_preprocessed.mkv")
            if os.path.exists(potential_preprocessed_path):
                preprocessed_path = potential_preprocessed_path
                # Also update the file_manager's state to be consistent
                self.app.file_manager.preprocessed_video_path = preprocessed_path

        if preprocessed_path:
            # Always validate the preprocessed file before using it
            self.logger.info(f"Found potential preprocessed file: {os.path.basename(preprocessed_path)}. Verifying...")

            # Basic validation first
            preprocessed_info = self._get_video_info(preprocessed_path)
            original_frames = self.video_info.get("total_frames", 0)
            original_fps = self.video_info.get("fps", 30.0)
            preprocessed_frames = preprocessed_info.get("total_frames", -1) if preprocessed_info else -1

            # Use comprehensive validation
            is_valid_preprocessed = self._validate_preprocessed_video(preprocessed_path, original_frames, original_fps)

            if is_valid_preprocessed and preprocessed_frames >= original_frames > 0:
                self._active_video_source_path = preprocessed_path
                self.logger.info(f"Preprocessed video validation passed. Using as active source.")
            else:
                self.logger.warning(
                    f"Preprocessed file is incomplete or invalid ({preprocessed_frames}/{original_frames} frames). "
                    f"Falling back to original video. Re-run Stage 1 with 'Save Preprocessed Video' enabled to fix."
                )
                # Clean up the invalid preprocessed file
                self._cleanup_invalid_preprocessed_file(preprocessed_path)

        if self._active_video_source_path == preprocessed_path:
            self.logger.info(f"VideoProcessor will use preprocessed video as its active source.")
        else:
            self.logger.info(f"VideoProcessor will use original video as its active source.")

        self._update_video_parameters()

        self.fps = self.video_info['fps']
        self.total_frames = self.video_info['total_frames']
        self.set_target_fps(self.fps)
        self.current_frame_index = 0
        self.frames_read_from_current_stream = 0
        self.current_stream_start_frame_abs = 0
        self.stop_event.clear()
        self.seek_request_frame_index = None
        self.current_frame = self._get_specific_frame(0)

        if self.tracker:
            reset_reason = "project_load_preserve_actions" if from_project_load else None
            self.tracker.reset(reason=reset_reason)

        active_source_name = os.path.basename(self._active_video_source_path)
        source_type = "preprocessed" if self._active_video_source_path != video_path else "original"
        self.logger.info(
            f"Opened: {active_source_name} ({source_type}, {self.determined_video_type}, "
            f"format: {self.vr_input_format if self.determined_video_type == 'VR' else 'N/A'}), "
            f"{self.total_frames}fr, {self.fps:.2f}fps, {self.video_info.get('bit_depth', 'N/A')}bit)")
        return True

    def _update_video_parameters(self):
        """
        [NEW HELPER] Consolidates logic for determining video type and building the FFmpeg filter string.
        Called from open_video and reapply_video_settings.
        """
        if not self.video_info:
            return

        width = self.video_info.get('width', 0)
        height = self.video_info.get('height', 0)
        is_sbs_resolution = width > 1000 and 1.8 * height <= width <= 2.2 * height
        is_tb_resolution = height > 1000 and 1.8 * width <= height <= 2.2 * width

        if self.video_type_setting == 'auto':
            upper_video_path = self.video_path.upper()
            vr_keywords = ['VR', '_180', '_360', 'SBS', '_TB', 'FISHEYE', 'EQUIRECTANGULAR', 'LR_', 'Oculus', '_3DH', 'MKX200']
            has_vr_keyword = any(kw in upper_video_path for kw in vr_keywords)

            self.determined_video_type = 'VR' if is_sbs_resolution or is_tb_resolution or has_vr_keyword else '2D'
            self.logger.info(
                f"Auto-detected video type: {self.determined_video_type} (SBS Res: {is_sbs_resolution}, TB Res: {is_tb_resolution}, Keyword: {has_vr_keyword})")
        else:
            self.determined_video_type = self.video_type_setting
            self.logger.info(f"Using configured video type: {self.determined_video_type}")

        if self.determined_video_type == 'VR':
            # Only perform auto-detection of the specific VR format if the main setting is 'auto'.
            # If the user has explicitly set the type to 'VR', this block is skipped,
            # thereby preserving the manually selected format.
            if self.video_type_setting == 'auto':
                suggested_base = 'he'
                suggested_layout = '_sbs'
                upper_video_path = self.video_path.upper()

                if is_tb_resolution:
                    suggested_layout = '_tb'
                    self.logger.info("Resolution (H > 1.8*W) suggests Top-Bottom (TB) layout.")

                fisheye_keywords = ['FISHEYE', 'MKX', 'RF52']
                if any(kw in upper_video_path for kw in fisheye_keywords):
                    suggested_base = 'fisheye'
                    self.logger.info("Filename keyword suggests 'fisheye' base format.")

                tb_keywords = ['_TB', 'TB_', 'TOPBOTTOM', 'OVERUNDER', '_OU', 'OU_']
                if any(kw in upper_video_path for kw in tb_keywords):
                    suggested_layout = '_tb'
                    self.logger.info("Filename keyword confirms 'Top-Bottom' layout.")
                elif 'SBS' in upper_video_path:
                    suggested_layout = '_sbs'
                    self.logger.info("Filename keyword confirms 'Side-by-Side' layout.")

                final_suggested_vr_input_format = f"{suggested_base}{suggested_layout}"

                self.logger.info(
                    f"Auto-detection suggests VR format: {final_suggested_vr_input_format} for '{os.path.basename(self.video_path)}'. Setting it."
                )
                self.vr_input_format = final_suggested_vr_input_format

                if 'MKX' in upper_video_path and 'fisheye' in self.vr_input_format and self.vr_fov != 200:
                    self.logger.info(
                        f"Filename suggests VR FOV: 200 (MKX) for fisheye. Overriding current: {self.vr_fov}")
                    self.vr_fov = 200

        self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
        self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
        self.logger.info(f"Frame size bytes updated to: {self.frame_size_bytes} for YOLO size {self.yolo_input_size}")

    def reapply_video_settings(self):
        if not self.video_path or not self.video_info:
            self.logger.info("No video loaded. Settings will apply when a video is opened.")
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
            return

        self.logger.info(f"Reapplying video settings (self.vr_input_format is currently: {self.vr_input_format})")
        was_processing = self.is_processing
        stored_frame_index = self.current_frame_index
        stored_end_limit = self.processing_end_frame_limit
        self.stop_processing()
        self._clear_cache()

        # [REDUNDANCY REMOVED] - Call the new helper method
        self._update_video_parameters()

        self.logger.info(f"Attempting to fetch frame {stored_frame_index} with new settings.")
        new_frame = self._get_specific_frame(stored_frame_index)
        if new_frame is not None:
            with self.frame_lock:
                self.current_frame = new_frame
            self.logger.info(f"Successfully fetched frame {self.current_frame_index} with new settings.")
        else:
            self.logger.warning(f"Failed to get frame {stored_frame_index} with new settings.")

        if was_processing:
            self.logger.info("Restarting processing with new settings...")
            self.start_processing(start_frame=self.current_frame_index, end_frame=stored_end_limit)
        else:
            self.logger.info("Settings applied. Video remains paused/stopped.")
        self.logger.info("Video settings reapplication complete.")

    def get_frames_batch(self, start_frame_num: int, num_frames_to_fetch: int) -> Dict[int, np.ndarray]:
        """
        Fetches a batch of frames using FFmpeg.
        This method now supports 2-pipe 10-bit CUDA processing.
        """
        frames_batch: Dict[int, np.ndarray] = {}
        if not self.video_path or not self.video_info or self.video_info.get('fps', 0) <= 0 or num_frames_to_fetch <= 0:
            self.logger.warning("get_frames_batch: Video not properly opened or invalid params.")
            return frames_batch

        local_p1_proc: Optional[subprocess.Popen] = None
        local_p2_proc: Optional[subprocess.Popen] = None

        start_time_seconds = start_frame_num / self.video_info['fps']
        common_ffmpeg_prefix = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']

        try:
            if self._is_10bit_cuda_pipe_needed():
                self.logger.debug(
                    f"get_frames_batch: Using 2-pipe FFmpeg for {num_frames_to_fetch} frames from {start_frame_num} (10-bit CUDA).")
                video_height_for_crop = self.video_info.get('height', 0)
                if video_height_for_crop <= 0:
                    self.logger.error("get_frames_batch (10-bit CUDA pipe 1): video height unknown.")
                    return frames_batch

                pipe1_vf = f"crop={int(video_height_for_crop)}:{int(video_height_for_crop)}:0:0,scale_cuda=1000:1000"
                cmd1 = common_ffmpeg_prefix[:]
                cmd1.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
                if start_time_seconds > 0.001: cmd1.extend(['-ss', str(start_time_seconds)])
                cmd1.extend(['-i', self._active_video_source_path, '-an', '-sn', '-vf', pipe1_vf])
                cmd1.extend(['-frames:v', str(num_frames_to_fetch)])
                cmd1.extend(['-c:v', 'hevc_nvenc', '-preset', 'fast', '-qp', '0', '-f', 'matroska', 'pipe:1'])

                cmd2 = common_ffmpeg_prefix[:]
                cmd2.extend(['-hwaccel', 'cuda', '-i', 'pipe:0', '-an', '-sn'])
                effective_vf_pipe2 = self.ffmpeg_filter_string
                if not effective_vf_pipe2: effective_vf_pipe2 = f"scale={self.yolo_input_size}:{self.yolo_input_size}"
                cmd2.extend(['-vf', effective_vf_pipe2])
                cmd2.extend(['-frames:v', str(num_frames_to_fetch)])
                cmd2.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"get_frames_batch Pipe 1 CMD: {' '.join(shlex.quote(str(x)) for x in cmd1)}")
                    self.logger.debug(f"get_frames_batch Pipe 2 CMD: {' '.join(shlex.quote(str(x)) for x in cmd2)}")

                # Windows fix: prevent terminal windows from spawning
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                local_p1_proc = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creation_flags)
                if local_p1_proc.stdout is None: raise IOError("get_frames_batch: Pipe 1 stdout is None.")

                local_p2_proc = subprocess.Popen(cmd2, stdin=local_p1_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_size_bytes * min(num_frames_to_fetch, 20), creationflags=creation_flags)
                local_p1_proc.stdout.close()

            else:  # Standard single FFmpeg process
                self.logger.debug(
                    f"get_frames_batch: Using single-pipe FFmpeg for {num_frames_to_fetch} frames from {start_frame_num}.")
                hwaccel_cmd_list = self._get_ffmpeg_hwaccel_args()
                ffmpeg_input_options = hwaccel_cmd_list[:]
                if start_time_seconds > 0.001: ffmpeg_input_options.extend(['-ss', str(start_time_seconds)])
                cmd_single = common_ffmpeg_prefix + ffmpeg_input_options + ['-i', self._active_video_source_path, '-an', '-sn']
                effective_vf = self.ffmpeg_filter_string
                if not effective_vf: effective_vf = f"scale={self.yolo_input_size}:{self.yolo_input_size}"
                cmd_single.extend(['-vf', effective_vf])
                cmd_single.extend(['-frames:v', str(num_frames_to_fetch)])
                cmd_single.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"get_frames_batch CMD (single pipe): {' '.join(shlex.quote(str(x)) for x in cmd_single)}")
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                local_p2_proc = subprocess.Popen(cmd_single, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_size_bytes * min(num_frames_to_fetch, 20), creationflags=creation_flags)

            if not local_p2_proc or local_p2_proc.stdout is None:
                self.logger.error("get_frames_batch: Output FFmpeg process or its stdout is None.")
                return frames_batch

            for i in range(num_frames_to_fetch):
                raw_frame_data = local_p2_proc.stdout.read(self.frame_size_bytes)
                if len(raw_frame_data) < self.frame_size_bytes:
                    p2_stderr_content = local_p2_proc.stderr.read().decode(
                        errors='ignore') if local_p2_proc.stderr else ""
                    self.logger.warning(
                        f"get_frames_batch: Incomplete data for frame {start_frame_num + i} (read {len(raw_frame_data)}/{self.frame_size_bytes}). P2 Stderr: {p2_stderr_content.strip()}")
                    if local_p1_proc and local_p1_proc.stderr:
                        p1_stderr_content = local_p1_proc.stderr.read().decode(errors='ignore')
                        self.logger.warning(f"get_frames_batch: P1 Stderr: {p1_stderr_content.strip()}")
                    break
                frames_batch[start_frame_num + i] = np.frombuffer(raw_frame_data, dtype=np.uint8).reshape(
                    self.yolo_input_size, self.yolo_input_size, 3)

        except Exception as e:
            self.logger.error(f"get_frames_batch: Error fetching batch @{start_frame_num}: {e}", exc_info=True)
        finally:
            # [REDUNDANCY REMOVED] - Use the new helper method for termination
            if local_p1_proc:
                self._terminate_process(local_p1_proc, "Batch Pipe 1")
            if local_p2_proc:
                self._terminate_process(local_p2_proc, "Batch Pipe 2/Main")

        self.logger.debug(
            f"get_frames_batch: Complete. Got {len(frames_batch)} frames for start {start_frame_num} (requested {num_frames_to_fetch}).")
        return frames_batch

    def _get_specific_frame(self, frame_index_abs: int) -> Optional[np.ndarray]:
        if not self.video_path or not self.video_info or self.video_info.get('fps', 0) <= 0:
            self.logger.warning("Cannot get frame: video not loaded/invalid FPS.")
            self.current_frame_index = frame_index_abs
            return None

        with self.frame_cache_lock:
            if frame_index_abs in self.frame_cache:
                self.logger.debug(f"Cache HIT for frame {frame_index_abs}")
                frame = self.frame_cache[frame_index_abs]
                self.frame_cache.move_to_end(frame_index_abs)
                self.current_frame_index = frame_index_abs
                return frame

        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"Cache MISS for frame {frame_index_abs}. Attempting batch fetch using get_frames_batch (batch size: {self.batch_fetch_size}).")

        batch_start_frame = max(0, frame_index_abs - self.batch_fetch_size // 2)
        if self.total_frames > 0:
            effective_end_frame_for_batch_calc = self.total_frames - 1
            if batch_start_frame + self.batch_fetch_size - 1 > effective_end_frame_for_batch_calc:
                batch_start_frame = max(0, effective_end_frame_for_batch_calc - self.batch_fetch_size + 1)

        num_frames_to_fetch_actual = self.batch_fetch_size
        if self.total_frames > 0:
            num_frames_to_fetch_actual = min(self.batch_fetch_size, self.total_frames - batch_start_frame)

        if num_frames_to_fetch_actual < 1 and self.total_frames > 0:
            num_frames_to_fetch_actual = 1
        elif num_frames_to_fetch_actual < 1 and self.total_frames == 0:
            num_frames_to_fetch_actual = self.batch_fetch_size

        fetched_batch = self.get_frames_batch(batch_start_frame, num_frames_to_fetch_actual)

        retrieved_frame: Optional[np.ndarray] = None
        with self.frame_cache_lock:
            for idx, frame_data in fetched_batch.items():
                if len(self.frame_cache) >= self.frame_cache_max_size:
                    try:
                        self.frame_cache.popitem(last=False)
                    except KeyError:
                        pass
                self.frame_cache[idx] = frame_data
                if idx == frame_index_abs:
                    retrieved_frame = frame_data

            if retrieved_frame is not None and frame_index_abs in self.frame_cache:
                self.frame_cache.move_to_end(frame_index_abs)

        self.current_frame_index = frame_index_abs
        if retrieved_frame is not None:
            self.logger.debug(f"Successfully retrieved frame {frame_index_abs} via get_frames_batch and cached.")
            return retrieved_frame
        else:
            self.logger.warning(
                f"Failed to retrieve specific frame {frame_index_abs} after batch fetch. FFmpeg might have failed or frame out of bounds.")
            with self.frame_cache_lock:
                if frame_index_abs in self.frame_cache:
                    self.logger.debug(f"Retrieved frame {frame_index_abs} from cache on fallback check.")
                    return self.frame_cache[frame_index_abs]
            return None

    @staticmethod
    def get_video_type_heuristic(video_path: str) -> str:
        """
        A lightweight heuristic to guess the video type (2D/VR) and format (SBS/TB)
        without fully opening the video. Uses ffprobe for metadata.
        """
        if not os.path.exists(video_path):
            return "Unknown"

        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=width,height', '-of', 'json', video_path]
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, timeout=5, creationflags=creation_flags)
            data = json.loads(result.stdout)
            stream_info = data.get('streams', [{}])[0]
            width = int(stream_info.get('width', 0))
            height = int(stream_info.get('height', 0))
        except Exception:
            return "Unknown"

        if width == 0 or height == 0:
            return "Unknown"

        is_sbs_resolution = width > 1000 and 1.8 * height <= width <= 2.2 * height
        is_tb_resolution = height > 1000 and 1.8 * width <= height <= 2.2 * width
        upper_video_path = video_path.upper()
        vr_keywords = ['VR', '_180', '_360', 'SBS', '_TB', 'FISHEYE', 'EQUIRECTANGULAR', 'LR_', 'Oculus', '_3DH', 'MKX200']
        has_vr_keyword = any(kw in upper_video_path for kw in vr_keywords)

        if not (is_sbs_resolution or is_tb_resolution or has_vr_keyword):
            return "2D"

        # If VR, guess the specific format
        suggested_base = 'he'
        suggested_layout = '_sbs'
        if is_tb_resolution or any(kw in upper_video_path for kw in ['_TB', 'TB_', 'TOPBOTTOM', 'OVERUNDER', '_OU', 'OU_']):
            suggested_layout = '_tb'
        if any(kw in upper_video_path for kw in ['FISHEYE', 'MKX', 'RF52']):
            suggested_base = 'fisheye'

        return f"VR ({suggested_base}{suggested_layout})"

    def _get_video_info(self, filename):
        # TODO: Add ffprobe detection and metadata extraction for YUV videos. Pass metadata to cv2 so it can use the correct decoder. Use metadata + cv2.cvtColor to convert to RGB.
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries',
               'stream=width,height,r_frame_rate,nb_frames,avg_frame_rate,duration,codec_type,pix_fmt,bits_per_raw_sample',
               '-show_entries', 'format=duration,size,bit_rate', '-of', 'json', filename]
        try:
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, creationflags=creation_flags)
            data = json.loads(result.stdout)
            stream_info = data.get('streams', [{}])[0]
            format_info = data.get('format', {})

            fr_str = stream_info.get('r_frame_rate', stream_info.get('avg_frame_rate', '30/1'))
            num, den = map(float, fr_str.split('/')) if '/' in fr_str else (float(fr_str), 1.0)
            fps = num / den if den != 0 else 30.0

            dur_str = stream_info.get('duration', format_info.get('duration', '0'))
            duration = float(dur_str) if dur_str and dur_str != 'N/A' else 0.0

            tf_str = stream_info.get('nb_frames')
            total_frames = int(tf_str) if tf_str and tf_str != 'N/A' else 0
            if total_frames == 0 and duration > 0 and fps > 0: total_frames = int(duration * fps)

            # --- New Fields ---
            file_size_bytes = int(format_info.get('size', 0))
            bitrate_bps = int(format_info.get('bit_rate', 0))
            file_name = os.path.basename(filename)

            # VFR check
            r_frame_rate_str = stream_info.get('r_frame_rate', '0/0')
            avg_frame_rate_str = stream_info.get('avg_frame_rate', '0/0')
            is_vfr = r_frame_rate_str != avg_frame_rate_str

            has_audio_ffprobe = False
            cmd_audio_check = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                               '-show_entries', 'stream=codec_type', '-of', 'json', filename]
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                result_audio = subprocess.run(cmd_audio_check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, creationflags=creation_flags)
                audio_data = json.loads(result_audio.stdout)
                if audio_data.get('streams') and audio_data['streams'][0].get('codec_type') == 'audio':
                    has_audio_ffprobe = True
            except Exception:
                pass

            if total_frames == 0:
                self.logger.warning("ffprobe gave 0 frames, trying OpenCV count...")
                cap = cv2.VideoCapture(filename)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if fps <= 0: fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
                    if duration <= 0 and total_frames > 0 and fps > 0: duration = total_frames / fps
                    cap.release()
                else:
                    self.logger.error(f"OpenCV could not open video file: {filename}")

            bit_depth = 8
            bits_per_raw_sample_str = stream_info.get('bits_per_raw_sample')
            if bits_per_raw_sample_str and bits_per_raw_sample_str != 'N/A':
                try:
                    bit_depth = int(bits_per_raw_sample_str)
                except ValueError:
                    self.logger.warning(f"Could not parse bits_per_raw_sample: {bits_per_raw_sample_str}")
            else:
                pix_fmt = stream_info.get('pix_fmt', '').lower()
                # Check for higher bit depths first
                if any(fmt in pix_fmt for fmt in ['12le', 'p012', '12be']):
                    bit_depth = 12
                elif any(fmt in pix_fmt for fmt in ['10le', 'p010', '10be']):
                    bit_depth = 10

            self.logger.info(
                f"Detected video properties: width={stream_info.get('width', 0)}, height={stream_info.get('height', 0)}, fps={fps:.2f}, bit_depth={bit_depth}")

            return {"duration": duration, "total_frames": total_frames, "fps": fps,
                    "width": int(stream_info.get('width', 0)), "height": int(stream_info.get('height', 0)),
                    "has_audio": has_audio_ffprobe, "bit_depth": bit_depth,
                    "file_size": file_size_bytes, "bitrate": bitrate_bps,
                    "is_vfr": is_vfr, "filename": file_name
                    }
        except Exception as e:
            self.logger.error(f"Error in _get_video_info for {filename}: {e}")
            return None

    def get_audio_waveform(self, num_samples: int = 1000) -> Optional[np.ndarray]:
        """
        [OPTIMIZED] Generates an audio waveform by streaming audio data directly
        from FFmpeg into memory, avoiding the need for a temporary file.
        """
        if not self.video_path or not self.video_info.get("has_audio"):
            self.logger.info("No video loaded or video has no audio stream for waveform generation.")
            return None
        if not SCIPY_AVAILABLE_FOR_AUDIO:
            self.logger.warning("Scipy is not available. Cannot generate audio waveform.")
            return None

        process = None
        try:
            ffmpeg_cmd = [
                'ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error',
                '-i', self.video_path,
                '-vn', '-ac', '1', '-ar', '44100', '-c:a', 'pcm_s16le', '-f', 's16le', 'pipe:1'
            ]
            self.logger.info(f"Extracting audio for waveform via memory pipe: {' '.join(shlex.quote(str(x)) for x in ffmpeg_cmd)}")

            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creation_flags)
            raw_audio, stderr = process.communicate(timeout=60)

            if process.returncode != 0:
                self.logger.error(f"FFmpeg failed to extract audio: {stderr.decode(errors='ignore')}")
                return None
            if not raw_audio:
                self.logger.error("FFmpeg produced no audio data.")
                return None

            data = np.frombuffer(raw_audio, dtype=np.int16)

            if data.size == 0:
                self.logger.warning("Audio data is empty after reading from FFmpeg pipe.")
                return None

            num_frames_audio = len(data)
            step = max(1, num_frames_audio // num_samples)
            waveform = [np.max(np.abs(data[i:i + step])) for i in range(0, num_frames_audio, step)]
            waveform_np = np.array(waveform)
            max_val = np.max(waveform_np)
            if max_val > 0:
                waveform_np = waveform_np / max_val

            self.logger.info(f"Generated waveform with {len(waveform_np)} samples.")
            return waveform_np

        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg timed out during audio extraction.")
            if process:
                process.kill()
                process.communicate()
            return None
        except Exception as e:
            self.logger.error(f"Error generating audio waveform: {e}", exc_info=True)
            return None

    def _is_10bit_cuda_pipe_needed(self) -> bool:
        # TODO: Add bitshift processing for 10-bit videos (fast 10-bit to 8-bit conversion).
        # Optional: Scale to 640x640 on GPU using tensorrt. This will not use lanczos. So if Lanczos is absolutely necessary, you will have to use other solution.
        """Checks if the special 2-pipe FFmpeg command for 10-bit CUDA should be used."""
        if not self.video_info:
            return False

        is_high_bit_depth = self.video_info.get('bit_depth', 8) > 8
        hwaccel_args = self._get_ffmpeg_hwaccel_args()
        # [OPTIMIZED] Simpler check
        is_cuda_hwaccel = 'cuda' in hwaccel_args

        if is_high_bit_depth and is_cuda_hwaccel:
            self.logger.info("Conditions for 10-bit CUDA pipe met.")
            return True
        return False

    def _is_using_preprocessed_video(self) -> bool:
        """Checks if the active video source is a preprocessed file."""
        is_using_preprocessed_by_path_diff = self._active_video_source_path != self.video_path
        is_preprocessed_by_name = self._active_video_source_path.endswith("_preprocessed.mkv")
        return is_using_preprocessed_by_path_diff or is_preprocessed_by_name

    def _needs_hw_download(self) -> bool:
        """Determines if the FFmpeg filter chain requires a 'hwdownload' filter."""
        current_hw_args = self._get_ffmpeg_hwaccel_args()
        if '-hwaccel_output_format' in current_hw_args:
            try:
                idx = current_hw_args.index('-hwaccel_output_format')
                hw_output_format = current_hw_args[idx + 1]
                # These formats are on the GPU and need to be downloaded for CPU-based filters.
                if hw_output_format in ['cuda', 'nv12', 'p010le', 'qsv', 'vaapi', 'd3d11va', 'dxva2_vld']:
                    return True
            except (ValueError, IndexError):
                self.logger.warning("Could not properly parse -hwaccel_output_format from hw_args.")
        return False

    def _get_2d_video_filters(self) -> List[str]:
        """Builds the list of FFmpeg filter segments for standard 2D video."""
        return [
            f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease",
            f"pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
        ]

    def _get_vr_video_filters(self) -> List[str]:
        """Builds the list of FFmpeg filter segments for VR video, including cropping and v360."""
        if not self.video_info:
            return []

        original_width = self.video_info.get('width', 0)
        original_height = self.video_info.get('height', 0)
        v_h_FOV = 90  # Default vertical and horizontal FOV for the output projection

        vr_filters = []
        is_sbs_format = '_sbs' in self.vr_input_format
        is_tb_format = '_tb' in self.vr_input_format

        if is_sbs_format and original_width > 0 and original_height > 0:
            crop_w = original_width / 2
            crop_h = original_height
            vr_filters.append(f"crop={int(crop_w)}:{int(crop_h)}:0:0")
            self.logger.info(f"Applying SBS pre-crop: w={int(crop_w)} h={int(crop_h)} x=0 y=0")
        elif is_tb_format and original_width > 0 and original_height > 0:
            crop_w = original_width
            crop_h = original_height / 2
            vr_filters.append(f"crop={int(crop_w)}:{int(crop_h)}:0:0")
            self.logger.info(f"Applying TB pre-crop: w={int(crop_w)} h={int(crop_h)} x=0 y=0")

        base_v360_input_format = self.vr_input_format.replace('_sbs', '').replace('_tb', '')
        v360_filter_core = (
            f"v360={base_v360_input_format}:in_stereo=0:output=sg:"
            f"iv_fov={self.vr_fov}:ih_fov={self.vr_fov}:"
            f"d_fov={self.vr_fov}:"
            f"v_fov={v_h_FOV}:h_fov={v_h_FOV}:"
            f"pitch={self.vr_pitch}:yaw=0:roll=0:"
            f"w={self.yolo_input_size}:h={self.yolo_input_size}:interp=lanczos"
        )
        vr_filters.append(v360_filter_core)
        return vr_filters

    def _build_ffmpeg_filter_string(self) -> str:
        if self._is_using_preprocessed_video():
            self.logger.info(f"Using preprocessed video source ('{os.path.basename(self._active_video_source_path)}'). No FFmpeg filters will be applied.")
            return ""

        if not self.video_info:
            return ''

        software_filter_segments = []
        if self.determined_video_type == '2D':
            software_filter_segments = self._get_2d_video_filters()
        elif self.determined_video_type == 'VR':
            software_filter_segments = self._get_vr_video_filters()

        final_filter_chain_parts = []
        if self._needs_hw_download() and software_filter_segments:
            final_filter_chain_parts.extend(["hwdownload", "format=nv12"])
            self.logger.info("Prepending 'hwdownload,format=nv12' to the software filter chain.")

        final_filter_chain_parts.extend(software_filter_segments)
        ffmpeg_filter = ",".join(final_filter_chain_parts)

        self.logger.info(
            f"Built FFmpeg filter (effective for single pipe, or pipe2 of 10bit-CUDA): {ffmpeg_filter if ffmpeg_filter else 'No explicit filter, direct output.'}")
        return ffmpeg_filter

    def _get_ffmpeg_hwaccel_args(self) -> List[str]:
        """Determines FFmpeg hardware acceleration arguments based on app settings."""
        hwaccel_args: List[str] = []
        selected_hwaccel = getattr(self.app, 'hardware_acceleration_method', 'auto') if self.app else "none"
        available_on_app = getattr(self.app, 'available_ffmpeg_hwaccels', []) if self.app else []

        system = platform.system().lower()
        self.logger.debug(
            f"Determining HWAccel. Selected: '{selected_hwaccel}', OS: {system}, App Available: {available_on_app}")

        if selected_hwaccel == "auto":
            if system == 'darwin' and 'videotoolbox' in available_on_app:
                hwaccel_args = ['-hwaccel', 'videotoolbox']
                self.logger.debug("Auto-selected 'videotoolbox' for macOS.")
            # [REDUNDANCY REMOVED] - Combined Linux/Windows logic
            elif system in ['linux', 'windows']:
                if 'nvdec' in available_on_app or 'cuda' in available_on_app:
                    chosen_nvidia_accel = 'nvdec' if 'nvdec' in available_on_app else 'cuda'
                    hwaccel_args = ['-hwaccel', chosen_nvidia_accel, '-hwaccel_output_format', 'cuda']
                    self.logger.debug(f"Auto-selected '{chosen_nvidia_accel}' (NVIDIA) for {system.capitalize()}.")
                elif 'qsv' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'qsv', '-hwaccel_output_format', 'qsv']
                    self.logger.debug(f"Auto-selected 'qsv' (Intel) for {system.capitalize()}.")
                elif system == 'linux' and 'vaapi' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'vaapi', '-hwaccel_output_format', 'vaapi']
                    self.logger.debug("Auto-selected 'vaapi' for Linux.")
                elif system == 'windows' and 'd3d11va' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'd3d11va']
                    self.logger.debug("Auto-selected 'd3d11va' for Windows.")
                elif system == 'windows' and 'dxva2' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'dxva2']
                    self.logger.debug("Auto-selected 'dxva2' for Windows.")

            if not hwaccel_args:
                self.logger.info("Auto hardware acceleration: No compatible method found, using CPU decoding.")
        elif selected_hwaccel != "none" and selected_hwaccel:
            if selected_hwaccel in available_on_app:
                hwaccel_args = ['-hwaccel', selected_hwaccel]
                if selected_hwaccel == 'qsv':
                    hwaccel_args.extend(['-hwaccel_output_format', 'qsv'])
                elif selected_hwaccel in ['cuda', 'nvdec']:
                    hwaccel_args.extend(['-hwaccel_output_format', 'cuda'])
                elif selected_hwaccel == 'vaapi':
                    hwaccel_args.extend(['-hwaccel_output_format', 'vaapi'])
                self.logger.info(f"User-selected hardware acceleration: '{selected_hwaccel}'. Args: {hwaccel_args}")
            else:
                self.logger.warning(
                    f"Selected HW accel '{selected_hwaccel}' not in FFmpeg's available list. Using CPU.")
        else:
            self.logger.debug("Hardware acceleration explicitly disabled (CPU decoding).")
        return hwaccel_args

    def _terminate_process(self, process: Optional[subprocess.Popen], process_name: str, timeout_sec: float = 2.0):
        """
        Terminate a process safely.
        """
        if process is not None and process.poll() is None:
            self.logger.debug(f"Terminating {process_name} process (PID: {process.pid}).")
            process.terminate()
            try:
                process.wait(timeout=timeout_sec)
                self.logger.debug(f"{process_name} process terminated gracefully.")
            except subprocess.TimeoutExpired:
                # Use reduced log level to avoid spam when streaming many short segments
                self.logger.debug(f"{process_name} process did not terminate in time. Killing.")
                process.kill()
                self.logger.debug(f"{process_name} process killed.")

        # Ensure all standard pipes are closed to release OS resources
        for stream in (getattr(process, 'stdout', None), getattr(process, 'stderr', None), getattr(process, 'stdin', None)):
            try:
                if stream is not None:
                    stream.close()
            except Exception:
                pass

    def _terminate_ffmpeg_processes(self):
        """Safely terminates all active FFmpeg processes using the helper."""
        self._terminate_process(self.ffmpeg_pipe1_process, "Pipe 1")
        self.ffmpeg_pipe1_process = None
        self._terminate_process(self.ffmpeg_process, "Main/Pipe 2")
        self.ffmpeg_process = None

    def _start_ffmpeg_process(self, start_frame_abs_idx=0, num_frames_to_output_ffmpeg=None):
        self._terminate_ffmpeg_processes()

        if not self.video_path or not self.video_info or self.video_info.get('fps', 0) <= 0:
            self.logger.warning("Cannot start FFmpeg: video not properly opened or invalid FPS.")
            return False

        start_time_seconds = start_frame_abs_idx / self.video_info['fps']
        self.current_stream_start_frame_abs = start_frame_abs_idx
        self.frames_read_from_current_stream = 0
        common_ffmpeg_prefix = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']

        if self._is_10bit_cuda_pipe_needed():
            self.logger.info("Using 2-pipe FFmpeg command for 10-bit CUDA video.")
            video_height_for_crop = self.video_info.get('height', 0)
            if video_height_for_crop <= 0:
                self.logger.error("Cannot construct 10-bit CUDA pipe 1: video height is unknown or invalid.")
                return False

            # This VF is a generic intermediate step to sanitize the stream.
            pipe1_vf = f"crop={int(video_height_for_crop)}:{int(video_height_for_crop)}:0:0,scale_cuda=1000:1000"
            cmd1 = common_ffmpeg_prefix[:]
            cmd1.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
            if start_time_seconds > 0.001: cmd1.extend(['-ss', str(start_time_seconds)])
            cmd1.extend(['-i', self._active_video_source_path, '-an', '-sn', '-vf', pipe1_vf])
            if num_frames_to_output_ffmpeg and num_frames_to_output_ffmpeg > 0:
                 cmd1.extend(['-frames:v', str(num_frames_to_output_ffmpeg)])
            cmd1.extend(['-c:v', 'hevc_nvenc', '-preset', 'fast', '-qp', '0', '-f', 'matroska', 'pipe:1'])

            cmd2 = common_ffmpeg_prefix[:]
            cmd2.extend(['-hwaccel', 'cuda', '-i', 'pipe:0', '-an', '-sn'])
            effective_vf_pipe2 = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}"
            cmd2.extend(['-vf', effective_vf_pipe2])
            if num_frames_to_output_ffmpeg and num_frames_to_output_ffmpeg > 0:
                cmd2.extend(['-frames:v', str(num_frames_to_output_ffmpeg)])
            cmd2.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

            self.logger.info(f"Pipe 1 CMD: {' '.join(shlex.quote(str(x)) for x in cmd1)}")
            self.logger.info(f"Pipe 2 CMD: {' '.join(shlex.quote(str(x)) for x in cmd2)}")
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                self.ffmpeg_pipe1_process = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creation_flags)
                if self.ffmpeg_pipe1_process.stdout is None:
                    raise IOError("Pipe 1 stdout is None.")
                self.ffmpeg_process = subprocess.Popen(cmd2, stdin=self.ffmpeg_pipe1_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_size_bytes * 5, creationflags=creation_flags)
                self.ffmpeg_pipe1_process.stdout.close()
                return True
            except Exception as e:
                self.logger.error(f"Failed to start 2-pipe FFmpeg: {e}", exc_info=True)
                self._terminate_ffmpeg_processes()
                return False
        else:
            # Standard single FFmpeg process
            hwaccel_cmd_list = self._get_ffmpeg_hwaccel_args()
            ffmpeg_input_options = hwaccel_cmd_list[:]
            if start_time_seconds > 0.001: ffmpeg_input_options.extend(['-ss', str(start_time_seconds)])

            cmd = common_ffmpeg_prefix + ffmpeg_input_options + ['-i', self._active_video_source_path, '-an', '-sn']
            effective_vf = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}"
            cmd.extend(['-vf', effective_vf])
            if num_frames_to_output_ffmpeg and num_frames_to_output_ffmpeg > 0:
                cmd.extend(['-frames:v', str(num_frames_to_output_ffmpeg)])
            cmd.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

            self.logger.info(f"Single Pipe CMD: {' '.join(shlex.quote(str(x)) for x in cmd)}")
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                self.ffmpeg_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_size_bytes * 5, creationflags=creation_flags)
                return True
            except Exception as e:
                self.logger.error(f"Failed to start FFmpeg: {e}", exc_info=True)
                self.ffmpeg_process = None
                return False

    def start_processing(self, start_frame=None, end_frame=None, cli_progress_callback=None):
        # If we are already processing but are in a paused state, just un-pause.
        if self.is_processing and self.pause_event.is_set():
            self.logger.info("Resuming video processing...")
            self.pause_event.clear()
            # Optional: callback to notify the main app UI
            if self.app and hasattr(self.app, 'on_processing_resumed'):
                self.app.on_processing_resumed()
            return

        if self.is_processing:
            self.logger.warning("Already processing.")
            return
        if not self.video_path or not self.video_info:
            self.logger.warning("Video not loaded.")
            return

        self.cli_progress_callback = cli_progress_callback

        effective_start_frame = self.current_frame_index
        # The check for `is_paused` is removed here, as the new block above handles it.
        if start_frame is not None:
            if 0 <= start_frame < self.total_frames:
                effective_start_frame = start_frame
            else:
                self.logger.warning(f"Start frame {start_frame} out of bounds ({self.total_frames} total). Not starting.")
                return

        self.logger.info(f"Starting processing from frame {effective_start_frame}.")

        self.processing_start_frame_limit = effective_start_frame
        self.processing_end_frame_limit = -1
        if end_frame is not None and end_frame >= 0:
            self.processing_end_frame_limit = min(end_frame, self.total_frames - 1)

        num_frames_to_process = None
        if self.processing_end_frame_limit != -1:
            num_frames_to_process = self.processing_end_frame_limit - self.processing_start_frame_limit + 1


        if not self._start_ffmpeg_process(start_frame_abs_idx=self.processing_start_frame_limit, num_frames_to_output_ffmpeg=num_frames_to_process):
            self.logger.error("Failed to start FFmpeg for processing start.")
            return

        self.is_processing = True
        self.pause_event.clear()
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.logger.info(
            f"Started GUI processing. Range: {self.processing_start_frame_limit} to "
            f"{self.processing_end_frame_limit if self.processing_end_frame_limit != -1 else 'EOS'}")

    def pause_processing(self):
        if not self.is_processing or self.pause_event.is_set():
            return

        self.logger.info("Pausing video processing...")
        self.pause_event.set()

        # Optional callback to update UI elements, like a play/pause button icon.
        if self.app and hasattr(self.app, 'on_processing_paused'):
            self.app.on_processing_paused()

    def stop_processing(self, join_thread=True):
        is_currently_processing = self.is_processing
        is_thread_alive = self.processing_thread and self.processing_thread.is_alive()

        if not is_currently_processing and not is_thread_alive:
            self._terminate_ffmpeg_processes()
            return

        self.logger.info("Stopping GUI processing...")
        was_scripting_session = self.tracker and self.tracker.tracking_active
        scripted_range = (self.processing_start_frame_limit, self.current_frame_index)

        self.is_processing = False
        self.pause_event.clear()
        self.stop_event.set()

        self._terminate_ffmpeg_processes()

        if join_thread:
            thread_to_join = self.processing_thread
            if thread_to_join and thread_to_join.is_alive():
                if threading.current_thread() is not thread_to_join:
                    self.logger.info(f"Joining processing thread: {thread_to_join.name} during stop.")
                    thread_to_join.join(timeout=2.0)
                    if thread_to_join.is_alive():
                        self.logger.warning("Processing thread did not join cleanly after stop signal.")
        self.processing_thread = None

        if self.tracker:
            self.logger.info("Signaling tracker to stop.")
            self.tracker.stop_tracking()

        self.enable_tracker_processing = False

        if self.app and hasattr(self.app, 'on_processing_stopped'):
            self.app.on_processing_stopped(was_scripting_session=was_scripting_session, scripted_frame_range=scripted_range)

        self.logger.info("GUI processing stopped.")

    def seek_video(self, frame_index: int):
        if not self.video_info or self.video_info.get('fps', 0) <= 0 or self.total_frames <= 0: return
        target_frame = max(0, min(frame_index, self.total_frames - 1))

        was_processing = self.is_processing
        was_paused = self.is_processing and self.pause_event.is_set()
        stored_end_limit = self.processing_end_frame_limit

        if was_processing:
            self.stop_processing(join_thread=True)

        self.logger.info(f"Seek requested to frame {target_frame}")
        new_frame = self._get_specific_frame(target_frame)

        with self.frame_lock:
            self.current_frame = new_frame

        if new_frame is None:
            self.logger.warning(f"Seek to frame {target_frame} failed to retrieve frame.")
            self.current_frame_index = target_frame

        if was_processing and not was_paused:
            self.start_processing(start_frame=self.current_frame_index, end_frame=stored_end_limit)
        # If was_paused, do not restart processing (remain paused after seek)

    def is_vr_active_or_potential(self) -> bool:
        if self.video_type_setting == 'VR':
            return True
        if self.video_type_setting == 'auto':
            if self.video_info and self.determined_video_type == 'VR':
                return True
        return False

    def display_current_frame(self):
        if not self.video_path or not self.video_info:
            return

        with self.frame_lock:
            raw_frame_to_process = self.current_frame
        if raw_frame_to_process is None: return
        if self.tracker and self.tracker.tracking_active:
            fps_for_timestamp = self.fps if self.fps > 0 else 30.0
            timestamp_ms = int(self.current_frame_index * (1000.0 / fps_for_timestamp))
            try:
                if not self.is_processing:
                    processed_frame_tuple = self.tracker.process_frame(raw_frame_to_process.copy(), timestamp_ms)
                    with self.frame_lock: self.current_frame = processed_frame_tuple[0]
            except Exception as e:
                self.logger.error(f"Error processing frame with tracker in display_current_frame: {e}", exc_info=True)

    def _processing_loop(self):
        if not self.ffmpeg_process or self.ffmpeg_process.stdout is None:
            self.logger.error("_processing_loop: FFmpeg process/stdout not available. Exiting.")
            self.is_processing = False
            return

        start_time = time.time()  # For calculating FPS and ETA in the callback

        loop_ffmpeg_process = self.ffmpeg_process
        next_frame_target_time = time.perf_counter()
        self.last_processed_chapter_id = None

        try:
            # The main processing loop
            while not self.stop_event.is_set():
                while self.pause_event.is_set():
                    if self.stop_event.is_set():
                        break
                    time.sleep(0.01)

                # If a stop was requested while we were paused, break the main loop.
                if self.stop_event.is_set():
                    break

                # The original logic of the loop continues below
                speed_mode = self.app.app_state_ui.selected_processing_speed_mode
                if speed_mode == constants.ProcessingSpeedMode.REALTIME:
                    target_delay = 1.0 / self.fps if self.fps > 0 else (1.0 / 30.0)
                elif speed_mode == constants.ProcessingSpeedMode.SLOW_MOTION:
                    target_delay = 1.0 / 10.0  # Fixed 10 FPS for slow-mo
                else:  # Max Speed
                    target_delay = 0.0

                current_chapter = self.app.funscript_processor.get_chapter_at_frame(self.current_frame_index)
                current_chapter_id = current_chapter.unique_id if current_chapter else None

                if current_chapter_id != self.last_processed_chapter_id:
                    if self.tracker:
                        if current_chapter and current_chapter.user_roi_fixed:
                            self.tracker.reconfigure_for_chapter(current_chapter)
                            if not self.tracker.tracking_active:
                                self.tracker.start_tracking()
                        elif current_chapter is None and self.tracker.tracking_active:
                            if self.tracker.tracking_mode != "USER_FIXED_ROI":
                                self.tracker.stop_tracking()
                                self.logger.info("Tracker stopped due to entering a gap between chapters.")
                    self.last_processed_chapter_id = current_chapter_id

                if current_chapter and self.tracker and not self.tracker.tracking_active and current_chapter.user_roi_fixed:
                    self.tracker.start_tracking()

                if self.ffmpeg_pipe1_process and self.ffmpeg_pipe1_process.poll() is not None:
                    pipe1_stderr = self.ffmpeg_pipe1_process.stderr.read(4096).decode(
                        errors='ignore') if self.ffmpeg_pipe1_process.stderr else ""
                    self.logger.warning(
                        f"FFmpeg Pipe 1 died. Exit: {self.ffmpeg_pipe1_process.returncode}. Stderr: {pipe1_stderr.strip()}. Stopping.")
                    self.is_processing = False
                    break

                if loop_ffmpeg_process.poll() is not None:
                    stderr_output = loop_ffmpeg_process.stderr.read(4096).decode(
                        errors='ignore') if loop_ffmpeg_process.stderr else ""
                    self.logger.info(
                        f"FFmpeg output process died unexpectedly. Exit: {loop_ffmpeg_process.returncode}. Stderr: {stderr_output.strip()}. Stopping.")
                    self.is_processing = False
                    break

                raw_frame_bytes = None
                if loop_ffmpeg_process.stdout is not None:
                    raw_frame_bytes = loop_ffmpeg_process.stdout.read(self.frame_size_bytes)
                raw_frame_len = len(raw_frame_bytes) if raw_frame_bytes is not None else 0
                if not raw_frame_bytes or raw_frame_len < self.frame_size_bytes:
                    self.logger.info(
                        f"End of FFmpeg GUI stream or incomplete frame (read {raw_frame_len}/{self.frame_size_bytes}).")
                    self.is_processing = False
                    # Clear tracker processing flag when stream ends naturally
                    self.enable_tracker_processing = False
                    if self.app:
                        was_scripting_at_end = self.tracker and self.tracker.tracking_active
                        end_range = (self.processing_start_frame_limit, self.current_frame_index)
                        self.app.on_processing_stopped(was_scripting_session=was_scripting_at_end, scripted_frame_range=end_range)
                    break

                self.current_frame_index = self.current_stream_start_frame_abs + self.frames_read_from_current_stream
                self.frames_read_from_current_stream += 1

                if self.cli_progress_callback:
                    # Throttle updates to avoid slowing down processing (e.g., update every 10 frames)
                    if self.current_frame_index % 10 == 0 or self.current_frame_index == self.total_frames - 1:
                        self.cli_progress_callback(self.current_frame_index, self.total_frames, start_time)

                if self.processing_end_frame_limit != -1 and self.current_frame_index > self.processing_end_frame_limit:
                    self.logger.info(f"Reached GUI end_frame_limit ({self.processing_end_frame_limit}). Stopping.")
                    self.is_processing = False
                    # Clear tracker processing flag when reaching end frame limit naturally
                    self.enable_tracker_processing = False
                    if self.app:
                        was_scripting_at_end_limit = self.tracker and self.tracker.tracking_active
                        end_range_limit = (self.processing_start_frame_limit, self.processing_end_frame_limit)
                        self.app.on_processing_stopped(was_scripting_session=was_scripting_at_end_limit, scripted_frame_range=end_range_limit)
                    break
                if self.total_frames > 0 and self.current_frame_index >= self.total_frames:
                    self.logger.info("Reached end of video. Stopping GUI processing.")
                    self.is_processing = False
                    # Clear tracker processing flag when reaching end of video naturally
                    self.enable_tracker_processing = False
                    if self.app:
                        was_scripting_at_eos = self.tracker and self.tracker.tracking_active
                        end_range_eos = (self.processing_start_frame_limit, self.current_frame_index)
                        self.app.on_processing_stopped(was_scripting_session=was_scripting_at_eos, scripted_frame_range=end_range_eos)
                    break

                frame_np = np.frombuffer(raw_frame_bytes, dtype=np.uint8).reshape(self.yolo_input_size, self.yolo_input_size, 3)
                processed_frame_for_gui = frame_np
                if self.tracker and self.tracker.tracking_active:
                    timestamp_ms = int(self.current_frame_index * (1000.0 / self.fps)) if self.fps > 0 else int(
                        time.time() * 1000)
                    try:
                        processed_frame_for_gui = self.tracker.process_frame(frame_np.copy(), timestamp_ms)[0]
                    except Exception as e:
                        self.logger.error(f"Error in tracker.process_frame during loop: {e}", exc_info=True)

                with self.frame_lock:
                    self.current_frame = processed_frame_for_gui

                self.frames_for_fps_calc += 1
                current_time_fps_calc = time.time()
                elapsed = current_time_fps_calc - self.last_fps_update_time
                if elapsed >= 1.0:
                    self.actual_fps = self.frames_for_fps_calc / elapsed
                    self.last_fps_update_time = current_time_fps_calc
                    self.frames_for_fps_calc = 0

                current_time = time.perf_counter()
                sleep_duration = next_frame_target_time - current_time

                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                if next_frame_target_time < current_time - target_delay:
                    next_frame_target_time = current_time + target_delay
                else:
                    next_frame_target_time += target_delay
        finally:
            self.logger.info(f"_processing_loop ending. is_processing: {self.is_processing}, stop_event: {self.stop_event.is_set()}")
            self._terminate_ffmpeg_processes()
            self.is_processing = False
            self.pause_event.set()
            self.last_processed_chapter_id = None

    def _start_ffmpeg_for_segment_streaming(self, start_frame_abs_idx: int, num_frames_to_stream_hint: Optional[int] = None) -> bool:
        self._terminate_ffmpeg_processes()

        if not self.video_path or not self.video_info or self.video_info.get('fps', 0) <= 0:
            self.logger.warning("Cannot start FFmpeg for segment: no video/invalid FPS.")
            return False

        start_time_seconds = start_frame_abs_idx / self.video_info['fps']
        common_ffmpeg_prefix = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']

        if self._is_10bit_cuda_pipe_needed():
            self.logger.info("Using 2-pipe FFmpeg command for 10-bit CUDA segment streaming.")
            video_height_for_crop = self.video_info.get('height', 0)
            if video_height_for_crop <= 0:
                self.logger.error("Cannot construct 10-bit CUDA pipe 1 for segment: video height is unknown.")
                return False

            pipe1_vf = f"crop={int(video_height_for_crop)}:{int(video_height_for_crop)}:0:0,scale_cuda=1000:1000"
            cmd1 = common_ffmpeg_prefix[:]
            cmd1.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
            if start_time_seconds > 0.001: cmd1.extend(['-ss', str(start_time_seconds)])
            cmd1.extend(['-i', self._active_video_source_path, '-an', '-sn', '-vf', pipe1_vf])
            if num_frames_to_stream_hint and num_frames_to_stream_hint > 0:
                cmd1.extend(['-frames:v', str(num_frames_to_stream_hint)])
            cmd1.extend(['-c:v', 'hevc_nvenc', '-preset', 'fast', '-qp', '0', '-f', 'matroska', 'pipe:1'])

            cmd2 = common_ffmpeg_prefix[:]
            cmd2.extend(['-hwaccel', 'cuda', '-i', 'pipe:0', '-an', '-sn'])
            effective_vf_pipe2 = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}"
            cmd2.extend(['-vf', effective_vf_pipe2])
            if num_frames_to_stream_hint and num_frames_to_stream_hint > 0:
                cmd2.extend(['-frames:v', str(num_frames_to_stream_hint)])
            cmd2.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

            self.logger.info(f"Segment Pipe 1 CMD: {' '.join(shlex.quote(str(x)) for x in cmd1)}")
            self.logger.info(f"Segment Pipe 2 CMD: {' '.join(shlex.quote(str(x)) for x in cmd2)}")
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                self.ffmpeg_pipe1_process = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creation_flags)
                if self.ffmpeg_pipe1_process.stdout is None:
                    raise IOError("Segment Pipe 1 stdout is None.")
                self.ffmpeg_process = subprocess.Popen(cmd2, stdin=self.ffmpeg_pipe1_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_size_bytes * 20, creationflags=creation_flags)
                self.ffmpeg_pipe1_process.stdout.close()
                return True
            except Exception as e:
                self.logger.error(f"Failed to start 2-pipe FFmpeg for segment: {e}", exc_info=True)
                self._terminate_ffmpeg_processes()
                return False
        else:
            # Standard single FFmpeg process for 8-bit or non-CUDA accelerated video
            hwaccel_cmd_list = self._get_ffmpeg_hwaccel_args()
            ffmpeg_input_options = hwaccel_cmd_list[:]
            if start_time_seconds > 0.001: ffmpeg_input_options.extend(['-ss', str(start_time_seconds)])
            ffmpeg_cmd = common_ffmpeg_prefix + ffmpeg_input_options + ['-i', self._active_video_source_path, '-an', '-sn']
            effective_vf = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}"
            ffmpeg_cmd.extend(['-vf', effective_vf])

            if num_frames_to_stream_hint and num_frames_to_stream_hint > 0:
                ffmpeg_cmd.extend(['-frames:v', str(num_frames_to_stream_hint)])

            ffmpeg_cmd.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])
            self.logger.info(f"Segment CMD (single pipe): {' '.join(shlex.quote(str(x)) for x in ffmpeg_cmd)}")
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_size_bytes * 20, creationflags=creation_flags)
                return True
            except Exception as e:
                self.logger.warning(f"Failed to start FFmpeg for segment: {e}", exc_info=True)
                self.ffmpeg_process = None
                return False

    def stream_frames_for_segment(self, start_frame_abs_idx: int, num_frames_to_read: int, stop_event: Optional[threading.Event] = None) -> Iterator[Tuple[int, np.ndarray]]:
        if num_frames_to_read <= 0:
            self.logger.warning("num_frames_to_read is not positive, no frames to stream.")
            return

        if not self._start_ffmpeg_for_segment_streaming(start_frame_abs_idx, num_frames_to_read):
            self.logger.warning(f"Failed to start FFmpeg for segment from {start_frame_abs_idx}.")
            return

        frames_yielded = 0
        segment_ffmpeg_process = self.ffmpeg_process
        try:
            for i in range(num_frames_to_read):
                if stop_event and stop_event.is_set():
                    self.logger.info("Stop event detected in stream_frames_for_segment. Aborting stream.")
                    break

                if not segment_ffmpeg_process or segment_ffmpeg_process.stdout is None:
                    self.logger.warning("FFmpeg process or stdout not available during segment streaming.")
                    break

                if segment_ffmpeg_process.poll() is not None:
                    stderr_output = segment_ffmpeg_process.stderr.read(4096).decode(errors='ignore') if segment_ffmpeg_process.stderr else ""
                    self.logger.warning(
                        f"FFmpeg process (segment) terminated prematurely. Exit: {segment_ffmpeg_process.returncode}. Stderr: '{stderr_output.strip()}'")
                    break

                raw_frame_bytes = segment_ffmpeg_process.stdout.read(self.frame_size_bytes)
                if len(raw_frame_bytes) < self.frame_size_bytes:
                    stderr_on_short_read = segment_ffmpeg_process.stderr.read(4096).decode(errors='ignore') if segment_ffmpeg_process.stderr else ""
                    self.logger.info(
                        f"End of FFmpeg stream or error (read {len(raw_frame_bytes)}/{self.frame_size_bytes}) "
                        f"after {frames_yielded} frames for segment (start {start_frame_abs_idx}). Stderr: '{stderr_on_short_read.strip()}'")
                    break

                frame_np = np.frombuffer(raw_frame_bytes, dtype=np.uint8).reshape(self.yolo_input_size, self.yolo_input_size, 3)
                current_frame_id = start_frame_abs_idx + frames_yielded
                yield current_frame_id, frame_np
                frames_yielded += 1
        finally:
            self._terminate_ffmpeg_processes()

    def set_target_fps(self, fps: float):
        self.target_fps = max(1.0, fps if fps > 0 else 1.0)
        self.logger.info(f"Target FPS set to: {self.target_fps}")

    def is_video_open(self) -> bool:
        """Checks if a video is currently loaded and has valid information."""
        return bool(self.video_path and self.video_info and self.video_info.get('total_frames', 0) > 0)

    def reset(self, close_video=False, skip_tracker_reset=False):
        self.logger.info("Resetting VideoProcessor...")
        self.stop_processing(join_thread=True)
        self._clear_cache()
        self.current_frame_index = 0
        self.frames_read_from_current_stream = 0
        self.current_stream_start_frame_abs = 0
        self.seek_request_frame_index = None
        if self.tracker and not skip_tracker_reset:
            self.tracker.reset()
        if close_video:
            self.video_path = ""
            self._active_video_source_path = ""
            self.video_info = {}
            self.determined_video_type = None
            self.ffmpeg_filter_string = ""
            self.logger.info("Video closed. Params reset.")
        with self.frame_lock:
            self.current_frame = None
        if self.video_path and self.video_info and not close_video:
            self.logger.info("Fetching frame 0 after reset (video still loaded).")
            self.current_frame = self._get_specific_frame(0)
        else:
            self.current_frame = None
        if self.app and hasattr(self.app, 'on_processing_stopped'):
            self.app.on_processing_stopped(was_scripting_session=False, scripted_frame_range=None)
        self.logger.info("VideoProcessor reset complete.")

    def _validate_preprocessed_video(self, video_path: str, expected_frames: int, expected_fps: float) -> bool:
        """
        Validates that a preprocessed video is complete and usable.

        Args:
            video_path: Path to the preprocessed video
            expected_frames: Expected number of frames
            expected_fps: Expected FPS

        Returns:
            True if video is valid, False otherwise
        """
        try:
            # Import validation function from stage_1_cd
            from detection.cd.stage_1_cd import _validate_preprocessed_video_completeness
            return _validate_preprocessed_video_completeness(video_path, expected_frames, expected_fps, self.logger)
        except Exception as e:
            self.logger.error(f"Error validating preprocessed video: {e}")
            return False

    def _cleanup_invalid_preprocessed_file(self, file_path: str) -> None:
        """
        Safely removes an invalid preprocessed file and notifies the user.

        Args:
            file_path: Path to the invalid file
        """
        try:
            from detection.cd.stage_1_cd import _cleanup_incomplete_file
            _cleanup_incomplete_file(file_path, self.logger)

            # Update app state to reflect that preprocessed file is no longer available
            if self.app and hasattr(self.app, 'file_manager'):
                if self.app.file_manager.preprocessed_video_path == file_path:
                    self.app.file_manager.preprocessed_video_path = None

            # Notify user about the cleanup
            if hasattr(self.app, 'set_status_message'):
                self.app.set_status_message(f"Removed invalid preprocessed file: {os.path.basename(file_path)}", level=logging.WARNING)

        except Exception as e:
            self.logger.error(f"Error cleaning up invalid preprocessed file: {e}")

    def get_preprocessed_video_status(self) -> Dict[str, Any]:
        """
        Returns the status of the preprocessed video for the current video.

        Returns:
            Dictionary with status information about preprocessed video availability
        """
        status = {
            "exists": False,
            "valid": False,
            "path": None,
            "using_preprocessed": False,
            "frame_count": 0,
            "expected_frames": 0
        }

        if not self.video_path or not self.video_info:
            return status

        try:
            if self.app and hasattr(self.app, 'file_manager'):
                preprocessed_path = self.app.file_manager.get_output_path_for_file(self.video_path, "_preprocessed.mkv")

                if os.path.exists(preprocessed_path):
                    status["exists"] = True
                    status["path"] = preprocessed_path

                    expected_frames = self.video_info.get("total_frames", 0)
                    expected_fps = self.video_info.get("fps", 30.0)
                    status["expected_frames"] = expected_frames

                    # Validate the file
                    if self._validate_preprocessed_video(preprocessed_path, expected_frames, expected_fps):
                        status["valid"] = True

                        # Get actual frame count
                        preprocessed_info = self._get_video_info(preprocessed_path)
                        if preprocessed_info:
                            status["frame_count"] = preprocessed_info.get("total_frames", 0)

                    # Check if we're currently using it
                    status["using_preprocessed"] = (self._active_video_source_path == preprocessed_path)

        except Exception as e:
            self.logger.error(f"Error getting preprocessed video status: {e}")

        return status

import time
import threading
import subprocess
import json
import shlex
import numpy as np
import cv2
import platform
from typing import Optional, Iterator, Tuple, List, Dict
import logging
import os
import tempfile
import wave as pywave  # For reading WAV file properties if scipy is not available

from scenedetect import open_video, SceneManager
from scenedetect.detectors import ContentDetector


try:
    from scipy.io import wavfile

    SCIPY_AVAILABLE_FOR_AUDIO = True
except ImportError:
    SCIPY_AVAILABLE_FOR_AUDIO = False
from collections import OrderedDict

class VideoProcessor:
    def __init__(self, app_instance, tracker: Optional = None, yolo_input_size=640,
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
        self.video_info = {}
        self.ffmpeg_process: Optional[subprocess.Popen] = None  # Main output process (pipe2 if active)
        self.ffmpeg_pipe1_process: Optional[subprocess.Popen] = None  # Pipe1 process, if active
        self.is_processing = False
        self.is_paused = True
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

        self.enable_tracker_processing = True
        if self.tracker is None:
            self.enable_tracker_processing = False
            if self.logger:
                self.logger.info("No tracker provided. Tracker processing will be disabled.")

        # Frame Caching
        self.frame_cache = OrderedDict()
        self.frame_cache_max_size = cache_size
        self.frame_cache_lock = threading.Lock()
        self.batch_fetch_size = 50

    # --- Scene Detection Method ---
    def detect_scenes(self, threshold: float = 27.0, progress_callback=None,
                      stop_event: Optional[threading.Event] = None) -> List[Tuple[int, int]]:
        """
        Uses PySceneDetect's modern API to find scene cuts with a manual processing loop
        to ensure progress reporting.
        Returns a list of (start_frame, end_frame) tuples for each scene.
        """
        if not self.video_path:
            self.logger.error("Cannot detect scenes: No video loaded.")
            return []

        self.logger.info("Starting scene detection with corrected manual frame processing loop...")
        if hasattr(self.app, 'set_status_message'):
            self.app.set_status_message("Detecting scenes...")

        try:
            video = open_video(self.video_path)
            total_frames = video.duration.get_frames()

            scene_manager = SceneManager()
            scene_manager.add_detector(ContentDetector(threshold=threshold))

            if progress_callback:
                progress_callback(0, total_frames, 0, 0, 0)

            while True:
                frame_image = video.read()
                if frame_image is False:
                    break

                if stop_event and stop_event.is_set():
                    raise InterruptedError("Scene detection cancelled by user.")

                # --- THIS IS THE FIX ---
                # Call the correct private method `_process_frame` as suggested by the error.
                # The expected arguments are (frame_image, frame_number).
                scene_manager._process_frame(frame_image, video.frame_number)

                if progress_callback:
                    progress_callback(video.frame_number, total_frames)

            # post_process is not a method on SceneManager, this is done when getting the list.
            # scene_manager.post_process()

            scene_list_raw = scene_manager.get_scene_list()

            if scene_list_raw and self.total_frames > 0:
                last_scene_end_frame = scene_list_raw[-1][1].get_frames()
                if last_scene_end_frame < self.total_frames:
                    end_timecode = video.get_duration()
                    last_scene_start_timecode = scene_list_raw[-1][0]
                    scene_list_raw[-1] = (last_scene_start_timecode, end_timecode)

            scene_list_frames = [(s[0].get_frames(), s[1].get_frames()) for s in scene_list_raw]

            self.logger.info(f"Scene detection complete. Found {len(scene_list_frames)} scenes.")
            return scene_list_frames

        except InterruptedError:
            self.logger.info("Scene detection was successfully aborted by the user.")
            return []
        except Exception as e:
            self.logger.error(f"An error occurred during scene detection: {e}", exc_info=True)
            if hasattr(self.app, 'set_status_message'):
                self.app.set_status_message("Error during scene detection.", level=logging.ERROR)
            return []

    def _clear_cache(self):
        with self.frame_cache_lock:
            if self.frame_cache:
                self.logger.debug(f"Clearing frame cache (had {len(self.frame_cache)} items).")
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

    def set_active_vr_parameters(self, fov: Optional[int] = None, pitch: Optional[int] = None,
                                 input_format: Optional[str] = None):
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
            self.logger.info(f"Tracker processing {'enabled' if enable else 'disabled'}.")

    def open_video(self, video_path: str, from_project_load: bool = False) -> bool:
        self.stop_processing()
        self.video_path = video_path
        self._clear_cache()
        self.video_info = self._get_video_info(video_path)
        if not self.video_info or self.video_info.get("total_frames", 0) == 0:
            self.logger.warning(f"Failed to get valid video info for {video_path}")
            self.video_path = ""
            self.video_info = {}
            return False

        info = self.video_info
        # Check for side-by-side (SBS) aspect ratio (approx. 2:1)
        is_sbs_resolution = (info['width'] >= 1.8 * info['height'] and
                             info['width'] <= 2.2 * info['height'] and
                             info['width'] > 1000)
        # Check for top-and-bottom (TB) aspect ratio (approx. 1:2)
        is_tb_resolution = (info['height'] >= 1.8 * info['width'] and
                            info['height'] <= 2.2 * info['width'] and
                            info['height'] > 1000)

        if self.video_type_setting == 'auto':
            # --- Heuristic 1: Determine if the video is '2D' or 'VR' ---
            upper_video_path = video_path.upper()
            # General keywords to identify a video as potentially VR
            vr_keywords = ['VR', '_180', '_360', 'SBS', '_TB', 'FISHEYE', 'EQUIRECTANGULAR', 'LR_', 'Oculus', '_3DH']
            has_vr_keyword = any(kw in upper_video_path for kw in vr_keywords)

            if is_sbs_resolution or is_tb_resolution or has_vr_keyword:
                self.determined_video_type = 'VR'
            else:
                self.determined_video_type = '2D'
            self.logger.info(
                f"Auto-detected video type: {self.determined_video_type} (SBS Res: {is_sbs_resolution}, TB Res: {is_tb_resolution}, Keyword: {has_vr_keyword})")
        else:
            self.determined_video_type = self.video_type_setting
            self.logger.info(f"Using configured video type: {self.determined_video_type}")

        if self.determined_video_type == 'VR':
            # --- Heuristic 2: Determine specific VR format (e.g., he_sbs, fisheye_tb) ---
            # Default to Equirectangular Side-by-Side (he_sbs) and override with evidence.
            suggested_base = 'he'  # 'he' corresponds to equirectangular
            suggested_layout = '_sbs'  # Default layout is SBS

            # Evidence from resolution (for layout)
            if is_tb_resolution:  # This will now work correctly
                suggested_layout = '_tb'
                self.logger.info("Resolution (H > 1.8*W) suggests Top-Bottom (TB) layout.")

            # Evidence from filename keywords (strongest evidence)
            upper_video_path = video_path.upper()

            # Keywords that indicate a fisheye format. 'LR_180' is now correctly excluded.
            fisheye_keywords = ['FISHEYE', 'MKX', 'RF52']
            if any(kw in upper_video_path for kw in fisheye_keywords):
                suggested_base = 'fisheye'
                self.logger.info(f"Filename keyword suggests 'fisheye' base format.")

            # Keywords that indicate a specific layout, overriding resolution hints.
            tb_keywords = ['_TB', 'TB_', 'TOPBOTTOM', 'OVERUNDER', '_OU', 'OU_']
            sbs_keywords = ['SBS']

            if any(kw in upper_video_path for kw in tb_keywords):
                suggested_layout = '_tb'  # Keyword override is definitive
                self.logger.info(f"Filename keyword confirms 'Top-Bottom' layout.")
            elif any(kw in upper_video_path for kw in sbs_keywords):
                suggested_layout = '_sbs'  # Keyword override is definitive
                self.logger.info(f"Filename keyword confirms 'Side-by-Side' layout.")

            final_suggested_vr_input_format = f"{suggested_base}{suggested_layout}"

            # Update the processor's setting for the new video.
            if self.vr_input_format != final_suggested_vr_input_format:
                self.logger.info(
                    f"For new video '{os.path.basename(video_path)}', auto-detection suggests VR format: {final_suggested_vr_input_format}. Updating from '{self.vr_input_format}'.")
                self.vr_input_format = final_suggested_vr_input_format

            # Heuristic for FOV based on filename can remain.
            if 'MKX' in upper_video_path and 'fisheye' in self.vr_input_format and self.vr_fov != 200:
                self.logger.info(
                    f"Filename suggests VR FOV: 200 (MKX) for fisheye. Overriding current: {self.vr_fov}")
                self.vr_fov = 200
        self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
        self.fps = self.video_info['fps']
        self.total_frames = self.video_info['total_frames']
        self.set_target_fps(self.fps)
        self.current_frame_index = 0
        self.frames_read_from_current_stream = 0
        self.current_stream_start_frame_abs = 0
        self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
        self.stop_event.clear()
        self.seek_request_frame_index = None
        self.current_frame = self._get_specific_frame(0)
        if self.tracker:
            reset_reason = "project_load_preserve_actions" if from_project_load else None
            self.tracker.reset(reason=reset_reason)
        self.logger.info(
            f"Opened: {os.path.basename(video_path)} ({self.determined_video_type}, format: {self.vr_input_format if self.determined_video_type == 'VR' else 'N/A'}), "
            f"{self.total_frames}fr, {self.fps:.2f}fps, {self.video_info.get('bit_depth', 'N/A')}bit)")
        return True

    def reapply_video_settings(self):
        if not self.video_path or not self.video_info:
            self.logger.info("No video loaded. Settings will apply when a video is opened.")
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
            return
        self.logger.info(
            "Reapplying video settings (self.vr_input_format is currently: " + self.vr_input_format + ")")
        was_processing = self.is_processing
        stored_frame_index = self.current_frame_index
        stored_end_limit = self.processing_end_frame_limit
        self.stop_processing()
        self._clear_cache()
        if self.video_type_setting == 'auto':
            self.determined_video_type = 'VR' if (self.video_info['width'] >= 1.9 * self.video_info['height'] and
                                                  self.video_info['width'] <= 2.1 * self.video_info['height'] and
                                                  self.video_info['width'] > 1000) else '2D'
            self.logger.info(f"Auto-re-determined video type: {self.determined_video_type}")
        else:
            self.determined_video_type = self.video_type_setting
            self.logger.info(f"Using configured video type: {self.determined_video_type}")
        if self.determined_video_type == 'VR':
            upper_video_path = self.video_path.upper()
            if 'MKX' in upper_video_path and 'fisheye' in self.vr_input_format and self.vr_fov != 200:
                self.logger.info(
                    f"Re-applying FOV heuristic during reapply: Filename suggests VR FOV: 200 (MKX). Overriding current: {self.vr_fov}")
                self.vr_fov = 200
            self.logger.info(f"Using user-set VR Input Format for reapply: {self.vr_input_format}")
        self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
        self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
        self.logger.info(
            f"Frame size bytes updated to: {self.frame_size_bytes} for YOLO size {self.yolo_input_size}")
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
        current_frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
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
                cmd1.extend(['-i', self.video_path, '-an', '-sn', '-vf', pipe1_vf])
                cmd1.extend(['-frames:v', str(num_frames_to_fetch)])
                cmd1.extend(['-c:v', 'hevc_nvenc', '-preset', 'fast', '-qp', '0', '-f', 'matroska', 'pipe:1'])

                cmd2 = common_ffmpeg_prefix[:]
                cmd2.extend(['-hwaccel', 'cuda', '-i', 'pipe:0', '-an', '-sn'])
                effective_vf_pipe2 = self.ffmpeg_filter_string
                if not effective_vf_pipe2: effective_vf_pipe2 = f"scale={self.yolo_input_size}:{self.yolo_input_size}"
                cmd2.extend(['-vf', effective_vf_pipe2])
                cmd2.extend(['-frames:v', str(num_frames_to_fetch)])
                cmd2.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

                self.logger.debug(f"get_frames_batch Pipe 1 CMD: {' '.join(shlex.quote(str(x)) for x in cmd1)}")
                self.logger.debug(f"get_frames_batch Pipe 2 CMD: {' '.join(shlex.quote(str(x)) for x in cmd2)}")

                local_p1_proc = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if local_p1_proc.stdout is None: raise IOError("get_frames_batch: Pipe 1 stdout is None.")

                local_p2_proc = subprocess.Popen(cmd2, stdin=local_p1_proc.stdout,
                                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                 bufsize=current_frame_size_bytes * min(num_frames_to_fetch, 20))
                local_p1_proc.stdout.close()

            else:  # Standard single FFmpeg process
                self.logger.debug(
                    f"get_frames_batch: Using single-pipe FFmpeg for {num_frames_to_fetch} frames from {start_frame_num}.")
                hwaccel_cmd_list = self._get_ffmpeg_hwaccel_args()
                ffmpeg_input_options = hwaccel_cmd_list[:]
                if start_time_seconds > 0.001: ffmpeg_input_options.extend(['-ss', str(start_time_seconds)])
                cmd_single = common_ffmpeg_prefix + ffmpeg_input_options + ['-i', self.video_path, '-an', '-sn']
                effective_vf = self.ffmpeg_filter_string
                if not effective_vf: effective_vf = f"scale={self.yolo_input_size}:{self.yolo_input_size}"
                cmd_single.extend(['-vf', effective_vf])
                cmd_single.extend(['-frames:v', str(num_frames_to_fetch)])
                cmd_single.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])
                self.logger.debug(
                    f"get_frames_batch CMD (single pipe): {' '.join(shlex.quote(str(x)) for x in cmd_single)}")
                local_p2_proc = subprocess.Popen(cmd_single, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                 bufsize=current_frame_size_bytes * min(num_frames_to_fetch, 20))

            # --- Read frames from local_p2_proc (final output pipe) ---
            if not local_p2_proc or local_p2_proc.stdout is None:
                self.logger.error("get_frames_batch: Output FFmpeg process or its stdout is None.")
                return frames_batch

            for i in range(num_frames_to_fetch):
                raw_frame_data = local_p2_proc.stdout.read(current_frame_size_bytes)
                if len(raw_frame_data) < current_frame_size_bytes:
                    p2_stderr_content = local_p2_proc.stderr.read().decode(
                        errors='ignore') if local_p2_proc.stderr else ""
                    self.logger.warning(
                        f"get_frames_batch: Incomplete data for frame {start_frame_num + i} (read {len(raw_frame_data)}/{current_frame_size_bytes}). P2 Stderr: {p2_stderr_content.strip()}")
                    if local_p1_proc and local_p1_proc.stderr:
                        p1_stderr_content = local_p1_proc.stderr.read().decode(errors='ignore')
                        self.logger.warning(f"get_frames_batch: P1 Stderr: {p1_stderr_content.strip()}")
                    break
                frames_batch[start_frame_num + i] = np.frombuffer(raw_frame_data, dtype=np.uint8).reshape(
                    self.yolo_input_size, self.yolo_input_size, 3)

        except Exception as e:
            self.logger.error(f"get_frames_batch: Error fetching batch @{start_frame_num}: {e}", exc_info=True)
        finally:
            # Terminate the local Popen objects used for this batch fetch
            if local_p1_proc:
                if local_p1_proc.stdout: local_p1_proc.stdout.close()
                if local_p1_proc.stderr: local_p1_proc.stderr.close()
                if local_p1_proc.poll() is None: local_p1_proc.terminate()
                try:
                    local_p1_proc.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    local_p1_proc.kill();
                    local_p1_proc.wait()

            if local_p2_proc:
                if local_p2_proc.stdout: local_p2_proc.stdout.close()
                if local_p2_proc.stderr: local_p2_proc.stderr.close()
                if local_p2_proc.poll() is None: local_p2_proc.terminate()
                try:
                    local_p2_proc.wait(timeout=0.2)
                except subprocess.TimeoutExpired:
                    local_p2_proc.kill();
                    local_p2_proc.wait()

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

        # Use the (now enhanced) get_frames_batch method
        fetched_batch = self.get_frames_batch(batch_start_frame, num_frames_to_fetch_actual)

        retrieved_frame: Optional[np.ndarray] = None
        with self.frame_cache_lock:
            for idx, frame_data in fetched_batch.items():
                if len(self.frame_cache) >= self.frame_cache_max_size:
                    try:
                        popped_key, _ = self.frame_cache.popitem(last=False)
                        # self.logger.debug(f"Cache full. Popped oldest frame: {popped_key}")
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

    def _get_video_info(self, filename):
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries',
               'stream=width,height,r_frame_rate,nb_frames,avg_frame_rate,duration,codec_type,pix_fmt,bits_per_raw_sample',
               # Added pix_fmt, bits_per_raw_sample
               '-show_entries', 'format=duration', '-of', 'json', filename]
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True)
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

            has_audio_ffprobe = False
            cmd_audio_check = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                               '-show_entries', 'stream=codec_type', '-of', 'json', filename]
            try:
                result_audio = subprocess.run(cmd_audio_check, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                              check=True, text=True)
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

            # Determine bit depth
            bit_depth = 8  # Default to 8
            bits_per_raw_sample_str = stream_info.get('bits_per_raw_sample')
            if bits_per_raw_sample_str and bits_per_raw_sample_str != 'N/A':
                try:
                    bit_depth = int(bits_per_raw_sample_str)
                except ValueError:
                    self.logger.warning(f"Could not parse bits_per_raw_sample: {bits_per_raw_sample_str}")
            else:  # Fallback to checking pix_fmt for common 10-bit formats
                pix_fmt = stream_info.get('pix_fmt', '').lower()
                if '10le' in pix_fmt or 'p010' in pix_fmt or '10be' in pix_fmt:  # yuv420p10le, p010le etc.
                    bit_depth = 10
                elif pix_fmt == 'yuv422p10le' or pix_fmt == 'yuv444p10le':  # Other specific 10-bit
                    bit_depth = 10
                # Add more 10-bit or higher pix_fmts if needed

            self.logger.info(
                f"Detected video properties: width={stream_info.get('width', 0)}, height={stream_info.get('height', 0)}, fps={fps:.2f}, bit_depth={bit_depth}")

            return {"duration": duration, "total_frames": total_frames, "fps": fps,  #
                    "width": int(stream_info.get('width', 0)), "height": int(stream_info.get('height', 0)),  #
                    "has_audio": has_audio_ffprobe, "bit_depth": bit_depth}
        except Exception as e:
            self.logger.error(f"Error in _get_video_info for {filename}: {e}")
            return None

    def get_audio_waveform(self, num_samples: int = 1000) -> Optional[np.ndarray]:
        if not self.video_path or not self.video_info.get("has_audio"):
            self.logger.info("No video loaded or video has no audio stream for waveform generation.")
            return None
        if not SCIPY_AVAILABLE_FOR_AUDIO:
            self.logger.warning("Scipy is not available. Cannot generate audio waveform.")
            return None

        temp_wav_file = None
        process = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
                temp_wav_file = tmpfile.name

            ffmpeg_cmd = [
                'ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error',
                '-i', self.video_path,
                '-vn', '-ac', '1', '-ar', '44100', '-c:a', 'pcm_s16le', '-y', temp_wav_file
            ]
            self.logger.info(f"Extracting audio for waveform: {' '.join(shlex.quote(str(x)) for x in ffmpeg_cmd)}")
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = process.communicate(timeout=60)

            if process.returncode != 0:
                self.logger.error(f"FFmpeg failed to extract audio: {stderr.decode(errors='ignore')}")
                return None

            if not os.path.exists(temp_wav_file) or os.path.getsize(temp_wav_file) == 0:
                self.logger.error("Temporary WAV file not created or is empty.")
                return None

            samplerate, data = wavfile.read(temp_wav_file)
            if data.ndim > 1:
                data = data.mean(axis=1)

            if data.size == 0:
                self.logger.warning("Audio data is empty after reading WAV file.")
                return None

            num_frames_audio = len(data)
            if num_frames_audio == 0: return np.array([])

            step = max(1, num_frames_audio // num_samples)

            waveform = []
            for i in range(0, num_frames_audio, step):
                segment = data[i:min(i + step, num_frames_audio)]
                if segment.size > 0:
                    waveform.append(np.max(np.abs(segment)))

            waveform_np = np.array(waveform)

            max_val = np.max(waveform_np)
            if max_val > 0:
                waveform_np = waveform_np / max_val

            self.logger.info(f"Generated waveform with {len(waveform_np)} samples.")
            return waveform_np

        except subprocess.TimeoutExpired:
            self.logger.error("FFmpeg timed out during audio extraction.")
            if process: process.kill()
            return None
        except Exception as e:
            self.logger.error(f"Error generating audio waveform: {e}", exc_info=True)
            return None
        finally:
            if temp_wav_file and os.path.exists(temp_wav_file):
                try:
                    os.remove(temp_wav_file)
                except Exception as e_rem:
                    self.logger.warning(f"Could not remove temporary WAV file {temp_wav_file}: {e_rem}")

    def _is_10bit_cuda_pipe_needed(self) -> bool:
        """Checks if the special 2-pipe FFmpeg command for 10-bit CUDA should be used."""
        if not self.video_info:
            return False

        # Check for 10-bit or higher bit depth
        bit_depth = self.video_info.get('bit_depth', 8)  #
        is_high_bit_depth = bit_depth > 8  #

        # Check if CUDA is the selected hardware acceleration
        # This relies on _get_ffmpeg_hwaccel_args reflecting the actual selection.
        hwaccel_args = self._get_ffmpeg_hwaccel_args()  #
        is_cuda_hwaccel = False
        if '-hwaccel' in hwaccel_args:  #
            try:
                hwaccel_idx = hwaccel_args.index('-hwaccel')  #
                if hwaccel_idx + 1 < len(hwaccel_args):
                    if hwaccel_args[hwaccel_idx + 1].lower() == 'cuda':  #
                        is_cuda_hwaccel = True
            except ValueError:
                pass  # '-hwaccel' not found or malformed

        if is_high_bit_depth and is_cuda_hwaccel:
            self.logger.info("Conditions for 10-bit CUDA pipe met.")
            return True
        return False

    def _build_ffmpeg_filter_string(self) -> str:
        ffmpeg_filter = ''
        if not self.video_info:
            return ''

        original_width = self.video_info.get('width', 0)
        original_height = self.video_info.get('height', 0)
        v_h_FOV = 90  # Default vertical and horizontal FOV for the output projection

        # Determine if hardware acceleration output needs downloading
        current_hw_args = self._get_ffmpeg_hwaccel_args()
        needs_hw_download = False
        hw_output_format_for_log = "N/A"

        if '-hwaccel_output_format' in current_hw_args:
            try:
                idx = current_hw_args.index('-hwaccel_output_format')
                hw_output_format = current_hw_args[idx + 1]
                hw_output_format_for_log = hw_output_format
                # Common GPU surface formats that would require hwdownload for software filters
                if hw_output_format in ['cuda', 'nv12', 'p010le', 'qsv', 'vaapi', 'd3d11va',
                                        'dxva2_vld']:
                    needs_hw_download = True
            except (ValueError, IndexError):
                self.logger.warning("Could not properly parse -hwaccel_output_format from hw_args.")
                pass

        self.logger.info(
            f"Hardware acceleration check for filter string: needs_hw_download={needs_hw_download} "
            f"(detected output format: {hw_output_format_for_log}). Determined video type: {self.determined_video_type}."
        )

        software_filter_segments = []

        if self.determined_video_type == '2D':
            # Scale and pad are software filters
            software_filter_segments.append(
                f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease"
            )
            software_filter_segments.append(
                f"pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
            )
        elif self.determined_video_type == 'VR':
            base_v360_input_format = self.vr_input_format.replace('_sbs', '').replace('_tb', '')
            is_sbs_format = '_sbs' in self.vr_input_format
            is_tb_format = '_tb' in self.vr_input_format

            pre_v360_filters_temp = []  # Temporary list for VR pre-filters

            if is_sbs_format:
                if original_width > 0 and original_height > 0:
                    crop_w = original_width / 2
                    crop_h = original_height
                    pre_v360_filters_temp.append(f"crop={int(crop_w)}:{int(crop_h)}:0:0")  #
                    self.logger.info(f"Applying SBS pre-crop: w={int(crop_w)} h={int(crop_h)} x=0 y=0")
                else:
                    self.logger.warning("Cannot apply SBS crop: original video dimensions unknown or invalid.")
            elif is_tb_format:
                if original_width > 0 and original_height > 0:
                    crop_w = original_width
                    crop_h = original_height / 2
                    pre_v360_filters_temp.append(f"crop={int(crop_w)}:{int(crop_h)}:0:0")
                    self.logger.info(f"Applying TB pre-crop: w={int(crop_w)} h={int(crop_h)} x=0 y=0")
                else:
                    self.logger.warning("Cannot apply TB crop: original video dimensions unknown or invalid.")

            software_filter_segments.extend(pre_v360_filters_temp)

            v360_filter_core = (
                f"v360={base_v360_input_format}:in_stereo=0:output=sg:"
                f"iv_fov={self.vr_fov}:ih_fov={self.vr_fov}:"
                f"d_fov={self.vr_fov}:"
                f"v_fov={v_h_FOV}:h_fov={v_h_FOV}:"
                f"pitch={self.vr_pitch}:yaw=0:roll=0:"
                f"w={self.yolo_input_size}:h={self.yolo_input_size}:interp=lanczos"
            )
            software_filter_segments.append(v360_filter_core)

        final_filter_chain_parts = []

        # If using 10-bit CUDA pipe, pipe1 handles initial HW work,
        # pipe2 decodes to CUDA, so hwdownload is still needed before SW filters in pipe2.
        if needs_hw_download and software_filter_segments:
            # Explicitly define the target format after downloading from hardware.
            # 'nv12' is a widely compatible format for subsequent software filters.
            final_filter_chain_parts.extend(["hwdownload", "format=nv12"])
            self.logger.info(
                "Prepending 'hwdownload,format=nv12' to the software filter chain."
            )

        final_filter_chain_parts.extend(software_filter_segments)
        ffmpeg_filter = ",".join(final_filter_chain_parts)

        self.logger.info(
            f"Built FFmpeg filter (effective for single pipe, or pipe2 of 10bit-CUDA): {ffmpeg_filter if ffmpeg_filter else 'No explicit filter, direct output.'}")
        return ffmpeg_filter

    def _get_ffmpeg_hwaccel_args(self) -> List[str]:
        """Determines FFmpeg hardware acceleration arguments based on app settings."""
        hwaccel_args: List[str] = []
        selected_hwaccel: str
        available_on_app: List[str] = []  # Default to empty list

        if self.app:
            # app_instance is provided, try to get settings from it.
            # Default to 'auto' for selection and an empty list for available accels
            # if the app_instance is present but missing these specific attributes.
            selected_hwaccel = getattr(self.app, 'hardware_acceleration_method', 'auto')
            available_on_app = getattr(self.app, 'available_ffmpeg_hwaccels', [])
            self.logger.debug(
                f"Getting HWAccel from app. Selected: '{selected_hwaccel}', App Available: {available_on_app}")
        else:
            # self.app is None. This means the VideoProcessor was not initialized with an app context.
            # Fallback to 'none' (CPU decoding) as the safest default because app-specific
            # hardware acceleration preferences and availability are unknown.
            self.logger.warning(
                "VideoProcessor's app instance is None. Hardware acceleration settings cannot be retrieved from app context. "
                "Defaulting to 'none' (CPU decoding) for safety.")
            selected_hwaccel = "none"
            # available_on_app remains [], which is consistent with a 'none' or unconfigurable state.

        system = platform.system().lower()
        machine = platform.machine().lower()  # e.g., 'x86_64', 'arm64', 'amd64'

        self.logger.debug(
            f"Determining HWAccel. Effective Selected: '{selected_hwaccel}', OS: {system}, Arch: {machine}, "
            f"Considering App Available: {available_on_app if self.app else 'N/A (no app instance)'}")

        if selected_hwaccel == "auto":
            # Auto-detection logic. This part relies on 'available_on_app'.
            # If 'self.app' was None, 'available_on_app' will be empty,
            # and this auto-detection block will likely not find any specific HW accel,
            # leading to CPU decoding, which is safe.
            if system == 'darwin':  # macOS
                if 'videotoolbox' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'videotoolbox']
                    self.logger.debug("Auto-selected 'videotoolbox' for macOS.")
            elif system == 'linux':
                if 'nvdec' in available_on_app:  # NVIDIA specific
                    hwaccel_args = ['-hwaccel', 'nvdec', '-hwaccel_output_format', 'cuda']
                    self.logger.debug("Auto-selected 'nvdec' (NVIDIA) for Linux.")
                elif 'cuda' in available_on_app:  # Older NVIDIA or if nvdec not listed
                    hwaccel_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
                    self.logger.debug("Auto-selected 'cuda' (NVIDIA) for Linux.")
                elif 'qsv' in available_on_app:  # Intel QuickSync
                    hwaccel_args = ['-hwaccel', 'qsv', '-hwaccel_output_format', 'qsv']
                    self.logger.debug("Auto-selected 'qsv' (Intel) for Linux.")
                elif 'vaapi' in available_on_app:  # Common for Intel/AMD on Linux
                    hwaccel_args = ['-hwaccel', 'vaapi', '-hwaccel_output_format', 'vaapi']
                    self.logger.debug("Auto-selected 'vaapi' for Linux.")
            elif system == 'windows':
                if 'nvdec' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'nvdec', '-hwaccel_output_format', 'cuda']
                    self.logger.debug("Auto-selected 'nvdec' (NVIDIA) for Windows.")
                elif 'cuda' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
                    self.logger.debug("Auto-selected 'cuda' (NVIDIA) for Windows.")
                elif 'qsv' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'qsv', '-hwaccel_output_format', 'qsv']
                    self.logger.debug("Auto-selected 'qsv' (Intel) for Windows.")
                elif 'd3d11va' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'd3d11va']
                    self.logger.debug("Auto-selected 'd3d11va' for Windows.")
                elif 'dxva2' in available_on_app:
                    hwaccel_args = ['-hwaccel', 'dxva2']
                    self.logger.debug("Auto-selected 'dxva2' for Windows.")

            if not hwaccel_args:
                self.logger.info(
                    "Auto hardware acceleration: No specific method matched based on app's available list (or app not provided/list empty), using CPU decoding.")
        elif selected_hwaccel != "none" and selected_hwaccel:
            # This case handles a specific hardware acceleration method selected by the user/app.
            # If 'available_on_app' (from app context) doesn't contain this selection,
            # or if 'available_on_app' is empty (e.g., because 'self.app' was None),
            # it will fall back to CPU.
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
                    f"Selected HW accel '{selected_hwaccel}' not in FFmpeg's list "
                    f"{'from app context' if self.app and available_on_app else '(app context not available or list empty/does not contain selection)'}. "
                    "Using CPU."
                )
        else:  # selected_hwaccel is "none" or an empty string
            self.logger.debug("Hardware acceleration explicitly disabled or not specified (CPU decoding).")
        return hwaccel_args

    def _terminate_ffmpeg_processes(self):
        """Safely terminates all active FFmpeg processes."""
        if self.ffmpeg_pipe1_process and self.ffmpeg_pipe1_process.poll() is None:
            self.logger.info("Terminating FFmpeg pipe 1 process.")
            if self.ffmpeg_pipe1_process.stdout: self.ffmpeg_pipe1_process.stdout.close()
            if self.ffmpeg_pipe1_process.stderr: self.ffmpeg_pipe1_process.stderr.close()
            self.ffmpeg_pipe1_process.terminate()
            try:
                self.ffmpeg_pipe1_process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.logger.warning("FFmpeg pipe 1 process kill timeout, killing.")
                self.ffmpeg_pipe1_process.kill()
                self.ffmpeg_pipe1_process.wait()
        self.ffmpeg_pipe1_process = None

        if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
            self.logger.info("Terminating main FFmpeg process (or pipe 2).")
            if self.ffmpeg_process.stdout: self.ffmpeg_process.stdout.close()
            if self.ffmpeg_process.stderr: self.ffmpeg_process.stderr.close()
            self.ffmpeg_process.terminate()
            try:
                self.ffmpeg_process.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.logger.warning("Main FFmpeg process kill timeout, killing.")
                self.ffmpeg_process.kill()
                self.ffmpeg_process.wait()
        self.ffmpeg_process = None

    def _start_ffmpeg_process(self, start_frame_abs_idx=0, num_frames_to_output_ffmpeg=None):
        self._terminate_ffmpeg_processes()  # Ensure any old processes are gone

        if not self.video_path or not self.video_info or self.video_info.get('fps', 0) <= 0:
            self.logger.warning("Cannot start FFmpeg: video not properly opened or invalid FPS.")
            return False

        start_time_seconds = start_frame_abs_idx / self.video_info['fps']
        self.current_stream_start_frame_abs = start_frame_abs_idx
        self.frames_read_from_current_stream = 0

        self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
        common_ffmpeg_prefix = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']

        if self._is_10bit_cuda_pipe_needed():
            self.logger.info("Using 2-pipe FFmpeg command for 10-bit CUDA video.")

            # Pipe 1: Pre-processing (crop, scale_cuda) and encode to HEVC
            # Inspired by user's get_ffmpeg_read_cmd example for 10-bit CUDA
            # The crop video.height:video.height:0:0 and scale_cuda=1000:1000 is very specific.
            # Adapting it to use self.video_info['height'].
            # This specific pre-processing might need to be more generic or configurable.
            video_height_for_crop = self.video_info.get('height', 0)
            if video_height_for_crop <= 0:
                self.logger.error("Cannot construct 10-bit CUDA pipe 1: video height is unknown or invalid.")
                return False

            # This VF for pipe 1 is taken from the example, its general applicability should be reviewed.
            # It crops to a square based on height, then scales to 1000x1000.
            pipe1_vf = f"crop={int(video_height_for_crop)}:{int(video_height_for_crop)}:0:0,scale_cuda=1000:1000"

            cmd1 = common_ffmpeg_prefix[:]
            # Input options for pipe 1 (HW accel for input, seek if needed)
            cmd1.extend(['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda'])
            if start_time_seconds > 0.001: cmd1.extend(['-ss', str(start_time_seconds)])
            cmd1.extend(['-i', self.video_path])
            cmd1.extend(['-an', '-sn'])
            cmd1.extend(['-vf', pipe1_vf])
            cmd1.extend(['-c:v', 'hevc_nvenc', '-preset', 'fast', '-qp', '0'])  # Output HEVC
            cmd1.extend(['-f', 'matroska', 'pipe:1'])  # Output to pipe

            # Pipe 2: Main processing (takes input from pipe 1)
            cmd2 = common_ffmpeg_prefix[:]
            cmd2.extend(['-hwaccel', 'cuda'])  # HW accel for decoding the HEVC from pipe1
            cmd2.extend(['-i', 'pipe:0'])  # Input from pipe1
            cmd2.extend(['-an', '-sn'])

            effective_vf_pipe2 = self.ffmpeg_filter_string
            if not effective_vf_pipe2: effective_vf_pipe2 = f"scale={self.yolo_input_size}:{self.yolo_input_size}"  #
            cmd2.extend(['-vf', effective_vf_pipe2])

            if num_frames_to_output_ffmpeg and num_frames_to_output_ffmpeg > 0:
                cmd2.extend(['-frames:v', str(num_frames_to_output_ffmpeg)])
            cmd2.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

            self.logger.info(f"Pipe 1 CMD: {' '.join(shlex.quote(str(x)) for x in cmd1)}")
            self.logger.info(f"Pipe 2 CMD: {' '.join(shlex.quote(str(x)) for x in cmd2)}")

            try:
                self.ffmpeg_pipe1_process = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                if self.ffmpeg_pipe1_process.stdout is None:  # Should not happen with stdout=PIPE
                    self.logger.error("Pipe 1 stdout is None. Cannot start FFmpeg.")
                    self._terminate_ffmpeg_processes()
                    return False

                self.ffmpeg_process = subprocess.Popen(cmd2, stdin=self.ffmpeg_pipe1_process.stdout,
                                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                       bufsize=self.frame_size_bytes * 5)  #
                # Allow p1 to receive a SIGPIPE if p2 exits.
                if self.ffmpeg_pipe1_process.stdout:  # Check again as it might be consumed by Popen
                    self.ffmpeg_pipe1_process.stdout.close()
                return True
            except Exception as e:
                self.logger.error(f"Failed to start 2-pipe FFmpeg: {e}", exc_info=True)  #
                self._terminate_ffmpeg_processes()
                return False
        else:
            # Standard single FFmpeg process
            hwaccel_cmd_list = self._get_ffmpeg_hwaccel_args()  #
            ffmpeg_input_options = hwaccel_cmd_list[:]  #
            if start_time_seconds > 0.001: ffmpeg_input_options.extend(['-ss', str(start_time_seconds)])  #

            cmd = common_ffmpeg_prefix + ffmpeg_input_options + ['-i', self.video_path, '-an', '-sn']  #

            effective_vf = self.ffmpeg_filter_string  #
            if not effective_vf: effective_vf = f"scale={self.yolo_input_size}:{self.yolo_input_size}"  #
            cmd.extend(['-vf', effective_vf])  #

            if num_frames_to_output_ffmpeg and num_frames_to_output_ffmpeg > 0:  #
                cmd.extend(['-frames:v', str(num_frames_to_output_ffmpeg)])  #
            cmd.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])  #

            self.logger.info(f"Single Pipe CMD: {' '.join(shlex.quote(str(x)) for x in cmd)}")  #
            try:
                self.ffmpeg_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,  #
                                                       bufsize=self.frame_size_bytes * 5)
                return True
            except Exception as e:
                self.logger.error(f"Failed to start FFmpeg: {e}", exc_info=True)  #
                self.ffmpeg_process = None  #
                return False

    def start_processing(self, start_frame=None, end_frame=None):
        if self.is_processing:
            self.logger.warning("Already processing.")
            return
        if not self.video_path or not self.video_info:
            self.logger.warning("Video not loaded.")
            return

        self.processing_start_frame_limit = self.current_frame_index  # Default to current
        if start_frame is not None and 0 <= start_frame < self.total_frames:
            self.processing_start_frame_limit = start_frame
        elif start_frame is not None:  # Invalid start_frame
            self.logger.warning(f"Start frame {start_frame} out of bounds ({self.total_frames} total). Not starting.")
            return

        self.processing_end_frame_limit = -1  # Means no end limit by default
        if end_frame is not None and end_frame >= 0:
            self.processing_end_frame_limit = min(end_frame, self.total_frames - 1)

        if not self._start_ffmpeg_process(start_frame_abs_idx=self.processing_start_frame_limit):
            self.logger.error("Failed to start FFmpeg for processing start.")
            return

        self.is_processing = True
        self.is_paused = False
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        if self.tracker:
            self.tracker.start_tracking()
        self.logger.info(
            f"Started GUI processing. Range: {self.processing_start_frame_limit} to "
            f"{self.processing_end_frame_limit if self.processing_end_frame_limit != -1 else 'EOS'}")

    def pause_processing(self):
        if not self.is_processing:
            return

        self.logger.info("Pausing video processing...")
        was_scripting_session = self.enable_tracker_processing and self.tracker and self.tracker.tracking_active
        scripted_range = (self.processing_start_frame_limit, self.current_frame_index)

        self.is_processing = False  # Signal loop to stop
        self.is_paused = True
        self.stop_event.set()  # Also signal via event

        thread_to_join = self.processing_thread
        if thread_to_join and thread_to_join.is_alive():
            if threading.current_thread() is not thread_to_join:
                self.logger.info(f"Joining processing thread: {thread_to_join.name}")
                thread_to_join.join(timeout=1.0)  # Wait for loop to exit
                if thread_to_join.is_alive():
                    self.logger.warning("Processing thread did not join cleanly after pause signal.")
        self.processing_thread = None

        if self.tracker and self.enable_tracker_processing:
            self.logger.info("Signaling tracker to stop/pause.")
            self.tracker.stop_tracking()

        if self.app:
            self.app.on_processing_stopped(was_scripting_session=was_scripting_session,
                                           scripted_frame_range=scripted_range)

        self.logger.info(f"Video processing paused. Current frame index: {self.current_frame_index}")

    def stop_processing(self, join_thread=True):
        is_currently_processing = self.is_processing
        is_thread_alive = self.processing_thread and self.processing_thread.is_alive()

        if not is_currently_processing and not is_thread_alive:
            self._terminate_ffmpeg_processes()
            return

        self.logger.info("Stopping GUI processing...")
        was_scripting_session = self.enable_tracker_processing and self.tracker and self.tracker.tracking_active
        scripted_range = (self.processing_start_frame_limit, self.current_frame_index)

        self.is_processing = False
        self.is_paused = True
        self.stop_event.set()

        if join_thread:
            thread_to_join = self.processing_thread
            if thread_to_join and thread_to_join.is_alive():
                if threading.current_thread() is not thread_to_join:
                    self.logger.info(f"Joining processing thread: {thread_to_join.name} during stop.")
                    thread_to_join.join(timeout=1.0)
                    if thread_to_join.is_alive():
                        self.logger.warning("Processing thread did not join cleanly after stop signal.")
        self.processing_thread = None

        self._terminate_ffmpeg_processes()

        if self.tracker:
            self.logger.info("Signaling tracker to stop.")
            self.tracker.stop_tracking()

        if self.app:
            self.app.on_processing_stopped(was_scripting_session=was_scripting_session,
                                           scripted_frame_range=scripted_range)
            if hasattr(self.app, 'on_processing_stopped') and callable(getattr(self.app, 'on_processing_stopped')):
                if 'was_scripting_session' not in self.app.on_processing_stopped.__code__.co_varnames:
                    self.app.on_processing_stopped()  # Fallback for older signature if needed

        self.logger.info("GUI processing stopped.")

    def seek_video(self, frame_index: int):
        if not self.video_info or self.video_info.get('fps', 0) <= 0 or self.total_frames <= 0: return
        target_frame = max(0, min(frame_index, self.total_frames - 1))

        was_processing = self.is_processing
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

        if was_processing:
            self.start_processing(start_frame=self.current_frame_index, end_frame=stored_end_limit)

    def is_vr_active_or_potential(self) -> bool:
        if self.video_type_setting == 'VR':
            return True
        if self.video_type_setting == 'auto':  #
            if self.video_info and self.determined_video_type == 'VR':  # If video loaded and determined VR
                return True
            # If no video is loaded, 'auto' might still mean VR settings are relevant if user expects to load VR
            # This depends on UX, True allows showing VR settings pre-load.
            # if not self.video_path: return True
        return False

    def display_current_frame(self):  # This method is called by GUI to get the frame for display
        if not self.video_path or not self.video_info:
            # self.logger.debug("display_current_frame: No video path or info.")
            return  # No video loaded

        with self.frame_lock:
            raw_frame_to_process = self.current_frame
        if raw_frame_to_process is None: return
        if self.tracker and self.enable_tracker_processing:
            fps_for_timestamp = self.fps if self.fps > 0 else 30.0
            timestamp_ms = int(self.current_frame_index * (1000.0 / fps_for_timestamp))
            try:
                if not self.is_processing:  # If paused, apply tracker for display of static frame
                    processed_frame_tuple = self.tracker.process_frame(raw_frame_to_process.copy(), timestamp_ms)
                    with self.frame_lock: self.current_frame = processed_frame_tuple[0]
            except Exception as e:
                self.logger.error(f"Error processing frame with tracker in display_current_frame: {e}", exc_info=True)

    def _processing_loop(self):  #
        if not self.ffmpeg_process or self.ffmpeg_process.stdout is None:
            self.logger.error("_processing_loop: FFmpeg process/stdout not available. Exiting.")
            self.is_processing = False
            if self.ffmpeg_process and self.ffmpeg_process.poll() is None:
                self.ffmpeg_process.kill()
            self.ffmpeg_process = None
            return

        loop_ffmpeg_process = self.ffmpeg_process

        # Initialize next_frame_target_time using perf_counter for higher precision
        next_frame_target_time = time.perf_counter()
        self.last_processed_chapter_id = None  # Reset chapter state at start of loop

        try:  #
            while not self.stop_event.is_set() and self.is_processing:

                # --- Calculate target_delay INSIDE the loop to use the latest self.target_fps ---
                target_delay = 1.0 / self.target_fps if self.target_fps > 0 else (1.0 / 30.0)

                # --- Context-aware tracker reconfiguration ---
                current_chapter = self.app.funscript_processor.get_chapter_at_frame(self.current_frame_index)
                current_chapter_id = current_chapter.unique_id if current_chapter else None

                if current_chapter_id != self.last_processed_chapter_id:
                    if self.tracker:
                        if current_chapter and current_chapter.user_roi_fixed:
                            self.tracker.reconfigure_for_chapter(current_chapter)
                        else:  # In a gap or a chapter without a configured ROI
                            self.tracker.set_tracking_mode("YOLO_ROI")  # Fallback mode
                            #self.tracker.roi = None  # Invalidate ROI
                            #self.tracker.stop_tracking()  # Stop generating actions
                    self.last_processed_chapter_id = current_chapter_id

                # If we entered a chapter and tracking is not active, start it
                if current_chapter and self.tracker and not self.tracker.tracking_active and current_chapter.user_roi_fixed:
                    self.tracker.start_tracking()


                # --- Start of Frame Work (Existing logic) ---
                if self.ffmpeg_pipe1_process and self.ffmpeg_pipe1_process.poll() is not None:
                    pipe1_stderr = ""
                    if self.ffmpeg_pipe1_process.stderr:
                        try:
                            pipe1_stderr = self.ffmpeg_pipe1_process.stderr.read(4096).decode(errors='ignore')
                        except:
                            pass
                    self.logger.warning(
                        f"FFmpeg Pipe 1 died. Exit: {self.ffmpeg_pipe1_process.returncode}. Stderr: {pipe1_stderr.strip()}. Stopping.")
                    self.is_processing = False
                    break

                if loop_ffmpeg_process.poll() is not None:
                    stderr_output = ""
                    if loop_ffmpeg_process.stderr:
                        try:
                            stderr_output = loop_ffmpeg_process.stderr.read(4096).decode(errors='ignore')
                        except:
                            pass
                    self.logger.info(
                        f"FFmpeg output process died unexpectedly in loop. Exit: {loop_ffmpeg_process.returncode}. Stderr: {stderr_output.strip()}. Stopping GUI processing.")
                    self.is_processing = False
                    break

                raw_frame_bytes = loop_ffmpeg_process.stdout.read(self.frame_size_bytes)
                if len(raw_frame_bytes) < self.frame_size_bytes:
                    self.logger.info(
                        f"End of FFmpeg GUI stream or incomplete frame (read {len(raw_frame_bytes)}/{self.frame_size_bytes}).")
                    self.is_processing = False

                    if self.app:  # Check if app exists before calling
                        was_scripting_at_end = self.enable_tracker_processing
                        end_range = (self.processing_start_frame_limit, self.current_frame_index)
                        self.app.on_processing_stopped(was_scripting_session=was_scripting_at_end,
                                                       scripted_frame_range=end_range)

                    break

                self.current_frame_index = self.current_stream_start_frame_abs + self.frames_read_from_current_stream
                self.frames_read_from_current_stream += 1

                if self.processing_end_frame_limit != -1 and self.current_frame_index > self.processing_end_frame_limit:
                    self.logger.info(f"Reached GUI end_frame_limit ({self.processing_end_frame_limit}). Stopping.")
                    self.is_processing = False
                    if self.app:
                        was_scripting_at_end_limit = self.enable_tracker_processing
                        end_range_limit = (self.processing_start_frame_limit, self.processing_end_frame_limit)
                        self.app.on_processing_stopped(was_scripting_session=was_scripting_at_end_limit,
                                                       scripted_frame_range=end_range_limit)
                    break
                if self.total_frames > 0 and self.current_frame_index >= self.total_frames:
                    self.logger.info("Reached end of video. Stopping GUI processing.")
                    self.is_processing = False
                    if self.app:
                        was_scripting_at_eos = self.enable_tracker_processing
                        end_range_eos = (self.processing_start_frame_limit, self.current_frame_index)
                        self.app.on_processing_stopped(was_scripting_session=was_scripting_at_eos,
                                                       scripted_frame_range=end_range_eos)
                    break

                frame_np = np.frombuffer(raw_frame_bytes, dtype=np.uint8).reshape(self.yolo_input_size,
                                                                                  self.yolo_input_size, 3)
                processed_frame_for_gui = frame_np
                if self.tracker and self.enable_tracker_processing:
                    timestamp_ms = int(self.current_frame_index * (1000.0 / self.fps)) if self.fps > 0 else int(
                        time.time() * 1000)
                    try:
                        processed_frame_for_gui = self.tracker.process_frame(frame_np.copy(), timestamp_ms)[0]
                    except Exception as e:
                        self.logger.error(f"Error in tracker.process_frame during loop: {e}", exc_info=True)

                with self.frame_lock:
                    self.current_frame = processed_frame_for_gui

                self.frames_for_fps_calc += 1
                current_time_fps_calc = time.time()  # Use time.time for this specific periodic update
                if current_time_fps_calc - self.last_fps_update_time >= 1.0:
                    self.actual_fps = self.frames_for_fps_calc / (current_time_fps_calc - self.last_fps_update_time)
                    self.last_fps_update_time = current_time_fps_calc
                    self.frames_for_fps_calc = 0
                # --- End of Frame Work ---

                # --- Revised Frame Rate Limiting ---
                current_time = time.perf_counter()
                sleep_duration = next_frame_target_time - current_time

                if sleep_duration > 0:
                    time.sleep(sleep_duration)

                # Adjust next_frame_target_time for the next iteration
                if next_frame_target_time < current_time - target_delay:
                    # Lagged significantly (more than one frame duration behind)
                    # Reset to avoid trying to "catch up" too aggressively with a burst
                    # self.logger.debug(f"Frame pacing reset: Lag was {current_time - next_frame_target_time:.4f}s")
                    next_frame_target_time = current_time + target_delay
                else:
                    # Schedule the next frame from the previous target time
                    next_frame_target_time += target_delay

        finally:
            self.logger.info(
                f"_processing_loop ending. is_processing: {self.is_processing}, stop_event: {self.stop_event.is_set()}")
            self._terminate_ffmpeg_processes()
            self.is_processing = False
            self.last_processed_chapter_id = None
            self.stop_event.clear()

    def _start_ffmpeg_for_segment_streaming(self, start_frame_abs_idx: int,
                                            num_frames_to_stream_hint: Optional[int] = None) -> bool:
        # This method currently does NOT support the 2-pipe 10-bit CUDA system.
        # It would require significant refactoring similar to _start_ffmpeg_process and _get_specific_frame.
        # For now, it will use the single-pipe logic, which might fail or be slow for 10-bit CUDA.
        self.logger.warning("Segment streaming currently does not use the 2-pipe 10-bit CUDA optimization.")
        self._terminate_ffmpeg_processes()

        if not self.video_path or not self.video_info or self.video_info.get('fps', 0) <= 0:
            self.logger.warning("Cannot start FFmpeg for segment: no video/invalid FPS.")
            return False
        start_time_seconds = start_frame_abs_idx / self.video_info['fps']
        hwaccel_cmd_list = self._get_ffmpeg_hwaccel_args()
        ffmpeg_cmd_prefix = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']
        ffmpeg_input_options = hwaccel_cmd_list[:]
        if start_time_seconds > 0.001: ffmpeg_input_options.extend(['-ss', str(start_time_seconds)])
        ffmpeg_cmd = ffmpeg_cmd_prefix + ffmpeg_input_options + ['-i', self.video_path, '-an', '-sn']
        effective_vf = self.ffmpeg_filter_string
        if not effective_vf: effective_vf = f"scale={self.yolo_input_size}:{self.yolo_input_size}"
        ffmpeg_cmd.extend(['-vf', effective_vf])

        if num_frames_to_stream_hint and num_frames_to_stream_hint > 0:
            ffmpeg_cmd.extend(['-frames:v', str(num_frames_to_stream_hint)])

        ffmpeg_cmd.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])
        # self.frame_size_bytes is already set

        self.logger.info(f"Segment CMD (single pipe only): {' '.join(shlex.quote(str(x)) for x in ffmpeg_cmd)}")  #
        try:
            # For segment streaming, a larger buffer might be good if reads are bursty
            self.ffmpeg_process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                                   bufsize=self.frame_size_bytes * 20)
            return True
        except Exception as e:
            self.logger.warning(f"Failed to start FFmpeg for segment: {e}", exc_info=True)
            self.ffmpeg_process = None
            return False

    def stream_frames_for_segment(self, start_frame_abs_idx: int, num_frames_to_read: int) -> Iterator[
        Tuple[int, np.ndarray]]:
        # This method currently does NOT support the 2-pipe 10-bit CUDA system due to _start_ffmpeg_for_segment_streaming limitation.
        if num_frames_to_read <= 0:
            self.logger.warning("num_frames_to_read is not positive, no frames to stream.")
            return  # Use return for empty iterator

        if not self._start_ffmpeg_for_segment_streaming(start_frame_abs_idx, num_frames_to_read):
            self.logger.warning(f"Failed to start FFmpeg for segment from {start_frame_abs_idx}.")
            return

        frames_yielded = 0
        segment_ffmpeg_process = self.ffmpeg_process  # Capture for this specific stream
        try:
            for i in range(num_frames_to_read):
                if not segment_ffmpeg_process or segment_ffmpeg_process.stdout is None:
                    self.logger.warning("FFmpeg process or stdout not available during segment streaming.")
                    break

                # Check if process terminated prematurely
                if segment_ffmpeg_process.poll() is not None:
                    stderr_output = ""
                    if segment_ffmpeg_process.stderr:
                        try:
                            stderr_output = segment_ffmpeg_process.stderr.read(4096).decode(errors='ignore')
                        except:
                            pass
                    self.logger.warning(
                        f"FFmpeg process (segment) terminated prematurely. Exit: {segment_ffmpeg_process.returncode}. Stderr: '{stderr_output.strip()}'")
                    break

                raw_frame_bytes = segment_ffmpeg_process.stdout.read(self.frame_size_bytes)
                if len(raw_frame_bytes) < self.frame_size_bytes:
                    stderr_on_short_read = ""
                    if segment_ffmpeg_process.stderr:
                        try:
                            stderr_on_short_read = segment_ffmpeg_process.stderr.read(4096).decode(errors='ignore')
                        except:
                            pass
                    self.logger.info(
                        f"End of FFmpeg stream or error (read {len(raw_frame_bytes)}/{self.frame_size_bytes}) "
                        f"after {frames_yielded} frames for segment (start {start_frame_abs_idx}). Stderr: '{stderr_on_short_read.strip()}'")
                    break

                frame_np = np.frombuffer(raw_frame_bytes, dtype=np.uint8).reshape(self.yolo_input_size,
                                                                                  self.yolo_input_size, 3)
                current_frame_id = start_frame_abs_idx + frames_yielded
                yield current_frame_id, frame_np
                frames_yielded += 1
        finally:
            self._terminate_ffmpeg_processes()

    def set_target_fps(self, fps: float):
        self.target_fps = max(1.0, fps if fps > 0 else 1.0)  # Ensure positive FPS
        self.logger.info(f"Target FPS set to: {self.target_fps}")

    def is_video_open(self) -> bool:
        """Checks if a video is currently loaded and has valid information."""
        return bool(self.video_path and self.video_info and self.video_info.get('total_frames', 0) > 0)

    def reset(self, close_video=False):
        self.logger.info("Resetting VideoProcessor...")
        self.stop_processing(join_thread=True)
        self._clear_cache()
        self.current_frame_index = 0
        self.frames_read_from_current_stream = 0
        self.current_stream_start_frame_abs = 0
        self.seek_request_frame_index = None
        if self.tracker:
            self.tracker.reset()
        if close_video:
            self.video_path = ""
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
            if 'was_scripting_session' in self.app.on_processing_stopped.__code__.co_varnames:
                self.app.on_processing_stopped(was_scripting_session=False)
            else:
                self.app.on_processing_stopped()
        self.logger.info("VideoProcessor reset complete.")

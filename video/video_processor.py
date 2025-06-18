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
    """
    Manages video loading, processing, and frame retrieval using FFmpeg.

    This class provides a comprehensive interface for handling video files,
    including:
    - Opening and parsing video metadata using ffprobe.
    - A flexible FFmpeg pipeline for frame decoding, supporting various hardware
      accelerations (CUDA, QSV, VAAPI, VideoToolbox).
    - Specialized 2-pipe processing for high-bit-depth (e.g., 10-bit) videos
      on CUDA to handle filtering requirements.
    - Heuristics for auto-detecting video type (2D vs. VR) and specific
      VR formats (Equirectangular/Fisheye, SBS/TB).
    - A real-time processing loop for continuous frame delivery, with rate
      limiting and support for a real-time 'tracker' object.
    - Efficient frame seeking with an intelligent caching mechanism.
    - On-demand batch frame fetching and segment streaming.
    - Scene detection and audio waveform generation capabilities.
    """
    # --- Constants for Configuration ---
    VIDEO_TYPE_AUTO = 'auto'
    VIDEO_TYPE_2D = '2D'
    VIDEO_TYPE_VR = 'VR'

    HWACCEL_NONE = 'none'
    HWACCEL_AUTO = 'auto'
    HWACCEL_CUDA = 'cuda'
    HWACCEL_NVDEC = 'nvdec'
    HWACCEL_QSV = 'qsv'
    HWACCEL_VAAPI = 'vaapi'
    HWACCEL_D3D11VA = 'd3d11va'
    HWACCEL_DXVA2 = 'dxva2'
    HWACCEL_VIDEOTOOLBOX = 'videotoolbox'

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

        self.enable_tracker_processing = False
        if self.tracker is None:
            if self.logger: self.logger.info("No tracker provided. Tracker processing will be disabled.")
        else:
            self.logger.info(
                "Tracker is available, but processing is DISABLED by default. An explicit call is needed to enable it.")

        # Frame Caching
        self.frame_cache = OrderedDict()
        self.frame_cache_max_size = cache_size
        self.frame_cache_lock = threading.Lock()
        self.batch_fetch_size = 50

    # --- Scene Detection Method ---
    def detect_scenes(self, threshold: float = 27.0, progress_callback=None,
                      stop_event: Optional[threading.Event] = None) -> List[Tuple[int, int]]:
        """
        Uses PySceneDetect to find scene cuts with a manual processing loop.

        This method supports progress reporting and cancellation. It includes a
        workaround to call a private method in PySceneDetect for frame processing,
        which may be necessary for certain library versions.

        Args:
            threshold: The threshold for the ContentDetector.
            progress_callback: A function to call with progress updates.
            stop_event: A threading.Event to signal cancellation.

        Returns:
            A list of (start_frame, end_frame) tuples for each detected scene.
        """
        if not self.video_path:
            self.logger.error("Cannot detect scenes: No video loaded.")
            return []

        self.logger.info("Starting scene detection...")
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
                    break  # End of video

                if stop_event and stop_event.is_set():
                    raise InterruptedError("Scene detection cancelled by user.")

                # This is a documented workaround for some versions of PySceneDetect
                # to enable manual processing loops with progress reporting.
                scene_manager._process_frame(frame_image, video.frame_number)

                if progress_callback:
                    progress_callback(video.frame_number, total_frames)

            scene_list_raw = scene_manager.get_scene_list()

            # Ensure the last scene extends to the very end of the video
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
        """Clears the internal frame cache."""
        with self.frame_cache_lock:
            if self.frame_cache:
                self.logger.debug(f"Clearing frame cache (had {len(self.frame_cache)} items).")
                self.frame_cache.clear()

    def set_active_video_type_setting(self, video_type: str):
        """Sets the user-configured video type ('auto', '2D', 'VR')."""
        if video_type not in [self.VIDEO_TYPE_AUTO, self.VIDEO_TYPE_2D, self.VIDEO_TYPE_VR]:
            self.logger.warning(f"Invalid video_type: {video_type}.")
            return
        if self.video_type_setting != video_type:
            self.video_type_setting = video_type
            self.logger.info(f"Video type setting changed to: {self.video_type_setting}.")

    def set_active_yolo_input_size(self, size: int):
        """Sets the resolution for frames being processed by YOLO."""
        if size <= 0:
            self.logger.warning(f"Invalid yolo_input_size: {size}.")
            return
        if self.yolo_input_size != size:
            self.yolo_input_size = size
            self.logger.info(f"YOLO input size changed to: {self.yolo_input_size}.")
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3

    def set_active_vr_parameters(self, fov: Optional[int] = None, pitch: Optional[int] = None,
                                 input_format: Optional[str] = None):
        """Updates VR processing parameters like FOV, pitch, and input format."""
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
        """Enables or disables the external tracker processing."""
        if enable and self.tracker is None:
            self.logger.warning("Cannot enable tracker processing because no tracker is available.")
            self.enable_tracker_processing = False
        else:
            self.enable_tracker_processing = enable
            self.logger.info(f"Tracker processing {'enabled' if enable else 'disabled'}.")

    def _autodetect_video_properties(self, video_path: str, video_info: dict) -> Tuple[str, Optional[str]]:
        """
        Analyzes video metadata and filename to determine its type (2D/VR) and VR format.

        Args:
            video_path: The full path to the video file.
            video_info: The dictionary of video properties from _get_video_info.

        Returns:
            A tuple containing:
            - The determined video type ('2D' or 'VR').
            - The suggested VR input format (e.g., 'he_sbs') or None if not VR.
        """
        # Heuristic 1: Determine if the video is '2D' or 'VR'
        is_sbs = (video_info['width'] >= 1.8 * video_info['height'] and
                  video_info['width'] <= 2.2 * video_info['height'] and
                  video_info['width'] > 1000)
        is_tb = (video_info['height'] >= 1.8 * video_info['width'] and
                 video_info['height'] <= 2.2 * video_info['width'] and
                 video_info['height'] > 1000)

        upper_path = video_path.upper()
        vr_keywords = ['VR', '_180', '_360', 'SBS', '_TB', 'FISHEYE', 'EQUIRECTANGULAR', 'LR_', 'Oculus', '_3DH']
        has_vr_keyword = any(kw in upper_path for kw in vr_keywords)

        determined_type = self.VIDEO_TYPE_2D
        if is_sbs or is_tb or has_vr_keyword:
            determined_type = self.VIDEO_TYPE_VR
        self.logger.info(
            f"Auto-detection found: Type={determined_type} (SBS Res: {is_sbs}, TB Res: {is_tb}, Keyword: {has_vr_keyword})")

        if determined_type == self.VIDEO_TYPE_2D:
            return self.VIDEO_TYPE_2D, None

        # Heuristic 2: Determine specific VR format (e.g., he_sbs, fisheye_tb)
        base = 'he'  # Default to 'he' for equirectangular
        layout = '_sbs'  # Default to side-by-side
        if is_tb:
            layout = '_tb'
            self.logger.info("Resolution suggests Top-Bottom (TB) layout.")

        fisheye_keywords = ['FISHEYE', 'MKX', 'RF52']
        if any(kw in upper_path for kw in fisheye_keywords):
            base = 'fisheye'
            self.logger.info("Filename keyword suggests 'fisheye' base format.")

        tb_keywords = ['_TB', 'TB_', 'TOPBOTTOM', 'OVERUNDER', '_OU', 'OU_']
        if any(kw in upper_path for kw in tb_keywords):
            layout = '_tb'
            self.logger.info("Filename keyword confirms 'Top-Bottom' layout.")
        elif any(kw in upper_path for kw in ['SBS']):
            layout = '_sbs'
            self.logger.info("Filename keyword confirms 'Side-by-Side' layout.")

        suggested_format = f"{base}{layout}"
        return self.VIDEO_TYPE_VR, suggested_format

    def open_video(self, video_path: str, from_project_load: bool = False) -> bool:
        """
        Opens a video file, gets its properties, and configures the processor.
        """
        self.stop_processing()
        self.video_path = video_path
        self._clear_cache()
        self.video_info = self._get_video_info(video_path)

        if not self.video_info or self.video_info.get("total_frames", 0) == 0:
            self.logger.warning(f"Failed to get valid video info for {video_path}")
            self.video_path = ""
            self.video_info = {}
            return False

        # --- Auto-detection logic ---
        if self.video_type_setting == self.VIDEO_TYPE_AUTO:
            det_type, sugg_vr_format = self._autodetect_video_properties(video_path, self.video_info)
            self.determined_video_type = det_type
            if det_type == self.VIDEO_TYPE_VR and sugg_vr_format:
                if self.vr_input_format != sugg_vr_format:
                    self.logger.info(
                        f"Auto-detection suggests VR format: {sugg_vr_format}. Updating from '{self.vr_input_format}'.")
                    self.vr_input_format = sugg_vr_format
        else:
            self.determined_video_type = self.video_type_setting
            self.logger.info(f"Using configured video type: {self.determined_video_type}")

        # --- Apply specific heuristics after main detection ---
        if self.determined_video_type == self.VIDEO_TYPE_VR:
            upper_path = video_path.upper()
            if 'MKX' in upper_path and 'fisheye' in self.vr_input_format and self.vr_fov != 200:
                self.logger.info(f"Filename suggests VR FOV: 200 (MKX). Overriding current: {self.vr_fov}")
                self.vr_fov = 200

        # --- Finalize setup ---
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
            f"Opened: {os.path.basename(video_path)} ({self.determined_video_type}, "
            f"format: {self.vr_input_format if self.determined_video_type == self.VIDEO_TYPE_VR else 'N/A'}), "
            f"{self.total_frames}fr, {self.fps:.2f}fps, {self.video_info.get('bit_depth', 'N/A')}bit)")
        return True

    def reapply_video_settings(self):
        """
        Re-evaluates and applies video settings (like VR format) to the currently loaded video.
        """
        if not self.is_video_open():
            self.logger.info("No video loaded. Settings will apply when a video is opened.")
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
            return

        self.logger.info(f"Reapplying video settings (current vr_input_format: {self.vr_input_format})")
        was_processing = self.is_processing
        stored_frame_index = self.current_frame_index
        stored_end_limit = self.processing_end_frame_limit
        self.stop_processing()
        self._clear_cache()

        # --- Re-run auto-detection ---
        if self.video_type_setting == self.VIDEO_TYPE_AUTO:
            det_type, _ = self._autodetect_video_properties(self.video_path, self.video_info)
            self.determined_video_type = det_type
        else:
            self.determined_video_type = self.video_type_setting
        self.logger.info(f"Re-determined video type as: {self.determined_video_type}")

        if self.determined_video_type == self.VIDEO_TYPE_VR:
            upper_path = self.video_path.upper()
            if 'MKX' in upper_path and 'fisheye' in self.vr_input_format and self.vr_fov != 200:
                self.logger.info("Re-applying FOV heuristic: Filename suggests VR FOV: 200 (MKX).")
                self.vr_fov = 200
            # Note: We do not re-detect vr_input_format here, as the user may have set it manually.
            # We respect the current self.vr_input_format.
            self.logger.info(f"Using user-set VR Input Format for reapply: {self.vr_input_format}")

        # --- Finalize and refresh frame ---
        self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
        self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
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
        self.logger.info("Video settings reapplication complete.")

    def get_frames_batch(self, start_frame_num: int, num_frames_to_fetch: int) -> Dict[int, np.ndarray]:
        """
        Fetches a batch of frames using a dedicated FFmpeg process.
        This method supports both standard and 2-pipe 10-bit CUDA processing.
        """
        frames_batch: Dict[int, np.ndarray] = {}
        if not self.is_video_open() or num_frames_to_fetch <= 0:
            self.logger.warning("get_frames_batch: Video not properly opened or invalid params.")
            return frames_batch

        local_p1_proc: Optional[subprocess.Popen] = None
        local_p2_proc: Optional[subprocess.Popen] = None
        current_frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3

        try:
            # _start_ffmpeg_stream is a unified launcher for both 1-pipe and 2-pipe
            local_p1_proc, local_p2_proc = self._start_ffmpeg_stream(
                start_frame_abs_idx=start_frame_num,
                num_frames_to_output_ffmpeg=num_frames_to_fetch,
                is_for_batch=True  # Indicates this is a short-lived process
            )

            # --- Read frames from the final output pipe (local_p2_proc) ---
            if not local_p2_proc or local_p2_proc.stdout is None:
                self.logger.error("get_frames_batch: Output FFmpeg process or its stdout is None.")
                return frames_batch

            for i in range(num_frames_to_fetch):
                raw_frame_data = local_p2_proc.stdout.read(current_frame_size_bytes)
                if len(raw_frame_data) < current_frame_size_bytes:
                    p2_stderr = local_p2_proc.stderr.read().decode(errors='ignore') if local_p2_proc.stderr else ""
                    self.logger.warning(
                        f"get_frames_batch: Incomplete data for frame {start_frame_num + i}. "
                        f"P2 Stderr: {p2_stderr.strip()}")
                    if local_p1_proc and local_p1_proc.stderr:
                        p1_stderr = local_p1_proc.stderr.read().decode(errors='ignore')
                        self.logger.warning(f"get_frames_batch: P1 Stderr: {p1_stderr.strip()}")
                    break
                frames_batch[start_frame_num + i] = np.frombuffer(raw_frame_data, dtype=np.uint8).reshape(
                    self.yolo_input_size, self.yolo_input_size, 3)

        except Exception as e:
            self.logger.error(f"get_frames_batch: Error fetching batch @{start_frame_num}: {e}", exc_info=True)
        finally:
            # Terminate the local Popen objects used for this batch fetch
            if local_p1_proc:
                self._terminate_process(local_p1_proc, "local_p1_proc")
            if local_p2_proc:
                self._terminate_process(local_p2_proc, "local_p2_proc")

        self.logger.debug(
            f"get_frames_batch: Complete. Got {len(frames_batch)} frames for start {start_frame_num}.")
        return frames_batch

    def _get_specific_frame(self, frame_index_abs: int) -> Optional[np.ndarray]:
        """
        Retrieves a single frame, using a cache and batch-fetching for efficiency.
        """
        if not self.is_video_open():
            self.logger.warning("Cannot get frame: video not properly loaded.")
            self.current_frame_index = frame_index_abs
            return None

        with self.frame_cache_lock:
            if frame_index_abs in self.frame_cache:
                self.logger.debug(f"Cache HIT for frame {frame_index_abs}")
                frame = self.frame_cache[frame_index_abs]
                self.frame_cache.move_to_end(frame_index_abs)
                self.current_frame_index = frame_index_abs
                return frame

        self.logger.debug(f"Cache MISS for frame {frame_index_abs}. Fetching new batch.")

        # Center the batch fetch around the requested frame
        batch_start_frame = max(0, frame_index_abs - self.batch_fetch_size // 2)
        if self.total_frames > 0:
            # Ensure the batch doesn't go past the end of the video
            effective_end = self.total_frames - 1
            if batch_start_frame + self.batch_fetch_size > effective_end:
                batch_start_frame = max(0, effective_end - self.batch_fetch_size + 1)

        num_to_fetch = min(self.batch_fetch_size,
                           self.total_frames - batch_start_frame) if self.total_frames > 0 else self.batch_fetch_size

        fetched_batch = self.get_frames_batch(batch_start_frame, num_to_fetch)

        retrieved_frame: Optional[np.ndarray] = None
        with self.frame_cache_lock:
            # Add all newly fetched frames to the cache
            for idx, frame_data in fetched_batch.items():
                if len(self.frame_cache) >= self.frame_cache_max_size:
                    self.frame_cache.popitem(last=False)  # Remove the oldest item
                self.frame_cache[idx] = frame_data
                if idx == frame_index_abs:
                    retrieved_frame = frame_data

            if retrieved_frame is not None and frame_index_abs in self.frame_cache:
                self.frame_cache.move_to_end(frame_index_abs)

        self.current_frame_index = frame_index_abs
        if retrieved_frame is not None:
            self.logger.debug(f"Successfully retrieved frame {frame_index_abs} via batch fetch.")
        else:
            self.logger.warning(f"Failed to retrieve frame {frame_index_abs} after batch fetch.")
            # Final check in case of race condition
            with self.frame_cache_lock:
                retrieved_frame = self.frame_cache.get(frame_index_abs)

        return retrieved_frame

    def _get_video_info(self, filename: str) -> Optional[Dict]:
        """
        Uses ffprobe to extract essential metadata from the video file.
        """
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries',
               'stream=width,height,r_frame_rate,nb_frames,avg_frame_rate,duration,codec_type,pix_fmt,bits_per_raw_sample',
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

            # Check for audio stream
            has_audio = False
            cmd_audio = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                         '-show_entries', 'stream=codec_type', '-of', 'json', filename]
            try:
                res_audio = subprocess.run(cmd_audio, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
                                           text=True)
                if json.loads(res_audio.stdout).get('streams'): has_audio = True
            except Exception:
                pass  # No audio stream found

            if total_frames == 0:
                self.logger.warning("ffprobe gave 0 frames, trying OpenCV count as fallback...")
                cap = cv2.VideoCapture(filename)
                if cap.isOpened():
                    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    if fps <= 0: fps = cap.get(cv2.CAP_PROP_FPS) if cap.get(cv2.CAP_PROP_FPS) > 0 else 30.0
                    cap.release()
                else:
                    self.logger.error(f"OpenCV could not open video file: {filename}")

            # Determine bit depth
            bit_depth = 8  # Default
            raw_sample_str = stream_info.get('bits_per_raw_sample')
            if raw_sample_str and raw_sample_str.isdigit():
                bit_depth = int(raw_sample_str)
            else:
                pix_fmt = stream_info.get('pix_fmt', '').lower()
                if '10le' in pix_fmt or 'p010' in pix_fmt or '10be' in pix_fmt:
                    bit_depth = 10

            self.logger.info(
                f"Detected properties: {stream_info.get('width')}x{stream_info.get('height')}, "
                f"{fps:.2f}fps, bit_depth={bit_depth}, has_audio={has_audio}")

            return {"duration": duration, "total_frames": total_frames, "fps": fps,
                    "width": int(stream_info.get('width', 0)), "height": int(stream_info.get('height', 0)),
                    "has_audio": has_audio, "bit_depth": bit_depth}
        except Exception as e:
            self.logger.error(f"Error in _get_video_info for {filename}: {e}")
            return None

    def get_audio_waveform(self, num_samples: int = 1000) -> Optional[np.ndarray]:
        """
        Extracts the audio track, converts it to WAV, and generates a normalized waveform.
        Requires Scipy to be installed.
        """
        if not self.is_video_open() or not self.video_info.get("has_audio"):
            self.logger.info("No audio stream available for waveform generation.")
            return None
        if not SCIPY_AVAILABLE_FOR_AUDIO:
            self.logger.warning("Scipy is not available. Cannot generate audio waveform.")
            return None

        temp_wav_file = None
        process = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                temp_wav_file = tmp.name

            ffmpeg_cmd = [
                'ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error', '-i', self.video_path,
                '-vn', '-ac', '1', '-ar', '44100', '-c:a', 'pcm_s16le', '-y', temp_wav_file
            ]
            self.logger.info(f"Extracting audio for waveform...")
            process = subprocess.run(ffmpeg_cmd, timeout=60, capture_output=True, text=True)

            if process.returncode != 0:
                self.logger.error(f"FFmpeg failed to extract audio: {process.stderr}")
                return None

            samplerate, data = wavfile.read(temp_wav_file)
            if data.ndim > 1: data = data.mean(axis=1)
            if data.size == 0: return np.array([])

            step = max(1, len(data) // num_samples)
            waveform = [np.max(np.abs(data[i:i + step])) for i in range(0, len(data), step)]
            waveform_np = np.array(waveform, dtype=np.float32)

            max_val = np.max(waveform_np)
            if max_val > 0: waveform_np /= max_val

            self.logger.info(f"Generated waveform with {len(waveform_np)} samples.")
            return waveform_np

        except Exception as e:
            self.logger.error(f"Error generating audio waveform: {e}", exc_info=True)
            return None
        finally:
            if temp_wav_file and os.path.exists(temp_wav_file):
                try:
                    os.remove(temp_wav_file)
                except OSError as e_rem:
                    self.logger.warning(f"Could not remove temp WAV file: {e_rem}")

    def _is_10bit_cuda_pipe_needed(self) -> bool:
        """Checks if the special 2-pipe FFmpeg command for 10-bit CUDA should be used."""
        if not self.video_info: return False

        is_high_bit_depth = self.video_info.get('bit_depth', 8) > 8
        hwaccel_args = self._get_ffmpeg_hwaccel_args()
        is_cuda_hwaccel = self.HWACCEL_CUDA in hwaccel_args

        return is_high_bit_depth and is_cuda_hwaccel

    def _build_ffmpeg_filter_string(self) -> str:
        """Constructs the -vf (video filter) argument for the FFmpeg command."""
        if not self.video_info: return ''

        original_width = self.video_info.get('width', 0)
        original_height = self.video_info.get('height', 0)
        v_h_FOV = 90  # Output projection FOV

        current_hw_args = self._get_ffmpeg_hwaccel_args()
        needs_hw_download = '-hwaccel_output_format' in current_hw_args

        self.logger.info(
            f"Filter build check: needs_hw_download={needs_hw_download}, type={self.determined_video_type}")

        sw_filters = []
        if self.determined_video_type == self.VIDEO_TYPE_2D:
            sw_filters.append(
                f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease")
            sw_filters.append(f"pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black")

        elif self.determined_video_type == self.VIDEO_TYPE_VR:
            crop_filter = ""
            if '_sbs' in self.vr_input_format and original_width > 0:
                crop_filter = f"crop={int(original_width / 2)}:{original_height}:0:0"
            elif '_tb' in self.vr_input_format and original_height > 0:
                crop_filter = f"crop={original_width}:{int(original_height / 2)}:0:0"
            if crop_filter: sw_filters.append(crop_filter)

            base_format = self.vr_input_format.replace('_sbs', '').replace('_tb', '')
            v360_filter = (
                f"v360={base_format}:in_stereo=0:output=sg:"
                f"iv_fov={self.vr_fov}:ih_fov={self.vr_fov}:d_fov={self.vr_fov}:"
                f"v_fov={v_h_FOV}:h_fov={v_h_FOV}:pitch={self.vr_pitch}:"
                f"w={self.yolo_input_size}:h={self.yolo_input_size}:interp=lanczos"
            )
            sw_filters.append(v360_filter)

        final_filters = []
        if needs_hw_download and sw_filters:
            # Must download from GPU memory to system memory before applying software filters.
            final_filters.extend(["hwdownload", "format=nv12"])
        final_filters.extend(sw_filters)

        ffmpeg_filter = ",".join(final_filters)
        self.logger.info(f"Built FFmpeg filter string: {ffmpeg_filter}")
        return ffmpeg_filter

    def _get_ffmpeg_hwaccel_args(self) -> List[str]:
        """Determines FFmpeg hardware acceleration arguments based on app settings."""
        selected = getattr(self.app, 'hardware_acceleration_method', self.HWACCEL_AUTO)
        available = getattr(self.app, 'available_ffmpeg_hwaccels', [])
        system, machine = platform.system().lower(), platform.machine().lower()

        auto_choice = ""
        if selected == self.HWACCEL_AUTO:
            if system == 'darwin' and self.HWACCEL_VIDEOTOOLBOX in available:
                auto_choice = self.HWACCEL_VIDEOTOOLBOX
            elif system == 'linux':
                if self.HWACCEL_NVDEC in available:
                    auto_choice = self.HWACCEL_NVDEC
                elif self.HWACCEL_CUDA in available:
                    auto_choice = self.HWACCEL_CUDA
                elif self.HWACCEL_VAAPI in available:
                    auto_choice = self.HWACCEL_VAAPI
                elif self.HWACCEL_QSV in available:
                    auto_choice = self.HWACCEL_QSV
            elif system == 'windows':
                if self.HWACCEL_NVDEC in available:
                    auto_choice = self.HWACCEL_NVDEC
                elif self.HWACCEL_CUDA in available:
                    auto_choice = self.HWACCEL_CUDA
                elif self.HWACCEL_D3D11VA in available:
                    auto_choice = self.HWACCEL_D3D11VA
                elif self.HWACCEL_QSV in available:
                    auto_choice = self.HWACCEL_QSV
            selected = auto_choice if auto_choice else self.HWACCEL_NONE

        if selected != self.HWACCEL_NONE and selected in available:
            args = ['-hwaccel', selected]
            if selected in [self.HWACCEL_CUDA, self.HWACCEL_NVDEC]:
                args.extend(['-hwaccel_output_format', 'cuda'])
            elif selected == self.HWACCEL_QSV:
                args.extend(['-hwaccel_output_format', 'qsv'])
            elif selected == self.HWACCEL_VAAPI:
                args.extend(['-hwaccel_output_format', 'vaapi'])
            self.logger.debug(f"Using HWAccel: {selected}, Args: {args}")
            return args

        self.logger.debug("Using CPU decoding (no hardware acceleration).")
        return []

    def _terminate_process(self, proc: Optional[subprocess.Popen], name: str):
        """Safely terminates a single subprocess."""
        if proc and proc.poll() is None:
            self.logger.debug(f"Terminating process '{name}' (PID: {proc.pid}).")
            if proc.stdout: proc.stdout.close()
            if proc.stderr: proc.stderr.close()
            proc.terminate()
            try:
                proc.wait(timeout=0.5)
            except subprocess.TimeoutExpired:
                self.logger.warning(f"Process '{name}' did not terminate gracefully. Killing.")
                proc.kill()
                proc.wait()

    def _terminate_ffmpeg_processes(self):
        """Safely terminates all active FFmpeg processes controlled by the class."""
        self._terminate_process(self.ffmpeg_pipe1_process, "ffmpeg_pipe1")
        self.ffmpeg_pipe1_process = None
        self._terminate_process(self.ffmpeg_process, "ffmpeg_main_or_pipe2")
        self.ffmpeg_process = None

    def _start_ffmpeg_stream(self, start_frame_abs_idx=0, num_frames_to_output_ffmpeg=None, is_for_batch=False):
        """
        Unified method to launch FFmpeg as either a 1-pipe or 2-pipe process.

        Returns:
            A tuple of (pipe1_process, pipe2_or_main_process). pipe1_process may be None.
        """
        if not self.is_video_open():
            self.logger.warning("Cannot start FFmpeg: video not properly opened.")
            return None, None

        start_time_sec = start_frame_abs_idx / self.video_info['fps']
        bufsize = self.frame_size_bytes * (10 if is_for_batch else 5)
        common_opts = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']

        p1_proc, p2_proc = None, None

        try:
            if self._is_10bit_cuda_pipe_needed():
                # --- 2-Pipe Command for 10-bit+ CUDA ---
                self.logger.debug("Launching 2-pipe FFmpeg for 10-bit CUDA video.")
                vid_h = self.video_info.get('height', 0)
                if vid_h <= 0:
                    self.logger.error("Cannot construct 10-bit CUDA pipe: video height unknown.")
                    return None, None

                # Pipe 1: Decodes to GPU, does GPU-side pre-filtering, re-encodes to HEVC
                pipe1_vf = f"crop={int(vid_h)}:{int(vid_h)}:0:0,scale_cuda=1000:1000"
                cmd1 = common_opts + ['-hwaccel', 'cuda', '-hwaccel_output_format', 'cuda']
                if start_time_sec > 0.001: cmd1.extend(['-ss', str(start_time_sec)])
                cmd1.extend(['-i', self.video_path, '-an', '-sn', '-vf', pipe1_vf])
                cmd1.extend(['-c:v', 'hevc_nvenc', '-preset', 'fast', '-qp', '0', '-f', 'matroska', 'pipe:1'])

                # Pipe 2: Decodes HEVC from pipe 1, applies final filters, outputs raw frames
                cmd2 = common_opts + ['-hwaccel', 'cuda', '-i', 'pipe:0', '-an', '-sn']
                effective_vf = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}"
                cmd2.extend(['-vf', effective_vf])
                if num_frames_to_output_ffmpeg: cmd2.extend(['-frames:v', str(num_frames_to_output_ffmpeg)])
                cmd2.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

                self.logger.debug(f"Pipe 1 CMD: {' '.join(shlex.quote(str(x)) for x in cmd1)}")
                self.logger.debug(f"Pipe 2 CMD: {' '.join(shlex.quote(str(x)) for x in cmd2)}")

                p1_proc = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                p2_proc = subprocess.Popen(cmd2, stdin=p1_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           bufsize=bufsize)
                if p1_proc.stdout: p1_proc.stdout.close()  # Allow SIGPIPE if p2 dies

            else:
                # --- Standard Single-Pipe Command ---
                self.logger.debug("Launching single-pipe FFmpeg.")
                hw_args = self._get_ffmpeg_hwaccel_args()
                input_opts = hw_args[:]
                if start_time_sec > 0.001: input_opts.extend(['-ss', str(start_time_sec)])

                cmd = common_opts + input_opts + ['-i', self.video_path, '-an', '-sn']
                effective_vf = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}"
                cmd.extend(['-vf', effective_vf])
                if num_frames_to_output_ffmpeg: cmd.extend(['-frames:v', str(num_frames_to_output_ffmpeg)])
                cmd.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

                self.logger.debug(f"Single Pipe CMD: {' '.join(shlex.quote(str(x)) for x in cmd)}")
                p2_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=bufsize)

            return p1_proc, p2_proc

        except Exception as e:
            self.logger.error(f"Failed to start FFmpeg stream: {e}", exc_info=True)
            if p1_proc: self._terminate_process(p1_proc, "failed_p1")
            if p2_proc: self._terminate_process(p2_proc, "failed_p2")
            return None, None

    def start_processing(self, start_frame=None, end_frame=None):
        """Starts the main processing loop in a separate thread."""
        if self.is_processing:
            self.logger.warning("Already processing.")
            return
        if not self.is_video_open():
            self.logger.warning("Video not loaded.")
            return

        effective_start = self.current_frame_index
        if not self.is_paused and start_frame is not None:  # New session
            effective_start = max(0, min(start_frame, self.total_frames - 1))

        self.processing_start_frame_limit = effective_start
        self.processing_end_frame_limit = -1
        if end_frame is not None and end_frame >= 0:
            self.processing_end_frame_limit = min(end_frame, self.total_frames - 1)

        self.current_stream_start_frame_abs = effective_start
        self.frames_read_from_current_stream = 0

        self.ffmpeg_pipe1_process, self.ffmpeg_process = self._start_ffmpeg_stream(
            start_frame_abs_idx=self.processing_start_frame_limit)

        if not self.ffmpeg_process:
            self.logger.error("Failed to start FFmpeg for processing.")
            self._terminate_ffmpeg_processes()
            return

        self.is_processing = True
        self.is_paused = False
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.logger.info(
            f"Started processing from frame {self.processing_start_frame_limit} to "
            f"{'end of video' if self.processing_end_frame_limit == -1 else self.processing_end_frame_limit}")

    def pause_processing(self):
        if not self.is_processing: return
        self.logger.info("Pausing video processing...")
        self.is_paused = True
        self.stop_processing(join_thread=True)  # Use stop logic but preserve paused state
        self.logger.info(f"Video processing paused at frame {self.current_frame_index}")

    def stop_processing(self, join_thread=True):
        if not self.is_processing and not self.is_paused and not (
                self.processing_thread and self.processing_thread.is_alive()):
            return  # Already fully stopped

        was_paused = self.is_paused
        self.logger.info("Stopping video processing...")
        self.is_processing = False
        self.is_paused = False
        self.stop_event.set()

        if join_thread:
            thread = self.processing_thread
            if thread and thread.is_alive() and threading.current_thread() is not thread:
                thread.join(timeout=1.0)
                if thread.is_alive(): self.logger.warning("Processing thread did not join cleanly.")
        self.processing_thread = None

        self._terminate_ffmpeg_processes()

        if self.tracker: self.tracker.stop_tracking()
        self.enable_tracker_processing = False

        if self.app and hasattr(self.app, 'on_processing_stopped') and not was_paused:
            was_scripting = self.tracker and self.tracker.tracking_active
            frame_range = (self.processing_start_frame_limit, self.current_frame_index)
            self.app.on_processing_stopped(was_scripting_session=was_scripting, scripted_frame_range=frame_range)
        self.logger.info("Video processing stopped.")

    def seek_video(self, frame_index: int):
        """Seeks the video to a specific frame index."""
        if not self.is_video_open(): return
        target_frame = max(0, min(frame_index, self.total_frames - 1))

        was_processing = self.is_processing
        stored_end_limit = self.processing_end_frame_limit
        if was_processing: self.stop_processing(join_thread=True)

        self.logger.info(f"Seek requested to frame {target_frame}")
        new_frame = self._get_specific_frame(target_frame)

        with self.frame_lock:
            self.current_frame = new_frame
            self.current_frame_index = target_frame  # Ensure index is updated even if frame fails

        if was_processing:
            self.start_processing(start_frame=self.current_frame_index, end_frame=stored_end_limit)

    def is_vr_active_or_potential(self) -> bool:
        """Checks if VR processing is active or could be activated."""
        if self.video_type_setting == self.VIDEO_TYPE_VR: return True
        if self.video_type_setting == self.VIDEO_TYPE_AUTO and self.determined_video_type == self.VIDEO_TYPE_VR: return True
        return False

    def display_current_frame(self):
        """
        Placeholder for GUI interaction to get the current frame.
        Can apply non-persistent processing for display purposes.
        """
        if self.current_frame is None or not self.tracker or not self.tracker.tracking_active or self.is_processing:
            return
        # If paused with tracker active, we can draw on the static frame for display
        timestamp_ms = int(self.current_frame_index * (1000.0 / self.fps))
        with self.frame_lock:
            processed_frame, _ = self.tracker.process_frame(self.current_frame.copy(), timestamp_ms)
            self.current_frame = processed_frame

    def _processing_loop(self):
        """The main threaded loop that reads and processes frames from FFmpeg."""
        if not self.ffmpeg_process or not self.ffmpeg_process.stdout:
            self.logger.error("_processing_loop: FFmpeg process invalid. Exiting.")
            self.is_processing = False
            return

        next_frame_time = time.perf_counter()

        try:
            while not self.stop_event.is_set():
                # --- Read Frame ---
                raw_bytes = self.ffmpeg_process.stdout.read(self.frame_size_bytes)
                if len(raw_bytes) < self.frame_size_bytes:
                    self.logger.info(f"End of FFmpeg stream (read {len(raw_bytes)}/{self.frame_size_bytes}).")
                    break

                # --- Update State ---
                self.current_frame_index = self.current_stream_start_frame_abs + self.frames_read_from_current_stream
                self.frames_read_from_current_stream += 1

                if (
                        self.processing_end_frame_limit != -1 and self.current_frame_index > self.processing_end_frame_limit) or \
                        (self.total_frames > 0 and self.current_frame_index >= self.total_frames):
                    self.logger.info("Reached end of processing range.")
                    break

                # --- Process Frame ---
                frame_np = np.frombuffer(raw_bytes, dtype=np.uint8).reshape(self.yolo_input_size, self.yolo_input_size,
                                                                            3)
                processed_frame = frame_np
                if self.tracker and self.tracker.tracking_active:
                    timestamp_ms = int(self.current_frame_index * (1000.0 / self.fps))
                    processed_frame, _ = self.tracker.process_frame(frame_np.copy(), timestamp_ms)

                with self.frame_lock:
                    self.current_frame = processed_frame

                # --- Frame Rate Limiting ---
                target_delay = 1.0 / self.target_fps
                current_time = time.perf_counter()
                sleep_duration = next_frame_time - current_time
                if sleep_duration > 0: time.sleep(sleep_duration)

                # Schedule next frame time; reset if we've fallen too far behind
                if current_time > next_frame_time + target_delay:
                    next_frame_time = current_time + target_delay
                else:
                    next_frame_time += target_delay

                # --- FPS Calculation ---
                self.frames_for_fps_calc += 1
                if time.time() - self.last_fps_update_time >= 1.0:
                    self.actual_fps = self.frames_for_fps_calc / (time.time() - self.last_fps_update_time)
                    self.last_fps_update_time = time.time()
                    self.frames_for_fps_calc = 0

        finally:
            self.logger.info(f"_processing_loop ending. Stop event: {self.stop_event.is_set()}")
            self.is_processing = False
            self.stop_event.clear()
            self._terminate_ffmpeg_processes()
            if self.app and hasattr(self.app, 'on_processing_stopped'):
                was_scripting = self.tracker and self.tracker.tracking_active
                frame_range = (self.processing_start_frame_limit, self.current_frame_index)
                self.app.on_processing_stopped(was_scripting_session=was_scripting, scripted_frame_range=frame_range)

    def stream_frames_for_segment(self, start_frame_abs_idx: int, num_frames_to_read: int) -> Iterator[
        Tuple[int, np.ndarray]]:
        """
        A generator that yields frames for a specific segment of the video.
        Now supports 10-bit CUDA processing.
        """
        if num_frames_to_read <= 0: return

        # Launch a dedicated FFmpeg process for this segment
        p1_proc, p2_proc = self._start_ffmpeg_stream(
            start_frame_abs_idx=start_frame_abs_idx,
            num_frames_to_output_ffmpeg=num_frames_to_read,
            is_for_batch=True
        )

        if not p2_proc or not p2_proc.stdout:
            self.logger.error(f"Failed to start FFmpeg for segment from frame {start_frame_abs_idx}.")
            return

        try:
            for i in range(num_frames_to_read):
                if p2_proc.poll() is not None:
                    stderr = p2_proc.stderr.read().decode(errors='ignore') if p2_proc.stderr else ""
                    self.logger.warning(f"FFmpeg process for segment terminated prematurely. Stderr: {stderr.strip()}")
                    break

                raw_frame_bytes = p2_proc.stdout.read(self.frame_size_bytes)
                if len(raw_frame_bytes) < self.frame_size_bytes:
                    self.logger.info(f"End of segment stream after {i} frames.")
                    break

                frame_np = np.frombuffer(raw_frame_bytes, dtype=np.uint8).reshape(self.yolo_input_size,
                                                                                  self.yolo_input_size, 3)
                yield (start_frame_abs_idx + i, frame_np)
        finally:
            self.logger.debug("Terminating FFmpeg processes for segment stream.")
            if p1_proc: self._terminate_process(p1_proc, "segment_p1")
            if p2_proc: self._terminate_process(p2_proc, "segment_p2")

    def set_target_fps(self, fps: float):
        """Sets the target frame rate for the main processing loop."""
        self.target_fps = max(1.0, fps if fps > 0 else 1.0)
        self.logger.info(f"Target FPS set to: {self.target_fps:.2f}")

    def is_video_open(self) -> bool:
        """Checks if a video is currently loaded with valid information."""
        return bool(self.video_path and self.video_info and self.video_info.get('total_frames', 0) > 0)

    def reset(self, close_video=False):
        """Resets the processor's state, optionally closing the video."""
        self.logger.info("Resetting VideoProcessor...")
        self.stop_processing(join_thread=True)
        self._clear_cache()
        self.current_frame_index = 0
        self.frames_read_from_current_stream = 0
        self.current_stream_start_frame_abs = 0

        if self.tracker: self.tracker.reset()

        if close_video:
            self.video_path = ""
            self.video_info = {}
            self.determined_video_type = None
            self.ffmpeg_filter_string = ""
            self.logger.info("Video closed and parameters reset.")

        with self.frame_lock:
            self.current_frame = None

        if self.is_video_open() and not close_video:
            self.logger.info("Fetching frame 0 after reset.")
            self.current_frame = self._get_specific_frame(0)

        self.logger.info("VideoProcessor reset complete.")

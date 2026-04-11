import time
import threading
import subprocess
import json
import shlex
import numpy as np
import cv2
import sys
from typing import Optional, Iterator, Tuple, List, Dict, Any
import logging
import os
from collections import OrderedDict, deque

from config import constants

# ML-based VR format detector
from video.vr_format_detector_ml_real import RealMLVRFormatDetector

# Thumbnail extractor for fast random frame access
from video.thumbnail_extractor import ThumbnailExtractor

# Decomposed mixin modules
from video._vp_nav_buffer import NavBufferMixin
from video._vp_format_detection import FormatDetectionMixin
from video._vp_ffmpeg_builders import FFmpegBuildersMixin
from video._vp_dual_output import DualOutputMixin
from video._vp_segment_streaming import SegmentStreamingMixin

try:
    import scipy  # noqa: F401 — presence check only (audio waveform generation)
    SCIPY_AVAILABLE_FOR_AUDIO = True
except ImportError:
    SCIPY_AVAILABLE_FOR_AUDIO = False


class VideoProcessor(
    NavBufferMixin,
    FormatDetectionMixin,
    FFmpegBuildersMixin,
    DualOutputMixin,
    SegmentStreamingMixin,
):
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

        self.logger.debug(f"VideoProcessor logger '{self.logger.name}' initialized.")

        self.video_path = ""
        self._active_video_source_path: str = ""
        self.video_info = {}
        self.ffmpeg_process: Optional[subprocess.Popen] = None  # Main output process (pipe2 if active)
        self.ffmpeg_pipe1_process: Optional[subprocess.Popen] = None  # Pipe1 process, if active
        self.is_processing = False
        self.pause_event = threading.Event()
        self.processing_thread = None
        self.current_frame = None
        self._frame_version = 0  # Incremented each time current_frame is replaced
        self.fps = 0.0
        self.playhead_override_ms = None  # Set by point-jump to display at exact action time
        self.target_fps = 30
        self.actual_fps = 0
        self.last_fps_update_time = time.time()
        self.frames_for_fps_calc = 0
        self.frame_lock = threading.Lock()
        self.seek_request_frame_index = None
        self.seek_in_progress = False  # Flag to track if seek operation is running
        self.seek_thread = None  # Thread for async seek operations
        self.arrow_nav_in_progress = False  # Flag to prevent arrow nav overload

        # Unified FFmpeg pipe for arrow key navigation and pipe reuse.
        # Single pipe survives across play/pause; buffer provides O(1) lookups.
        self._ffmpeg_pipe: Optional[subprocess.Popen] = None
        self._ffmpeg_read_position: int = 0  # Next frame the pipe will yield
        self.frame_buffer_progress = 0.0  # Progress of frame buffer creation (0.0 to 1.0)
        self.frame_buffer_total = 0  # Total frames to buffer
        self.frame_buffer_current = 0  # Current frames buffered
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

        # HD display: decode at higher resolution for GUI, downsample to yolo_input_size for processing
        self.hd_display_enabled = False
        self._display_frame_w = self.yolo_input_size
        self._display_frame_h = self.yolo_input_size
        self._is_hd_active = False
        self._processing_frame_buf = np.zeros((self.yolo_input_size, self.yolo_input_size, 3), dtype=np.uint8)
        self._proc_resize_dims = (self.yolo_input_size, self.yolo_input_size)
        self._proc_pad_offset = (0, 0)
        if self.app and hasattr(self.app, 'app_settings'):
            self.hd_display_enabled = self.app.app_settings.get('hd_video_display', True)

        # GPU Unwarp Worker for VR optimization
        self.gpu_unwarp_worker = None
        self.gpu_unwarp_enabled = False

        # VR Unwarp method override (from UI dropdown)
        # Options: 'auto', 'metal', 'opengl', 'v360', 'none'
        self.vr_unwarp_method_override = 'auto'
        # VR crop panel selection for "none" (crop only) mode: 'first' or 'second'
        self.vr_crop_panel = 'first'
        if self.app and hasattr(self.app, 'app_settings'):
            self.vr_unwarp_method_override = self.app.app_settings.get('vr_unwarp_method', 'auto')
            self.vr_crop_panel = self.app.app_settings.get('vr_crop_panel', 'first')

        # Thumbnail Extractor for fast random frame access (OpenCV-based)
        self.thumbnail_extractor = None

        # Performance timing metrics (for UI display)
        # Update once per second with mean values
        self._last_decode_time_ms = 0.0
        self._last_unwarp_time_ms = 0.0
        self._last_yolo_time_ms = 0.0

        # Sample accumulators for 1-second averaging
        self._decode_samples = []
        self._unwarp_samples = []
        self._yolo_samples = []
        self._last_timing_update = time.time()

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

        # Frame Caching with rolling backward buffer for arrow navigation
        self.frame_cache = OrderedDict()
        self.frame_cache_max_size = cache_size
        self.frame_cache_lock = threading.Lock()
        self.batch_fetch_size = 600  # For explicit batch fetches only

        # Unified frame buffer for arrow key navigation (deque for efficient FIFO)
        # Populated by BOTH arrow-nav pipe reads AND the processing loop,
        # so backward nav works seamlessly regardless of how frames were produced.
        # Buffer miss = pipe restart at new position.
        self._frame_buffer: deque = deque(maxlen=self._compute_nav_buffer_size())
        self._frame_buffer_lock = threading.Lock()
        
        # Single FFmpeg dual-output processor integration
        from video.dual_frame_processor import SingleFFmpegDualOutputProcessor
        self.dual_output_processor = SingleFFmpegDualOutputProcessor(self)
        self.dual_output_enabled = False

        # ML format detector (lazy loaded)
        self.ml_detector = None
        self.ml_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vr_detector_model_rf.pkl')

        # Event callbacks (for optional features like streamer, device_control)
        self._seek_callbacks = []  # List of callbacks: func(frame_index: int) -> None
        self._playback_state_callbacks = []  # List of callbacks: func(is_playing: bool, current_time_ms: float) -> None

    def register_seek_callback(self, callback):
        """
        Register a callback to be notified when video seeks.

        Callback signature: func(frame_index: int) -> None

        This allows optional features (like streamer) to observe seek events
        without VideoProcessor knowing about them.

        Args:
            callback: Callable that takes frame_index as parameter
        """
        if callback not in self._seek_callbacks:
            self._seek_callbacks.append(callback)
            self.logger.debug(f"Registered seek callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'} (total callbacks: {len(self._seek_callbacks)})")

    def unregister_seek_callback(self, callback):
        """
        Unregister a seek callback.

        Args:
            callback: Previously registered callback to remove
        """
        if callback in self._seek_callbacks:
            self._seek_callbacks.remove(callback)
            self.logger.debug(f"Unregistered seek callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")

    def _notify_seek_callbacks(self, frame_index: int):
        """
        Notify all registered callbacks that a seek occurred.

        Args:
            frame_index: Frame that was seeked to
        """
        if self._seek_callbacks:
            self.logger.debug(f"Notifying {len(self._seek_callbacks)} seek callbacks for frame {frame_index}")
        for callback in self._seek_callbacks:
            try:
                callback(frame_index)
            except Exception as e:
                self.logger.error(f"Error in seek callback {callback}: {e}")

    def register_playback_state_callback(self, callback):
        """
        Register a callback to be notified of playback state changes.

        Callback signature: func(is_playing: bool, current_time_ms: float) -> None

        This allows optional features (like device_control) to observe playback
        state without VideoProcessor knowing about them.

        Args:
            callback: Callable that takes is_playing and current_time_ms as parameters
        """
        if callback not in self._playback_state_callbacks:
            self._playback_state_callbacks.append(callback)
            self.logger.debug(f"Registered playback state callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'} (total callbacks: {len(self._playback_state_callbacks)})")

    def unregister_playback_state_callback(self, callback):
        """
        Unregister a playback state callback.

        Args:
            callback: Previously registered callback to remove
        """
        if callback in self._playback_state_callbacks:
            self._playback_state_callbacks.remove(callback)
            self.logger.debug(f"Unregistered playback state callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'}")

    def _notify_playback_state_callbacks(self, is_playing: bool, current_time_ms: float):
        """
        Notify all registered callbacks of playback state change.

        Args:
            is_playing: Whether video is currently playing
            current_time_ms: Current time in milliseconds
        """
        for callback in self._playback_state_callbacks:
            try:
                callback(is_playing, current_time_ms)
            except Exception as e:
                self.logger.error(f"Error in playback state callback {callback}: {e}")

    def _update_timing_metrics(self):
        """Update display timing metrics from accumulated samples (once per second)."""
        current_time = time.time()
        if current_time - self._last_timing_update >= 1.0:
            # Calculate means
            if self._decode_samples:
                self._last_decode_time_ms = sum(self._decode_samples) / len(self._decode_samples)
                self._decode_samples = []

            if self._unwarp_samples:
                self._last_unwarp_time_ms = sum(self._unwarp_samples) / len(self._unwarp_samples)
                self._unwarp_samples = []
            else:
                self._last_unwarp_time_ms = 0.0

            if self._yolo_samples:
                self._last_yolo_time_ms = sum(self._yolo_samples) / len(self._yolo_samples)
                self._yolo_samples = []
            else:
                self._last_yolo_time_ms = 0.0

            self._last_timing_update = current_time


    # ------------------------------------------------------------------ #
    #  HD Display Helpers                                                 #
    # ------------------------------------------------------------------ #

    def _compute_display_dimensions(self):
        """Compute display frame dimensions based on video info and HD setting.

        When HD is enabled for 2D video, scales to max 1920px on longest edge
        preserving aspect ratio with even dimensions. Otherwise falls back to
        yolo_input_size square.
        """
        size = self.yolo_input_size
        self._display_frame_w = size
        self._display_frame_h = size
        self._is_hd_active = False
        # Pre-allocated processing frame buffer + cached resize params
        self._processing_frame_buf = np.zeros((size, size, 3), dtype=np.uint8)
        self._proc_resize_dims = (size, size)
        self._proc_pad_offset = (0, 0)

        # Re-read setting each time (user may toggle between video loads)
        if self.app and hasattr(self.app, 'app_settings'):
            self.hd_display_enabled = self.app.app_settings.get('hd_video_display', True)

        if not self.hd_display_enabled:
            return
        if not self.video_info:
            return
        if self.determined_video_type != '2D':
            return
        if hasattr(self, '_is_using_preprocessed_video') and self._is_using_preprocessed_video():
            return

        src_w = self.video_info.get('width', 0)
        src_h = self.video_info.get('height', 0)
        if src_w <= 0 or src_h <= 0:
            return

        max_dim = 1920
        # Scale so the longest edge is at most max_dim
        if max(src_w, src_h) <= max_dim:
            out_w, out_h = src_w, src_h
        elif src_w >= src_h:
            out_w = max_dim
            out_h = int(src_h * max_dim / src_w)
        else:
            out_h = max_dim
            out_w = int(src_w * max_dim / src_h)

        # Ensure even dimensions (FFmpeg requirement)
        out_w = out_w & ~1
        out_h = out_h & ~1

        # Never smaller than yolo_input_size on longest edge
        if max(out_w, out_h) < size:
            return

        self._display_frame_w = out_w
        self._display_frame_h = out_h
        self._is_hd_active = True
        self.frame_size_bytes = out_w * out_h * 3

        # Pre-compute processing frame resize params (avoids per-frame math)
        scale = size / max(out_w, out_h)
        new_w = int(out_w * scale) & ~1
        new_h = int(out_h * scale) & ~1
        new_w = min(new_w, size)
        new_h = min(new_h, size)
        self._proc_resize_dims = (new_w, new_h)
        self._proc_pad_offset = ((size - new_w) // 2, (size - new_h) // 2)

        self.logger.info(f"HD display: {out_w}x{out_h} ({out_w * out_h * 3} bytes/frame)")

    def _make_processing_frame(self, display_frame):
        """Resize an HD display frame down to yolo_input_size square with padding for YOLO/tracker.

        Uses pre-allocated buffer and cached resize parameters to avoid
        per-frame allocation overhead.
        """
        h, w = display_frame.shape[:2]
        size = self.yolo_input_size
        if h == size and w == size:
            return display_frame

        # Use cached resize params (computed once in _compute_display_dimensions)
        new_w, new_h = self._proc_resize_dims
        x_off, y_off = self._proc_pad_offset

        import cv2
        resized = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Reuse pre-allocated buffer (zero once, then overwrite content region)
        buf = self._processing_frame_buf
        buf[:] = 0
        buf[y_off:y_off + new_h, x_off:x_off + new_w] = resized
        return buf

    @property
    def is_hd_active(self):
        """True when display frame is larger than processing frame."""
        return self._is_hd_active

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
            self._compute_display_dimensions()
            self.frame_size_bytes = self._display_frame_w * self._display_frame_h * 3

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
        video_filename = os.path.basename(video_path)
        self.logger.info(f"Opening video: {video_filename}...", extra={'status_message': True, 'duration': 2.0})

        # Invalidate content UV cache for new video dimensions
        if self.app and hasattr(self.app, 'app_state_ui'):
            self.app.app_state_ui.invalidate_content_uv_cache()

        self.stop_processing()
        self.video_path = video_path # This will always be the ORIGINAL video path
        self._clear_cache()
        # Clear ML detection cache when opening new video
        if hasattr(self, '_ml_detection_cached'):
            delattr(self, '_ml_detection_cached')

        # Re-read VR unwarp method and crop panel from settings in case they changed
        if self.app and hasattr(self.app, 'app_settings'):
            self.vr_unwarp_method_override = self.app.app_settings.get('vr_unwarp_method', 'auto')
            self.vr_crop_panel = self.app.app_settings.get('vr_crop_panel', 'first')
            self.logger.debug(f"VR unwarp method from settings: {self.vr_unwarp_method_override}")

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
            potential_preprocessed_path = self.app.file_manager.get_output_path_for_file(self.video_path, "_preprocessed.mp4")
            if os.path.exists(potential_preprocessed_path):
                preprocessed_path = potential_preprocessed_path
                # Also update the file_manager's state to be consistent
                self.app.file_manager.preprocessed_video_path = preprocessed_path

        if preprocessed_path:
            # Always validate the preprocessed file before using it
            self.logger.debug(f"Found potential preprocessed file: {os.path.basename(preprocessed_path)}. Verifying...")

            # Basic validation first
            preprocessed_info = self._get_video_info(preprocessed_path)
            original_frames = self.video_info.get("total_frames", 0)
            original_fps = self.video_info.get("fps", 30.0)
            preprocessed_frames = preprocessed_info.get("total_frames", -1) if preprocessed_info else -1

            # Use comprehensive validation
            is_valid_preprocessed = self._validate_preprocessed_video(preprocessed_path, original_frames, original_fps)

            if is_valid_preprocessed and preprocessed_frames >= original_frames > 0:
                self._active_video_source_path = preprocessed_path
                self.logger.debug(f"Preprocessed video validation passed. Using as active source.")
            else:
                self.logger.warning(
                    f"Preprocessed file is incomplete or invalid ({preprocessed_frames}/{original_frames} frames). "
                    f"Falling back to original video. Re-run Stage 1 with 'Save Preprocessed Video' enabled to fix."
                )
                # Clean up the invalid preprocessed file
                self._cleanup_invalid_preprocessed_file(preprocessed_path)

        if self._active_video_source_path == preprocessed_path:
            self.logger.debug(f"VideoProcessor will use preprocessed video as its active source.")
        else:
            self.logger.debug(f"VideoProcessor will use original video as its active source.")

        self._update_video_parameters()

        # Initialize GPU unwarp worker for VR videos (needed for seek-to-frame before playback starts)
        self._init_gpu_unwarp_worker()

        self.fps = self.video_info['fps']
        self.total_frames = self.video_info['total_frames']

        # Initialize thumbnail extractor for fast random frame access (FFmpeg-based)
        self._init_thumbnail_extractor()
        self.set_target_fps(self.fps)
        self.current_frame_index = 0
        self.frames_read_from_current_stream = 0
        self.current_stream_start_frame_abs = 0
        self.stop_event.clear()
        self.seek_request_frame_index = None
        # OPTIMIZATION: Load first frame with thumbnail for instant startup
        try:
            self.current_frame = self._get_specific_frame(0, use_thumbnail=True)
            self._frame_version += 1
        except Exception as e:
            self.logger.warning(f"Could not load initial frame: {e}")
            self.current_frame = None

        if self.tracker:
            reset_reason = "project_load_preserve_actions" if from_project_load else None
            self.tracker.reset(reason=reset_reason)

        active_source_name = os.path.basename(self._active_video_source_path)
        source_type = "preprocessed" if self._active_video_source_path != video_path else "original"
        self.logger.info(
            f"Opened: {active_source_name} ({source_type}, {self.determined_video_type}, "
            f"format: {self.vr_input_format if self.determined_video_type == 'VR' else 'N/A'}), "
            f"{self.total_frames}fr, {self.fps:.2f}fps, {self.video_info.get('bit_depth', 'N/A')}bit)")

        # Notify sync server (streamer) that video was loaded in desktop FunGen
        # This broadcasts to ALL connected browser clients (VR viewer, etc.)
        # even if the video was loaded from XBVR/Stash browser
        if hasattr(self, 'sync_server') and self.sync_server and hasattr(self.sync_server, 'loop') and self.sync_server.loop:
            try:
                import asyncio
                is_remote_video = video_path.startswith(('http://', 'https://'))
                source_desc = "remote" if is_remote_video else "local"
                self.logger.debug(f"Notifying streamer of {source_desc} video load: {os.path.basename(video_path)}")
                asyncio.run_coroutine_threadsafe(
                    self.sync_server.broadcast_video_loaded(video_path),
                    self.sync_server.loop
                )
            except Exception as e:
                self.logger.warning(f"Could not notify sync server: {e}")
                import traceback
                self.logger.warning(traceback.format_exc())
        else:
            self.logger.debug(f"Streamer not available (sync_server: {hasattr(self, 'sync_server')})")

        return True


    def reapply_video_settings(self):
        # Invalidate content UV cache so GUI picks up new dimensions
        if self.app and hasattr(self.app, 'app_state_ui'):
            self.app.app_state_ui.invalidate_content_uv_cache()

        if not self.video_path or not self.video_info:
            self.logger.info("No video loaded. Settings will apply when a video is opened.")
            self._compute_display_dimensions()
            self.frame_size_bytes = self._display_frame_w * self._display_frame_h * 3
            return

        self.logger.info(f"Applying video settings...", extra={'status_message': True})
        self.logger.info(f"Reapplying video settings (self.vr_input_format is currently: {self.vr_input_format})")
        was_processing = self.is_processing
        stored_frame_index = self.current_frame_index
        stored_end_limit = self.processing_end_frame_limit
        self.stop_processing()
        self._clear_cache()

        # [REDUNDANCY REMOVED] - Call the new helper method
        self._update_video_parameters()

        # Reinitialize GPU unwarp worker in case unwarp method changed
        self._init_gpu_unwarp_worker()

        # Reinitialize thumbnail extractor with new display dimensions
        self._init_thumbnail_extractor()

        self.logger.info(f"Attempting to fetch frame {stored_frame_index} with new settings.")
        new_frame = self._get_specific_frame(stored_frame_index, use_thumbnail=True)
        if new_frame is not None:
            with self.frame_lock:
                self.current_frame = new_frame
                self._frame_version += 1
            self.logger.info(f"Successfully fetched frame {self.current_frame_index} with new settings.")
        else:
            self.logger.warning(f"Failed to get frame {stored_frame_index} with new settings.")

        if was_processing:
            self.logger.info("Restarting processing with new settings...")
            self.start_processing(start_frame=self.current_frame_index, end_frame=stored_end_limit)
        else:
            self.logger.info("Settings applied. Video remains paused/stopped.")
        self.logger.info("Video settings applied successfully", extra={'status_message': True})

    def get_frames_batch(self, start_frame_num: int, num_frames_to_fetch: int, immediate_display_frame: Optional[int] = None) -> Dict[int, np.ndarray]:
        """
        Fetches a batch of frames using FFmpeg.
        This method now supports 2-pipe 10-bit CUDA processing.

        Args:
            immediate_display_frame: If specified, immediately display this frame when decoded
        """
        decode_start = time.perf_counter()  # Performance tracking
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
                if not effective_vf_pipe2: effective_vf_pipe2 = f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease,pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
                cmd2.extend(['-vf', effective_vf_pipe2])
                cmd2.extend(['-frames:v', str(num_frames_to_fetch)])
                # Always use BGR24 - GPU unwarp worker handles BGR->RGBA conversion internally
                cmd2.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(f"get_frames_batch Pipe 1 CMD: {' '.join(shlex.quote(str(x)) for x in cmd1)}")
                    self.logger.debug(f"get_frames_batch Pipe 2 CMD: {' '.join(shlex.quote(str(x)) for x in cmd2)}")

                # Windows fix: prevent terminal windows from spawning
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                local_p1_proc = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, creationflags=creation_flags)
                if local_p1_proc.stdout is None: raise IOError("get_frames_batch: Pipe 1 stdout is None.")

                # Always use BGR24 (3 bytes per pixel)
                buffer_frame_size = self.yolo_input_size * self.yolo_input_size * 3
                local_p2_proc = subprocess.Popen(cmd2, stdin=local_p1_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, creationflags=creation_flags)
                local_p1_proc.stdout.close()

            else:  # Standard single FFmpeg process
                self.logger.debug(
                    f"get_frames_batch: Using single-pipe FFmpeg for {num_frames_to_fetch} frames from {start_frame_num}.")
                hwaccel_cmd_list = self._get_ffmpeg_hwaccel_args()
                ffmpeg_input_options = hwaccel_cmd_list[:]
                if start_time_seconds > 0.001: ffmpeg_input_options.extend(['-ss', str(start_time_seconds)])
                cmd_single = common_ffmpeg_prefix + ffmpeg_input_options + ['-i', self._active_video_source_path, '-an', '-sn']
                effective_vf = self.ffmpeg_filter_string
                if not effective_vf: effective_vf = f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease,pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
                cmd_single.extend(['-vf', effective_vf])
                cmd_single.extend(['-frames:v', str(num_frames_to_fetch)])
                # Always use BGR24 - GPU unwarp worker handles BGR->RGBA conversion internally
                cmd_single.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])
                if self.logger.isEnabledFor(logging.DEBUG):
                    self.logger.debug(
                        f"get_frames_batch CMD (single pipe): {' '.join(shlex.quote(str(x)) for x in cmd_single)}")
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                # Always use BGR24 (3 bytes per pixel)
                buffer_frame_size = self._display_frame_w * self._display_frame_h * 3
                local_p2_proc = subprocess.Popen(cmd_single, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, creationflags=creation_flags)

            if not local_p2_proc or local_p2_proc.stdout is None:
                self.logger.error("get_frames_batch: Output FFmpeg process or its stdout is None.")
                return frames_batch

            # Always use BGR24 (3 bytes per pixel)
            frame_size = self._display_frame_w * self._display_frame_h * 3

            # Initialize progress tracking for frame buffer creation
            # Only show overlay for large batches — single-frame seeks should be invisible
            show_progress = num_frames_to_fetch > 2
            if show_progress:
                self.frame_buffer_total = num_frames_to_fetch
                self.frame_buffer_current = 0
                self.frame_buffer_progress = 0.0

            for i in range(num_frames_to_fetch):
                raw_frame_data = local_p2_proc.stdout.read(frame_size)
                if len(raw_frame_data) < frame_size:
                    p2_stderr_content = local_p2_proc.stderr.read().decode(
                        errors='ignore') if local_p2_proc.stderr else ""
                    self.logger.warning(
                        f"get_frames_batch: Incomplete data for frame {start_frame_num + i} (read {len(raw_frame_data)}/{frame_size}). P2 Stderr: {p2_stderr_content.strip()}")
                    if local_p1_proc and local_p1_proc.stderr:
                        p1_stderr_content = local_p1_proc.stderr.read().decode(errors='ignore')
                        self.logger.warning(f"get_frames_batch: P1 Stderr: {p1_stderr_content.strip()}")
                    break

                frame = np.frombuffer(raw_frame_data, dtype=np.uint8).reshape(
                    self._display_frame_h, self._display_frame_w, 3)  # BGR24

                # Apply GPU unwarp for VR frames if enabled
                if self.gpu_unwarp_enabled and self.gpu_unwarp_worker:
                    frame_idx = start_frame_num + i
                    self.gpu_unwarp_worker.submit_frame(frame_idx, frame,
                                                       timestamp_ms=frame_idx * (1000.0 / self.fps) if self.fps > 0 else 0.0,
                                                       timeout=0.1)
                    unwarp_result = self.gpu_unwarp_worker.get_unwrapped_frame(timeout=0.5)
                    if unwarp_result is not None:
                        _, frame, _ = unwarp_result
                    else:
                        self.logger.warning(f"GPU unwarp timeout for batch frame {frame_idx}")

                frames_batch[start_frame_num + i] = frame

                # Immediate display: update current frame as soon as target is decoded
                if immediate_display_frame is not None and (start_frame_num + i) == immediate_display_frame:
                    with self.frame_lock:
                        self.current_frame = frame
                        self.current_frame_index = immediate_display_frame
                        self._frame_version += 1

                # Update frame buffer progress
                if show_progress:
                    self.frame_buffer_current = i + 1
                    self.frame_buffer_progress = self.frame_buffer_current / self.frame_buffer_total if self.frame_buffer_total > 0 else 1.0

        except Exception as e:
            self.logger.error(f"get_frames_batch: Error fetching batch @{start_frame_num}: {e}", exc_info=True)
        finally:
            # [REDUNDANCY REMOVED] - Use the new helper method for termination
            if local_p1_proc:
                self._terminate_process(local_p1_proc, "Batch Pipe 1")
            if local_p2_proc:
                self._terminate_process(local_p2_proc, "Batch Pipe 2/Main")

        # Performance tracking completion
        decode_time = (time.perf_counter() - decode_start) * 1000
        if hasattr(self.app, 'gui_instance') and self.app.gui_instance:
            self.app.gui_instance.track_video_decode_time(decode_time)

        # Reset progress tracking
        self.frame_buffer_progress = 0.0
        self.frame_buffer_total = 0
        self.frame_buffer_current = 0

        self.logger.debug(
            f"get_frames_batch: Complete. Got {len(frames_batch)} frames for start {start_frame_num} (requested {num_frames_to_fetch}). Decode time: {decode_time:.2f}ms")
        return frames_batch

    def _get_specific_frame(self, frame_index_abs: int, update_current_index: bool = True, immediate_display: bool = False, use_thumbnail: bool = False) -> Optional[np.ndarray]:
        if not self.video_path or not self.video_info or self.video_info.get('fps', 0) <= 0:
            self.logger.warning("Cannot get frame: video not loaded/invalid FPS.")
            if update_current_index:
                self.current_frame_index = frame_index_abs
            return None

        with self.frame_cache_lock:
            if frame_index_abs in self.frame_cache:
                self.logger.debug(f"Cache HIT for frame {frame_index_abs}")
                frame = self.frame_cache[frame_index_abs]
                self.frame_cache.move_to_end(frame_index_abs)
                if update_current_index:
                    self.current_frame_index = frame_index_abs
                return frame

        # For instant seek preview, use fast thumbnail extractor
        # This provides immediate visual feedback (~20ms) while batch buffer loads in background
        if use_thumbnail and self.thumbnail_extractor is not None:
            self.logger.debug(f"Using fast thumbnail extractor for instant seek preview of frame {frame_index_abs}")
            frame = self.thumbnail_extractor.get_frame(frame_index_abs, use_gpu_unwarp=False)

            if frame is not None:
                # Cache the thumbnail frame
                with self.frame_cache_lock:
                    if len(self.frame_cache) >= self.frame_cache_max_size:
                        try:
                            self.frame_cache.popitem(last=False)
                        except KeyError:
                            pass
                    self.frame_cache[frame_index_abs] = frame
                    self.frame_cache.move_to_end(frame_index_abs)

                if update_current_index:
                    self.current_frame_index = frame_index_abs
                return frame
            else:
                self.logger.warning(f"Thumbnail extractor failed for frame {frame_index_abs}, falling back to FFmpeg")

        # When use_thumbnail was requested but extractor failed, fetch just 1 frame
        # instead of the full batch_fetch_size (600) to avoid long stalls
        fetch_size = 1 if use_thumbnail else self.batch_fetch_size

        # Standard FFmpeg fetch for cache misses
        if self.logger.isEnabledFor(logging.DEBUG):
            self.logger.debug(
                f"Cache MISS for frame {frame_index_abs}. Attempting batch fetch using get_frames_batch (batch size: {fetch_size}).")

        batch_start_frame = max(0, frame_index_abs - fetch_size // 2)
        if self.total_frames > 0:
            effective_end_frame_for_batch_calc = self.total_frames - 1
            if batch_start_frame + fetch_size - 1 > effective_end_frame_for_batch_calc:
                batch_start_frame = max(0, effective_end_frame_for_batch_calc - fetch_size + 1)

        num_frames_to_fetch_actual = fetch_size
        if self.total_frames > 0:
            num_frames_to_fetch_actual = min(fetch_size, self.total_frames - batch_start_frame)

        if num_frames_to_fetch_actual < 1 and self.total_frames > 0:
            num_frames_to_fetch_actual = 1
        elif num_frames_to_fetch_actual < 1 and self.total_frames == 0:
            num_frames_to_fetch_actual = fetch_size

        # Pass immediate_display flag for responsive seeking
        immediate_frame = frame_index_abs if immediate_display else None
        fetched_batch = self.get_frames_batch(batch_start_frame, num_frames_to_fetch_actual, immediate_display_frame=immediate_frame)

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

        if update_current_index:
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
        # TODO: Add ffprobe detection and metadata extraction for YUV videos. Pass metadata to cv2 so it can use the correct decoder. Use metadata + cv2.cvtColor to convert to RGB.
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
               '-show_entries',
               'stream=width,height,r_frame_rate,nb_frames,avg_frame_rate,duration,codec_name,codec_long_name,codec_type,pix_fmt,bits_per_raw_sample',
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
            audio_codec_name = ''
            audio_codec_long_name = ''
            audio_bitrate = 0
            audio_sample_rate = 0
            audio_channels = 0
            cmd_audio_check = ['ffprobe', '-v', 'error', '-select_streams', 'a:0',
                               '-show_entries', 'stream=codec_type,codec_name,codec_long_name,sample_rate,channels,bit_rate',
                               '-of', 'json', filename]
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                result_audio = subprocess.run(cmd_audio_check, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, creationflags=creation_flags)
                audio_data = json.loads(result_audio.stdout)
                if audio_data.get('streams') and audio_data['streams'][0].get('codec_type') == 'audio':
                    has_audio_ffprobe = True
                    a_stream = audio_data['streams'][0]
                    audio_codec_name = a_stream.get('codec_name', '')
                    audio_codec_long_name = a_stream.get('codec_long_name', '')
                    try:
                        audio_bitrate = int(a_stream.get('bit_rate', 0))
                    except (ValueError, TypeError):
                        audio_bitrate = 0
                    try:
                        audio_sample_rate = int(a_stream.get('sample_rate', 0))
                    except (ValueError, TypeError):
                        audio_sample_rate = 0
                    try:
                        audio_channels = int(a_stream.get('channels', 0))
                    except (ValueError, TypeError):
                        audio_channels = 0
            except (subprocess.SubprocessError, json.JSONDecodeError, KeyError, IndexError, OSError):
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

            self.logger.debug(
                f"Detected video properties: width={stream_info.get('width', 0)}, height={stream_info.get('height', 0)}, fps={fps:.2f}, bit_depth={bit_depth}")

            return {"duration": duration, "total_frames": total_frames, "fps": fps,
                    "width": int(stream_info.get('width', 0)), "height": int(stream_info.get('height', 0)),
                    "has_audio": has_audio_ffprobe, "bit_depth": bit_depth,
                    "file_size": file_size_bytes, "bitrate": bitrate_bps,
                    "is_vfr": is_vfr, "filename": file_name,
                    "codec_name": stream_info.get('codec_name', 'N/A'),
                    "codec_long_name": stream_info.get('codec_long_name', 'N/A'),
                    "audio_codec_name": audio_codec_name,
                    "audio_codec_long_name": audio_codec_long_name,
                    "audio_bitrate": audio_bitrate,
                    "audio_sample_rate": audio_sample_rate,
                    "audio_channels": audio_channels,
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
            process = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, creationflags=creation_flags)
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
            except OSError:
                pass

    def _terminate_ffmpeg_processes(self):
        """Safely terminates all active FFmpeg processes using the helper."""
        self._terminate_process(self.ffmpeg_pipe1_process, "Pipe 1")
        self.ffmpeg_pipe1_process = None
        self._terminate_process(self.ffmpeg_process, "Main/Pipe 2")
        self.ffmpeg_process = None
        self._stop_unified_pipe()

    def _start_ffmpeg_process(self, start_frame_abs_idx=0, num_frames_to_output_ffmpeg=None):
        self._terminate_ffmpeg_processes()

        if not self.video_path or not self.video_info or self.video_info.get('fps', 0) <= 0:
            self.logger.warning("Cannot start FFmpeg: video not properly opened or invalid FPS.")
            return False
        
        # Check if dual-output mode is enabled
        if self.dual_output_enabled:
            return self._start_dual_output_ffmpeg_process(start_frame_abs_idx, num_frames_to_output_ffmpeg)

        start_time_seconds = start_frame_abs_idx / self.video_info['fps']
        self.current_stream_start_frame_abs = start_frame_abs_idx
        self.frames_read_from_current_stream = 0
        
        # Optimize ffmpeg for MAX_SPEED processing
        common_ffmpeg_prefix = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']
        
        # Add MAX_SPEED optimizations if in MAX_SPEED mode
        if (hasattr(self.app, 'app_state_ui') and 
            hasattr(self.app.app_state_ui, 'selected_processing_speed_mode') and
            self.app.app_state_ui.selected_processing_speed_mode == constants.ProcessingSpeedMode.MAX_SPEED):
            # Optimize ffmpeg for maximum decode speed:
            # Hardware acceleration: Handled by individual pipe paths (don't add to common prefix)
            # -fflags +genpts+fastseek: Generate timestamps and enable fast seeking
            # -threads 0: Use optimal number of threads
            # -preset ultrafast: Fastest decode preset
            # -tune zerolatency: Minimize decode latency
            # -probesize 32: Smaller probe for faster startup
            # -analyzeduration 1: Faster stream analysis
            # No -re flag: Don't limit to real-time (decode as fast as possible)
            
            # Add speed optimizations (hardware acceleration handled by pipe-specific code)
            # NOTE: -preset and -tune are encoding options, not decoding options
            common_ffmpeg_prefix.extend([
                '-fflags', '+genpts+fastseek', 
                '-threads', '0',
                '-probesize', '32',
                '-analyzeduration', '1'
            ])
            self.logger.info("FFmpeg optimized for MAX_SPEED processing with fast decode")

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
            effective_vf_pipe2 = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease,pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
            cmd2.extend(['-vf', effective_vf_pipe2])
            if num_frames_to_output_ffmpeg and num_frames_to_output_ffmpeg > 0:
                cmd2.extend(['-frames:v', str(num_frames_to_output_ffmpeg)])
            # Always use BGR24 - GPU unwarp worker handles BGR->RGBA conversion internally
            cmd2.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

            self.logger.info(f"Pipe 1 CMD: {' '.join(shlex.quote(str(x)) for x in cmd1)}")
            self.logger.info(f"Pipe 2 CMD: {' '.join(shlex.quote(str(x)) for x in cmd2)}")
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                self.ffmpeg_pipe1_process = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, creationflags=creation_flags)
                if self.ffmpeg_pipe1_process.stdout is None:
                    raise IOError("Pipe 1 stdout is None.")
                self.ffmpeg_process = subprocess.Popen(cmd2, stdin=self.ffmpeg_pipe1_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, creationflags=creation_flags)
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
            effective_vf = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease,pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
            cmd.extend(['-vf', effective_vf])
            if num_frames_to_output_ffmpeg and num_frames_to_output_ffmpeg > 0:
                cmd.extend(['-frames:v', str(num_frames_to_output_ffmpeg)])
            # Always use BGR24 - GPU unwarp worker handles BGR->RGBA conversion internally
            cmd.extend(['-pix_fmt', 'bgr24', '-f', 'rawvideo', 'pipe:1'])

            self.logger.info(f"Single Pipe CMD: {' '.join(shlex.quote(str(x)) for x in cmd)}")
            try:
                creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
                self.ffmpeg_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=-1, creationflags=creation_flags)
                return True
            except Exception as e:
                self.logger.error(f"Failed to start FFmpeg: {e}", exc_info=True)
                self.ffmpeg_process = None
                return False

    def start_processing(self, start_frame=None, end_frame=None, cli_progress_callback=None):
        # If already processing but paused, just clear the pause event.
        # The GUI thread NEVER touches FFmpeg processes.
        # The processing thread self-heals: after waking from pause it checks
        # cursor vs pipe position and skips/restarts as needed.
        if self.is_processing and self.pause_event.is_set():
            self.logger.info(f"Resuming playback from frame {self.current_frame_index}")
            self.pause_event.clear()

            # Notify playback state observers (e.g., device_control) that playback resumed
            if self._playback_state_callbacks:
                current_time_ms = (self.current_frame_index / self.fps) * 1000.0 if self.fps > 0 else 0.0
                self._notify_playback_state_callbacks(True, current_time_ms)

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

        self._hwaccel_fallback_attempted = False
        self.processing_start_frame_limit = effective_start_frame
        self.processing_end_frame_limit = -1
        if end_frame is not None and end_frame >= 0:
            self.processing_end_frame_limit = min(end_frame, self.total_frames - 1)

        num_frames_to_process = None
        if self.processing_end_frame_limit != -1:
            num_frames_to_process = self.processing_end_frame_limit - self.processing_start_frame_limit + 1

        # Try to adopt the unified pipe for zero-startup processing.
        # The unified pipe uses the same filter chain (v360, scale, etc.),
        # so frames are identical. Only works for single-pipe (non-10-bit-CUDA).
        adopted = False
        if (self._ffmpeg_pipe is not None
                and self._ffmpeg_pipe.poll() is None
                and not self._is_10bit_cuda_pipe_needed()
                and not self.dual_output_enabled):
            # Check if pipe is positioned at the effective start frame
            if self._ffmpeg_read_position == effective_start_frame:
                # Clean up any stale processing pipes
                self._terminate_process(self.ffmpeg_pipe1_process, "Pipe 1")
                self.ffmpeg_pipe1_process = None
                self._terminate_process(self.ffmpeg_process, "Main/Pipe 2")
                # Transfer unified pipe → processing pipe
                self.ffmpeg_process = self._ffmpeg_pipe
                self._ffmpeg_pipe = None
                self.current_stream_start_frame_abs = self._ffmpeg_read_position
                self.frames_read_from_current_stream = 0
                adopted = True
                self.logger.info(f"Adopted unified pipe for processing (continuing from frame {self.current_stream_start_frame_abs})")
            else:
                # Pipe exists but at wrong position — kill it
                self._stop_unified_pipe()

        if not adopted:
            if not self._start_ffmpeg_process(start_frame_abs_idx=self.processing_start_frame_limit, num_frames_to_output_ffmpeg=num_frames_to_process):
                self.logger.error("Failed to start FFmpeg for processing start.")
                return

        # Initialize GPU unwarp worker for VR if enabled
        self._init_gpu_unwarp_worker()

        self.is_processing = True
        self.pause_event.clear()
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop, name="VideoProcessingThread")
        self.processing_thread.daemon = True
        self.processing_thread.start()

        self.logger.debug(
            f"Started GUI processing. Range: {self.processing_start_frame_limit} to "
            f"{self.processing_end_frame_limit if self.processing_end_frame_limit != -1 else 'EOS'}")

    def pause_processing(self):
        if not self.is_processing or self.pause_event.is_set():
            return

        self.logger.info(f"Pausing playback at frame {self.current_frame_index}")
        self.pause_event.set()

        # Notify playback state observers (e.g., device_control) that playback stopped
        if self._playback_state_callbacks:
            current_time_ms = (self.current_frame_index / self.fps) * 1000.0 if self.fps > 0 else 0.0
            self._notify_playback_state_callbacks(False, current_time_ms)

        # Optional callback to update UI elements, like a play/pause button icon.
        if self.app and hasattr(self.app, 'on_processing_paused'):
            self.app.on_processing_paused()

    def stop_processing(self, join_thread=True):
        is_currently_processing = self.is_processing
        is_thread_alive = self.processing_thread and self.processing_thread.is_alive()

        if not is_currently_processing and not is_thread_alive:
            self._terminate_ffmpeg_processes()
            return

        self.logger.debug("Stopping GUI processing...")
        was_scripting_session = self.tracker and self.tracker.tracking_active
        scripted_range = (self.processing_start_frame_limit, self.current_frame_index)

        # Notify playback state observers (e.g., device_control) that playback stopped
        if self._playback_state_callbacks:
            current_time_ms = (self.current_frame_index / self.fps) * 1000.0 if self.fps > 0 else 0.0
            self._notify_playback_state_callbacks(False, current_time_ms)

        self.is_processing = False
        self.pause_event.clear()
        self.stop_event.set()

        if join_thread:
            # Full blocking cleanup — used for tracking stop, app shutdown, etc.
            if hasattr(self, 'dual_output_processor'):
                self.dual_output_processor.disable_dual_output_mode()

            self._terminate_ffmpeg_processes()

            thread_to_join = self.processing_thread
            if thread_to_join and thread_to_join.is_alive():
                if threading.current_thread() is not thread_to_join:
                    self.logger.info(f"Joining processing thread: {thread_to_join.name} during stop.")
                    thread_to_join.join(timeout=2.0)
                    if thread_to_join.is_alive():
                        self.logger.warning("Processing thread did not join cleanly after stop signal.")
            self.processing_thread = None

            # Stop GPU unwarp worker if active
            if self.gpu_unwarp_worker:
                self.logger.info("Stopping GPU unwarp worker...")
                self.gpu_unwarp_worker.stop()
                self.gpu_unwarp_worker = None
                self.gpu_unwarp_enabled = False

            if self.tracker:
                self.logger.debug("Signaling tracker to stop.")
                self.tracker.stop_tracking()

            self.enable_tracker_processing = False

            if self.app and hasattr(self.app, 'on_processing_stopped'):
                self.app.on_processing_stopped(was_scripting_session=was_scripting_session, scripted_frame_range=scripted_range)
        else:
            # Non-blocking path (used by _seek_video_worker during seek-while-playing):
            # Send SIGTERM to unblock the processing thread's stdout.read().
            # Pre-spawn unified pipe at cursor position for immediate arrow nav.
            for proc in (self.ffmpeg_pipe1_process, self.ffmpeg_process):
                if proc is not None and proc.poll() is None:
                    try:
                        proc.terminate()
                    except OSError:
                        pass
            self.enable_tracker_processing = False

        self.logger.debug("GUI processing stopped.")

    def seek_video(self, frame_index: int):
        """
        Seek to a specific frame. This runs asynchronously to avoid blocking the UI.
        If a seek is already in progress, it will be cancelled and replaced with the new seek.
        """
        if not self.video_info or self.video_info.get('fps', 0) <= 0 or self.total_frames <= 0:
            return

        target_frame = max(0, min(frame_index, self.total_frames - 1))
        self.playhead_override_ms = None  # Clear on any non-point seek
        self.logger.debug(f"Seek to frame {target_frame} (from {self.current_frame_index})")

        # Update frame index immediately so the timeline doesn't snap back
        # to the old position while the async decode is in progress
        self.current_frame_index = target_frame

        # If the target frame is cached, update current_frame immediately
        # so the display shows the correct frame without waiting for the worker.
        # Otherwise, keep showing the previous frame to avoid flicker.
        with self.frame_cache_lock:
            cached = self.frame_cache.get(target_frame)
        if cached is not None:
            with self.frame_lock:
                self.current_frame = cached
                self._frame_version += 1

        # Invalidate arrow-nav state since the user jumped to a new position
        self._clear_nav_state()

        # Notify registered observers (e.g., streamer) of seek event
        self._notify_seek_callbacks(target_frame)

        # If a seek is already in progress, wait for it to finish (or cancel it)
        if self.seek_in_progress and self.seek_thread and self.seek_thread.is_alive():
            self.logger.debug(f"Seek coalesced: target={target_frame}")
            # Don't block - just set the new target and let the existing thread handle it
            self.seek_request_frame_index = target_frame
            return

        # Mark seek as in progress
        self.seek_in_progress = True
        self.seek_request_frame_index = target_frame

        # Run seek operation in background thread to avoid blocking UI
        self.seek_thread = threading.Thread(
            target=self._seek_video_worker,
            args=(target_frame,),
            daemon=True,
            name=f"SeekThread-{target_frame}"
        )
        self.seek_thread.start()

    def _seek_video_worker(self, target_frame: int):
        """Worker thread for async seek operations. Runs blocking operations without freezing UI.

        Drains coalesced seek requests: after each seek completes, checks if a newer
        target arrived via seek_request_frame_index and loops to serve it.
        """
        try:
            was_processing = self.is_processing
            was_paused = self.is_processing and self.pause_event.is_set()
            stored_end_limit = self.processing_end_frame_limit
            # Remember if tracker was active before stopping (important for streamer mode)
            was_tracking = self.tracker and self.tracker.tracking_active

            if was_processing:
                self.stop_processing(join_thread=False)

            # Coalesce loop: drain any pending seek requests that arrived during decode
            while True:
                # Show instant thumbnail preview only (no buffer loading for scrubbing)
                # update_current_index=False: seek_video() already set current_frame_index
                # immediately. Letting _get_specific_frame overwrite it would race with
                # newer coalesced seeks and cause the timeline to snap back.
                seek_t0 = time.perf_counter()
                new_frame = self._get_specific_frame(target_frame, immediate_display=False, use_thumbnail=True, update_current_index=False)
                seek_ms = (time.perf_counter() - seek_t0) * 1000

                # Report scrub decode cost to perf monitor
                if hasattr(self.app, 'gui_instance') and self.app.gui_instance:
                    self.app.gui_instance.track_frame_seek_time(seek_ms, path="scrub")

                with self.frame_lock:
                    self.current_frame = new_frame
                    self._frame_version += 1

                # Check if a newer seek target arrived while we were decoding
                pending = self.seek_request_frame_index
                if pending is not None and pending != target_frame:
                    target_frame = pending
                    self.seek_request_frame_index = None
                    continue  # Loop to serve the newer target

                # Only update frame index for the FINAL decoded frame.
                # During the loop, seek_video() already set current_frame_index
                # immediately for each new seek request — overwriting here with
                # an intermediate (stale) target would snap the timeline back.
                self.current_frame_index = target_frame
                break  # No pending request, we're done

            if was_processing and not was_paused:
                self.start_processing(start_frame=self.current_frame_index, end_frame=stored_end_limit)
                # Restart tracker if it was active before seeking (e.g., in streamer mode without chapters)
                if was_tracking and self.tracker and not self.tracker.tracking_active:
                    self.tracker.start_tracking()

        except Exception as e:
            self.logger.error(f"Error during seek operation: {e}", exc_info=True)
        finally:
            self.seek_in_progress = False
            self.seek_request_frame_index = None

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
                    # Create 640x640 processing frame for tracker (HD frame is display-only)
                    if self.is_hd_active:
                        processing_frame = self._make_processing_frame(raw_frame_to_process)
                    else:
                        processing_frame = raw_frame_to_process.copy()
                    # Tracker populates live_overlay; display frame stays clean
                    self.tracker.process_frame(processing_frame, timestamp_ms, self.current_frame_index)
            except Exception as e:
                self.logger.error(f"Error processing frame with tracker in display_current_frame: {e}", exc_info=True)


    def _processing_loop(self):
        if not self.ffmpeg_process or self.ffmpeg_process.stdout is None:
            self.logger.error("_processing_loop: FFmpeg process/stdout not available. Exiting.")
            self.is_processing = False
            return

        start_time = time.time()  # For calculating FPS and ETA in the callback

        next_frame_target_time = time.perf_counter()
        self.last_processed_chapter_id = None

        _SKIP_THRESHOLD = 90  # Frames: skip via read-discard if cursor ahead by <= this

        try:
            # The main processing loop
            was_paused = False
            while not self.stop_event.is_set():
                # --- Entering pause: hand off pipe for arrow nav ---
                if self.pause_event.is_set() and not was_paused:
                    was_paused = True
                    # Transfer processing pipe → unified pipe so arrow nav can use it
                    if self.ffmpeg_process is not None:
                        self._ffmpeg_pipe = self.ffmpeg_process
                        self._ffmpeg_read_position = (
                            self.current_stream_start_frame_abs + self.frames_read_from_current_stream)
                        self.ffmpeg_process = None
                        self.logger.info(
                            f"Pause at frame {self.current_frame_index}, "
                            f"pipe handed off (read_pos={self._ffmpeg_read_position})")
                    else:
                        self.logger.info(f"Pause at frame {self.current_frame_index} (no pipe to hand off)")

                while self.pause_event.is_set():
                    if self.stop_event.is_set():
                        break
                    self.stop_event.wait(0.01)

                if self.stop_event.is_set():
                    break

                # --- Waking from pause: reclaim pipe, self-heal cursor divergence ---
                if was_paused:
                    was_paused = False
                    cursor = self.current_frame_index
                    heal_t0 = time.perf_counter()

                    # Try to reclaim the unified pipe (arrow nav may have used/replaced it)
                    if self._ffmpeg_pipe is not None and self._ffmpeg_pipe.poll() is None:
                        self.ffmpeg_process = self._ffmpeg_pipe
                        self._ffmpeg_pipe = None
                        pipe_pos = self._ffmpeg_read_position
                        gap = cursor - pipe_pos  # Normal is -1 (pipe 1 ahead of cursor)

                        if -1 <= gap <= 0:
                            # Normal: pipe is at or 1 ahead of cursor. Just sync tracking.
                            self.current_stream_start_frame_abs = pipe_pos
                            self.frames_read_from_current_stream = 0
                            heal_ms = (time.perf_counter() - heal_t0) * 1000
                            self.logger.info(
                                f"Resumed at frame {cursor} (gap={gap}, pipe at {pipe_pos}, {heal_ms:.1f}ms)")
                        elif 0 < gap <= _SKIP_THRESHOLD:
                            # Cursor ahead of pipe: skip-discard frames (fast)
                            self.current_stream_start_frame_abs = pipe_pos
                            self.frames_read_from_current_stream = 0
                            skipped = 0
                            if self.ffmpeg_process.stdout:
                                for _ in range(gap):
                                    d = self.ffmpeg_process.stdout.read(self.frame_size_bytes)
                                    if not d or len(d) < self.frame_size_bytes:
                                        break
                                    self.frames_read_from_current_stream += 1
                                    skipped += 1
                            heal_ms = (time.perf_counter() - heal_t0) * 1000
                            self.logger.info(
                                f"Resumed: skipped {skipped}/{gap} frames ({heal_ms:.1f}ms)")
                        else:
                            # Large backward or forward: kill reclaimed pipe, start fresh
                            self._terminate_process(self.ffmpeg_process, "Main")
                            self.ffmpeg_process = None
                            if not self._start_ffmpeg_process(start_frame_abs_idx=cursor):
                                self.logger.error("Pipe restart failed after pause, ending loop.")
                                break
                            heal_ms = (time.perf_counter() - heal_t0) * 1000
                            self.logger.info(
                                f"Resumed: pipe restarted at frame {cursor} (gap={gap}, {heal_ms:.1f}ms)")
                    else:
                        # Pipe died or was replaced — start fresh
                        self._stop_unified_pipe()
                        if not self._start_ffmpeg_process(start_frame_abs_idx=cursor):
                            self.logger.error("Fresh pipe failed after pause, ending loop.")
                            break
                        heal_ms = (time.perf_counter() - heal_t0) * 1000
                        self.logger.info(
                            f"Resumed: fresh pipe at frame {cursor} ({heal_ms:.1f}ms)")

                # Re-read pipe ref after potential restart
                loop_ffmpeg_process = self.ffmpeg_process
                if loop_ffmpeg_process is None or loop_ffmpeg_process.stdout is None:
                    self.logger.info("Processing pipe is None, ending loop.")
                    break

                # The original logic of the loop continues below
                speed_mode = self.app.app_state_ui.selected_processing_speed_mode
                
                # Debug: Log speed mode selection for MAX_SPEED troubleshooting
                if hasattr(self, '_last_logged_speed_mode') and self._last_logged_speed_mode != speed_mode:
                    self.logger.info(f"Processing speed mode changed to: {speed_mode}")
                    self._last_logged_speed_mode = speed_mode
                elif not hasattr(self, '_last_logged_speed_mode'):
                    self.logger.info(f"Initial processing speed mode: {speed_mode}")
                    self._last_logged_speed_mode = speed_mode
                
                if speed_mode == constants.ProcessingSpeedMode.REALTIME:
                    target_delay = 1.0 / self.fps if self.fps > 0 else (1.0 / 30.0)
                elif speed_mode == constants.ProcessingSpeedMode.SLOW_MOTION:
                    slo_mo_fps = getattr(self.app.app_state_ui, 'slow_motion_fps', 10.0)
                    target_delay = 1.0 / max(1.0, slo_mo_fps)
                else:  # Max Speed
                    target_delay = 0.0
                    
                # Debug: Log target_delay for MAX_SPEED troubleshooting
                if speed_mode == constants.ProcessingSpeedMode.MAX_SPEED and target_delay != 0.0:
                    self.logger.error(f"MAX_SPEED mode but target_delay = {target_delay} (should be 0.0)")
                elif speed_mode == constants.ProcessingSpeedMode.MAX_SPEED and not hasattr(self, '_max_speed_logged'):
                    self.logger.info(f"MAX_SPEED mode active: target_delay = {target_delay}")
                    self._max_speed_logged = True

                current_chapter = self.app.funscript_processor.get_chapter_at_frame(self.current_frame_index)
                current_chapter_id = current_chapter.unique_id if current_chapter else None

                if current_chapter_id != self.last_processed_chapter_id:
                    # Only auto-start/stop tracker if enable_tracker_processing is True
                    # This prevents the play button from triggering live tracking after offline analysis
                    if self.tracker and self.enable_tracker_processing:
                        # Check if we should track in this chapter based on category
                        from config.constants import POSITION_INFO_MAPPING
                        should_track = True

                        if current_chapter:
                            # Check chapter category
                            position_info = POSITION_INFO_MAPPING.get(current_chapter.position_short_name, {})
                            category = position_info.get('category', 'Position')
                            should_track = (category == "Position")  # Only track Position category

                            # Reconfigure if chapter has user ROI
                            if should_track and current_chapter.user_roi_fixed:
                                self.tracker.reconfigure_for_chapter(current_chapter)
                        # No chapter (unchaptered) = should track (default behavior)

                        # Start/stop tracker based on category
                        if should_track and not self.tracker.tracking_active:
                            self.tracker.start_tracking()
                            if current_chapter:
                                self.logger.info(f"Tracker resumed for Position chapter: {current_chapter.position_short_name}")
                            else:
                                self.logger.info("Tracker active in unchaptered section")
                        elif not should_track and self.tracker.tracking_active:
                            self.tracker.stop_tracking()
                            if current_chapter:
                                self.logger.info(f"Tracker paused for Not Relevant chapter: {current_chapter.position_short_name}")

                    self.last_processed_chapter_id = current_chapter_id

                # Only auto-start tracker for user ROI if enable_tracker_processing is True
                if current_chapter and self.tracker and self.enable_tracker_processing and not self.tracker.tracking_active and current_chapter.user_roi_fixed:
                    self.tracker.start_tracking()

                if self.ffmpeg_pipe1_process and self.ffmpeg_pipe1_process.poll() is not None:
                    pipe1_stderr = self.ffmpeg_pipe1_process.stderr.read(4096).decode(
                        errors='ignore') if self.ffmpeg_pipe1_process.stderr else ""
                    rc1 = self.ffmpeg_pipe1_process.returncode
                    if rc1 != 0:
                        self.logger.warning(
                            f"FFmpeg Pipe 1 died. Exit: {rc1}. Stderr: {pipe1_stderr.strip()}.")
                    else:
                        self.logger.debug("FFmpeg Pipe 1 exited normally.")
                    self.is_processing = False
                    break

                if loop_ffmpeg_process.poll() is not None:
                    rc = loop_ffmpeg_process.returncode
                    if rc != 0:
                        stderr_output = loop_ffmpeg_process.stderr.read(4096).decode(
                            errors='ignore') if loop_ffmpeg_process.stderr else ""
                        self.logger.warning(
                            f"FFmpeg process died (pid={loop_ffmpeg_process.pid}). Exit: {rc}. Stderr: {stderr_output.strip()}.")
                    else:
                        self.logger.debug(f"FFmpeg process exited normally (pid={loop_ffmpeg_process.pid}).")
                    self.is_processing = False
                    break

                # Get frame from dual output processor or standard FFmpeg
                raw_frame_bytes = None
                decode_start = time.perf_counter()
                if self.dual_output_enabled:
                    # Use dual output processor
                    processing_frame = self.dual_output_processor.get_processing_frame()
                    if processing_frame is not None:
                        # Convert numpy array back to bytes for compatibility
                        raw_frame_bytes = processing_frame.tobytes()
                    else:
                        raw_frame_bytes = None
                    decode_time = (time.perf_counter() - decode_start) * 1000.0
                    self._decode_samples.append(decode_time)
                else:
                    # Standard FFmpeg reading
                    if loop_ffmpeg_process.stdout is not None:
                        raw_frame_bytes = loop_ffmpeg_process.stdout.read(self.frame_size_bytes)
                        decode_time = (time.perf_counter() - decode_start) * 1000.0
                        self._decode_samples.append(decode_time)
                        if decode_time > 200:
                            self.logger.warning(f"Slow frame read: {decode_time:.0f}ms at frame {self.current_frame_index}")
                    else:
                        raw_frame_bytes = None

                raw_frame_len = len(raw_frame_bytes) if raw_frame_bytes is not None else 0
                if not raw_frame_bytes or raw_frame_len < self.frame_size_bytes:
                    if self.dual_output_enabled:
                        self.logger.info("End of dual-output stream or no frames available.")
                    else:
                        self.logger.info(
                            f"End of FFmpeg GUI stream or incomplete frame (read {raw_frame_len}/{self.frame_size_bytes}).")
                        # Log FFmpeg stderr to help diagnose why it produced no output
                        # (e.g., filter errors, codec issues, file access problems)
                        ffmpeg_stderr = ""
                        if loop_ffmpeg_process.stderr:
                            try:
                                ffmpeg_stderr = loop_ffmpeg_process.stderr.read(8192).decode(errors='ignore').strip()
                                if ffmpeg_stderr:
                                    self.logger.warning(f"FFmpeg stderr: {ffmpeg_stderr}")
                            except Exception:
                                pass

                        # Auto-fallback: if FFmpeg failed on the very first frame and hardware
                        # acceleration is active, retry with CPU-only decoding. This handles
                        # cases where ffmpeg reports a hwaccel as available (compiled-in) but
                        # the actual GPU/driver doesn't support it at runtime.
                        if (self.frames_read_from_current_stream == 0
                                and self._get_ffmpeg_hwaccel_args()
                                and not self.dual_output_enabled
                                and not getattr(self, '_hwaccel_fallback_attempted', False)):
                            self._hwaccel_fallback_attempted = True
                            self.logger.warning(
                                "Hardware-accelerated FFmpeg failed on first frame. "
                                "Retrying with CPU-only decoding...",
                                extra={'status_message': True, 'duration': 5.0})
                            # Force CPU-only and persist the change (hwaccel doesn't work on this hardware)
                            if self.app:
                                self.app.hardware_acceleration_method = "none"
                                if hasattr(self.app, 'app_settings'):
                                    self.app.app_settings.set("hardware_acceleration_method", "none")
                            self._terminate_process(loop_ffmpeg_process, "HWAccel-failed")
                            self.ffmpeg_process = None
                            if self._start_ffmpeg_process(
                                    start_frame_abs_idx=self.current_stream_start_frame_abs):
                                self.logger.info("CPU-only FFmpeg fallback started successfully.")
                                continue  # Retry the loop with the new process
                            else:
                                self.logger.error("CPU-only FFmpeg fallback also failed.")

                    self.is_processing = False
                    self.enable_tracker_processing = False
                    if self.app:
                        was_scripting_at_end = self.tracker and self.tracker.tracking_active
                        end_range = (self.processing_start_frame_limit, self.current_frame_index)
                        if self.tracker and self.tracker.tracking_active:
                            self.tracker.stop_tracking()
                        self.app.on_processing_stopped(was_scripting_session=was_scripting_at_end, scripted_frame_range=end_range)
                    break

                self.current_frame_index = self.current_stream_start_frame_abs + self.frames_read_from_current_stream
                self.frames_read_from_current_stream += 1
                self._ffmpeg_read_position = self.current_frame_index + 1

                # Notify playback state observers (e.g., device_control)
                if self._playback_state_callbacks:
                    is_currently_playing = self.is_processing and not self.pause_event.is_set()
                    current_time_ms = (self.current_frame_index / self.fps) * 1000.0 if self.fps > 0 else 0.0
                    self._notify_playback_state_callbacks(is_currently_playing, current_time_ms)

                if self.cli_progress_callback:
                    # Throttle updates to avoid slowing down processing (e.g., update every 10 frames)
                    if self.current_frame_index % 10 == 0 or self.current_frame_index == self.total_frames - 1:
                        self.cli_progress_callback(self.current_frame_index, self.total_frames, start_time)

                if self.processing_end_frame_limit != -1 and self.current_frame_index > self.processing_end_frame_limit:
                    self.logger.info(f"Reached GUI end_frame_limit ({self.processing_end_frame_limit}). Stopping.")
                    self.is_processing = False
                    self.enable_tracker_processing = False
                    if self.app:
                        was_scripting_at_end_limit = self.tracker and self.tracker.tracking_active
                        end_range_limit = (self.processing_start_frame_limit, self.processing_end_frame_limit)
                        if self.tracker and self.tracker.tracking_active:
                            self.tracker.stop_tracking()
                        self.app.on_processing_stopped(was_scripting_session=was_scripting_at_end_limit, scripted_frame_range=end_range_limit)
                    break
                if self.total_frames > 0 and self.current_frame_index >= self.total_frames:
                    self.logger.info("Reached end of video. Stopping GUI processing.")
                    self.is_processing = False
                    self.enable_tracker_processing = False
                    if self.app:
                        was_scripting_at_eos = self.tracker and self.tracker.tracking_active
                        end_range_eos = (self.processing_start_frame_limit, self.current_frame_index)
                        if self.tracker and self.tracker.tracking_active:
                            self.tracker.stop_tracking()
                        self.app.on_processing_stopped(was_scripting_session=was_scripting_at_eos, scripted_frame_range=end_range_eos)
                    break

                # Always use BGR24 format (3 bytes per pixel)
                expected_size = self._display_frame_w * self._display_frame_h * 3
                actual_bytes = len(raw_frame_bytes)

                # Validate frame size
                if actual_bytes != expected_size:
                    self.logger.error(f"Invalid frame size: {actual_bytes} bytes (expected {expected_size}). Skipping frame.")
                    continue

                frame_np = np.frombuffer(raw_frame_bytes, dtype=np.uint8).reshape(self._display_frame_h, self._display_frame_w, 3)

                # Apply GPU unwarp for VR frames if enabled (before processing frame creation)
                if self.gpu_unwarp_enabled and self.gpu_unwarp_worker:
                    unwarp_start = time.perf_counter()

                    # Submit current frame to GPU worker (non-blocking)
                    submit_success = self.gpu_unwarp_worker.submit_frame(self.current_frame_index, frame_np,
                                                       timestamp_ms=self.current_frame_index * (1000.0 / self.fps) if self.fps > 0 else 0.0,
                                                       timeout=0.05)

                    # For MAX_SPEED: wait synchronously for current frame
                    # For realtime: use async pattern (get previous frame from queue)
                    is_max_speed = target_delay == 0.0
                    timeout = 0.2 if is_max_speed else 0.01

                    if submit_success:
                        unwarp_result = self.gpu_unwarp_worker.get_unwrapped_frame(timeout=timeout)
                        if unwarp_result is not None:
                            _, frame_np, _ = unwarp_result
                    # else: Use fisheye frame_np as-is (queue full or timeout)

                    unwarp_time = (time.perf_counter() - unwarp_start) * 1000.0
                    self._unwarp_samples.append(unwarp_time)

                # Create processing frame for tracker (640x640 with padding) when HD active.
                # Must be AFTER GPU unwarp so VR frames are already unwrapped.
                if self.is_hd_active:
                    processing_frame = self._make_processing_frame(frame_np)
                else:
                    processing_frame = frame_np

                # Feed clean decoded frame into unified buffer (before tracker overlays)
                # so arrow-key backward nav works seamlessly after play stops.
                self._buffer_append(self.current_frame_index, frame_np.copy())

                if self.tracker and self.tracker.tracking_active and self.enable_tracker_processing:
                    timestamp_ms = int(self.current_frame_index * (1000.0 / self.fps)) if self.fps > 0 else int(
                        time.time() * 1000)

                    try:
                        yolo_start = time.perf_counter()
                        # Tracker processes the 640x640 processing_frame (not the display frame).
                        # Overlays are rendered via ImGui draw_list, not burned into the frame.
                        self.tracker.process_frame(processing_frame.copy(), timestamp_ms, self.current_frame_index)
                        yolo_time = (time.perf_counter() - yolo_start) * 1000.0
                        self._yolo_samples.append(yolo_time)
                    except Exception as e:
                        self.logger.error(f"Error in tracker.process_frame during loop: {e}", exc_info=True)

                # Update timing metrics display (once per second)
                self._update_timing_metrics()

                with self.frame_lock:
                    # Always display the clean frame (HD or standard) — overlays via ImGui
                    self.current_frame = frame_np
                    self._frame_version += 1

                self.frames_for_fps_calc += 1
                current_time_fps_calc = time.time()
                elapsed = current_time_fps_calc - self.last_fps_update_time
                if elapsed >= 1.0:
                    self.actual_fps = self.frames_for_fps_calc / elapsed
                    self.last_fps_update_time = current_time_fps_calc
                    self.frames_for_fps_calc = 0

                # Apply timing control only if not in MAX_SPEED mode
                if target_delay > 0:
                    # Check if we should skip frame delay (when behind by 3+ frames)
                    should_skip = False
                    if hasattr(self, 'sync_server') and self.sync_server:
                        should_skip = self.sync_server.should_skip_frame()

                    if not should_skip:
                        current_time = time.perf_counter()
                        sleep_duration = next_frame_target_time - current_time

                        if sleep_duration > 0:
                            time.sleep(sleep_duration)

                        if next_frame_target_time < current_time - target_delay:
                            next_frame_target_time = current_time + target_delay
                        else:
                            next_frame_target_time += target_delay
                    else:
                        # Skipping frame delay to catch up
                        current_time = time.perf_counter()
                        next_frame_target_time = current_time + target_delay
        finally:
            self.logger.info(f"_processing_loop ending. is_processing: {self.is_processing}, stop_event: {self.stop_event.is_set()}")

            # Notify playback state observers that playback stopped.
            # stop_processing() does this for explicit stops, but natural exits
            # (end of video, end frame limit, FFmpeg pipe death) skip stop_processing()
            # entirely — leaving AudioVideoSync and DeviceControl thinking playback
            # is still active, which causes audio/device activation on subsequent seeks.
            if self._playback_state_callbacks:
                current_time_ms = (self.current_frame_index / self.fps) * 1000.0 if self.fps > 0 else 0.0
                self._notify_playback_state_callbacks(False, current_time_ms)

            self._terminate_process(self.ffmpeg_pipe1_process, "Pipe 1")
            self.ffmpeg_pipe1_process = None
            # Transfer processing pipe to unified pipe for reuse if still alive.
            # If pipe was already handed off during pause, ffmpeg_process is None
            # and _ffmpeg_pipe already has the pipe — leave it alone.
            if self.ffmpeg_process is not None:
                if self.ffmpeg_process.poll() is None:
                    self._ffmpeg_pipe = self.ffmpeg_process
                    self._ffmpeg_read_position = (
                        self.current_stream_start_frame_abs + self.frames_read_from_current_stream)
                    self.logger.debug("Transferred processing pipe to unified pipe for arrow nav reuse")
                else:
                    self._terminate_process(self.ffmpeg_process, "Main")
            self.ffmpeg_process = None
            self.is_processing = False
            self.last_processed_chapter_id = None


    # ============================================================================
    # Single FFmpeg Dual-Output Integration Methods
    # ============================================================================
    
    
    def is_video_open(self) -> bool:
        """Checks if a video is currently loaded and has valid information."""
        return bool(self.video_path and self.video_info and self.video_info.get('total_frames', 0) > 0)

    def reset(self, close_video=False, skip_tracker_reset=False):
        self.logger.debug("Resetting VideoProcessor...")
        self.stop_processing(join_thread=True)
        self._clear_cache()
        self.current_frame_index = 0
        self.frames_read_from_current_stream = 0
        self.current_stream_start_frame_abs = 0
        self.seek_request_frame_index = None
        if self.tracker and not skip_tracker_reset:
            self.tracker.reset()
        if close_video:
            if self.thumbnail_extractor:
                self.thumbnail_extractor.close()
                self.thumbnail_extractor = None
            self.video_path = ""
            self._active_video_source_path = ""
            self.video_info = {}
            self.determined_video_type = None
            self.ffmpeg_filter_string = ""
            self.logger.debug("Video closed. Params reset.")
        with self.frame_lock:
            self.current_frame = None
        if self.video_path and self.video_info and not close_video:
            self.logger.info("Fetching frame 0 after reset (video still loaded).")
            self.current_frame = self._get_specific_frame(0, use_thumbnail=True)
            self._frame_version += 1
        else:
            self.current_frame = None
        if self.app and hasattr(self.app, 'on_processing_stopped'):
            self.app.on_processing_stopped(was_scripting_session=False, scripted_frame_range=None)
        self.logger.debug("VideoProcessor reset complete.")

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
                preprocessed_path = self.app.file_manager.get_output_path_for_file(self.video_path, "_preprocessed.mp4")

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
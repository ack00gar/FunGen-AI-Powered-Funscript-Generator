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

# ML-based VR format detector
from video.vr_format_detector_ml_real import RealMLVRFormatDetector

# Thumbnail extractor for fast random frame access
from video.thumbnail_extractor import ThumbnailExtractor

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
        self.seek_in_progress = False  # Flag to track if seek operation is running
        self.seek_thread = None  # Thread for async seek operations
        self.arrow_nav_in_progress = False  # Flag to prevent arrow nav overload
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

        # GPU Unwarp Worker for VR optimization
        self.gpu_unwarp_worker = None
        self.gpu_unwarp_enabled = False

        # VR Unwarp method override (from UI dropdown)
        # Options: 'auto', 'metal', 'opengl', 'v360'
        self.vr_unwarp_method_override = 'auto'
        if self.app and hasattr(self.app, 'app_settings'):
            self.vr_unwarp_method_override = self.app.app_settings.get('vr_unwarp_method', 'auto')

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

        # Rolling backward buffer for arrow key navigation (deque for efficient FIFO)
        from collections import deque
        # Get buffer size from settings (default 600)
        buffer_size = 600  # Default
        if self.app and hasattr(self.app, 'app_settings'):
            buffer_size = self.app.app_settings.get('arrow_nav_buffer_size', 600)
        self.arrow_nav_backward_buffer = deque(maxlen=buffer_size)
        self.arrow_nav_backward_buffer_lock = threading.Lock()
        self.arrow_nav_refill_threshold = max(120, buffer_size // 5)  # Refill when < 20% of buffer size
        self.arrow_nav_refilling = False  # Flag to prevent concurrent refills
        
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
            self.logger.info(f"✅ Registered seek callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'} (total callbacks: {len(self._seek_callbacks)})")

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
            self.logger.debug(f"🔔 Notifying {len(self._seek_callbacks)} seek callbacks for frame {frame_index}")
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
            self.logger.info(f"✅ Registered playback state callback: {callback.__name__ if hasattr(callback, '__name__') else 'anonymous'} (total callbacks: {len(self._playback_state_callbacks)})")

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
        video_filename = os.path.basename(video_path)
        self.logger.info(f"Opening video: {video_filename}...", extra={'status_message': True, 'duration': 2.0})

        self.stop_processing()
        self.video_path = video_path # This will always be the ORIGINAL video path
        self._clear_cache()
        # Clear ML detection cache when opening new video
        if hasattr(self, '_ml_detection_cached'):
            delattr(self, '_ml_detection_cached')

        # Re-read VR unwarp method from settings in case it changed
        if self.app and hasattr(self.app, 'app_settings'):
            self.vr_unwarp_method_override = self.app.app_settings.get('vr_unwarp_method', 'auto')
            self.logger.info(f"VR unwarp method from settings: {self.vr_unwarp_method_override}")

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

        # Initialize GPU unwarp worker for VR videos (needed for seek-to-frame before playback starts)
        self._init_gpu_unwarp_worker()

        # Initialize thumbnail extractor for fast random frame access (OpenCV-based)
        self._init_thumbnail_extractor()

        self.fps = self.video_info['fps']
        self.total_frames = self.video_info['total_frames']
        self.set_target_fps(self.fps)
        self.current_frame_index = 0
        self.frames_read_from_current_stream = 0
        self.current_stream_start_frame_abs = 0
        self.stop_event.clear()
        self.seek_request_frame_index = None
        # OPTIMIZATION: Load first frame with minimal processing to avoid startup delay
        # Use a smaller batch size for initial frame to speed up video opening
        original_batch_size = self.batch_fetch_size
        self.batch_fetch_size = 1  # Fetch only 1 frame for startup
        try:
            self.current_frame = self._get_specific_frame(0)
        except Exception as e:
            self.logger.warning(f"Could not load initial frame: {e}")
            self.current_frame = None
        finally:
            self.batch_fetch_size = original_batch_size  # Restore normal batch size

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
                self.logger.info(f"📹 Notifying streamer of {source_desc} video load: {os.path.basename(video_path)}")
                asyncio.run_coroutine_threadsafe(
                    self.sync_server.broadcast_video_loaded(video_path),
                    self.sync_server.loop
                )
            except Exception as e:
                self.logger.warning(f"Could not notify sync server: {e}")
                import traceback
                self.logger.warning(traceback.format_exc())
        else:
            self.logger.info(f"📹 Streamer not available (sync_server: {hasattr(self, 'sync_server')})")

        return True

    @staticmethod
    def _detect_format_from_filename(filename: str) -> dict:
        """
        Detects video format information from filename suffixes.

        Returns:
            dict with keys:
            - 'type': 'VR', '2D', or None (if cannot determine)
            - 'projection': projection type if VR (e.g., 'fisheye', 'he', 'eac')
            - 'layout': stereoscopic layout if VR (e.g., '_sbs', '_tb', '_lr')
            - 'fov': FOV value if specific lens detected (e.g., 200 for MKX200)
        """
        upper_filename = filename.upper()
        result = {
            'type': None,
            'projection': None,
            'layout': None,
            'fov': None
        }

        # Check for 2D markers first
        if '_2D' in upper_filename or '_FLAT' in upper_filename:
            result['type'] = '2D'
            return result

        # Check for custom fisheye lenses
        if '_MKX200' in upper_filename or 'MKX200' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 200
            return result
        elif '_MKX220' in upper_filename or 'MKX220' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 220
            return result
        elif '_RF52' in upper_filename or 'RF52' in upper_filename or '_VRCA220' in upper_filename or 'VRCA220' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            return result

        # Check for standard fisheye (flexible matching - with or without underscore)
        if '_F180' in upper_filename or 'F180_' in upper_filename or '_180F' in upper_filename or '180F_' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 180
            return result
        if 'FISHEYE190' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 190
            return result
        if 'FISHEYE200' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 200
            return result
        if 'FISHEYE220' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 220
            return result
        if 'FISHEYE' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'fisheye'
            result['layout'] = '_sbs'
            result['fov'] = 190
            return result

        # Check for equiangular cubemap
        if '_EAC360' in upper_filename or '_360EAC' in upper_filename or 'EAC360' in upper_filename or '360EAC' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'eac'
            if '_LR' in upper_filename:
                result['layout'] = '_lr'
            elif '_RL' in upper_filename:
                result['layout'] = '_rl'
            elif '_TB' in upper_filename or '_BT' in upper_filename:
                result['layout'] = '_tb'
            elif '_3DH' in upper_filename:
                result['layout'] = '_sbs'
            elif '_3DV' in upper_filename:
                result['layout'] = '_tb'
            else:
                result['layout'] = '_sbs'
            return result

        # Check for equirectangular 360
        if '_360' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'he'
            if '_LR' in upper_filename:
                result['layout'] = '_lr'
            elif '_RL' in upper_filename:
                result['layout'] = '_rl'
            elif '_TB' in upper_filename or '_BT' in upper_filename:
                result['layout'] = '_tb'
            elif '_3DH' in upper_filename:
                result['layout'] = '_sbs'
            elif '_3DV' in upper_filename:
                result['layout'] = '_tb'
            else:
                result['layout'] = '_sbs'
            return result

        # Check for equirectangular 180
        if '_180' in upper_filename:
            result['type'] = 'VR'
            result['projection'] = 'he'
            if '_LR' in upper_filename:
                result['layout'] = '_lr'
            elif '_RL' in upper_filename:
                result['layout'] = '_rl'
            elif '_TB' in upper_filename or '_BT' in upper_filename:
                result['layout'] = '_tb'
            elif '_3DH' in upper_filename:
                result['layout'] = '_sbs'
            elif '_3DV' in upper_filename:
                result['layout'] = '_tb'
            else:
                result['layout'] = '_sbs'
            return result

        return result

    @staticmethod
    def _classify_by_resolution(width: int, height: int) -> str:
        """
        Classifies video as '2D', 'most_likely_VR', or 'uncertain' based on resolution.

        Returns:
            '2D': Definitely 2D based on resolution
            'most_likely_VR': Resolution suggests VR (should trigger ML)
            'uncertain': Cannot determine (should check other heuristics)
        """
        # < 1080p -> 2D
        if height < 1080 and width < 1920:
            return '2D'

        # Exactly 1920x1080p or 3840x2160p -> 2D
        if (width == 1920 and height == 1080) or (width == 3840 and height == 2160):
            return '2D'

        # Check if width = 2x height or height = 2x width (VR aspect ratios)
        is_sbs_aspect = width > 1000 and 1.8 <= (width / height) <= 2.2
        is_tb_aspect = height > 1000 and 1.8 <= (height / width) <= 2.2

        if is_sbs_aspect or is_tb_aspect:
            return 'most_likely_VR'

        # Bigger than 2160p -> most likely VR
        if height > 2160 or width > 3840:
            return 'most_likely_VR'

        return 'uncertain'

    def _update_video_parameters(self):
        """
        Consolidates logic for determining video type and building the FFmpeg filter string.
        Called from open_video and reapply_video_settings.

        Detection priority:
        1. Filename-based detection (most specific)
        2. Resolution-based classification
        3. ML detection (only if resolution suggests VR)
        """
        if not self.video_info:
            return

        width = self.video_info.get('width', 0)
        height = self.video_info.get('height', 0)

        # Skip detection if user has manually set the video type
        if self.video_type_setting != 'auto':
            self.determined_video_type = self.video_type_setting
            # Clear VR metadata if manually set to 2D
            if self.video_type_setting == '2D':
                self.vr_input_format = ""
                self.vr_fov = 0
            self.logger.info(f"Using configured video type: {self.determined_video_type}")
            self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
            return

        # STEP 1: Try filename-based detection first
        filename_result = self._detect_format_from_filename(self.video_path)

        if filename_result['type'] == '2D':
            self.logger.info(f"Filename indicates 2D video (contains _2D or _FLAT)")
            self.determined_video_type = '2D'
            # Clear VR metadata for 2D videos
            self.vr_input_format = ""
            self.vr_fov = 0
            self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
            return

        if filename_result['type'] == 'VR':
            self.logger.info(f"Filename indicates VR video: projection={filename_result['projection']}, layout={filename_result['layout']}, fov={filename_result['fov']}")
            self.determined_video_type = 'VR'

            # Apply detected format
            if filename_result['projection'] and filename_result['layout']:
                self.vr_input_format = f"{filename_result['projection']}{filename_result['layout']}"
                self.logger.info(f"Set VR format to: {self.vr_input_format}")

            # Apply detected FOV if available
            if filename_result['fov']:
                self.vr_fov = filename_result['fov']
                self.logger.info(f"Set VR FOV to: {self.vr_fov}")

            self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
            return

        # STEP 2: Filename inconclusive - check resolution
        resolution_classification = self._classify_by_resolution(width, height)

        if resolution_classification == '2D':
            self.logger.info(f"Resolution {width}x{height} classified as 2D (< 1080p or standard 2D resolution)")
            self.determined_video_type = '2D'
            # Clear VR metadata for 2D videos
            self.vr_input_format = ""
            self.vr_fov = 0
            self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
            return

        # STEP 3: Resolution suggests VR - run ML detection
        if resolution_classification == 'most_likely_VR':
            self.logger.info(f"Resolution {width}x{height} suggests VR - running ML detection")

            # Try ML detection if model available
            # Cache ML detection to avoid re-running expensive inference on every settings change
            if os.path.exists(self.ml_model_path) and not hasattr(self, '_ml_detection_cached'):
                try:
                    # Lazy load detector
                    if self.ml_detector is None:
                        self.logger.info("Loading ML format detector...")
                        self.ml_detector = RealMLVRFormatDetector(logger=self.logger)
                        self.ml_detector.load_model(self.ml_model_path)
                        self.logger.info("ML format detector loaded successfully")

                    # Detect format
                    ml_result = self.ml_detector.detect(self.video_path, self.video_info, num_frames=3)

                    if ml_result and ml_result.get('confidence', 0) > 0.5:
                        self.logger.info(f"ML detected format: {ml_result.get('format_string')} "
                                       f"(confidence: {ml_result.get('confidence'):.2f})")

                        # Apply ML results
                        self.determined_video_type = ml_result['video_type']

                        if ml_result['video_type'] == 'VR':
                            self.vr_input_format = ml_result['format_string']
                            if ml_result.get('fov'):
                                self.vr_fov = ml_result['fov']
                        else:
                            # ML detected 2D - clear VR metadata
                            self.vr_input_format = ""
                            self.vr_fov = 0

                        self._ml_detection_cached = True  # Cache result to avoid re-running on settings changes
                        self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
                        self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
                        self.logger.info(f"Frame size bytes updated to: {self.frame_size_bytes} for YOLO size {self.yolo_input_size}")
                        return
                    else:
                        self.logger.info("ML detection confidence low, falling back to resolution heuristics")

                except Exception as e:
                    self.logger.warning(f"ML detection failed: {e}, falling back to resolution heuristics")

        # STEP 4: Fallback - use resolution-based heuristics
        # Check for VR-like aspect ratios
        is_sbs_resolution = width > 1000 and 1.8 <= (width / height) <= 2.2
        is_tb_resolution = height > 1000 and 1.8 <= (height / width) <= 2.2

        if is_sbs_resolution or is_tb_resolution:
            self.logger.info(f"Resolution aspect ratio suggests VR (SBS: {is_sbs_resolution}, TB: {is_tb_resolution})")
            self.determined_video_type = 'VR'

            # Determine format based on aspect ratio
            suggested_base = 'he'
            suggested_layout = '_tb' if is_tb_resolution else '_sbs'

            self.vr_input_format = f"{suggested_base}{suggested_layout}"
            self.logger.info(f"Auto-detected VR format: {self.vr_input_format}")
        else:
            self.logger.info(f"Resolution {width}x{height} does not suggest VR - defaulting to 2D")
            self.determined_video_type = '2D'
            # Clear VR metadata for 2D videos
            self.vr_input_format = ""
            self.vr_fov = 0

        self.ffmpeg_filter_string = self._build_ffmpeg_filter_string()
        self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
        self.logger.info(f"Frame size bytes updated to: {self.frame_size_bytes} for YOLO size {self.yolo_input_size}")

    def reapply_video_settings(self):
        if not self.video_path or not self.video_info:
            self.logger.info("No video loaded. Settings will apply when a video is opened.")
            self.frame_size_bytes = self.yolo_input_size * self.yolo_input_size * 3
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
                local_p1_proc = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creation_flags)
                if local_p1_proc.stdout is None: raise IOError("get_frames_batch: Pipe 1 stdout is None.")

                # Always use BGR24 (3 bytes per pixel)
                buffer_frame_size = self.yolo_input_size * self.yolo_input_size * 3
                local_p2_proc = subprocess.Popen(cmd2, stdin=local_p1_proc.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=buffer_frame_size * min(num_frames_to_fetch, 20), creationflags=creation_flags)
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
                buffer_frame_size = self.yolo_input_size * self.yolo_input_size * 3
                local_p2_proc = subprocess.Popen(cmd_single, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=buffer_frame_size * min(num_frames_to_fetch, 20), creationflags=creation_flags)

            if not local_p2_proc or local_p2_proc.stdout is None:
                self.logger.error("get_frames_batch: Output FFmpeg process or its stdout is None.")
                return frames_batch

            # Always use BGR24 (3 bytes per pixel)
            frame_size = self.yolo_input_size * self.yolo_input_size * 3

            # Initialize progress tracking for frame buffer creation
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
                    self.yolo_input_size, self.yolo_input_size, 3)  # BGR24

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

                # Update frame buffer progress
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

    def _get_specific_frame(self, frame_index_abs: int, update_current_index: bool = True, immediate_display: bool = False) -> Optional[np.ndarray]:
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

        # For instant seek preview (batch_size=1), use fast thumbnail extractor
        # This provides immediate visual feedback (~20ms) while batch buffer loads in background
        if self.batch_fetch_size == 1 and self.thumbnail_extractor is not None:
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

        # Standard FFmpeg batch fetch for all other cases
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

    @staticmethod
    def get_video_type_heuristic(video_path: str, use_ml: bool = False) -> str:
        """
        A lightweight heuristic to guess the video type (2D/VR) and format (SBS/TB)
        without fully opening the video. Uses ffprobe for metadata.

        Args:
            video_path: Path to video file
            use_ml: If True, attempt ML detection first (requires model in /models)

        Returns:
            String like "2D", "VR (he_sbs)", "VR (fisheye_tb)", or "Unknown"
        """
        if not os.path.exists(video_path):
            return "Unknown"

        try:
            cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0',
                   '-show_entries', 'stream=width,height,pix_fmt', '-of', 'json', video_path]
            creation_flags = subprocess.CREATE_NO_WINDOW if sys.platform == 'win32' else 0
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, text=True, timeout=5, creationflags=creation_flags)
            data = json.loads(result.stdout)
            stream_info = data.get('streams', [{}])[0]
            width = int(stream_info.get('width', 0))
            height = int(stream_info.get('height', 0))
            pix_fmt = stream_info.get('pix_fmt', '')
        except Exception:
            return "Unknown"

        if width == 0 or height == 0:
            return "Unknown"

        # Try ML detection if requested
        if use_ml:
            try:
                model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vr_detector_model_rf.pkl')
                if os.path.exists(model_path):
                    detector = RealMLVRFormatDetector(logger=None)
                    detector.load_model(model_path)

                    video_info = {'width': width, 'height': height, 'pix_fmt': pix_fmt}
                    ml_result = detector.detect(video_path, video_info, num_frames=3)

                    if ml_result and ml_result.get('confidence', 0) > 0.5:
                        if ml_result['video_type'] == '2D':
                            return "2D"
                        else:
                            return f"VR ({ml_result['format_string']})"
            except Exception:
                pass  # Fall back to filename heuristics

        # Fallback to filename heuristics
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

            self.logger.debug(
                f"Detected video properties: width={stream_info.get('width', 0)}, height={stream_info.get('height', 0)}, fps={fps:.2f}, bit_depth={bit_depth}")

            return {"duration": duration, "total_frames": total_frames, "fps": fps,
                    "width": int(stream_info.get('width', 0)), "height": int(stream_info.get('height', 0)),
                    "has_audio": has_audio_ffprobe, "bit_depth": bit_depth,
                    "file_size": file_size_bytes, "bitrate": bitrate_bps,
                    "is_vfr": is_vfr, "filename": file_name,
                    "codec_name": stream_info.get('codec_name', 'N/A'),
                    "codec_long_name": stream_info.get('codec_long_name', 'N/A')
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
        is_preprocessed_by_name = self._active_video_source_path.endswith("_preprocessed.mp4")
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
        if not self.video_info:
            # Fallback if no video info
            return [
                f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease",
                f"pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
            ]

        width = self.video_info.get('width', 0)
        height = self.video_info.get('height', 0)

        if width == 0 or height == 0:
            # Fallback if dimensions unknown
            return [
                f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease",
                f"pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
            ]

        aspect_ratio = width / height

        # Check if video is square (or nearly square)
        if 0.95 < aspect_ratio < 1.05:
            # Square video - just scale, no padding needed
            return [f"scale={self.yolo_input_size}:{self.yolo_input_size}"]
        elif aspect_ratio > 1.05:
            # Wider than tall (landscape) - scale and pad top/bottom
            return [
                f"scale={self.yolo_input_size}:-1:force_original_aspect_ratio=decrease",
                f"pad={self.yolo_input_size}:{self.yolo_input_size}:0:(oh-ih)/2:black"
            ]
        else:
            # Taller than wide (portrait) - scale and pad left/right
            return [
                f"scale=-1:{self.yolo_input_size}:force_original_aspect_ratio=decrease",
                f"pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:0:black"
            ]

    def _init_gpu_unwarp_worker(self):
        """Initialize GPU unwarp worker for VR video processing."""
        from config.constants import ENABLE_GPU_UNWARP, GPU_UNWARP_BACKEND

        # Check if user wants CPU v360 instead of GPU unwarp
        if self.vr_unwarp_method_override == 'v360':
            self.logger.info("User selected CPU v360 unwarp method - GPU unwarp disabled")
            # Clean up existing GPU unwarp worker if switching from GPU to CPU v360
            if self.gpu_unwarp_worker:
                self.logger.info("Stopping existing GPU unwarp worker (switching to v360)")
                self.gpu_unwarp_worker.stop()
                self.gpu_unwarp_worker = None
            self.gpu_unwarp_enabled = False
            return

        # Only initialize for VR videos when GPU unwarp is enabled
        if self.determined_video_type != 'VR' or not ENABLE_GPU_UNWARP:
            # Clean up GPU unwarp worker if it exists but shouldn't be used
            if self.gpu_unwarp_worker:
                self.logger.info("Stopping GPU unwarp worker (not VR or GPU unwarp disabled)")
                self.gpu_unwarp_worker.stop()
                self.gpu_unwarp_worker = None
            self.gpu_unwarp_enabled = False
            return

        # If already initialized, skip (prevents duplicate workers)
        if self.gpu_unwarp_enabled and self.gpu_unwarp_worker:
            self.logger.debug("GPU unwarp worker already initialized, skipping")
            return

        try:
            from video.gpu_unwarp_worker import GPUUnwarpWorker

            # Get projection type from VR input format
            projection_type = self.vr_input_format.replace('_sbs', '').replace('_tb', '')
            if 'fisheye' in projection_type or 'he' in projection_type:
                # Map VR format to projection type
                if projection_type == 'fisheye':
                    projection_type = f'fisheye{int(self.vr_fov)}'
                elif projection_type == 'he':
                    projection_type = 'equirect180'

            # Determine backend based on user override or default
            if self.vr_unwarp_method_override in ['metal', 'opengl']:
                backend = self.vr_unwarp_method_override
            else:
                backend = GPU_UNWARP_BACKEND  # Use default (auto)

            self.gpu_unwarp_worker = GPUUnwarpWorker(
                projection_type=projection_type,
                output_size=self.yolo_input_size,
                queue_size=16,  # Increased for batch processing
                backend=backend,
                pitch=self.vr_pitch,  # Pass directly (matches benchmark behavior)
                yaw=0.0,
                roll=0.0,
                batch_size=4,  # Enable batch processing (12% faster)
                input_format='bgr24'  # Input is BGR24 from FFmpeg, worker converts to RGBA internally
            )
            self.gpu_unwarp_worker.start()
            self.gpu_unwarp_enabled = True
            self.logger.info(f"GPU unwarp worker started (backend={backend}, projection={projection_type}, pitch={self.vr_pitch})")

        except Exception as e:
            self.logger.warning(f"Failed to initialize GPU unwarp worker: {e}. Falling back to CPU v360.")
            self.gpu_unwarp_worker = None
            self.gpu_unwarp_enabled = False

    def _init_thumbnail_extractor(self):
        """Initialize OpenCV-based thumbnail extractor for fast random frame access."""
        # Close existing extractor if present
        if self.thumbnail_extractor:
            self.thumbnail_extractor.close()
            self.thumbnail_extractor = None

        # Only initialize if we have a valid video
        if not self._active_video_source_path or not self.video_info:
            return

        try:
            # Don't apply VR cropping if using preprocessed video (already cropped/unwrapped)
            vr_format = None
            if self.determined_video_type == 'VR' and not self._is_using_preprocessed_video():
                vr_format = self.vr_input_format

            self.thumbnail_extractor = ThumbnailExtractor(
                video_path=self._active_video_source_path,
                logger=self.logger,
                gpu_unwarp_worker=self.gpu_unwarp_worker,  # Optional: use GPU unwarp for VR thumbnails
                output_size=self.yolo_input_size,
                vr_input_format=vr_format
            )
            source_type = "preprocessed" if self._is_using_preprocessed_video() else "original"
            self.logger.info(f"Thumbnail extractor initialized (OpenCV-based, {source_type} video)")

        except Exception as e:
            self.logger.warning(f"Failed to initialize thumbnail extractor: {e}")
            self.thumbnail_extractor = None

    def get_thumbnail_frame(self, frame_index: int, use_gpu_unwarp: bool = False) -> Optional[np.ndarray]:
        """
        Get a thumbnail frame using OpenCV-based extractor (much faster than FFmpeg spawning).

        This method is optimized for random frame access (e.g., timeline tooltips) and uses
        OpenCV VideoCapture which maintains a persistent video connection.

        Args:
            frame_index: Frame index to extract
            use_gpu_unwarp: Whether to apply GPU unwarp for VR content (slower but accurate)

        Returns:
            Frame as BGR24 numpy array (yolo_input_size x yolo_input_size) or None
        """
        if self.thumbnail_extractor is None:
            # Fallback to FFmpeg-based extraction if thumbnail extractor not available
            self.logger.debug("Thumbnail extractor not available, falling back to FFmpeg")
            return self._get_specific_frame(frame_index, update_current_index=False)

        try:
            frame = self.thumbnail_extractor.get_frame(frame_index, use_gpu_unwarp=use_gpu_unwarp)
            return frame

        except Exception as e:
            self.logger.warning(f"Thumbnail extraction failed: {e}, falling back to FFmpeg")
            # Fallback to FFmpeg if OpenCV fails
            return self._get_specific_frame(frame_index, update_current_index=False)

    def _get_vr_video_filters(self) -> List[str]:
        """Builds the list of FFmpeg filter segments for VR video, including cropping and v360."""
        from config.constants import ENABLE_GPU_UNWARP

        if not self.video_info:
            return []

        original_width = self.video_info.get('width', 0)
        original_height = self.video_info.get('height', 0)
        v_h_FOV = 90  # Default vertical and horizontal FOV for the output projection

        vr_filters = []
        is_sbs_format = '_sbs' in self.vr_input_format
        is_tb_format = '_tb' in self.vr_input_format
        is_lr_format = '_lr' in self.vr_input_format
        is_rl_format = '_rl' in self.vr_input_format

        if is_sbs_format and original_width > 0 and original_height > 0:
            crop_w = original_width / 2
            crop_h = original_height
            vr_filters.append(f"crop={int(crop_w)}:{int(crop_h)}:0:0")
            self.logger.debug(f"Applying SBS pre-crop: w={int(crop_w)} h={int(crop_h)} x=0 y=0")
        elif is_tb_format and original_width > 0 and original_height > 0:
            crop_w = original_width
            crop_h = original_height / 2
            vr_filters.append(f"crop={int(crop_w)}:{int(crop_h)}:0:0")
            self.logger.info(f"Applying TB pre-crop: w={int(crop_w)} h={int(crop_h)} x=0 y=0")
        elif is_lr_format and original_width > 0 and original_height > 0:
            # LR format: left and right panels side-by-side, select left panel
            crop_w = original_width / 2
            crop_h = original_height
            vr_filters.append(f"crop={int(crop_w)}:{int(crop_h)}:0:0")
            self.logger.info(f"Applying LR pre-crop (left panel): w={int(crop_w)} h={int(crop_h)} x=0 y=0")
        elif is_rl_format and original_width > 0 and original_height > 0:
            # RL format: right and left panels side-by-side, select right panel
            crop_w = original_width / 2
            crop_h = original_height
            crop_x = int(original_width / 2)
            vr_filters.append(f"crop={int(crop_w)}:{int(crop_h)}:{crop_x}:0")
            self.logger.info(f"Applying RL pre-crop (right panel): w={int(crop_w)} h={int(crop_h)} x={crop_x} y=0")

        # Check unwarp method override to decide between GPU unwarp and CPU v360
        if self.vr_unwarp_method_override == 'v360':
            # User selected CPU v360 - use FFmpeg v360 filter for unwrapping
            base_v360_input_format = self.vr_input_format.replace('_sbs', '').replace('_tb', '').replace('_lr', '').replace('_rl', '')
            v360_filter_core = (
                f"v360={base_v360_input_format}:in_stereo=0:output=sg:"
                f"iv_fov={self.vr_fov}:ih_fov={self.vr_fov}:"
                f"d_fov={self.vr_fov}:"
                f"v_fov={v_h_FOV}:h_fov={v_h_FOV}:"
                f"pitch={self.vr_pitch}:yaw=0:roll=0:"
                f"w={self.yolo_input_size}:h={self.yolo_input_size}:interp=linear"
            )
            vr_filters.append(v360_filter_core)
            self.logger.info(f"Using CPU v360 filter (user override): {v360_filter_core}")
        elif ENABLE_GPU_UNWARP:
            # GPU unwarp enabled (auto/metal/opengl) - skip v360, just scale
            # Unwrapping will be done by GPU worker in tracker
            vr_filters.append(f"scale={self.yolo_input_size}:{self.yolo_input_size}")
            self.logger.info(f"GPU unwarp enabled (method={self.vr_unwarp_method_override}) - using crop+scale (v360 skipped)")
        else:
            # Fallback: GPU unwarp disabled globally
            base_v360_input_format = self.vr_input_format.replace('_sbs', '').replace('_tb', '').replace('_lr', '').replace('_rl', '')
            v360_filter_core = (
                f"v360={base_v360_input_format}:in_stereo=0:output=sg:"
                f"iv_fov={self.vr_fov}:ih_fov={self.vr_fov}:"
                f"d_fov={self.vr_fov}:"
                f"v_fov={v_h_FOV}:h_fov={v_h_FOV}:"
                f"pitch={self.vr_pitch}:yaw=0:roll=0:"
                f"w={self.yolo_input_size}:h={self.yolo_input_size}:interp=linear"
            )
            vr_filters.append(v360_filter_core)
            self.logger.info(f"Using CPU v360 filter (GPU unwarp disabled): {v360_filter_core}")

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

        self.logger.debug(
            f"Built FFmpeg filter (effective for single pipe, or pipe2 of 10bit-CUDA): {ffmpeg_filter if ffmpeg_filter else 'No explicit filter, direct output.'}")
        return ffmpeg_filter

    def _get_ffmpeg_hwaccel_args(self) -> List[str]:
        """Determines FFmpeg hardware acceleration arguments based on app settings."""
        hwaccel_args: List[str] = []
        selected_hwaccel = getattr(self.app, 'hardware_acceleration_method', 'none') if self.app else "none"
        available_on_app = getattr(self.app, 'available_ffmpeg_hwaccels', []) if self.app else []

        # Force hardware acceleration to "none" for 10-bit or preprocessed videos
        is_10bit_video = self.video_info.get('bit_depth', 8) > 8
        is_preprocessed_video = self._is_using_preprocessed_video()
        
        if is_10bit_video or is_preprocessed_video:
            if is_10bit_video and is_preprocessed_video:
                self.logger.info("Hardware acceleration forced to 'none' for 10-bit preprocessed video (compatibility)")
            elif is_10bit_video:
                self.logger.info("Hardware acceleration forced to 'none' for 10-bit video (compatibility)")
            elif is_preprocessed_video:
                self.logger.info("Hardware acceleration forced to 'none' for preprocessed video (compatibility)")
            return []  # Return empty args = no hardware acceleration

        system = platform.system().lower()
        self.logger.debug(
            f"Determining HWAccel. Selected: '{selected_hwaccel}', OS: {system}, App Available: {available_on_app}")

        if selected_hwaccel == "auto":
            # macOS: CPU-only is 6x faster than VideoToolbox for filter chains
            # Benchmark: CPU 293 FPS vs VideoToolbox 47 FPS
            if system == 'darwin':
                hwaccel_args = []  # Use CPU-only decoding
                self.logger.debug("Auto-selected CPU-only for macOS (6x faster than VideoToolbox for sequential processing with filters).")
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
                self.ffmpeg_pipe1_process = subprocess.Popen(cmd1, stdout=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=creation_flags)
                if self.ffmpeg_pipe1_process.stdout is None:
                    raise IOError("Pipe 1 stdout is None.")
                # Use larger buffer for MAX_SPEED mode to improve throughput
                if (hasattr(self.app, 'app_state_ui') and 
                    hasattr(self.app.app_state_ui, 'selected_processing_speed_mode') and
                    self.app.app_state_ui.selected_processing_speed_mode == constants.ProcessingSpeedMode.MAX_SPEED):
                    buffer_multiplier = 20  # Match CLI streaming buffer size
                else:
                    buffer_multiplier = 5   # Normal buffer size
                    
                self.ffmpeg_process = subprocess.Popen(cmd2, stdin=self.ffmpeg_pipe1_process.stdout, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_size_bytes * buffer_multiplier, creationflags=creation_flags)
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
                # Use larger buffer for MAX_SPEED mode to improve throughput
                if (hasattr(self.app, 'app_state_ui') and 
                    hasattr(self.app.app_state_ui, 'selected_processing_speed_mode') and
                    self.app.app_state_ui.selected_processing_speed_mode == constants.ProcessingSpeedMode.MAX_SPEED):
                    buffer_multiplier = 20  # Match CLI streaming buffer size
                else:
                    buffer_multiplier = 5   # Normal buffer size
                    
                self.ffmpeg_process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=self.frame_size_bytes * buffer_multiplier, creationflags=creation_flags)
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

            # Notify playback state observers (e.g., device_control) that playback resumed
            if self._playback_state_callbacks:
                current_time_ms = (self.current_frame_index / self.fps) * 1000.0 if self.fps > 0 else 0.0
                self._notify_playback_state_callbacks(True, current_time_ms)

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

        # Initialize GPU unwarp worker for VR if enabled
        self._init_gpu_unwarp_worker()

        self.is_processing = True
        self.pause_event.clear()
        self.stop_event.clear()
        self.processing_thread = threading.Thread(target=self._processing_loop, name="VideoProcessingThread")
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

        # Notify playback state observers (e.g., device_control) that playback stopped
        if self._playback_state_callbacks:
            current_time_ms = (self.current_frame_index / self.fps) * 1000.0 if self.fps > 0 else 0.0
            self._notify_playback_state_callbacks(False, current_time_ms)

        # Optional callback to update UI elements, like a play/pause button icon.
        if self.app and hasattr(self.app, 'on_processing_paused'):
            self.app.on_processing_paused()

    def stop_processing(self, join_thread=True):
        # Ensure dual output is disabled to kill any hanging pipe readers/processes
        if hasattr(self, 'dual_output_processor'):
            self.dual_output_processor.disable_dual_output_mode()

        is_currently_processing = self.is_processing
        is_thread_alive = self.processing_thread and self.processing_thread.is_alive()

        if not is_currently_processing and not is_thread_alive:
            self._terminate_ffmpeg_processes()
            return

        self.logger.info("Stopping GUI processing...")
        was_scripting_session = self.tracker and self.tracker.tracking_active
        scripted_range = (self.processing_start_frame_limit, self.current_frame_index)

        # Notify playback state observers (e.g., device_control) that playback stopped
        if self._playback_state_callbacks:
            current_time_ms = (self.current_frame_index / self.fps) * 1000.0 if self.fps > 0 else 0.0
            self._notify_playback_state_callbacks(False, current_time_ms)

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

        # Stop GPU unwarp worker if active
        if self.gpu_unwarp_worker:
            self.logger.info("Stopping GPU unwarp worker...")
            self.gpu_unwarp_worker.stop()
            self.gpu_unwarp_worker = None
            self.gpu_unwarp_enabled = False

        # Close thumbnail extractor if active
        if self.thumbnail_extractor:
            self.logger.info("Closing thumbnail extractor...")
            self.thumbnail_extractor.close()
            self.thumbnail_extractor = None

        if self.tracker:
            self.logger.info("Signaling tracker to stop.")
            self.tracker.stop_tracking()

        self.enable_tracker_processing = False

        if self.app and hasattr(self.app, 'on_processing_stopped'):
            self.app.on_processing_stopped(was_scripting_session=was_scripting_session, scripted_frame_range=scripted_range)

        self.logger.info("GUI processing stopped.")

    def seek_video(self, frame_index: int):
        """
        Seek to a specific frame. This runs asynchronously to avoid blocking the UI.
        If a seek is already in progress, it will be cancelled and replaced with the new seek.
        """
        if not self.video_info or self.video_info.get('fps', 0) <= 0 or self.total_frames <= 0:
            return

        target_frame = max(0, min(frame_index, self.total_frames - 1))

        # Notify registered observers (e.g., streamer) of seek event
        self._notify_seek_callbacks(target_frame)

        # If a seek is already in progress, wait for it to finish (or cancel it)
        if self.seek_in_progress and self.seek_thread and self.seek_thread.is_alive():
            self.logger.debug(f"Seek already in progress, new seek to frame {target_frame} will wait")
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
        """Worker thread for async seek operations. Runs blocking operations without freezing UI."""
        try:
            was_processing = self.is_processing
            was_paused = self.is_processing and self.pause_event.is_set()
            stored_end_limit = self.processing_end_frame_limit
            # Remember if tracker was active before stopping (important for streamer mode)
            was_tracking = self.tracker and self.tracker.tracking_active

            if was_processing:
                # Stop processing without joining thread to avoid blocking
                self.stop_processing(join_thread=False)
                # Give it a moment to stop gracefully
                time.sleep(0.05)

            self.logger.info(f"Seeking to frame {target_frame}")

            # Show instant thumbnail preview only (no buffer loading for scrubbing)
            # Rolling backward buffer is only for arrow key navigation
            original_batch_size = self.batch_fetch_size
            self.batch_fetch_size = 1  # Triggers fast thumbnail extractor path
            new_frame = self._get_specific_frame(target_frame, immediate_display=False)
            self.batch_fetch_size = original_batch_size

            with self.frame_lock:
                self.current_frame = new_frame

            if new_frame is None:
                self.logger.warning(f"Seek to frame {target_frame} failed to retrieve frame.")
                self.current_frame_index = target_frame

            if was_processing and not was_paused:
                self.start_processing(start_frame=self.current_frame_index, end_frame=stored_end_limit)
                # Restart tracker if it was active before seeking (e.g., in streamer mode without chapters)
                if was_tracking and self.tracker and not self.tracker.tracking_active:
                    self.logger.info("Restarting tracker after seek")
                    self.tracker.start_tracking()
            # If was_paused, do not restart processing (remain paused after seek)

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
                    processed_frame_tuple = self.tracker.process_frame(raw_frame_to_process.copy(), timestamp_ms)
                    with self.frame_lock: self.current_frame = processed_frame_tuple[0]
            except Exception as e:
                self.logger.error(f"Error processing frame with tracker in display_current_frame: {e}", exc_info=True)

    def arrow_nav_forward(self, target_frame: int) -> Optional[np.ndarray]:
        """
        Navigate forward by reading next frame sequentially.
        Adds frame to rolling backward buffer as we go.
        Returns the frame at target_frame.
        """
        # Skip if already fetching a frame (prevents CPU overload when holding arrow keys)
        if self.arrow_nav_in_progress:
            return None

        self.arrow_nav_in_progress = True
        try:
            # Use fast thumbnail path for instant response during arrow navigation
            # This avoids the 600-frame batch fetch which is very slow for VR videos
            original_batch_size = self.batch_fetch_size
            self.batch_fetch_size = 1  # Triggers fast thumbnail extractor path
            try:
                frame = self._get_specific_frame(target_frame, update_current_index=True)
            finally:
                self.batch_fetch_size = original_batch_size
            return frame
        finally:
            self.arrow_nav_in_progress = False

    def arrow_nav_backward(self, target_frame: int) -> Optional[np.ndarray]:
        """
        Navigate backward using rolling buffer.
        Returns frame from buffer if available, triggers async refill if needed.
        """
        # Check if frame is in backward buffer
        with self.arrow_nav_backward_buffer_lock:
            # Buffer stores (frame_index, frame_data) tuples
            for frame_index, frame_data in reversed(self.arrow_nav_backward_buffer):
                if frame_index == target_frame:
                    self.current_frame_index = target_frame
                    return frame_data

            # Check buffer size for refill trigger
            buffer_size = len(self.arrow_nav_backward_buffer)
            if buffer_size > 0:
                oldest_frame_index = self.arrow_nav_backward_buffer[0][0]
                frames_behind = target_frame - oldest_frame_index

                # Trigger async refill if < 120 frames in buffer behind us
                if frames_behind < self.arrow_nav_refill_threshold and not self.arrow_nav_refilling:
                    self._trigger_backward_buffer_refill(oldest_frame_index)

        # Frame not in buffer, fetch it using fast thumbnail path
        # This shouldn't happen often but when it does, we need instant response
        # Skip if already fetching (prevents CPU overload when holding arrow keys)
        if self.arrow_nav_in_progress:
            return None

        self.arrow_nav_in_progress = True
        try:
            self.logger.debug(f"Frame {target_frame} not in backward buffer, fetching via fast path")
            original_batch_size = self.batch_fetch_size
            self.batch_fetch_size = 1  # Triggers fast thumbnail extractor path
            try:
                return self._get_specific_frame(target_frame, update_current_index=True)
            finally:
                self.batch_fetch_size = original_batch_size
        finally:
            self.arrow_nav_in_progress = False

    def _trigger_backward_buffer_refill(self, oldest_frame_index: int):
        """Async refill backward buffer with 480 frames."""
        self.arrow_nav_refilling = True

        def refill_worker():
            try:
                # Fetch 480 frames backward from oldest buffered frame
                refill_end = oldest_frame_index - 1
                refill_start = max(0, refill_end - 479)  # 480 frames
                num_frames = refill_end - refill_start + 1

                if num_frames <= 0:
                    self.arrow_nav_refilling = False
                    return

                self.logger.debug(f"Refilling backward buffer: {num_frames} frames from {refill_start}")

                # Fetch batch
                fetched_batch = self.get_frames_batch(refill_start, num_frames)

                # Add to front of backward buffer (in correct order)
                with self.arrow_nav_backward_buffer_lock:
                    # Convert to list of tuples sorted by frame index
                    new_frames = sorted([(idx, frame) for idx, frame in fetched_batch.items()])

                    # Add to front of buffer (prepend in order)
                    for frame_tuple in new_frames:
                        self.arrow_nav_backward_buffer.appendleft(frame_tuple)

                    self.logger.debug(f"Backward buffer refilled: {len(new_frames)} frames added, buffer size: {len(self.arrow_nav_backward_buffer)}")

            except Exception as e:
                self.logger.error(f"Error refilling backward buffer: {e}", exc_info=True)
            finally:
                self.arrow_nav_refilling = False

        # Run in background thread
        refill_thread = threading.Thread(target=refill_worker, daemon=True)
        refill_thread.start()

    def add_frame_to_backward_buffer(self, frame_index: int, frame_data: np.ndarray):
        """Add a frame to the rolling backward buffer (used during forward navigation)."""
        with self.arrow_nav_backward_buffer_lock:
            # Deque automatically removes oldest when at maxlen
            self.arrow_nav_backward_buffer.append((frame_index, frame_data))

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
                    target_delay = 1.0 / 10.0  # Fixed 10 FPS for slow-mo
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
                    else:
                        raw_frame_bytes = None

                raw_frame_len = len(raw_frame_bytes) if raw_frame_bytes is not None else 0
                if not raw_frame_bytes or raw_frame_len < self.frame_size_bytes:
                    if self.dual_output_enabled:
                        self.logger.info("End of dual-output stream or no frames available.")
                    else:
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

                # Always use BGR24 format (3 bytes per pixel)
                expected_size = self.yolo_input_size * self.yolo_input_size * 3
                actual_bytes = len(raw_frame_bytes)

                # Validate frame size
                if actual_bytes != expected_size:
                    self.logger.error(f"Invalid frame size: {actual_bytes} bytes (expected {expected_size}). Skipping frame.")
                    continue

                frame_np = np.frombuffer(raw_frame_bytes, dtype=np.uint8).reshape(self.yolo_input_size, self.yolo_input_size, 3)

                # Apply GPU unwarp for VR frames if enabled
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

                processed_frame_for_gui = frame_np
                if self.tracker and self.tracker.tracking_active:
                    timestamp_ms = int(self.current_frame_index * (1000.0 / self.fps)) if self.fps > 0 else int(
                        time.time() * 1000)

                    try:
                        yolo_start = time.perf_counter()
                        processed_frame_for_gui = self.tracker.process_frame(frame_np.copy(), timestamp_ms)[0]
                        yolo_time = (time.perf_counter() - yolo_start) * 1000.0
                        self._yolo_samples.append(yolo_time)
                    except Exception as e:
                        self.logger.error(f"Error in tracker.process_frame during loop: {e}", exc_info=True)

                # Update timing metrics display (once per second)
                self._update_timing_metrics()

                with self.frame_lock:
                    self.current_frame = processed_frame_for_gui

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
        
        # Optimize ffmpeg for MAX_SPEED processing (segment streaming)
        common_ffmpeg_prefix = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']
        
        # Add MAX_SPEED optimizations if in MAX_SPEED mode
        if (hasattr(self.app, 'app_state_ui') and 
            hasattr(self.app.app_state_ui, 'selected_processing_speed_mode') and
            self.app.app_state_ui.selected_processing_speed_mode == constants.ProcessingSpeedMode.MAX_SPEED):
            # Same aggressive optimizations for segment streaming
            # Hardware acceleration: Handled by individual pipe paths (don't add to common prefix)
            
            # Add speed optimizations (hardware acceleration handled by pipe-specific code)
            # NOTE: -preset and -tune are encoding options, not decoding options
            common_ffmpeg_prefix.extend([
                '-fflags', '+genpts+fastseek', 
                '-threads', '0',
                '-probesize', '32',
                '-analyzeduration', '1'
            ])
            self.logger.info("FFmpeg segment streaming optimized for MAX_SPEED with fast decode")

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
            effective_vf_pipe2 = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease,pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
            cmd2.extend(['-vf', effective_vf_pipe2])
            if num_frames_to_stream_hint and num_frames_to_stream_hint > 0:
                cmd2.extend(['-frames:v', str(num_frames_to_stream_hint)])
            # Always use BGR24 - GPU unwarp worker handles BGR->RGBA conversion internally
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
            effective_vf = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease,pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
            ffmpeg_cmd.extend(['-vf', effective_vf])

            if num_frames_to_stream_hint and num_frames_to_stream_hint > 0:
                ffmpeg_cmd.extend(['-frames:v', str(num_frames_to_stream_hint)])

            # Use RGBA for GPU unwarp to skip BGR->RGBA conversion
            pix_fmt = 'rgba' if self.gpu_unwarp_enabled else 'bgr24'
            ffmpeg_cmd.extend(['-pix_fmt', pix_fmt, '-f', 'rawvideo', 'pipe:1'])
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

                # Always use BGR24 format (3 bytes per pixel)
                expected_size = self.yolo_input_size * self.yolo_input_size * 3
                actual_bytes = len(raw_frame_bytes)

                # Validate frame size
                if actual_bytes != expected_size:
                    self.logger.error(f"Invalid frame size: {actual_bytes} bytes (expected {expected_size}). Skipping frame.")
                    continue

                frame_np = np.frombuffer(raw_frame_bytes, dtype=np.uint8).reshape(self.yolo_input_size, self.yolo_input_size, 3)

                # Apply GPU unwarp for VR frames if enabled
                if self.gpu_unwarp_enabled and self.gpu_unwarp_worker:
                    current_frame_id = start_frame_abs_idx + frames_yielded
                    self.gpu_unwarp_worker.submit_frame(current_frame_id, frame_np,
                                                       timestamp_ms=current_frame_id * (1000.0 / self.fps) if self.fps > 0 else 0.0,
                                                       timeout=0.1)
                    unwarp_result = self.gpu_unwarp_worker.get_unwrapped_frame(timeout=0.5)
                    if unwarp_result is not None:
                        _, frame_np, _ = unwarp_result
                    else:
                        self.logger.warning(f"GPU unwarp timeout for segment frame {current_frame_id}")

                current_frame_id = start_frame_abs_idx + frames_yielded
                yield current_frame_id, frame_np
                frames_yielded += 1
        finally:
            self._terminate_ffmpeg_processes()

    def set_target_fps(self, fps: float):
        self.target_fps = max(1.0, fps if fps > 0 else 1.0)
    
    # ============================================================================
    # Single FFmpeg Dual-Output Integration Methods
    # ============================================================================
    
    def enable_dual_output_mode(self, fullscreen_resolution: Optional[Tuple[int, int]] = None) -> bool:
        """
        Enable single FFmpeg dual-output mode for perfect synchronization.
        
        Args:
            fullscreen_resolution: Target resolution for fullscreen frames
            
        Returns:
            True if enabled successfully
        """
        try:
            if self.dual_output_enabled:
                self.logger.warning("Dual-output mode already enabled")
                return True
            
            # Enable dual output processor
            self.dual_output_processor.enable_dual_output_mode(fullscreen_resolution)
            
            if self.dual_output_processor.dual_output_enabled:
                self.dual_output_enabled = True
                self.logger.info("🎯 VideoProcessor dual-output mode enabled")
                return True
            else:
                self.logger.error("Failed to enable dual-output processor")
                return False
                
        except Exception as e:
            self.logger.error(f"Error enabling dual-output mode: {e}")
            return False
    
    def disable_dual_output_mode(self) -> bool:
        """
        Disable dual-output mode and return to standard processing.
        
        Returns:
            True if disabled successfully
        """
        try:
            if not self.dual_output_enabled:
                self.logger.info("Dual-output mode already disabled")
                return True
            
            # Disable dual output processor
            self.dual_output_processor.disable_dual_output_mode()
            self.dual_output_enabled = False
            
            self.logger.info("✅ VideoProcessor dual-output mode disabled")
            return True
            
        except Exception as e:
            self.logger.error(f"Error disabling dual-output mode: {e}")
            return False
    
    def is_dual_output_active(self) -> bool:
        """Check if dual-output mode is active."""
        return (self.dual_output_enabled and 
                self.dual_output_processor.is_dual_output_active())
    
    def get_dual_output_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get synchronized processing and fullscreen frames from dual output.
        
        Returns:
            Tuple of (processing_frame, fullscreen_frame)
        """
        if not self.dual_output_enabled:
            return None, None
        return self.dual_output_processor.get_dual_frames()
    
    def get_fullscreen_frame(self) -> Optional[np.ndarray]:
        """Get the latest fullscreen frame for display."""
        if not self.dual_output_enabled:
            return None
        return self.dual_output_processor.get_fullscreen_frame()
    
    def get_audio_buffer(self) -> Optional[np.ndarray]:
        """Get the latest audio buffer for sound."""
        if not self.dual_output_enabled:
            return None
        return self.dual_output_processor.get_audio_buffer()
    
    def get_dual_output_stats(self) -> Dict[str, Any]:
        """Get statistics about dual-output processing."""
        if not self.dual_output_enabled:
            return {'dual_output_enabled': False}
        return self.dual_output_processor.get_frame_stats()
    
    def _start_dual_output_ffmpeg_process(self, start_frame_abs_idx=0, num_frames_to_output_ffmpeg=None) -> bool:
        """
        Start FFmpeg process using the single FFmpeg dual-output architecture.
        
        Args:
            start_frame_abs_idx: Starting frame index
            num_frames_to_output_ffmpeg: Number of frames to output (optional)
            
        Returns:
            True if started successfully
        """
        try:
            if not self.dual_output_processor.dual_output_enabled:
                self.logger.error("Dual output processor not enabled")
                return False
            
            start_time_seconds = start_frame_abs_idx / self.video_info['fps']
            self.current_stream_start_frame_abs = start_frame_abs_idx
            self.frames_read_from_current_stream = 0
            
            # Build base FFmpeg command
            base_cmd = self._build_base_ffmpeg_command(start_time_seconds, num_frames_to_output_ffmpeg)
            
            # Enhance command for dual output
            dual_output_cmd = self.dual_output_processor.build_single_ffmpeg_dual_output_command(base_cmd)
            
            # Start the single FFmpeg process with dual outputs
            success = self.dual_output_processor.start_single_ffmpeg_process(dual_output_cmd)
            
            if success:
                self.logger.info("✅ Single FFmpeg dual-output process started successfully")
                return True
            else:
                self.logger.error("❌ Failed to start single FFmpeg dual-output process")
                return False
                
        except Exception as e:
            self.logger.error(f"Error starting dual-output FFmpeg process: {e}")
            return False
    
    def _build_base_ffmpeg_command(self, start_time_seconds: float, num_frames_to_output: Optional[int] = None) -> List[str]:
        """
        Build base FFmpeg command with input arguments and filters.
        
        Args:
            start_time_seconds: Start time in seconds
            num_frames_to_output: Number of frames to output (optional)
            
        Returns:
            Base FFmpeg command list
        """
        cmd = ['ffmpeg', '-hide_banner', '-nostats', '-loglevel', 'error']
        
        # Add hardware acceleration arguments
        hwaccel_args = self._get_ffmpeg_hwaccel_args()
        cmd.extend(hwaccel_args)
        
        # Add input file with seeking
        cmd.extend(['-ss', str(start_time_seconds), '-i', self.video_path])
        
        # Add frame limiting if specified
        if num_frames_to_output and num_frames_to_output > 0:
            cmd.extend(['-frames:v', str(num_frames_to_output)])
        
        # Add audio and subtitle options
        cmd.extend(['-an', '-sn'])  # No audio, no subtitles initially (dual processor handles audio separately)
        
        # Add video filter for processing
        effective_vf = self.ffmpeg_filter_string or f"scale={self.yolo_input_size}:{self.yolo_input_size}:force_original_aspect_ratio=decrease,pad={self.yolo_input_size}:{self.yolo_input_size}:(ow-iw)/2:(oh-ih)/2:black"
        cmd.extend(['-vf', effective_vf])
        
        return cmd

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

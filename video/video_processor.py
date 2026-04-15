import time
import threading
import json
import av
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
from video.frame_source.pyav_source import PyAVFrameSource, SourceConfig

# Decomposed mixin modules
from video._vp_nav_buffer import NavBufferMixin
from video._vp_format_detection import FormatDetectionMixin
from video._vp_ffmpeg_builders import FFmpegBuildersMixin

try:
    import scipy  # noqa: F401 — presence check only (audio waveform generation)
    SCIPY_AVAILABLE_FOR_AUDIO = True
except ImportError:
    SCIPY_AVAILABLE_FOR_AUDIO = False


class VideoProcessor(
    NavBufferMixin,
    FormatDetectionMixin,
    FFmpegBuildersMixin,
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
        self.arrow_nav_in_progress = False  # Flag to prevent arrow nav overload

        self.total_frames = 0
        self.current_frame_index = 0

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

        # VR Unwarp method. Only two options now:
        #   'v360' — default, libavfilter v360 in the PyAV graph (accurate dewarp)
        #   'none' — crop-only, skip v360 (useful for debugging or preprocessed
        #            videos where dewarp was already baked in).
        # GPU unwarp paths (metal/opengl/auto) were removed with the subprocess
        # backend; PyAV applies v360 in-process with negligible cost.
        self.vr_unwarp_method_override = 'v360'
        # VR crop panel selection for "none" mode: 'first' or 'second'
        self.vr_crop_panel = 'first'
        if self.app and hasattr(self.app, 'app_settings'):
            self.vr_unwarp_method_override = self.app.app_settings.get('vr_unwarp_method', 'v360')
            # Migrate legacy values ('auto', 'metal', 'opengl') to 'v360'.
            if self.vr_unwarp_method_override not in ('v360', 'none'):
                self.vr_unwarp_method_override = 'v360'
            self.vr_crop_panel = self.app.app_settings.get('vr_crop_panel', 'first')

        # Thumbnail Extractor for fast random frame access (OpenCV-based)
        self.thumbnail_extractor = None

        # Performance timing metrics (for UI display)
        # Update once per second with mean values
        self._last_decode_time_ms = 0.0
        self._last_unwarp_time_ms = 0.0
        self._last_yolo_time_ms = 0.0
        self._last_flow_time_ms = 0.0

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

        # Frame cache (LRU OrderedDict) — hit by random-access fetches
        # (_get_specific_frame, timeline scrub previews, arrow-nav misses).
        self.frame_cache = OrderedDict()
        self.frame_cache_max_size = cache_size
        self.frame_cache_lock = threading.Lock()

        # Nav-buffer for arrow-key navigation. Populated by the processing
        # loop as frames are decoded; O(1) lookups for sequential arrow hits.
        self._frame_buffer: deque = deque(maxlen=self._compute_nav_buffer_size())
        self._frame_buffer_lock = threading.Lock()

        # ML format detector (lazy loaded)
        self.ml_detector = None
        self.ml_model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'vr_detector_model_rf.pkl')

        # Event callbacks (for optional features like streamer, device_control)
        self._seek_callbacks = []  # List of callbacks: func(frame_index: int) -> None
        self._playback_state_callbacks = []  # List of callbacks: func(is_playing: bool, current_time_ms: float) -> None

        # PyAV decoder source (in-process libav; sole decode path).
        self.pyav_source: Optional[PyAVFrameSource] = None
        # Bumped on every seek_video. The processing loop captures it
        # before pulling each frame and refuses to publish frames whose epoch
        # is stale — protects against races where a seek lands in between the
        # loop's "pull frame" and "write current_frame_index" steps.
        self._pyav_seek_epoch: int = 0
        # Set by seek_video to the user's target frame. While non-None, the
        # loop pins current_frame_index to this target (so the timeline snaps
        # instantly) but still publishes decoded frames to current_frame so
        # the video catches up visually from the GOP keyframe.
        self._pyav_pending_seek_target: Optional[int] = None

    # ----------------------------------------------------------- PyAV source

    def _open_pyav_source(self) -> bool:
        """Build and open a PyAVFrameSource using a complete in-process
        filter chain. Must be called AFTER _update_video_parameters.

        Note: the PyAV chain is NOT identical to ``ffmpeg_filter_string``.
        When GPU unwarp is enabled for VR, the subprocess path's ffmpeg chain
        omits v360 and the GPU unwarp worker handles dewarping post-decode.
        PyAV does everything in libavfilter in one process, so its chain must
        always include v360 for VR — otherwise downstream consumers (overlays,
        live trackers) would see the still-warped fisheye source.
        """
        self._close_pyav_source()
        chain = self._build_pyav_filter_chain()
        cfg = SourceConfig(
            video_path=self._active_video_source_path or self.video_path,
            filter_chain=chain,
            output_w=self._display_frame_w,
            output_h=self._display_frame_h,
        )
        src = PyAVFrameSource(cfg, logger=self.logger)
        if not src.open():
            return False
        self.pyav_source = src
        # Mirror seek/state observers so existing callbacks keep firing.
        for cb in list(self._seek_callbacks):
            src.register_seek_callback(cb)
        for cb in list(self._playback_state_callbacks):
            src.register_playback_state_callback(cb)
        # Position callback drives current_frame_index/version sync for any
        # one-shot get_frame from the GUI (the playback loop sets these directly).
        src.register_position_callback(self._on_pyav_position)
        return True

    def _build_pyav_filter_chain(self) -> str:
        """Build a self-contained libavfilter chain for PyAV. Mirrors
        FFmpegBuildersMixin's logic but unconditionally includes v360 for VR
        (no separate GPU unwarp pass exists on this path).

        Output size is ``_display_frame_w`` / ``_display_frame_h`` — same as
        the subprocess path's HD-aware output. The tracker's 640x640 frame
        is derived from this via ``_make_processing_frame`` if HD is active.
        """
        out_w = self._display_frame_w
        out_h = self._display_frame_h
        if not self.video_info:
            return f"scale={out_w}:{out_h}"

        if self.determined_video_type == 'VR':
            # Pre-crop to one stereo panel.
            ow = self.video_info.get('width', 0)
            oh = self.video_info.get('height', 0)
            parts = []
            if '_sbs' in self.vr_input_format and ow > 0 and oh > 0:
                parts.append(f"crop={ow // 2}:{oh}:0:0")
            elif '_tb' in self.vr_input_format and ow > 0 and oh > 0:
                parts.append(f"crop={ow}:{oh // 2}:0:0")
            elif '_lr' in self.vr_input_format and ow > 0 and oh > 0:
                parts.append(f"crop={ow // 2}:{oh}:0:0")
            elif '_rl' in self.vr_input_format and ow > 0 and oh > 0:
                parts.append(f"crop={ow // 2}:{oh}:{ow // 2}:0")
            base_fmt = (self.vr_input_format
                        .replace('_sbs', '').replace('_tb', '')
                        .replace('_lr', '').replace('_rl', ''))
            v_h_fov = 90
            parts.append(
                f"v360={base_fmt}:in_stereo=0:output=sg:"
                f"iv_fov={self.vr_fov}:ih_fov={self.vr_fov}:"
                f"d_fov={self.vr_fov}:"
                f"v_fov={v_h_fov}:h_fov={v_h_fov}:"
                f"pitch={self.vr_pitch}:yaw=0:roll=0:"
                f"w={out_w}:h={out_h}:interp=linear"
            )
            return ",".join(parts)

        # 2D path: scale-and-pad to the display size.
        return (f"scale={out_w}:{out_h}"
                f":force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}"
                f":(ow-iw)/2:(oh-ih)/2:black")

    def _close_pyav_source(self) -> None:
        if self.pyav_source is not None:
            try:
                self.pyav_source.close()
            except Exception as e:
                self.logger.debug(f"pyav source close error: {e}")
            self.pyav_source = None

    def _on_pyav_position(self, frame_index: int) -> None:
        """Position callback from the source — kept light-weight."""
        # current_frame_index is updated by the playback loop to keep the
        # write path consistent. This hook is a placeholder for future use
        # (e.g., feeding device sync from random seeks).
        pass

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

        HD for 2D: scales to max 1920px on the longest edge, preserving
        aspect ratio with even dimensions. HD for VR: the v360 filter
        renders a square projection at 1920x1920. In both cases the tracker
        frame is derived via _make_processing_frame, which downsamples to
        yolo_input_size. ~3% cost measured vs 640x640 VR output.
        """
        size = self.yolo_input_size
        self._display_frame_w = size
        self._display_frame_h = size
        self._is_hd_active = False
        self._processing_frame_buf = np.zeros((size, size, 3), dtype=np.uint8)
        self._proc_resize_dims = (size, size)
        self._proc_pad_offset = (0, 0)

        if self.app and hasattr(self.app, 'app_settings'):
            self.hd_display_enabled = self.app.app_settings.get('hd_video_display', True)

        if not self.hd_display_enabled:
            return
        if not self.video_info:
            return
        if hasattr(self, '_is_using_preprocessed_video') and self._is_using_preprocessed_video():
            return

        max_dim = 1920

        if self.determined_video_type == 'VR':
            # v360 output aspect is user-configurable; wider aspect widens the
            # stereographic projection's horizontal field of view so VR content
            # uses more of the available horizontal display space.
            try:
                aspect = float(self.app.app_settings.get('vr_display_aspect', 1.0))
            except Exception:
                aspect = 1.78
            aspect = max(1.0, min(2.4, aspect))
            out_h = max_dim
            out_w = int(max_dim * aspect) & ~1
        else:
            src_w = self.video_info.get('width', 0)
            src_h = self.video_info.get('height', 0)
            if src_w <= 0 or src_h <= 0:
                return
            if max(src_w, src_h) <= max_dim:
                out_w, out_h = src_w, src_h
            elif src_w >= src_h:
                out_w = max_dim
                out_h = int(src_h * max_dim / src_w)
            else:
                out_h = max_dim
                out_w = int(src_w * max_dim / src_h)
            out_w = out_w & ~1
            out_h = out_h & ~1
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
    
    def set_target_fps(self, fps: float):
        """Target frame rate for the display pacing loop."""
        self.target_fps = max(1.0, fps if fps > 0 else 1.0)

    def stream_frames_for_segment(self, start_frame_abs_idx: int,
                                  num_frames_to_read: int,
                                  stop_event=None):
        """Yield ``(frame_id, frame_ndarray, timing_dict)`` for a contiguous
        segment of the video. Used by the Stage 1/Stage 3 offline pipelines
        and offline trackers.

        Output frames are guaranteed to be at yolo_input_size (640x640) — the
        tracker / detection pipelines were designed around that coordinate
        space. If HD display is currently on (for GUI playback), this method
        temporarily flips it off, rebuilds the filter graph, and restores
        state on exit. Safe to call from the main thread; the tracker loops
        drive one-shot decoding via the PyAV source's internal primitives.

        Backed by the PyAV source, no subprocess ffmpeg.
        """
        import time as _time
        if self.pyav_source is None:
            self.logger.error("stream_frames_for_segment: no PyAV source")
            return
        src = self.pyav_source

        # Make sure the decode thread is not running; we drive the source
        # directly via its internal primitives for one-shot-per-call control.
        if src.is_running:
            src.stop()

        # Force 640x640 output for the duration of the stream so trackers
        # (chapter_maker, vr_hybrid_chapter, Stage 1 CD, Stage 3 flow) see
        # the coordinate space they were written against. Restored in finally.
        _saved_hd = getattr(self, 'hd_display_enabled', False)
        _hd_was_on = _saved_hd and getattr(self, '_is_hd_active', False)
        if _hd_was_on:
            self.hd_display_enabled = False
            # _update_video_parameters recomputes display dims + filter string
            # but does NOT rebuild the PyAV filter graph. Do that explicitly.
            self._update_video_parameters()
            try:
                new_chain = self._build_pyav_filter_chain()
                new_cfg = SourceConfig(
                    video_path=src.cfg.video_path,
                    filter_chain=new_chain,
                    output_w=self._display_frame_w,
                    output_h=self._display_frame_h,
                    decoder_threads=src.cfg.decoder_threads,
                )
                src.reapply_settings(new_cfg)
            except Exception as e:
                self.logger.error(f"stream_frames_for_segment: graph rebuild failed: {e}")
                self.hd_display_enabled = _saved_hd
                return

        # Seek once at the start. Accurate seek to the requested frame.
        src._seek_to_target(start_frame_abs_idx)

        frames_yielded = 0
        _hd_restore_needed = _hd_was_on
        # Drive decode via the source's background thread so YOLO / DIS /
        # cvtColor on the consumer thread overlap with decode (avoids the
        # single-thread serialization that dropped us to ~100 FPS).
        src.start(start_frame=start_frame_abs_idx)
        try:
            last_pull_ts = _time.perf_counter()
            while frames_yielded < num_frames_to_read:
                if stop_event is not None and stop_event.is_set():
                    break
                item = src.next_frame(timeout=2.0)
                if item is None:
                    if src.is_eos:
                        break
                    continue  # transient queue miss, retry
                idx, arr = item
                if idx < start_frame_abs_idx:
                    continue  # pre-seek tail from keyframe-aligned seek
                now = _time.perf_counter()
                decode_ms = (now - last_pull_ts) * 1000.0
                last_pull_ts = now
                current_id = start_frame_abs_idx + frames_yielded
                yield current_id, arr, {'decode_ms': decode_ms, 'unwarp_ms': 0.0}
                frames_yielded += 1
        except Exception as e:
            self.logger.warning(f"stream_frames_for_segment error: {e}")
        finally:
            try:
                src.stop()
            except Exception:
                pass
        # HD restore is a secondary finally block; the first one above
        # already stopped the decode thread. Keep this as a top-level
        # concern (no nested try to avoid indentation confusion).
        if _hd_restore_needed:
            try:
                self.hd_display_enabled = _saved_hd
                self._update_video_parameters()
                src2 = self.pyav_source
                if src2 is not None:
                    new_chain = self._build_pyav_filter_chain()
                    new_cfg = SourceConfig(
                        video_path=src2.cfg.video_path,
                        filter_chain=new_chain,
                        output_w=self._display_frame_w,
                        output_h=self._display_frame_h,
                        decoder_threads=src2.cfg.decoder_threads,
                    )
                    src2.reapply_settings(new_cfg)
            except Exception:
                pass

    def stream_frames_prefetched(self, start_frame_abs_idx: int,
                                 num_frames_to_read: int,
                                 stop_event=None,
                                 prefetch: int = 4):
        """Same yield contract as stream_frames_for_segment but decodes in a
        background thread with an N-frame prefetch queue. Lets the caller
        run compute (YOLO, optical flow, etc.) in parallel with decode."""
        import queue as _q
        import threading
        _SENTINEL = object()
        q: _q.Queue = _q.Queue(maxsize=max(1, prefetch))

        def _producer():
            try:
                for item in self.stream_frames_for_segment(
                        start_frame_abs_idx, num_frames_to_read, stop_event=stop_event):
                    if stop_event is not None and stop_event.is_set():
                        break
                    q.put(item)
            except Exception as e:
                self.logger.warning(f"prefetch producer error: {e}")
            finally:
                q.put(_SENTINEL)

        t = threading.Thread(target=_producer, name="FramePrefetch", daemon=True)
        t.start()
        try:
            while True:
                item = q.get()
                if item is _SENTINEL:
                    return
                if stop_event is not None and stop_event.is_set():
                    return
                yield item
        finally:
            # Drain the queue so the producer doesn't block on put().
            try:
                while not q.empty():
                    q.get_nowait()
            except Exception:
                pass

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

        self.fps = self.video_info['fps']
        self.total_frames = self.video_info['total_frames']

        # Initialize thumbnail extractor for fast random frame access (FFmpeg-based)
        self._init_thumbnail_extractor()
        # PyAV decoder backend (opt-in) — built after _update_video_parameters
        if not self._open_pyav_source():
            self.logger.error("Failed to open PyAV source; playback will not work")
        else:
            # Pre-start the decoder in paused state so arrow-nav / scrub hits
            # the fast +1 pump path without paying the decode-thread spin-up
            # on every keypress. Consume the seek-response frame so the
            # source's _current_frame_index is published; that primes the
            # +1 fast path for the first arrow press.
            self.pyav_source.start(0)
            self.pyav_source.wait_seek(timeout=2.0)
            self.pyav_source.next_frame(timeout=2.0)
            self.pyav_source.pause()
        self.set_target_fps(self.fps)
        self.current_frame_index = 0
        self.stop_event.clear()
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

        # Reinitialize thumbnail extractor with new display dimensions
        self._init_thumbnail_extractor()
        if self._open_pyav_source():
            # Pre-start paused (see open_video path for rationale).
            self.pyav_source.start(max(0, stored_frame_index))
            self.pyav_source.wait_seek(timeout=2.0)
            self.pyav_source.next_frame(timeout=2.0)
            self.pyav_source.pause()

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

    def _get_specific_frame(self, frame_index_abs: int, update_current_index: bool = True, immediate_display: bool = False, use_thumbnail: bool = False) -> Optional[np.ndarray]:
        """Random-access frame fetch. Cache hit → instant; cache miss →
        PyAV source.get_frame (~300ms on 8K VR)."""
        if not self.video_path or not self.video_info or self.video_info.get('fps', 0) <= 0:
            self.logger.warning("Cannot get frame: video not loaded/invalid FPS.")
            if update_current_index:
                self.current_frame_index = frame_index_abs
            return None

        with self.frame_cache_lock:
            if frame_index_abs in self.frame_cache:
                frame = self.frame_cache[frame_index_abs]
                self.frame_cache.move_to_end(frame_index_abs)
                if update_current_index:
                    self.current_frame_index = frame_index_abs
                return frame

        if self.pyav_source is None:
            self.logger.warning(f"_get_specific_frame({frame_index_abs}): no PyAV source")
            return None
        frame = self.pyav_source.get_frame(frame_index_abs, timeout=2.0)
        if frame is None:
            self.logger.warning(f"PyAV get_frame failed for {frame_index_abs}")
            return None
        with self.frame_cache_lock:
            if len(self.frame_cache) >= self.frame_cache_max_size:
                try: self.frame_cache.popitem(last=False)
                except KeyError: pass
            self.frame_cache[frame_index_abs] = frame
            self.frame_cache.move_to_end(frame_index_abs)
        if update_current_index:
            self.current_frame_index = frame_index_abs
        return frame


    def _get_video_info(self, filename):
        """Probe video metadata via PyAV (libav in-process)."""
        try:
            container = av.open(filename)
        except Exception as e:
            self.logger.error(f"PyAV could not open {filename}: {e}")
            return None
        try:
            try:
                vstream = container.streams.video[0]
            except (IndexError, AttributeError):
                self.logger.error(f"No video stream in {filename}")
                return None

            fps = float(vstream.average_rate or vstream.guessed_rate or 30.0)
            avg_rate = float(vstream.average_rate or 0.0)
            r_rate = float(vstream.guessed_rate or avg_rate or 0.0)
            is_vfr = (avg_rate > 0 and r_rate > 0 and abs(avg_rate - r_rate) / max(avg_rate, 1e-9) > 0.01)

            duration = 0.0
            if container.duration:
                duration = float(container.duration / 1_000_000)
            elif vstream.duration and vstream.time_base:
                duration = float(vstream.duration * vstream.time_base)

            total_frames = int(vstream.frames or 0)
            if total_frames == 0 and duration > 0 and fps > 0:
                total_frames = int(duration * fps)

            try:
                file_size_bytes = os.path.getsize(filename)
            except OSError:
                file_size_bytes = 0
            bitrate_bps = int(getattr(container, 'bit_rate', 0) or 0)

            codec_ctx = vstream.codec_context
            codec_name = codec_ctx.name or 'unknown'
            codec_long_name = codec_ctx.codec.long_name if codec_ctx.codec else codec_name

            bit_depth = 8
            pix_fmt = (codec_ctx.pix_fmt or '').lower()
            if any(fmt in pix_fmt for fmt in ('12le', 'p012', '12be')):
                bit_depth = 12
            elif any(fmt in pix_fmt for fmt in ('10le', 'p010', '10be')):
                bit_depth = 10

            # Audio stream info (optional)
            has_audio = False
            audio_codec_name = ''
            audio_codec_long_name = ''
            audio_bitrate = 0
            audio_sample_rate = 0
            audio_channels = 0
            try:
                astream = container.streams.audio[0]
                has_audio = True
                actx = astream.codec_context
                audio_codec_name = actx.name or ''
                audio_codec_long_name = actx.codec.long_name if actx.codec else ''
                audio_bitrate = int(actx.bit_rate or 0)
                audio_sample_rate = int(actx.sample_rate or 0)
                audio_channels = int(actx.channels or 0)
            except (IndexError, AttributeError):
                pass

            self.logger.debug(
                f"Detected video properties: width={vstream.width}, height={vstream.height}, "
                f"fps={fps:.2f}, bit_depth={bit_depth}")

            return {
                "duration": duration, "total_frames": total_frames, "fps": fps,
                "width": int(vstream.width), "height": int(vstream.height),
                "has_audio": has_audio, "bit_depth": bit_depth,
                "file_size": file_size_bytes, "bitrate": bitrate_bps,
                "is_vfr": is_vfr, "filename": os.path.basename(filename),
                "codec_name": codec_name, "codec_long_name": codec_long_name,
                "audio_codec_name": audio_codec_name,
                "audio_codec_long_name": audio_codec_long_name,
                "audio_bitrate": audio_bitrate,
                "audio_sample_rate": audio_sample_rate,
                "audio_channels": audio_channels,
            }
        except Exception as e:
            self.logger.error(f"Error in _get_video_info for {filename}: {e}")
            return None
        finally:
            try: container.close()
            except Exception: pass

    def get_audio_waveform(self, num_samples: int = 1000) -> Optional[np.ndarray]:
        """Generate a waveform by decoding the audio stream via PyAV and
        downsampling to ``num_samples`` peak-amplitude buckets."""
        if not self.video_path or not self.video_info.get("has_audio"):
            self.logger.info("No video loaded or video has no audio stream for waveform generation.")
            return None
        if not SCIPY_AVAILABLE_FOR_AUDIO:
            self.logger.warning("Scipy is not available. Cannot generate audio waveform.")
            return None

        container = None
        try:
            container = av.open(self.video_path)
            try:
                astream = container.streams.audio[0]
            except (IndexError, AttributeError):
                self.logger.error("No audio stream to read waveform from.")
                return None
            astream.thread_type = "AUTO"

            resampler = av.audio.resampler.AudioResampler(
                format='s16', layout='mono', rate=44100,
            )
            pcm = bytearray()
            for packet in container.demux(astream):
                for frame in packet.decode():
                    for out_frame in resampler.resample(frame):
                        pcm.extend(out_frame.to_ndarray().tobytes())

            if not pcm:
                self.logger.error("PyAV produced no audio samples.")
                return None

            data = np.frombuffer(bytes(pcm), dtype=np.int16)
            if data.size == 0:
                return None
            step = max(1, len(data) // num_samples)
            waveform = [np.max(np.abs(data[i:i + step])) for i in range(0, len(data), step)]
            waveform_np = np.array(waveform, dtype=np.float32)
            max_val = np.max(waveform_np)
            if max_val > 0:
                waveform_np = waveform_np / max_val
            self.logger.info(f"Generated waveform with {len(waveform_np)} samples.")
            return waveform_np
        except Exception as e:
            self.logger.error(f"Error generating audio waveform: {e}", exc_info=True)
            return None
        finally:
            if container is not None:
                try: container.close()
                except Exception: pass


    def start_processing(self, start_frame=None, end_frame=None, cli_progress_callback=None):
        # If already processing but paused, just clear the pause event.
        # The GUI thread NEVER touches FFmpeg processes.
        # The processing thread self-heals: after waking from pause it checks
        # cursor vs pipe position and skips/restarts as needed.
        if self.is_processing and self.pause_event.is_set():
            self.logger.info(f"Resuming playback from frame {self.current_frame_index}")
            self.pause_event.clear()
            # Resume pyav source so the decoder starts pumping frames for the
            # processing loop again (was paused when playback paused to keep
            # arrow-nav fast-path correct — no speculative pump-ahead).
            if self.pyav_source is not None:
                self.pyav_source.resume()

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

        self.processing_start_frame_limit = effective_start_frame
        self.processing_end_frame_limit = -1
        if end_frame is not None and end_frame >= 0:
            self.processing_end_frame_limit = min(end_frame, self.total_frames - 1)

        self.is_processing = True
        self.pause_event.clear()
        self.stop_event.clear()
        self.processing_thread = threading.Thread(
            target=self._processing_loop, name="VideoProcessingThread")
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.logger.debug(
            f"Started PyAV processing. Range: {self.processing_start_frame_limit} to "
            f"{self.processing_end_frame_limit if self.processing_end_frame_limit != -1 else 'EOS'}")

    def pause_processing(self):
        if not self.is_processing or self.pause_event.is_set():
            return

        self.logger.info(f"Pausing playback at frame {self.current_frame_index}")
        self.pause_event.set()
        # Pause pyav source so the decoder stops speculatively pumping frames
        # into the queue ahead of current_frame_index. Without this, arrow
        # nav's +1 fast path reads the wrong (speculated) frame.
        if self.pyav_source is not None:
            self.pyav_source.pause()

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
            return

        self.logger.debug("Stopping GUI processing...")
        was_scripting_session = self.tracker and self.tracker.tracking_active
        scripted_range = (self.processing_start_frame_limit, self.current_frame_index)

        if self._playback_state_callbacks:
            current_time_ms = (self.current_frame_index / self.fps) * 1000.0 if self.fps > 0 else 0.0
            self._notify_playback_state_callbacks(False, current_time_ms)

        self.is_processing = False
        self.pause_event.clear()
        self.stop_event.set()

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
                self.logger.debug("Signaling tracker to stop.")
                self.tracker.stop_tracking()

            self.enable_tracker_processing = False

            if self.app and hasattr(self.app, 'on_processing_stopped'):
                self.app.on_processing_stopped(was_scripting_session=was_scripting_session, scripted_frame_range=scripted_range)
        else:
            self.enable_tracker_processing = False

        self.logger.debug("GUI processing stopped.")

    def seek_video(self, frame_index: int):
        """Seek to a specific frame via the PyAV source.

        When playing: fast (keyframe) seek — timeline snaps to target,
        display catches up visually in the processing loop.
        When paused: accurate seek — exact target frame is decoded.
        """
        if not self.video_info or self.video_info.get('fps', 0) <= 0 or self.total_frames <= 0:
            return
        if self.pyav_source is None:
            self.logger.error("seek_video: no PyAV source")
            return

        target_frame = max(0, min(frame_index, self.total_frames - 1))
        self.playhead_override_ms = None
        self._pyav_seek_epoch += 1
        self.current_frame_index = target_frame
        self._pyav_pending_seek_target = target_frame
        self._clear_nav_state()
        self._notify_seek_callbacks(target_frame)

        if self.is_processing and not self.pause_event.is_set():
            self.pyav_source.seek(target_frame, accurate=False)
        else:
            frame = self.pyav_source.get_frame(target_frame, timeout=8.0)
            if frame is not None:
                with self.frame_lock:
                    self.current_frame = frame
                    self._frame_version += 1
            self._pyav_pending_seek_target = None

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
        """Unified playback loop (PyAV). Subprocess backend removed."""
        return self._pyav_processing_loop()

    def _pyav_processing_loop(self):
        """Playback loop driven by PyAVFrameSource. No subprocess pipes; seek
        is in-process and ~5-12x faster than the ffmpeg-respawn path on heavy
        formats. Keeps every observable invariant the subprocess loop holds:
        current_frame / current_frame_index / _frame_version are written here,
        nav buffer is populated, tracker.process_frame is called with the same
        signature, playback-state callbacks fire on each frame.
        """
        src = self.pyav_source
        if src is None:
            self.logger.error("_pyav_processing_loop: no source")
            self.is_processing = False
            return

        # Start the source at the requested frame.
        start_frame = max(0, int(self.processing_start_frame_limit))

        # Always use threaded decoder. EOS is detected exactly via
        # src.is_eos (set when decoder thread emits _EOS), so trackers see
        # every frame regardless of B-frame reorder tail.
        if src.is_running:
            # Source was pre-started (paused) at video open. Seek to the
            # requested frame and resume instead of starting a new thread.
            src.seek(start_frame, accurate=False)
            src.resume()
        else:
            src.start(start_frame=start_frame)

        start_time = time.time()
        next_frame_target_time = time.perf_counter()
        self.last_processed_chapter_id = None

        try:
            while not self.stop_event.is_set():
                # ---- pause handling: source-native, no pipe handoff ----
                if self.pause_event.is_set():
                    if not src.is_paused:
                        src.pause()
                        if self.app and hasattr(self.app, 'on_processing_paused'):
                            self.app.on_processing_paused()
                    while self.pause_event.is_set() and not self.stop_event.is_set():
                        self.stop_event.wait(0.01)
                    if self.stop_event.is_set():
                        break
                    # Sync source to cursor: while paused the user may have
                    # arrow-nav'd or scrubbed, which updates current_frame_index
                    # directly without the PyAV source knowing. If we resumed
                    # without seeking, the source would deliver stale frames
                    # from the old pause point and drag the timeline back.
                    if src.current_frame_index != self.current_frame_index:
                        self._pyav_seek_epoch += 1
                        self._pyav_pending_seek_target = self.current_frame_index
                        src.seek(self.current_frame_index, accurate=False)
                    if src.is_paused:
                        src.resume()
                        if self.app and hasattr(self.app, 'on_processing_resumed'):
                            self.app.on_processing_resumed()
                    next_frame_target_time = time.perf_counter()

                # ---- pacing ----
                speed_mode = self.app.app_state_ui.selected_processing_speed_mode
                if speed_mode == constants.ProcessingSpeedMode.REALTIME:
                    target_delay = 1.0 / self.fps if self.fps > 0 else (1.0 / 30.0)
                elif speed_mode == constants.ProcessingSpeedMode.SLOW_MOTION:
                    slo_mo_fps = getattr(self.app.app_state_ui, 'slow_motion_fps', 10.0)
                    target_delay = 1.0 / max(1.0, slo_mo_fps)
                else:
                    target_delay = 0.0

                # ---- chapter-aware tracker start/stop (mirror subprocess loop) ----
                current_chapter = self.app.funscript_processor.get_chapter_at_frame(self.current_frame_index)
                current_chapter_id = current_chapter.unique_id if current_chapter else None
                if current_chapter_id != self.last_processed_chapter_id:
                    if self.tracker and self.enable_tracker_processing:
                        from config.constants import POSITION_INFO_MAPPING
                        should_track = True
                        if current_chapter:
                            position_info = POSITION_INFO_MAPPING.get(current_chapter.position_short_name, {})
                            category = position_info.get('category', 'Position')
                            should_track = (category == "Position")
                            if should_track and current_chapter.user_roi_fixed:
                                self.tracker.reconfigure_for_chapter(current_chapter)
                        if should_track and not self.tracker.tracking_active:
                            self.tracker.start_tracking()
                        elif not should_track and self.tracker.tracking_active:
                            self.tracker.stop_tracking()
                    self.last_processed_chapter_id = current_chapter_id
                if current_chapter and self.tracker and self.enable_tracker_processing \
                        and not self.tracker.tracking_active and current_chapter.user_roi_fixed:
                    self.tracker.start_tracking()

                # ---- pull next frame ----
                # Capture epoch BEFORE pulling so we can detect a seek that
                # happens during the (potentially blocking) next_frame call
                # and discard the resulting stale frame.
                epoch_before = self._pyav_seek_epoch
                decode_start = time.perf_counter()
                item = src.next_frame(timeout=2.0)
                decode_time = (time.perf_counter() - decode_start) * 1000.0
                self._decode_samples.append(decode_time)
                if decode_time > 200:
                    self.logger.warning(f"Slow PyAV frame at {self.current_frame_index}: {decode_time:.0f}ms")

                if item is None:
                    if self.stop_event.is_set():
                        break
                    # Differentiate true EOS (decoder produced _EOS sentinel)
                    # from a transient pull timeout (decoder still working).
                    if src.is_eos:
                        self.logger.info(
                            f"PyAV loop: end of stream (idx={src.current_frame_index})")
                        self.is_processing = False
                        self.enable_tracker_processing = False
                        if self.app:
                            was_scripting = self.tracker and self.tracker.tracking_active
                            end_range = (self.processing_start_frame_limit, self.current_frame_index)
                            if self.tracker and self.tracker.tracking_active:
                                self.tracker.stop_tracking()
                            self.app.on_processing_stopped(was_scripting_session=was_scripting, scripted_frame_range=end_range)
                        break
                    continue  # transient miss; decoder is still busy

                idx, frame_np = item
                # Drop frames decoded with a stale epoch — a seek_video happened
                # between when we asked for this frame and when we got it.
                if epoch_before != self._pyav_seek_epoch:
                    continue

                # Post-seek catch-up: the user clicked frame T, source landed on
                # the GOP keyframe at T'<T and is decoding forward. Show those
                # frames so the video plays through the catch-up (good UX), but
                # keep current_frame_index pinned to the user's intent (T) until
                # the source actually reaches it. After that, resume normal
                # index updates from the decoded frame index.
                seek_target = self._pyav_pending_seek_target
                if seek_target is not None and idx < seek_target:
                    if self.is_hd_active:
                        processing_frame = self._make_processing_frame(frame_np)
                    else:
                        processing_frame = frame_np
                    self._buffer_append(idx, frame_np.copy())
                    with self.frame_lock:
                        self.current_frame = frame_np
                        self._frame_version += 1
                    continue
                self._pyav_pending_seek_target = None
                self.current_frame_index = idx

                # ---- HD downsample for tracker if needed ----
                if self.is_hd_active:
                    processing_frame = self._make_processing_frame(frame_np)
                else:
                    processing_frame = frame_np

                # Feed nav buffer so backward arrow keys keep working
                self._buffer_append(self.current_frame_index, frame_np.copy())

                # ---- tracker ----
                if self.tracker and self.tracker.tracking_active and self.enable_tracker_processing:
                    timestamp_ms = int(self.current_frame_index * (1000.0 / self.fps)) if self.fps > 0 else 0
                    try:
                        yolo_start = time.perf_counter()
                        self.tracker.process_frame(processing_frame.copy(), timestamp_ms, self.current_frame_index)
                        self._yolo_samples.append((time.perf_counter() - yolo_start) * 1000.0)
                    except Exception as e:
                        self.logger.error(f"PyAV loop tracker error: {e}", exc_info=True)

                self._update_timing_metrics()

                # ---- expose frame to display ----
                with self.frame_lock:
                    self.current_frame = frame_np
                    self._frame_version += 1

                # ---- playback state observers ----
                if self._playback_state_callbacks:
                    is_playing = self.is_processing and not self.pause_event.is_set()
                    ts_ms = (self.current_frame_index / self.fps) * 1000.0 if self.fps > 0 else 0.0
                    self._notify_playback_state_callbacks(is_playing, ts_ms)

                if self.cli_progress_callback and self.current_frame_index % 10 == 0:
                    self.cli_progress_callback(self.current_frame_index, self.total_frames, start_time)

                if self.processing_end_frame_limit != -1 and self.current_frame_index > self.processing_end_frame_limit:
                    self.logger.info(f"Reached PyAV end_frame_limit ({self.processing_end_frame_limit})")
                    self.is_processing = False
                    self.enable_tracker_processing = False
                    if self.app:
                        was_scripting = self.tracker and self.tracker.tracking_active
                        end_range = (self.processing_start_frame_limit, self.processing_end_frame_limit)
                        if self.tracker and self.tracker.tracking_active:
                            self.tracker.stop_tracking()
                        self.app.on_processing_stopped(was_scripting_session=was_scripting, scripted_frame_range=end_range)
                    break

                # ---- pacing sleep ----
                if target_delay > 0:
                    next_frame_target_time += target_delay
                    sleep_for = next_frame_target_time - time.perf_counter()
                    if sleep_for > 0:
                        self.stop_event.wait(min(sleep_for, 0.1))
                    else:
                        next_frame_target_time = time.perf_counter()

        except Exception as e:
            self.logger.error(f"PyAV processing loop crashed: {e}", exc_info=True)
        finally:
            try:
                src.stop()
            except Exception:
                pass
            self.is_processing = False
            self.enable_tracker_processing = False

    def is_video_open(self) -> bool:
        """Checks if a video is currently loaded and has valid information."""
        return bool(self.video_path and self.video_info and self.video_info.get('total_frames', 0) > 0)

    def _close_video_resources(self) -> None:
        """Tear down per-video resources: PyAV source, thumbnail extractor."""
        self._close_pyav_source()

    def reset(self, close_video=False, skip_tracker_reset=False):
        self.logger.debug("Resetting VideoProcessor...")
        self.stop_processing(join_thread=True)
        self._clear_cache()
        self.current_frame_index = 0
        if self.tracker and not skip_tracker_reset:
            self.tracker.reset()
        if close_video:
            if self.thumbnail_extractor:
                self.thumbnail_extractor.close()
                self.thumbnail_extractor = None
            self._close_pyav_source()
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
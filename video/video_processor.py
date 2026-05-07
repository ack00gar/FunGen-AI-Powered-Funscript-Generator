import time
import threading
import json
import numpy as np
import cv2
import sys
from typing import Optional, Iterator, Tuple, List, Dict, Any
import logging
import os
from collections import OrderedDict, deque

from config import constants
from common import paths

# ML-based VR format detector
from video.vr_format_detector_ml_real import RealMLVRFormatDetector

# Thumbnail extractor for fast random frame access
from video.thumbnail_extractor import ThumbnailExtractor
from video.frame_source._types import SourceConfig
from video.frame_source.ffmpeg_source import FFmpegFrameSource

# Decomposed mixin modules
from video._vp_nav_buffer import NavBufferMixin
from video._vp_format_detection import FormatDetectionMixin
from video._vp_ffmpeg_builders import FFmpegBuildersMixin

try:
    import scipy  # noqa: F401 - presence check only (audio waveform generation)
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
        self._ms_per_frame = 0.0
        self.playhead_override_ms = None  # Set by point-jump to display at exact action time
        self._seek_in_progress_since = 0.0  # monotonic; display_route shows "Seeking..." while fresh
        self._video_open_in_progress = False
        self.target_fps = 30
        self.actual_fps = 0
        self.last_fps_update_time = time.time()
        self.frames_for_fps_calc = 0
        self.frame_lock = threading.Lock()
        self.arrow_nav_in_progress = False  # Flag to prevent arrow nav overload

        self.total_frames = 0
        self.current_frame_index = 0

        self._last_applied_speed_mode = None
        self._last_applied_slow_mo_fps = None

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
        # Live-tracking HD suspend state (see suspend_hd_for_live_tracking).
        self._hd_suspended_for_live = False
        self._hd_suspended_saved_flag = False
        self._hd_display_force_off = False
        if self.app and hasattr(getattr(self.app, 'app_settings', None), 'config'):
            self.hd_display_enabled = self.app.app_settings.config.ui.hd_video_display

        # VR Unwarp method. Only two options now:
        #   'v360' (default): libavfilter v360 applied inside the ffmpeg
        #            subprocess -vf chain
        #   'none': crop-only, skip v360 (useful for debugging or preprocessed
        #            videos where dewarp was already baked in).
        self.vr_unwarp_method_override = 'v360'
        if self.app and hasattr(getattr(self.app, 'app_settings', None), 'config'):
            self.vr_unwarp_method_override = self.app.app_settings.config.performance.vr_unwarp_method
            if self.vr_unwarp_method_override not in ('v360', 'none'):
                self.vr_unwarp_method_override = 'v360'

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

        # Unified frame cache (byte-budgeted LRU). Serves both the
        # arrow-nav hit path and random-access fetches; survives seeks so
        # bouncing between regions reuses decoded frames for free until
        # LRU eviction. Created here; the prefetcher thread is started
        # after open_video succeeds and stopped on reset(close_video=True).
        self._init_nav_cache()
        self._init_arrow_async()

        # ML format detector (lazy loaded)
        self.ml_detector = None
        self.ml_model_path = str(paths.MODELS_DIR / 'vr_detector_model_rf.pkl')

        # Event callbacks (for optional features like streamer, device_control)
        self._seek_callbacks = []  # List of callbacks: func(frame_index: int) -> None
        self._playback_state_callbacks = []  # List of callbacks: func(is_playing: bool, current_time_ms: float) -> None

        # Active ffmpeg-subprocess frame source. Produces numpy BGR frames
        # for the GUI display + tracker inference. One subprocess per video.
        self.frame_source = None
        # Bumped on every seek_video. The processing loop captures it
        # before pulling each frame and refuses to publish frames whose epoch
        # is stale - protects against races where a seek lands in between the
        # loop's "pull frame" and "write current_frame_index" steps.
        self._frame_source_seek_epoch: int = 0
        # Set by seek_video to the user's target frame. While non-None, the
        # loop pins current_frame_index to this target (so the timeline snaps
        # instantly) but still publishes decoded frames to current_frame so
        # the video catches up visually from the GOP keyframe.
        self._pending_seek_target: Optional[int] = None

    # ---------------------------------------------------------- frame source

    def _open_frame_source(self) -> bool:
        """Build and open the ffmpeg-subprocess frame source.

        The filter chain (crop + v360 for VR, scale+pad for 2D) is passed
        to ffmpeg via ``-vf``. Preprocessed sources route through the 2D
        path (see _build_filter_chain).

        Hwaccel args come from _get_ffmpeg_hwaccel_args, which already
        disables hwaccel on macOS and for 10-bit / preprocessed sources
        where the CPU filter chain would need a GPU->CPU download per
        frame (benchmarked 6x slower than CPU decode on macOS).
        """
        self._close_frame_source()
        chain = self._build_tracker_filter_chain()
        cfg = SourceConfig(
            video_path=self._active_video_source_path or self.video_path,
            filter_chain=chain,
            output_w=self._display_frame_w,
            output_h=self._display_frame_h,
        )

        hwaccel = []
        try:
            hwaccel = self._get_ffmpeg_hwaccel_args()
        except Exception as e:
            self.logger.debug(f"hwaccel probe failed, using CPU: {e}")
        src = FFmpegFrameSource(cfg, logger=self.logger, hwaccel_args=hwaccel)
        self.logger.debug(f"Frame source: ffmpeg subprocess (hwaccel={hwaccel or 'cpu'})")

        if not src.open():
            return False
        self.frame_source = src
        # Mirror seek/state observers so existing callbacks keep firing.
        for cb in list(self._seek_callbacks):
            src.register_seek_callback(cb)
        for cb in list(self._playback_state_callbacks):
            src.register_playback_state_callback(cb)
        # Position callback drives current_frame_index/version sync for any
        # one-shot get_frame from the GUI (the playback loop sets these directly).
        src.register_position_callback(self._on_frame_source_position)
        return True

    def _active_single_eye(self):
        """Return the user's panel selection clamped to 'left' | 'right'.

        Consumers that feed into a single-eye pipeline (v360, trackers)
        cannot meaningfully handle 'full'; they fall back to 'left'.
        """
        from video import vr_panel
        app_settings = getattr(self.app, 'app_settings', None)
        eye = vr_panel.read_setting(app_settings, default=vr_panel.EYE_LEFT)
        if eye == vr_panel.EYE_FULL:
            eye = vr_panel.EYE_LEFT
        return eye

    def _build_tracker_filter_chain(self) -> str:
        """libavfilter chain for the ffmpeg frame source feeding trackers.

        Always bakes v360 at canonical angles for VR (yaw=0, roll=0,
        configured pitch) so YOLO/DIS see consistent rectilinear input.
        Display-side filters are NOT controlled here; see
        _build_display_filter_chain.
        """
        from video import vr_panel
        out_w = self._display_frame_w
        out_h = self._display_frame_h
        if not self.video_info:
            return f"scale={out_w}:{out_h}"

        # Preprocessed videos are already cropped to one panel and dewarped.
        # Applying crop+v360 again produced libavfilter "Invalid argument"
        # crashes when vr_fov was unset on a reload path; route them through
        # the 2D scale-and-pad path.
        if self.determined_video_type == 'VR' and not self._is_using_preprocessed_video():
            ow = self.video_info.get('width', 0)
            oh = self.video_info.get('height', 0)
            parts = []
            crop = vr_panel.resolve_eye(self.vr_input_format,
                                        self._active_single_eye()).ffmpeg_crop(ow, oh)
            if crop:
                parts.append(crop)
            base_fmt = (self.vr_input_format
                        .replace('_sbs', '').replace('_tb', '')
                        .replace('_lr', '').replace('_rl', ''))
            v_h_fov = 90
            vr_fov = self.vr_fov if getattr(self, 'vr_fov', 0) and self.vr_fov > 0 else 190
            parts.append(
                f"v360={base_fmt}:in_stereo=0:output=sg:"
                f"iv_fov={vr_fov}:ih_fov={vr_fov}:"
                f"d_fov={vr_fov}:"
                f"v_fov={v_h_fov}:h_fov={v_h_fov}:"
                f"pitch={self.vr_pitch}:yaw=0:roll=0:"
                f"w={out_w}:h={out_h}:interp=linear"
            )
            return ",".join(parts)

        return (f"scale={out_w}:{out_h}"
                f":force_original_aspect_ratio=decrease,"
                f"pad={out_w}:{out_h}"
                f":(ow-iw)/2:(oh-ih)/2:black")

    # Back-compat alias for any external callers that still reference the
    # old single-chain method. Internal call sites have moved to the
    # explicit tracker / display variants.
    def _build_filter_chain(self) -> str:
        return self._build_tracker_filter_chain()

    # ---- HD gating for live tracking ----
    # In v0.8.0 a GPUUnwarpWorker produced 640x640 tracker frames directly.
    # v0.9.0 decodes at HD (up to 1920x1920) and CPU-resizes per frame in
    # _make_processing_frame (~0.5-1 ms per frame, 30-65 ms/sec at 60 fps).
    # Display uses the cpu_tracker route while tracking, so HD buys nothing.
    # Drop it for the duration of live tracking.
    def suspend_hd_for_live_tracking(self) -> None:
        if self._hd_suspended_for_live:
            return
        if not self._is_hd_active:
            return
        src = self.frame_source
        if src is None:
            return
        self._hd_suspended_saved_flag = getattr(self, 'hd_display_enabled', False)
        try:
            self._hd_display_force_off = True
            self.hd_display_enabled = False
            self._update_video_parameters()  # recomputes _display_frame_w/h
            new_chain = self._build_tracker_filter_chain()
            new_cfg = SourceConfig(
                video_path=src.cfg.video_path,
                filter_chain=new_chain,
                output_w=self._display_frame_w,
                output_h=self._display_frame_h,
                decoder_threads=src.cfg.decoder_threads,
            )
            src.reapply_settings(new_cfg)
            self._hd_suspended_for_live = True
            self.logger.debug(
                f"HD suspended for live tracking: output {self._display_frame_w}x{self._display_frame_h}")
        except Exception as e:
            self.logger.warning(f"suspend_hd_for_live_tracking failed: {e}")
            self._hd_display_force_off = False
            self.hd_display_enabled = self._hd_suspended_saved_flag

    def resume_hd_after_live_tracking(self) -> None:
        if not self._hd_suspended_for_live:
            return
        src = self.frame_source
        if src is None:
            self._hd_suspended_for_live = False
            return
        try:
            self._hd_display_force_off = False
            self.hd_display_enabled = self._hd_suspended_saved_flag
            self._update_video_parameters()
            new_chain = self._build_tracker_filter_chain()
            new_cfg = SourceConfig(
                video_path=src.cfg.video_path,
                filter_chain=new_chain,
                output_w=self._display_frame_w,
                output_h=self._display_frame_h,
                decoder_threads=src.cfg.decoder_threads,
            )
            src.reapply_settings(new_cfg)
            self.logger.debug(
                f"HD restored after live tracking: output {self._display_frame_w}x{self._display_frame_h}")
        except Exception as e:
            self.logger.warning(f"resume_hd_after_live_tracking failed: {e}")
        finally:
            self._hd_suspended_for_live = False

    def _vr_display_mode(self) -> str:
        """Returns 'passthrough' | 'shader_dewarp'."""
        try:
            mode = self.app.app_settings.config.vr_display.mode
        except Exception:
            return 'shader_dewarp'
        if mode == 'v360_baked':
            return 'shader_dewarp'
        return mode

    @property
    def _DISPLAY_CHAIN_MAX(self) -> int:
        """Cap for the libmpv display-side filter output.

        SW backend: capped at 1080 (CPU-bound on 8K sources).
        GL backend: 2048 in passthrough; 4096 in shader_dewarp so the
          shader has enough source texels to sample a narrow VR window
          without turning into upsampled mush.
        """
        try:
            cfg = self.app.app_settings.config
            backend = cfg.mpv.render_backend
        except Exception:
            return 2048
        if backend != 'gl':
            return 1080
        mode = cfg.vr_display.mode
        is_vr = self.determined_video_type == 'VR'
        return 4096 if (is_vr and mode == 'shader_dewarp') else 2048

    def _build_display_filter_chain(self) -> str:
        """VF string for mpv's display render path.

        Backend-dependent:
          - GL backend: return an empty filter chain. libmpv's OpenGL
            render API scales the decoded frame to the target FBO
            dimensions on GPU, so CPU filters (crop/scale/pad/v360) are
            wasted work that becomes the frame-time bottleneck on 8K
            sources. Panel extraction and dewarp happen in the shader.
          - SW backend: pre-scale on CPU to avoid asking the SW render
            path to handle full-resolution frames, and keep the v360
            baked option available.
        """
        cap = self._DISPLAY_CHAIN_MAX
        out_w = min(int(self._display_frame_w), cap)
        out_h = min(int(self._display_frame_h), cap)
        if not self.video_info:
            return ""

        try:
            backend = self.app.app_settings.config.mpv.render_backend
        except Exception:
            backend = 'gl'

        from video import vr_panel
        fmt = self.vr_input_format or ''

        if backend == 'gl':
            return ""

        if self.determined_video_type != 'VR' or self._is_using_preprocessed_video():
            return (f"scale={out_w}:{out_h}"
                    f":force_original_aspect_ratio=decrease,"
                    f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black")
        ow = self.video_info.get('width', 0)
        oh = self.video_info.get('height', 0)
        parts = []
        crop = vr_panel.resolve_eye(fmt, self._active_single_eye()).ffmpeg_crop(ow, oh)
        if crop:
            parts.append(crop)
        parts.append(
            f"scale={out_w}:{out_h}:force_original_aspect_ratio=decrease,"
            f"pad={out_w}:{out_h}:(ow-iw)/2:(oh-ih)/2:black")
        return ",".join(parts)

    def _close_frame_source(self) -> None:
        if self.frame_source is not None:
            try:
                self.frame_source.close()
            except Exception as e:
                self.logger.debug(f"frame source close error: {e}")
            self.frame_source = None

    def _on_frame_source_position(self, frame_index: int) -> None:
        """Position callback from the source - kept light-weight."""
        # current_frame_index is updated by the playback loop to keep the
        # write path consistent. This hook is a placeholder for future use
        # (e.g., feeding device sync from random seeks).
        pass

    # --------------------------------------------------------- mpv display sync

    def _get_mpv_display(self):
        """Return the GUI's MpvDisplay instance, or None if not available."""
        gui = getattr(self.app, "gui_instance", None) if self.app else None
        return getattr(gui, "mpv_display", None) if gui else None

    def _load_into_mpv_display(self) -> None:
        # Pass the same -vf chain as the frame source so overlays align
        # with what mpv paints. Don't pause immediately: vo=libmpv needs
        # one render pass before any frame is ready.
        disp = self._get_mpv_display()
        if disp is None:
            self.logger.debug("MpvDisplay load skipped: no disp instance")
            return
        if not disp.is_alive:
            self.logger.debug("MpvDisplay load skipped: disp not alive")
            return
        path = self._active_video_source_path or self.video_path
        if not path:
            self.logger.debug("MpvDisplay load skipped: no video path yet")
            return
        try:
            vf = self._build_display_filter_chain() or None
            hwdec_override = None
            if vf and ('v360=' in vf or 'crop=' in vf or 'scale=' in vf):
                base_hwdec = getattr(disp, 'hwdec', '') or ''
                if base_hwdec and not base_hwdec.endswith('-copy') and base_hwdec != 'no':
                    hwdec_override = base_hwdec + '-copy'
            self.logger.debug(
                f"MpvDisplay.load() starting: path={path} "
                f"mode={self._vr_display_mode()} vf={vf!r} "
                f"hwdec_override={hwdec_override!r}")
            t0 = time.perf_counter()
            ok = disp.load(path, vf=vf, hwdec=hwdec_override)
            dur = (time.perf_counter() - t0) * 1000.0
            if ok:
                # Hand mpv our ffprobe-known fps if its own demuxer probe
                # came back empty (seen on some FISHEYE190 mp4s). Without
                # this the time-pos observer can't compute a frame index
                # and current_frame_index stays stuck at 0.
                if disp.fps <= 0 and self.fps and self.fps > 0:
                    try:
                        disp.set_fps_fallback(self.fps)
                    except Exception:
                        pass
                # Sync mpv pause state with the processor's intent. load()
                # internally calls play() which unpauses mpv; without this,
                # switching display modes (passthrough <-> shader) silently
                # unpauses while the UI still reads 'paused'.
                try:
                    should_play = (self.is_processing
                                   and not self.pause_event.is_set())
                    if should_play:
                        disp.play()
                    else:
                        disp.pause()
                except Exception as _e:
                    self.logger.debug(f"mpv pause sync failed: {_e}")
                self.logger.debug(
                    f"MpvDisplay.load() ok in {dur:.0f}ms "
                    f"(is_loaded={disp.is_loaded}, fps={disp.fps:.2f}, "
                    f"frames={disp.total_frames})")
            else:
                err = getattr(disp, 'last_load_error', None) or 'unknown error'
                self.logger.warning(
                    f"MpvDisplay.load() failed in {dur:.0f}ms ({err}); "
                    f"falling back to CPU upload path")
                if self.app is not None and hasattr(self.app, 'notify'):
                    try:
                        self.app.notify(
                            f"Smooth display unavailable: {err}", "warning", 4.0)
                    except Exception:
                        pass
        except Exception as e:
            self.logger.warning(f"MpvDisplay.load() raised: {e}", exc_info=True)

    def _mpv_seek_to_frame(self, frame_index: int) -> None:
        """Forward a frame-indexed seek to the libmpv display."""
        self._mpv_seek_to_frame_ex(frame_index, exact=True)

    def _mpv_seek_to_frame_ex(self, frame_index: int, exact: bool = True) -> None:
        disp = self._get_mpv_display()
        if disp is None or not disp.is_alive:
            return
        if self.fps and self.fps > 0:
            try:
                disp.seek(frame_index / self.fps, exact=exact)
            except Exception as e:
                self.logger.debug(f"MpvDisplay seek failed: {e}")

    def on_mpv_position(self, frame_index: int) -> None:
        """mpv time-pos observer. Drives current_frame_index during pure
        playback and clears the seek-in-progress flag once mpv catches up."""
        self._seek_in_progress_since = 0.0
        if not self.is_processing or self.pause_event.is_set():
            return
        if self.enable_tracker_processing:
            return
        tracker = getattr(self.app, 'tracker', None) if self.app else None
        if tracker is not None and getattr(tracker, 'tracking_active', False):
            return
        try:
            self.current_frame_index = int(frame_index)
        except (TypeError, ValueError):
            pass

    def _mpv_play(self) -> None:
        disp = self._get_mpv_display()
        if disp is not None and disp.is_alive:
            try: disp.play()
            except Exception: pass
        self._sync_mpv_speed_from_mode()

    def _sync_mpv_speed_from_mode(self) -> None:
        # Toolbar speed buttons only set app_state.selected_processing_speed_mode;
        # mpv plays at 1.0x unless we push the factor in. Called from _mpv_play
        # and the processing loop so user changes take effect without a seek.
        # MAX_SPEED is for tracker / offline-analysis pacing only (the loop's
        # target_delay), not for mpv display speed: 4x on 8K stutters because
        # the decoder cannot sustain it.
        disp = self._get_mpv_display()
        if disp is None or not disp.is_alive:
            return
        app_state = getattr(self.app, 'app_state_ui', None) if self.app else None
        if app_state is None:
            return
        mode = getattr(app_state, 'selected_processing_speed_mode', None)
        if mode is None:
            return
        if mode == constants.ProcessingSpeedMode.SLOW_MOTION:
            slo_mo_fps = float(getattr(app_state, 'slow_motion_fps', 10.0))
            factor = slo_mo_fps / self.fps if self.fps > 0 else 1.0
        else:
            factor = 1.0
        try:
            disp.set_speed(factor)
        except Exception:
            pass

    def _mpv_pause(self) -> None:
        disp = self._get_mpv_display()
        if disp is not None and disp.is_alive:
            try: disp.pause()
            except Exception: pass

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

        if self.app and hasattr(getattr(self.app, 'app_settings', None), 'config'):
            # Honor a caller-forced override (suspend_hd_for_live_tracking)
            # without clobbering it from persistent settings.
            if not getattr(self, '_hd_display_force_off', False):
                self.hd_display_enabled = self.app.app_settings.config.ui.hd_video_display

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
                aspect = self.app.app_settings.config.vr_display.display_aspect
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

        self.logger.debug(f"HD display: {out_w}x{out_h} ({out_w * out_h * 3} bytes/frame)")

    def _get_resize_device(self) -> str:
        """Lazily resolve the device used by the HD-path resize.

        Reads `hd_resize_device` from settings (default "cpu"). "auto" picks
        CUDA > MPS > CPU. On any GPU-init failure the path falls back to CPU
        and stays there for the life of the processor.
        """
        cached = getattr(self, "_resize_device", None)
        if cached is not None:
            return cached
        try:
            setting = str(self.app.app_settings.get("hd_resize_device", "cpu") or "cpu").lower()
        except Exception:
            setting = "cpu"
        resolved = "cpu"
        if setting in ("cuda", "mps", "auto"):
            try:
                import torch
                if setting == "cuda" and torch.cuda.is_available():
                    resolved = "cuda"
                elif setting == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                    resolved = "mps"
                elif setting == "auto":
                    if torch.cuda.is_available():
                        resolved = "cuda"
                    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                        resolved = "mps"
            except Exception:
                resolved = "cpu"
        self._resize_device = resolved
        return resolved

    def _resize_gpu(self, display_frame, new_w: int, new_h: int, device: str):
        """Resize an HWC uint8 frame on a torch device. Returns HWC uint8 numpy.

        Uses bilinear interpolation. Antialiasing is preferred for quality but
        the op is not implemented on MPS, so we sticky-fall-back to the
        non-AA path after the first exception and remember the choice for the
        life of the processor.
        """
        import torch
        import torch.nn.functional as F
        t = torch.from_numpy(display_frame).to(device).permute(2, 0, 1).unsqueeze(0).float()
        use_aa = getattr(self, "_resize_gpu_antialias", True)
        try:
            out = F.interpolate(t, size=(new_h, new_w), mode="bilinear",
                                antialias=use_aa, align_corners=False)
        except Exception:
            if use_aa:
                self._resize_gpu_antialias = False
                out = F.interpolate(t, size=(new_h, new_w), mode="bilinear",
                                    antialias=False, align_corners=False)
            else:
                raise
        out = out.clamp_(0, 255).to(torch.uint8).squeeze(0).permute(1, 2, 0).contiguous()
        return out.cpu().numpy()

    def _make_processing_frame(self, display_frame):
        """Resize an HD display frame down to yolo_input_size square with padding for YOLO/tracker.

        Uses pre-allocated buffer and cached resize parameters to avoid
        per-frame allocation overhead. The resize itself runs on CPU
        (cv2.INTER_AREA) by default, or on the torch device selected by the
        `hd_resize_device` setting ("cpu" | "auto" | "cuda" | "mps").
        """
        h, w = display_frame.shape[:2]
        size = self.yolo_input_size
        if h == size and w == size:
            return display_frame

        # Use cached resize params (computed once in _compute_display_dimensions)
        new_w, new_h = self._proc_resize_dims
        x_off, y_off = self._proc_pad_offset

        device = self._get_resize_device()
        if device != "cpu":
            try:
                resized = self._resize_gpu(display_frame, new_w, new_h, device)
            except Exception as e:
                self.logger.debug(f"GPU resize failed on {device}, pinning to CPU: {e}")
                self._resize_device = "cpu"
                resized = cv2.resize(display_frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
        else:
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

        Output frames are guaranteed to be at yolo_input_size (640x640) - the
        tracker / detection pipelines were designed around that coordinate
        space. If HD display is currently on (for GUI playback), this method
        temporarily flips it off, rebuilds the filter graph, and restores
        state on exit. Safe to call from the main thread; the tracker loops
        drive one-shot decoding via the frame source's internal primitives.
        """
        import time as _time
        if self.frame_source is None:
            self.logger.error("stream_frames_for_segment: no frame source")
            return
        src = self.frame_source

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
            # but does NOT rebuild the ffmpeg filter graph. Do that explicitly.
            self._update_video_parameters()
            try:
                new_chain = self._build_tracker_filter_chain()
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

        frames_yielded = 0
        _hd_restore_needed = _hd_was_on
        # Drive decode via the source's background thread so YOLO / DIS /
        # cvtColor on the consumer thread overlap with decode (avoids the
        # single-thread serialization that dropped us to ~100 FPS).
        # start(start_frame=N) seeks internally before streaming.
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
                src2 = self.frame_source
                if src2 is not None:
                    new_chain = self._build_tracker_filter_chain()
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
        self._video_open_in_progress = True

        if self.app and hasattr(self.app, 'app_state_ui'):
            self.app.app_state_ui.invalidate_content_uv_cache()

        self.stop_processing()
        self.video_path = video_path
        self._clear_cache()
        # Clear ML detection cache when opening new video
        if hasattr(self, '_ml_detection_cached'):
            delattr(self, '_ml_detection_cached')

        if self.app and hasattr(getattr(self.app, 'app_settings', None), 'config'):
            self.vr_unwarp_method_override = self.app.app_settings.config.performance.vr_unwarp_method
            if self.vr_unwarp_method_override not in ('v360', 'none'):
                self.vr_unwarp_method_override = 'v360'
            self.logger.debug(f"VR unwarp method from settings: {self.vr_unwarp_method_override}")

        self.video_info = self._get_video_info(video_path)
        if not self.video_info or self.video_info.get("total_frames", 0) == 0:
            self.logger.warning(f"Failed to get valid video info for {video_path}")
            self.video_path = ""
            self.video_info = {}
            self._video_open_in_progress = False
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
        self._ms_per_frame = (1000.0 / self.fps) if self.fps and self.fps > 0 else 0.0
        self.total_frames = self.video_info['total_frames']

        # Initialize thumbnail extractor for fast random frame access (FFmpeg-based)
        self._init_thumbnail_extractor()
        # Hand the active source path to the libmpv display (when the GUI
        # has spun one up). Display path runs in parallel with the
        # ffmpeg-subprocess frame source: mpv decodes for the screen with
        # HW acceleration; the frame source decodes numpy frames for the
        # tracker / detection pipelines. Seeks are forwarded to both so
        # the displayed picture matches the tracker's view.
        self._load_into_mpv_display()
        if not self._open_frame_source():
            self.logger.error("Failed to open frame source; playback will not work")
        else:
            # Pre-start the decoder in paused state so arrow-nav / scrub hits
            # the fast +1 pump path without paying the decode-thread spin-up
            # on every keypress. Consume the seek-response frame so the
            # source's _current_frame_index is published; that primes the
            # +1 fast path for the first arrow press.
            self.frame_source.start(0)
            self.frame_source.wait_seek(timeout=2.0)
            self.frame_source.next_frame(timeout=2.0)
            self.frame_source.pause()
        self.set_target_fps(self.fps)
        self.current_frame_index = 0
        self.stop_event.clear()
        # Reset VR pan to center on each video open.
        if self.app and hasattr(self.app, 'app_state_ui'):
            try:
                self.app.app_state_ui.vr_pan_yaw = 0.0
                self.app.app_state_ui.vr_pan_pitch = 0.0
            except Exception:
                pass
        # Boot the nav prefetcher now that the frame source is warm.
        # Idle-gated internally: it only runs while paused + tracker-idle.
        self._start_nav_prefetcher()
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
        fmt = self.vr_input_format if self.determined_video_type == 'VR' else self.determined_video_type
        w = self.video_info.get('width', 0)
        h = self.video_info.get('height', 0)
        self.logger.info(
            f"Opened: {active_source_name} [{fmt}, {w}x{h}, {self.total_frames}f @ {self.fps:.2f}fps]")

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

        self._video_open_in_progress = False
        return True


    def reapply_display_settings(self):
        """Reload mpv with a fresh vf chain, preserving the playhead."""
        if not self.video_path:
            return
        saved_frame = int(self.current_frame_index or 0)
        self._load_into_mpv_display()
        if saved_frame > 0 and saved_frame < (self.total_frames or 0):
            try:
                self._mpv_seek_to_frame(saved_frame)
            except Exception:
                pass

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
        # Settings change means the output frame shape may change (HD toggle,
        # VR format switch). Stop the prefetcher, clear the cache, and
        # restart once the new source is live.
        self._stop_nav_prefetcher()
        self._clear_cache()

        # [REDUNDANCY REMOVED] - Call the new helper method
        self._update_video_parameters()

        # Reinitialize thumbnail extractor with new display dimensions
        self._init_thumbnail_extractor()
        if self._open_frame_source():
            # Pre-start paused (see open_video path for rationale).
            self.frame_source.start(max(0, stored_frame_index))
            self.frame_source.wait_seek(timeout=2.0)
            self.frame_source.next_frame(timeout=2.0)
            self.frame_source.pause()
            self._start_nav_prefetcher()

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
        """Random-access frame fetch. Cache hit -> instant; cache miss ->
        frame_source.get_frame (~120 ms cold-spawn on subprocess backend,
        <1 ms on hot)."""
        if not self.video_path or not self.video_info or self.video_info.get('fps', 0) <= 0:
            self.logger.warning("Cannot get frame: video not loaded/invalid FPS.")
            if update_current_index:
                self.current_frame_index = frame_index_abs
            return None

        frame = self._nav_cache.get(int(frame_index_abs))
        if frame is not None:
            if update_current_index:
                self.current_frame_index = frame_index_abs
            return frame

        if self.frame_source is None:
            self.logger.warning(f"_get_specific_frame({frame_index_abs}): no frame source")
            return None
        frame = self.frame_source.get_frame(frame_index_abs, timeout=2.0)
        if frame is None:
            self.logger.warning(f"frame_source.get_frame failed for {frame_index_abs}")
            return None
        self._nav_cache.put(int(frame_index_abs), frame)
        if update_current_index:
            self.current_frame_index = frame_index_abs
        return frame


    def _get_video_info(self, filename):
        """Probe comprehensive video + audio metadata via ffprobe.

        Returns the same dict shape the rest of the app expects (callers
        pull width/height/fps/duration/codec_name/bit_depth/has_audio/etc.).
        Ported from v0.8.0's _get_video_info pattern that leveraged ffprobe
        + filename/resolution heuristics without opening a decoder.
        """
        import json as _json
        import subprocess as _subprocess
        from video.ffmpeg_helpers import find_ffprobe as _find_ffprobe, subprocess_flags as _flags

        cmd = [
            _find_ffprobe(), "-v", "error", "-show_streams", "-show_format",
            "-of", "json", filename,
        ]
        try:
            result = _subprocess.run(
                cmd, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE,
                timeout=10.0, creationflags=_flags(),
            )
            if result.returncode != 0:
                self.logger.error(
                    f"ffprobe failed for {filename}: "
                    f"{result.stderr.decode('utf-8', errors='replace').strip()[:200]}")
                return None
            data = _json.loads(result.stdout.decode("utf-8", errors="replace"))
        except (OSError, _subprocess.TimeoutExpired, ValueError) as e:
            self.logger.error(f"ffprobe error for {filename}: {e}")
            return None

        vstream = None
        astream = None
        for s in data.get("streams") or []:
            ct = s.get("codec_type")
            if ct == "video" and vstream is None:
                vstream = s
            elif ct == "audio" and astream is None:
                astream = s
        if vstream is None:
            self.logger.error(f"No video stream in {filename}")
            return None

        def _parse_rat(expr, fallback=0.0):
            if not expr:
                return fallback
            if "/" in expr:
                n, _, d = expr.partition("/")
                try:
                    n, d = float(n), float(d)
                    return n / d if d > 0 else fallback
                except ValueError:
                    return fallback
            try:
                return float(expr)
            except ValueError:
                return fallback

        r_rate = _parse_rat(vstream.get("r_frame_rate"), 0.0)
        avg_rate = _parse_rat(vstream.get("avg_frame_rate"), 0.0)
        fps = r_rate or avg_rate or 30.0
        is_vfr = (avg_rate > 0 and r_rate > 0
                  and abs(avg_rate - r_rate) / max(avg_rate, 1e-9) > 0.01)

        fmt_info = data.get("format") or {}
        duration = 0.0
        for raw in (vstream.get("duration"), fmt_info.get("duration")):
            if raw and raw != "N/A":
                try:
                    duration = float(raw)
                    break
                except ValueError:
                    pass

        total_frames = 0
        nb_raw = vstream.get("nb_frames")
        if nb_raw and nb_raw != "N/A":
            try:
                total_frames = int(nb_raw)
            except ValueError:
                total_frames = 0
        if total_frames == 0 and duration > 0 and fps > 0:
            total_frames = int(duration * fps)

        try:
            file_size_bytes = os.path.getsize(filename)
        except OSError:
            file_size_bytes = 0
        try:
            bitrate_bps = int(fmt_info.get("bit_rate") or 0)
        except ValueError:
            bitrate_bps = 0

        codec_name = vstream.get("codec_name") or "unknown"
        codec_long_name = vstream.get("codec_long_name") or codec_name

        bit_depth = 8
        pix_fmt = (vstream.get("pix_fmt") or "").lower()
        if any(fmt in pix_fmt for fmt in ("12le", "p012", "12be")):
            bit_depth = 12
        elif any(fmt in pix_fmt for fmt in ("10le", "p010", "10be")):
            bit_depth = 10
        # ffprobe also reports bits_per_raw_sample on many containers; prefer
        # it when available since pix_fmt can miss exotic 10/12-bit profiles.
        bps_raw = vstream.get("bits_per_raw_sample")
        if bps_raw:
            try:
                bps_int = int(bps_raw)
                if bps_int > bit_depth:
                    bit_depth = bps_int
            except ValueError:
                pass

        has_audio = astream is not None
        audio_codec_name = astream.get("codec_name", "") if astream else ""
        audio_codec_long_name = astream.get("codec_long_name", "") if astream else ""
        try:
            audio_bitrate = int((astream or {}).get("bit_rate") or 0)
        except ValueError:
            audio_bitrate = 0
        try:
            audio_sample_rate = int((astream or {}).get("sample_rate") or 0)
        except ValueError:
            audio_sample_rate = 0
        try:
            audio_channels = int((astream or {}).get("channels") or 0)
        except ValueError:
            audio_channels = 0

        self.logger.debug(
            f"Detected video properties: width={vstream.get('width')}, "
            f"height={vstream.get('height')}, fps={fps:.2f}, bit_depth={bit_depth}")

        return {
            "duration": duration, "total_frames": total_frames, "fps": fps,
            "width": int(vstream.get("width") or 0),
            "height": int(vstream.get("height") or 0),
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

    def get_audio_waveform(self, num_samples: int = 1000) -> Optional[np.ndarray]:
        """Generate a waveform by decoding the audio stream via an ffmpeg
        subprocess and downsampling to ``num_samples`` peak-amplitude buckets.

        ffmpeg downsamples to a low-rate mono 16-bit PCM stream (8 kHz by
        default) before piping to Python. At this rate a feature-length film
        is ~50 MB instead of ~600 MB, and every downstream step -- pipe I/O,
        numpy frombuffer, reshape, max-reduce -- scales linearly with that
        data. 8 kHz still captures envelope peaks for every viewable-width
        bucket (the UI pool is ~2k samples, so each bucket covers tens of
        thousands of audio samples; peak fidelity is preserved).
        """
        if not self.video_path or not self.video_info.get("has_audio"):
            self.logger.info("No video loaded or video has no audio stream for waveform generation.")
            return None
        if not SCIPY_AVAILABLE_FOR_AUDIO:
            self.logger.warning("Scipy is not available. Cannot generate audio waveform.")
            return None

        import subprocess as _subprocess
        from video.ffmpeg_helpers import find_ffmpeg as _find_ffmpeg, subprocess_flags as _flags

        target_rate = 8000
        cmd = [
            _find_ffmpeg(), "-hide_banner", "-nostats", "-loglevel", "error",
            "-i", self.video_path,
            "-vn", "-sn",
            "-ac", "1", "-ar", str(target_rate),
            "-c:a", "pcm_s16le", "-f", "s16le", "pipe:1",
        ]
        try:
            # 5-minute cap; long enough for feature-length VR content, short
            # enough that a stuck ffmpeg won't hang the caller forever.
            result = _subprocess.run(
                cmd, stdout=_subprocess.PIPE, stderr=_subprocess.PIPE,
                timeout=300.0, creationflags=_flags(),
            )
            if result.returncode != 0 or not result.stdout:
                self.logger.error(
                    f"ffmpeg audio decode failed: rc={result.returncode} "
                    f"stderr={result.stderr.decode('utf-8', errors='replace').strip()[:200]}")
                return None
            data = np.frombuffer(result.stdout, dtype=np.int16)
        except (OSError, _subprocess.TimeoutExpired) as e:
            self.logger.error(f"Error generating audio waveform: {e}")
            return None

        if data.size == 0:
            return None
        step = max(1, len(data) // num_samples)
        # Vectorized max-abs per step-sized window. Trim to a whole multiple
        # of `step` so we can reshape to (n, step); any trailing partial
        # window is computed separately and appended.
        n_full = (len(data) // step) * step
        abs_data = np.abs(data)
        main = abs_data[:n_full].reshape(-1, step).max(axis=1)
        if n_full < len(data):
            tail = np.asarray([abs_data[n_full:].max()], dtype=main.dtype)
            waveform_np = np.concatenate([main, tail]).astype(np.float32)
        else:
            waveform_np = main.astype(np.float32)
        max_val = float(waveform_np.max())
        if max_val > 0:
            waveform_np /= max_val
        self.logger.info(f"Generated waveform with {len(waveform_np)} samples.")
        return waveform_np


    def start_processing(self, start_frame=None, end_frame=None, cli_progress_callback=None):
        if self.is_processing and self.pause_event.is_set():
            self.logger.debug(f"Resuming playback from frame {self.current_frame_index}")
            self.pause_event.clear()
            tracker_active = self.tracker and getattr(self.tracker, 'tracking_active', False)
            # Skip the ffmpeg seek+resume when mpv is going to drive display.
            # Without a tracker pulling frames, the playback loop's mpv branch
            # immediately re-pauses the source anyway, so a play-time seek on
            # an 8K source costs hundreds of ms for nothing.
            if tracker_active and self.frame_source is not None:
                fs_idx = getattr(self.frame_source, 'current_frame_index', self.current_frame_index)
                if fs_idx != self.current_frame_index:
                    self._pending_seek_target = self.current_frame_index
                    self._frame_source_seek_epoch += 1
                    self.frame_source.seek(self.current_frame_index, accurate=True)
                self.frame_source.resume()
            if not tracker_active:
                self._mpv_seek_to_frame(self.current_frame_index)
                self._mpv_play()

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
        if start_frame is not None:
            if 0 <= start_frame < self.total_frames:
                effective_start_frame = start_frame
            else:
                self.logger.warning(f"Start frame {start_frame} out of bounds ({self.total_frames} total). Not starting.")
                return

        self.logger.debug(f"Starting processing from frame {effective_start_frame}.")

        self.processing_start_frame_limit = effective_start_frame
        self.processing_end_frame_limit = -1
        if end_frame is not None and end_frame >= 0:
            self.processing_end_frame_limit = min(end_frame, self.total_frames - 1)

        self.is_processing = True
        self.pause_event.clear()
        self.stop_event.clear()
        # Skip waking mpv when a live tracker is driving decode; otherwise
        # mpv decodes the same video in parallel with the tracker's ffmpeg
        # source, burns CPU, and drifts into A/V desync.
        if not (self.tracker and getattr(self.tracker, 'tracking_active', False)):
            self._mpv_seek_to_frame(effective_start_frame)
            self._mpv_play()
        self.processing_thread = threading.Thread(
            target=self._processing_loop, name="VideoProcessingThread")
        self.processing_thread.daemon = True
        self.processing_thread.start()
        self.logger.debug(
            f"Started processing. Range: {self.processing_start_frame_limit} to "
            f"{self.processing_end_frame_limit if self.processing_end_frame_limit != -1 else 'EOS'}")

    def pause_processing(self):
        if not self.is_processing or self.pause_event.is_set():
            return

        self.logger.debug(f"Pausing playback at frame {self.current_frame_index}")
        self.pause_event.set()
        # Pause the source so it stops pumping frames past current_frame_index;
        # otherwise arrow nav's +1 fast path reads the speculated-ahead frame.
        if self.frame_source is not None:
            self.frame_source.pause()
        self._mpv_pause()
        if hasattr(self, "_nav_prefetcher"):
            self._nav_prefetcher.notify()

        if self._playback_state_callbacks:
            current_time_ms = (self.current_frame_index / self.fps) * 1000.0 if self.fps > 0 else 0.0
            self._notify_playback_state_callbacks(False, current_time_ms)

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

        # Stop must also halt the decoder and mpv display -- otherwise
        # mpv-driven playback keeps rendering frames even though the
        # processing thread has exited.
        if self.frame_source is not None:
            try:
                self.frame_source.pause()
            except Exception:
                pass
        self._mpv_pause()

        if join_thread:
            thread_to_join = self.processing_thread
            if thread_to_join and thread_to_join.is_alive():
                if threading.current_thread() is not thread_to_join:
                    thread_to_join.join(timeout=2.0)
                    if thread_to_join.is_alive():
                        self.logger.warning("Processing thread did not join cleanly after stop signal.")
            self.processing_thread = None

            if self.tracker:
                self.tracker.stop_tracking()

            self.enable_tracker_processing = False

            if self.app and hasattr(self.app, 'on_processing_stopped'):
                self.app.on_processing_stopped(was_scripting_session=was_scripting_session, scripted_frame_range=scripted_range)
        else:
            self.enable_tracker_processing = False

        self.logger.debug("GUI processing stopped.")

    def seek_video(self, frame_index: int, accurate: bool = True):
        """Seek to a specific frame via the active frame source.

        accurate=True (default): mpv lands on exact target frame so single
        seeks (jump-to-point, click, chapter, bookmark) display correctly.
        accurate=False: keyframe-fast seek; only the drag-scrub path opts
        in for responsiveness during continuous mouse drag.
        """
        if not self.video_info or self.video_info.get('fps', 0) <= 0 or self.total_frames <= 0:
            return
        if self.frame_source is None:
            self.logger.error("seek_video: no frame source")
            return

        _t0 = time.perf_counter()
        target_frame = max(0, min(frame_index, self.total_frames - 1))
        self.playhead_override_ms = None
        self._frame_source_seek_epoch += 1
        self.current_frame_index = target_frame
        self._pending_seek_target = target_frame
        self._seek_in_progress_since = time.monotonic()
        # NOTE: we deliberately DO NOT clear the nav cache on seek anymore.
        # The LRU cache is keyed by frame index and survives jumps, so
        # bouncing between regions of the video (timeline scrub, chapter
        # jump, up/down point jump) keeps every frame the user already
        # decoded until byte-budget pressure evicts it. Cache clears only
        # on video change or frame-shape change (see _clear_cache).
        if hasattr(self, "_nav_detector"):
            self._nav_detector.record(target_frame)
        if hasattr(self, "_nav_prefetcher"):
            self._nav_prefetcher.notify()
        self._notify_seek_callbacks(target_frame)

        # Skip the ffmpeg seek when no tracker consumes frames: mpv drives
        # the display, the playback loop has already paused the source, and
        # the loop's transition path will re-seek + resume on tracker
        # activation. On 8K VR with v360 dewarp this avoids hundreds of ms
        # of keyframe-decode latency per scrub.
        tracker_needs_frames = (
            self.enable_tracker_processing
            or (self.tracker is not None
                and getattr(self.tracker, 'tracking_active', False))
        )
        if tracker_needs_frames:
            self.frame_source.seek(target_frame, accurate=False)
            if not (self.is_processing and not self.pause_event.is_set()):
                self._pending_seek_target = None
                self._seek_in_progress_since = 0.0
        else:
            self._pending_seek_target = None
            self._seek_in_progress_since = 0.0

        _t1 = time.perf_counter()
        # Keep libmpv display in sync with tracker's source.
        self._mpv_seek_to_frame_ex(target_frame, exact=accurate)
        _t2 = time.perf_counter()
        if (_t2 - _t0) * 1000.0 > 30.0:
            self.logger.debug(
                f"seek {target_frame}: total={(_t2-_t0)*1000:.1f}ms "
                f"ffmpeg={(_t1-_t0)*1000:.1f}ms mpv={(_t2-_t1)*1000:.1f}ms "
                f"tracker_needs={tracker_needs_frames} accurate={accurate}")

        # Mirror to external fullscreen mpv so scrub updates current_frame_index.
        mpv_ctrl = getattr(self.app, '_mpv_controller', None) if self.app else None
        if mpv_ctrl is not None and getattr(mpv_ctrl, 'is_active', False):
            try:
                mpv_ctrl.seek(target_frame)
            except Exception as e:
                self.logger.debug(f"external mpv seek failed: {e}")

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
                        # Skip copy when the active tracker has opted out via
                        # BaseTracker.mutates_input_frame = False.
                        cur = getattr(self.tracker, '_current_tracker', None)
                        needs_copy = getattr(cur, 'mutates_input_frame', True)
                        processing_frame = raw_frame_to_process.copy() if needs_copy else raw_frame_to_process
                    # Tracker populates live_overlay; display frame stays clean
                    self.tracker.process_frame(processing_frame, timestamp_ms, self.current_frame_index)
            except Exception as e:
                self.logger.error(f"Error processing frame with tracker in display_current_frame: {e}", exc_info=True)


    def _processing_loop(self):
        """Unified playback loop."""
        return self._playback_loop()

    def _playback_loop(self):
        """Playback loop driven by the ffmpeg-subprocess frame source. Seek
        is in-process and ~5-12x faster than the ffmpeg-respawn path on heavy
        formats. Keeps every observable invariant the subprocess loop holds:
        current_frame / current_frame_index / _frame_version are written here,
        nav buffer is populated, tracker.process_frame is called with the same
        signature, playback-state callbacks fire on each frame.
        """
        src = self.frame_source
        if src is None:
            self.logger.error("_playback_loop: no source")
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
        # Skip the slow-frame warning on the first pull of a run: ffmpeg
        # always pays keyframe-decode cost on the first frame after a fresh
        # spawn, and that's not a regression-worthy signal.
        _steady_state_frames = 0
        _prev_source_idle = None  # log once per state transition

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
                    # directly without the frame source knowing. If we resumed
                    # without seeking, the source would deliver stale frames
                    # from the old pause point and drag the timeline back.
                    if src.current_frame_index != self.current_frame_index:
                        self._frame_source_seek_epoch += 1
                        self._pending_seek_target = self.current_frame_index
                        src.seek(self.current_frame_index, accurate=False)
                    if src.is_paused:
                        src.resume()
                        if self.app and hasattr(self.app, 'on_processing_resumed'):
                            self.app.on_processing_resumed()
                    next_frame_target_time = time.perf_counter()

                # ---- pacing ----
                _ui = self.app.app_state_ui
                speed_mode = _ui.selected_processing_speed_mode
                if speed_mode == constants.ProcessingSpeedMode.REALTIME:
                    target_delay = 1.0 / self.fps if self.fps > 0 else (1.0 / 30.0)
                elif speed_mode == constants.ProcessingSpeedMode.SLOW_MOTION:
                    slo_mo_fps = getattr(_ui, 'slow_motion_fps', 10.0)
                    target_delay = 1.0 / max(1.0, slo_mo_fps)
                else:
                    target_delay = 0.0
                if speed_mode != self._last_applied_speed_mode or (
                    speed_mode == constants.ProcessingSpeedMode.SLOW_MOTION
                    and getattr(_ui, 'slow_motion_fps', 10.0) != self._last_applied_slow_mo_fps
                ):
                    self._sync_mpv_speed_from_mode()
                    self._last_applied_speed_mode = speed_mode
                    self._last_applied_slow_mo_fps = getattr(_ui, 'slow_motion_fps', 10.0)

                # ---- chapter-aware tracker start/stop ----
                # Runs BEFORE the decode gate so an NR -> Position transition
                # re-arms tracking_active on the same iteration; otherwise
                # mpv-driven playback would never restart the tracker.
                if self.tracker and self.enable_tracker_processing:
                    current_chapter = self.app.funscript_processor.get_chapter_at_frame(self.current_frame_index)
                    current_chapter_id = current_chapter.unique_id if current_chapter else None
                    if current_chapter_id != self.last_processed_chapter_id:
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
                    if current_chapter and not self.tracker.tracking_active and current_chapter.user_roi_fixed:
                        self.tracker.start_tracking()

                # ---- gate ffmpeg decode on whether anyone consumes its frames ----
                # Pure playback (no tracker, mpv loaded) doesn't need numpy
                # frames; pause the source so its decoder thread is idle.
                # mpv drives current_frame_index via on_mpv_position.
                tracker_needs_frames = (
                    self.enable_tracker_processing
                    or (self.tracker is not None
                        and getattr(self.tracker, 'tracking_active', False))
                )
                disp = self._get_mpv_display()
                disp_loaded = bool(getattr(disp, 'is_loaded', False)) if disp else False
                mpv_drives_display = (
                    disp is not None
                    and disp_loaded
                    and not tracker_needs_frames
                )
                if _prev_source_idle != mpv_drives_display:
                    self.logger.debug(
                        f"Playback decode mode: source_idle={mpv_drives_display} "
                        f"(disp={disp is not None}, loaded={disp_loaded}, "
                        f"tracker_needs={tracker_needs_frames}, "
                        f"mpv_fps={getattr(disp, 'fps', 0):.2f})")
                    _prev_source_idle = mpv_drives_display
                if mpv_drives_display:
                    if not src.is_paused:
                        src.pause()
                    # Periodic diagnostic: are mpv renders actually firing
                    # and is the time-pos advancing? Helps when the FBO
                    # is mysteriously black.
                    now = time.perf_counter()
                    if now - getattr(self, '_mpv_diag_t', 0.0) > 2.0:
                        self._mpv_diag_t = now
                        try:
                            renders = disp.render_stats
                            tp = getattr(disp, '_last_time_pos', 0.0)
                            self.logger.debug(
                                f"mpv driving: renders={renders[1]}/{renders[0]} "
                                f"time_pos={tp:.2f}s cur_idx={self.current_frame_index}")
                        except Exception:
                            pass
                    if (self.total_frames > 0
                            and self.current_frame_index >= self.total_frames - 1):
                        self.logger.info(
                            f"mpv reached end of stream "
                            f"(idx={self.current_frame_index})")
                        self.is_processing = False
                        if self.app and hasattr(self.app, 'on_processing_stopped'):
                            try:
                                self.app.on_processing_stopped(was_scripting_session=False)
                            except Exception:
                                pass
                        break
                    self.stop_event.wait(max(0.02, target_delay) if target_delay > 0 else 0.05)
                    continue

                # Transition into tracker-needed: resume source at the cursor
                # mpv was driving so the next frame matches the displayed one.
                if src.is_paused:
                    if src.current_frame_index != self.current_frame_index:
                        self._frame_source_seek_epoch += 1
                        self._pending_seek_target = self.current_frame_index
                        src.seek(self.current_frame_index, accurate=False)
                    src.resume()
                    next_frame_target_time = time.perf_counter()

                # ---- pull next frame ----
                # Capture epoch BEFORE pulling so we can detect a seek that
                # happens during the (potentially blocking) next_frame call
                # and discard the resulting stale frame.
                epoch_before = self._frame_source_seek_epoch
                decode_start = time.perf_counter()
                item = src.next_frame(timeout=2.0)
                decode_time = (time.perf_counter() - decode_start) * 1000.0
                self._decode_samples.append(decode_time)
                # First frame after a seek/start includes keyframe-decode cost;
                # only warn when a steady-state decode (frame actually
                # returned, not right after a spawn) is this slow.
                if item is not None:
                    _steady_state_frames += 1
                if (item is not None and decode_time > 200
                        and self._pending_seek_target is None
                        and _steady_state_frames > 2):
                    self.logger.warning(
                        f"Slow ffmpeg frame at {self.current_frame_index}: {decode_time:.0f}ms")

                if item is None:
                    if self.stop_event.is_set():
                        break
                    # Differentiate true EOS (decoder produced _EOS sentinel)
                    # from a transient pull timeout (decoder still working).
                    if src.is_eos:
                        # idx<=0 + positive total_frames = silent decode failure.
                        if (getattr(src, 'current_frame_index', -1) <= 0
                                and (self.total_frames or 0) > 1):
                            self.logger.error(
                                "ffmpeg produced 0 frames before end-of-stream; "
                                "decode likely failed. Try setting hwaccel to 'none' or 'auto' in settings.")
                        else:
                            self.logger.info(
                                f"ffmpeg loop: end of stream (idx={src.current_frame_index})")
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
                # Drop frames decoded with a stale epoch - a seek_video happened
                # between when we asked for this frame and when we got it.
                if epoch_before != self._frame_source_seek_epoch:
                    continue

                # Post-seek catch-up: the user clicked frame T, source landed on
                # the GOP keyframe at T'<T and is decoding forward. Show those
                # frames so the video plays through the catch-up (good UX), but
                # keep current_frame_index pinned to the user's intent (T) until
                # the source actually reaches it. After that, resume normal
                # index updates from the decoded frame index.
                # MAX_SPEED is the offline analysis path; the user isn't
                # scrubbing mid-run so the nav cache would just burn memory
                # bandwidth (75 MB per frame on 8K VR) for frames that will
                # never be revisited.
                cache_this_frame = speed_mode != constants.ProcessingSpeedMode.MAX_SPEED

                seek_target = self._pending_seek_target
                if seek_target is not None and idx < seek_target:
                    # Fast-skip: only cache and bump version.
                    if cache_this_frame:
                        self._buffer_append(idx, frame_np)
                    with self.frame_lock:
                        self.current_frame = frame_np
                        self._frame_version += 1
                    continue
                self._pending_seek_target = None
                self._seek_in_progress_since = 0.0
                self.current_frame_index = idx

                # Only resize when a tracker will consume the frame.
                tracker_will_run = bool(
                    self.tracker and self.tracker.tracking_active
                    and self.enable_tracker_processing
                )
                if tracker_will_run and self.is_hd_active:
                    processing_frame = self._make_processing_frame(frame_np)
                else:
                    processing_frame = frame_np

                if cache_this_frame:
                    self._buffer_append(self.current_frame_index, frame_np)

                # ---- tracker ----
                if tracker_will_run:
                    timestamp_ms = int(self.current_frame_index * self._ms_per_frame)
                    # Skip 75MB memcpy when the active tracker has opted out
                    # via BaseTracker.mutates_input_frame = False.
                    cur = getattr(self.tracker, '_current_tracker', None)
                    needs_copy = getattr(cur, 'mutates_input_frame', True)
                    frame_arg = processing_frame.copy() if needs_copy else processing_frame
                    try:
                        yolo_start = time.perf_counter()
                        self.tracker.process_frame(frame_arg, timestamp_ms, self.current_frame_index)
                        self._yolo_samples.append((time.perf_counter() - yolo_start) * 1000.0)
                    except Exception as e:
                        self.logger.error(f"playback loop tracker error: {e}", exc_info=True)

                self._update_timing_metrics()

                # ---- actual decode/processing fps (rolling 0.5 s window) ----
                # Surfaces in the EXPERT > Video Pipeline panel so MAX_SPEED
                # users can see the real throughput vs the target.
                self.frames_for_fps_calc += 1
                _now_fps = time.perf_counter()
                _fps_dt = _now_fps - self.last_fps_update_time
                if _fps_dt >= 0.5:
                    self.actual_fps = self.frames_for_fps_calc / _fps_dt
                    self.frames_for_fps_calc = 0
                    self.last_fps_update_time = _now_fps

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
                    self.logger.info(f"Reached end_frame_limit ({self.processing_end_frame_limit})")
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
            self.logger.error(f"processing loop crashed: {e}", exc_info=True)
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
        """Tear down per-video resources: frame source, thumbnail extractor."""
        self._close_frame_source()

    def reset(self, close_video=False, skip_tracker_reset=False):
        self.logger.debug("Resetting VideoProcessor...")
        self.stop_processing(join_thread=True)
        self._clear_cache()
        self.current_frame_index = 0
        if self.tracker and not skip_tracker_reset:
            self.tracker.reset()
        if close_video:
            # Stop the prefetcher BEFORE closing the frame source so it
            # doesn't race against a torn-down decoder.
            self._stop_nav_prefetcher()
            self._stop_arrow_async()
            if self.thumbnail_extractor:
                self.thumbnail_extractor.close()
                self.thumbnail_extractor = None
            self._close_frame_source()
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
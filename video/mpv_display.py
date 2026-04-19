"""libmpv SW render API backend (legacy fallback; GL is the default)."""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from typing import Callable, List, Optional

import numpy as np

from video.mpv_loader import mpv, mpv_available


SeekCallback = Callable[[int], None]
PlaybackStateCallback = Callable[[bool, float], None]
PositionCallback = Callable[[int], None]


# ---------------------------------------------------------------------------
# Patch python-mpv to expose the SW render params (IDs 17-20 from render.h).
# ---------------------------------------------------------------------------
def _patch_mpv_sw_params():
    if mpv is None:
        return
    if 'sw_size' in getattr(mpv.MpvRenderParam, 'TYPES', {}):
        return  # already patched

    class MpvSwSize(ctypes.Structure):
        _fields_ = [('w', ctypes.c_int), ('h', ctypes.c_int)]
        def __init__(self, w=0, h=0):
            super().__init__()
            self.w = int(w)
            self.h = int(h)

    class MpvSwStride(ctypes.Structure):
        _fields_ = [('stride', ctypes.c_size_t)]
        def __init__(self, stride=0):
            super().__init__()
            self.stride = int(stride)

    mpv._MpvSwSize = MpvSwSize
    mpv._MpvSwStride = MpvSwStride
    # Mpv render param IDs from include/mpv/render.h
    mpv.MpvRenderParam.TYPES['sw_size'] = (17, MpvSwSize)
    mpv.MpvRenderParam.TYPES['sw_format'] = (18, str)
    mpv.MpvRenderParam.TYPES['sw_stride'] = (19, MpvSwStride)
    mpv.MpvRenderParam.TYPES['sw_pointer'] = (20, ctypes.c_void_p)


_patch_mpv_sw_params()


class MpvDisplay:

    def __init__(
        self,
        video_path: str,
        vf: Optional[str] = None,
        hwdec: str = "videotoolbox-copy",
        with_audio: bool = True,
        logger: Optional[logging.Logger] = None,
    ):
        self.video_path = video_path
        self.vf = vf or ""
        self.hwdec = hwdec
        self.with_audio = with_audio
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._player = None
        self._ctx = None

        self._last_time_pos: float = 0.0
        self._fps: float = 0.0
        self._total_frames: int = 0
        self._duration_s: float = 0.0

        self._seek_callbacks: List[SeekCallback] = []
        self._playback_state_callbacks: List[PlaybackStateCallback] = []
        self._position_callbacks: List[PositionCallback] = []

        self._state_lock = threading.Lock()
        self._is_paused = False
        self._loaded = False
        self._last_load_error: Optional[str] = None

        # SW render output buffer. (h, w, 4) uint8, format "0bgr" -> the
        # buffer holds [pad, B, G, R] per pixel; we sample with GL_BGRA.
        self._sw_buffer: Optional[np.ndarray] = None
        self._sw_w: int = 0
        self._sw_h: int = 0
        self._sw_format: bytes = b"0bgr\0"  # null-terminated, kept alive

        self._render_count = 0
        self._render_ok_count = 0

    @property
    def is_alive(self) -> bool:
        return self._player is not None and self._ctx is not None

    @property
    def is_loaded(self) -> bool:
        return self.is_alive and self._loaded

    @property
    def render_stats(self) -> tuple:
        return (self._render_count, self._render_ok_count)

    @property
    def last_load_error(self) -> Optional[str]:
        return self._last_load_error

    def open(self, get_proc_address=None) -> bool:
        """``get_proc_address`` is accepted for API compatibility but unused
        in SW mode (mpv does not need a GL context here)."""
        if not mpv_available:
            self.logger.warning("libmpv not available; MpvDisplay disabled")
            return False
        if self._player is not None:
            return True

        try:
            def _mpv_log(level, prefix, msg):
                line = f"[mpv:{prefix}] {msg.rstrip() if isinstance(msg, str) else msg}"
                if level in ("fatal", "error"):
                    self.logger.error(line)
                elif level == "warn":
                    self.logger.warning(line)
                elif level == "info":
                    self.logger.info(line)
                else:
                    self.logger.debug(line)

            kwargs = {
                "vo": "libmpv",
                "audio": "auto" if self.with_audio else "no",
                "keep_open": "yes",
                "hwdec": self.hwdec,
                # SW render path is CPU-bound; use mpv's fast profile.
                "profile": "fast",
                # Bilinear scaling everywhere; quality scalers waste CPU
                # at our display sizes.
                "scale": "bilinear",
                "cscale": "bilinear",
                "dscale": "bilinear",
                # Use all decoder threads for the CPU portion of the
                # pipeline (post-hwdec filters and the SW output stage).
                "vd_lavc_threads": "0",   # 0 means auto = all cores
                # Cap audio cache so we don't fall further and further
                # behind on slow machines.
                "audio_buffer": "0.05",
                "log_handler": _mpv_log,
                "loglevel": "info",
            }
            if self.vf:
                kwargs["vf"] = self.vf
            self._player = mpv.MPV(**kwargs)
        except Exception as e:
            self.logger.error(f"mpv.MPV init failed: {e}")
            return False

        try:
            self._ctx = mpv.MpvRenderContext(self._player, "sw")
        except Exception as e:
            self.logger.error(f"MpvRenderContext (sw) init failed: {e}")
            self._cleanup()
            return False

        self._install_observers()
        self.logger.info("MpvDisplay ready (SW render API)")
        return True

    def load(self, video_path: str, vf: Optional[str] = None) -> bool:
        if self._player is None:
            self.logger.warning("MpvDisplay.load called before open()")
            self._last_load_error = "open() not called"
            return False
        self._fps = 0.0
        self._total_frames = 0
        self._duration_s = 0.0
        self._last_time_pos = 0.0
        self._loaded = False
        self._last_load_error = None
        self.video_path = video_path
        if vf is not None:
            self.vf = vf
            try:
                self._player["vf"] = vf
            except Exception as e:
                self.logger.debug(f"set vf failed: {e}")
        try:
            self._player.play(video_path)
        except Exception as e:
            self.logger.error(f"mpv play({video_path}) failed: {e}")
            self._last_load_error = f"play() failed: {e}"
            return False

        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline:
            try:
                d = self._player.duration
            except Exception:
                d = None
            if d is not None and d > 0:
                self._duration_s = float(d)
                for prop in ('container-fps', 'estimated-vf-fps', 'fps', 'video-fps'):
                    try:
                        v = self._player[prop]
                    except Exception:
                        v = None
                    try:
                        if v is not None and float(v) > 0:
                            self._fps = float(v)
                            break
                    except (TypeError, ValueError):
                        continue
                if self._fps > 0:
                    self._total_frames = int(self._duration_s * self._fps)
                # Pause + seek to start so audio doesn't leak before play.
                try:
                    self._player.pause = True
                    self._is_paused = True
                    self._player.command("seek", 0.0, "absolute", "keyframes")
                except Exception as e:
                    self.logger.debug(f"post-load pause/seek failed: {e}")
                self._loaded = True
                self.logger.info(
                    f"MpvDisplay loaded {video_path} "
                    f"(duration={self._duration_s:.2f}s fps={self._fps:.3f})")
                return True
            time.sleep(0.02)

        tail = None
        try:
            tail = self._player.video_codec or self._player.file_format
        except Exception:
            pass
        self._last_load_error = (
            f"no duration within 5s (codec/format={tail!r})"
            if tail else "no duration reported within 5s"
        )
        self.logger.error(f"MpvDisplay.load: {self._last_load_error} for {video_path}")
        self._loaded = False
        return False

    def set_fps_fallback(self, fps: float) -> None:
        try:
            f = float(fps)
        except (TypeError, ValueError):
            return
        if f > 0 and self._fps <= 0:
            self._fps = f
            if self._duration_s > 0:
                self._total_frames = int(self._duration_s * f)
            self.logger.info(f"MpvDisplay fps fallback applied: {f:.3f}")

    def close(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        with self._state_lock:
            if self._ctx is not None:
                try:
                    self._ctx.free()
                except Exception:
                    pass
                self._ctx = None
            if self._player is not None:
                try:
                    self._player.terminate()
                except Exception:
                    pass
                self._player = None
        self._sw_buffer = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def play(self) -> None:
        if self._player is None:
            return
        self._player.pause = False
        self._is_paused = False

    def pause(self) -> None:
        if self._player is None:
            return
        self._player.pause = True
        self._is_paused = True

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def seek(self, time_seconds: float, exact: bool = True) -> None:
        if self._player is None:
            return
        precision = "exact" if exact else "keyframes"
        try:
            self._player.command("seek", float(time_seconds), "absolute", precision)
        except Exception as e:
            self.logger.debug(f"MpvDisplay seek({time_seconds}) failed: {e}")
            return
        if self._fps > 0:
            frame_idx = int(round(time_seconds * self._fps))
            for cb in list(self._seek_callbacks):
                try:
                    cb(frame_idx)
                except Exception as cb_err:
                    self.logger.debug(f"seek cb error: {cb_err}")

    def step_forward(self) -> None:
        if self._player is None:
            return
        try:
            self._player.command("frame-step")
        except Exception as e:
            self.logger.debug(f"frame-step failed: {e}")

    def step_backward(self) -> None:
        if self._player is None:
            return
        try:
            self._player.command("frame-back-step")
        except Exception as e:
            self.logger.debug(f"frame-back-step failed: {e}")

    def set_speed(self, factor: float) -> None:
        if self._player is None:
            return
        try:
            self._player.speed = max(0.1, min(4.0, float(factor)))
        except Exception:
            pass

    def render_to_buffer(self, w: int, h: int) -> Optional[np.ndarray]:
        """Decode the current frame into a CPU RGBA buffer.
        Returns ``(h, w, 4)`` uint8 ndarray when mpv had a new frame to
        give us, else None (caller skips the texture upload)."""
        if self._ctx is None:
            return None

        self._render_count += 1
        try:
            if not bool(self._ctx.update()):
                return None
        except Exception:
            pass

        w = max(1, int(w))
        h = max(1, int(h))
        stride = w * 4
        if (self._sw_buffer is None or self._sw_w != w or self._sw_h != h
                or not self._sw_buffer.flags['C_CONTIGUOUS']):
            self._sw_buffer = np.zeros((h, w, 4), dtype=np.uint8)
            self._sw_w, self._sw_h = w, h

        ptr = self._sw_buffer.ctypes.data
        try:
            t0 = time.perf_counter()
            self._ctx.render(
                sw_size={"w": w, "h": h},
                sw_format="rgba",
                sw_stride={"stride": stride},
                sw_pointer=ptr,
                block_for_target_time=False,
            )
            dur_ms = (time.perf_counter() - t0) * 1000.0
            self._render_ok_count += 1
            if self._render_ok_count % 60 == 1:
                self.logger.info(
                    f"mpv SW render {w}x{h} took {dur_ms:.1f}ms "
                    f"(ok={self._render_ok_count}/{self._render_count})")
            return self._sw_buffer
        except Exception as e:
            self.logger.debug(f"sw render failed: {e}")
            return None

    @property
    def fps(self) -> float:
        if self._player is None:
            return 0.0
        if self._fps > 0:
            return self._fps
        try:
            val = self._player.container_fps or 0.0
            self._fps = float(val or 0.0)
        except Exception:
            self._fps = 0.0
        return self._fps

    @property
    def total_frames(self) -> int:
        if self._total_frames > 0:
            return self._total_frames
        if self._player is None:
            return 0
        try:
            dur = float(self._player.duration or 0.0)
            self._duration_s = dur
            if dur > 0 and self.fps > 0:
                self._total_frames = int(dur * self.fps)
        except Exception:
            pass
        return self._total_frames

    @property
    def current_frame_index(self) -> int:
        if self._player is None or self.fps <= 0:
            return 0
        try:
            pos = float(self._player.time_pos or 0.0)
        except Exception:
            pos = self._last_time_pos
        return max(0, int(round(pos * self._fps)))

    def register_seek_callback(self, cb: SeekCallback) -> None:
        if cb not in self._seek_callbacks:
            self._seek_callbacks.append(cb)

    def unregister_seek_callback(self, cb: SeekCallback) -> None:
        if cb in self._seek_callbacks:
            self._seek_callbacks.remove(cb)

    def register_playback_state_callback(self, cb: PlaybackStateCallback) -> None:
        if cb not in self._playback_state_callbacks:
            self._playback_state_callbacks.append(cb)

    def unregister_playback_state_callback(self, cb: PlaybackStateCallback) -> None:
        if cb in self._playback_state_callbacks:
            self._playback_state_callbacks.remove(cb)

    def register_position_callback(self, cb: PositionCallback) -> None:
        if cb not in self._position_callbacks:
            self._position_callbacks.append(cb)

    def unregister_position_callback(self, cb: PositionCallback) -> None:
        if cb in self._position_callbacks:
            self._position_callbacks.remove(cb)

    def _install_observers(self) -> None:
        p = self._player
        if p is None:
            return

        @p.property_observer("time-pos")
        def _on_time_pos(_name, value):
            if value is None:
                return
            try:
                self._last_time_pos = float(value)
            except (TypeError, ValueError):
                return
            fps = self._fps
            if fps <= 0:
                for prop in ('container-fps', 'estimated-vf-fps', 'fps'):
                    try:
                        v = self._player[prop]
                    except Exception:
                        v = None
                    try:
                        if v is not None and float(v) > 0:
                            self._fps = float(v)
                            fps = self._fps
                            break
                    except (TypeError, ValueError):
                        continue
            if fps > 0:
                idx = int(round(self._last_time_pos * fps))
                for cb in list(self._position_callbacks):
                    try:
                        cb(idx)
                    except Exception:
                        pass

        @p.property_observer("pause")
        def _on_pause(_name, value):
            if value is None:
                return
            is_playing = not bool(value)
            ts_ms = self._last_time_pos * 1000.0
            for cb in list(self._playback_state_callbacks):
                try:
                    cb(is_playing, ts_ms)
                except Exception:
                    pass

"""libmpv display via the OpenGL render API.

Parallel to ``MpvDisplay`` (SW render) but renders directly into a GL FBO
owned by the caller. The pipeline becomes:

    VideoToolbox decode -> mpv GPU filter/scale -> FBO color attachment

No CPU round-trip, no glTexSubImage2D upload. On 8K VR this is the
difference between ~10 fps and display-cadence smoothness.

Key correctness rules (learned from the previous failed attempt that
produced "clear color only" FBO output):

  * The ``get_proc_address`` callable MUST be wrapped in
    ``mpv.MpvGlGetProcAddressFn`` (CFUNCTYPE) and the wrapper held in a
    strong reference for the lifetime of the render context. A plain
    Python callable gets auto-wrapped at assignment but the wrapper is
    not retained — mpv then resolves every GL symbol to NULL and only
    its fixed-path glClear survives, which looks exactly like the
    previous "only clear color reaches FBO" symptom.
  * The FBO color texture must be ``GL_RGBA8``. ``GL_RGB`` is accepted
    by glCheckFramebufferStatus but mpv's shaders write RGBA and some
    drivers drop the alpha-write, leaving clear-color artifacts.
  * mpv only promises to restore the bound FBO and viewport
    (advanced_control=off, which is the default). The shader program,
    VAO binding, blend/scissor/depth test, active texture, and
    ``glPixelStorei`` state can all be left dirty. Callers that draw
    immediately after ``render_to_fbo`` should re-bind what they need.

API compatibility with ``MpvDisplay`` (SW) is deliberate: everything is
the same except ``render_to_buffer`` is replaced by ``render_to_fbo``.
"""

from __future__ import annotations

import ctypes
import logging
import threading
import time
from typing import Callable, List, Optional

from video.mpv_loader import mpv, mpv_available


SeekCallback = Callable[[int], None]
PlaybackStateCallback = Callable[[bool, float], None]
PositionCallback = Callable[[int], None]


class MpvDisplayGL:

    def __init__(
        self,
        video_path: str,
        vf: Optional[str] = None,
        hwdec: str = "auto",
        with_audio: bool = True,
        display_fps: float = 60.0,
        logger: Optional[logging.Logger] = None,
    ):
        self.video_path = video_path
        self.vf = vf or ""
        self.hwdec = hwdec
        self.with_audio = with_audio
        self.display_fps = max(30.0, float(display_fps))
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._player = None
        self._ctx = None
        self._video_sync_downgraded = False

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

        self._render_count = 0
        self._render_ok_count = 0
        self._last_render_ms: float = 0.0

        self._proc_address_cfunc = None

        self._new_frame_pending = True

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

    def open(self, get_proc_address: Callable[[object, bytes], int]) -> bool:
        """Create the mpv player and OpenGL render context."""
        if not mpv_available:
            self.logger.warning("libmpv not available; MpvDisplayGL disabled")
            return False
        if self._player is not None:
            return True
        if get_proc_address is None:
            self.logger.error("MpvDisplayGL.open: get_proc_address is required")
            return False

        try:
            def _mpv_log(level, prefix, msg):
                line = f"[mpv:{prefix}] {msg.rstrip() if isinstance(msg, str) else msg}"
                if level in ("fatal", "error"):
                    self.logger.error(line)
                elif level == "warn":
                    text = msg.lower() if isinstance(msg, str) else ""
                    # Demote mpv's desync chatter to debug; we auto-recover.
                    if "desynchroni" in text or "a/v status" in text:
                        self.logger.debug(line)
                        if not self._video_sync_downgraded:
                            try:
                                self._player["video-sync"] = "audio"
                                self._video_sync_downgraded = True
                                self.logger.info(
                                    "[mpv] video-sync switched to audio (auto-recover)")
                            except Exception:
                                pass
                    else:
                        self.logger.warning(line)
                else:
                    self.logger.debug(line)

            kwargs = {
                "vo": "libmpv",
                "audio": "auto" if self.with_audio else "no",
                "keep_open": "yes",
                "pause": "yes",
                "hwdec": self.hwdec,
                # gpu-hq: ewa_lanczossharp + mitchell
                # + linear-light upscaling. Cost is acceptable when the FBO
                # is viewport-sized (ss=1.0); the previous fail was ss=2.0
                # which made the FBO 4x and starved 8K.
                "profile": "gpu-hq",
                "video_sync": "display-resample",
                "display_fps_override": str(self.display_fps),
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
            # Pin the cfunc so libmpv's callback doesn't see a GC'd wrapper.
            self._proc_address_cfunc = mpv.MpvGlGetProcAddressFn(get_proc_address)
            self._ctx = mpv.MpvRenderContext(
                self._player,
                "opengl",
                opengl_init_params={"get_proc_address": self._proc_address_cfunc},
            )
        except Exception as e:
            self.logger.error(f"MpvRenderContext (opengl) init failed: {e}")
            self._cleanup()
            return False

        def _on_update():
            self._new_frame_pending = True
        try:
            self._ctx.update_cb = _on_update
        except Exception as e:
            self.logger.debug(f"update_cb install failed: {e}")

        self._install_observers()
        self.logger.debug("MpvDisplayGL ready (OpenGL render API)")
        return True

    def load(self, video_path: str, vf: Optional[str] = None,
             hwdec: Optional[str] = None) -> bool:
        if self._player is None:
            self.logger.warning("MpvDisplayGL.load called before open()")
            self._last_load_error = "open() not called"
            return False
        self._fps = 0.0
        self._total_frames = 0
        self._duration_s = 0.0
        self._last_time_pos = 0.0
        self._loaded = False
        self._last_load_error = None
        self.video_path = video_path
        if hwdec is not None and hwdec != self.hwdec:
            try:
                self._player["hwdec"] = hwdec
                self.hwdec = hwdec
            except Exception as e:
                self.logger.debug(f"set hwdec={hwdec} failed: {e}")
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
                try:
                    self._player.pause = True
                    self._is_paused = True
                    self._player.command("seek", 0.0, "absolute", "keyframes")
                    # Prime the decoder so the first render_to_fbo has a frame.
                    self._player.command("frame-step")
                except Exception as e:
                    self.logger.debug(f"post-load pause/seek failed: {e}")
                self._loaded = True
                self._new_frame_pending = True
                self.logger.info(
                    f"Video display ready: {video_path} "
                    f"(duration={self._duration_s:.2f}s, fps={self._fps:.3f})")
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
        self.logger.error(f"MpvDisplayGL.load: {self._last_load_error} for {video_path}")
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
            self.logger.info(f"MpvDisplayGL fps fallback applied: {f:.3f}")

    def close(self) -> None:
        self._cleanup()

    def _cleanup(self) -> None:
        with self._state_lock:
            if self._ctx is not None:
                try:
                    self._ctx.update_cb = None
                except Exception:
                    pass
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
        self._proc_address_cfunc = None

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    def _async_cmd(self, *args) -> bool:
        # Fire-and-forget mpv command. command_async returns a Future the
        # event loop fulfils; we drop the reference so the UI thread never
        # waits on libmpv. Falls back to sync command on older mpv builds
        # that don't expose command_async.
        p = self._player
        if p is None:
            return False
        try:
            ca = getattr(p, "command_async", None)
            if ca is not None:
                ca(*args)
            else:
                p.command(*args)
            return True
        except Exception as e:
            self.logger.debug(f"async cmd {args!r} failed: {e}")
            return False

    def play(self) -> None:
        if self._player is None:
            return
        self._is_paused = False
        self._async_cmd("set", "pause", "no")

    def pause(self) -> None:
        if self._player is None:
            return
        self._is_paused = True
        self._async_cmd("set", "pause", "yes")

    @property
    def is_paused(self) -> bool:
        return self._is_paused

    def seek(self, time_seconds: float, exact: bool = True) -> None:
        if self._player is None:
            return
        precision = "exact" if exact else "keyframes"
        if not self._async_cmd("seek", float(time_seconds), "absolute", precision):
            return
        # Force a render on the next tick; mpv's update_cb won't always
        # fire immediately on an exact-seek to a keyframe close to pos.
        self._new_frame_pending = True
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
        if self._async_cmd("frame-step"):
            self._new_frame_pending = True

    def step_backward(self) -> None:
        if self._player is None:
            return
        if self._async_cmd("frame-back-step"):
            self._new_frame_pending = True

    def set_speed(self, factor: float) -> None:
        if self._player is None:
            return
        clamped = max(0.1, min(4.0, float(factor)))
        self._async_cmd("set", "speed", str(clamped))

    def render_to_fbo(self, fbo_id: int, w: int, h: int) -> bool:
        """Render mpv's current frame into the caller-owned FBO.

        Callers must restore GL state (shader/VAO/blend/etc.) after the call.
        """
        if self._ctx is None:
            return False
        if not fbo_id or fbo_id == 0:
            return False

        self._render_count += 1
        self._new_frame_pending = False

        w = max(1, int(w))
        h = max(1, int(h))
        try:
            t0 = time.perf_counter()
            self._ctx.render(
                opengl_fbo={"w": w, "h": h, "fbo": int(fbo_id)},
                flip_y=False,
                block_for_target_time=False,
            )
            self._last_render_ms = (time.perf_counter() - t0) * 1000.0
            self._render_ok_count += 1
            if self._render_ok_count % 120 == 1:
                self.logger.debug(
                    f"mpv GL render {w}x{h} took {self._last_render_ms:.1f}ms "
                    f"(ok={self._render_ok_count}/{self._render_count})")
            return True
        except Exception as e:
            self.logger.debug(f"gl render failed: {e}")
            return False

    def report_swap(self) -> None:
        """Call after glfw.swap_buffers so framedrop=vo sees actual presents."""
        if self._ctx is None:
            return
        try:
            self._ctx.report_swap()
        except Exception:
            pass

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
            # time-pos advance means a new frame is in mpv's pipeline.
            # update_cb does not always fire on paused frame-step / exact
            # seek (see seek()), so the host loop would otherwise sit on a
            # stale FBO until something else triggers a re-render.
            self._new_frame_pending = True
            fps = self._fps
            # Hot path first: skip the property fetches once fps is known.
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
                if fps <= 0:
                    return
            idx = int(round(self._last_time_pos * fps))
            cbs = self._position_callbacks
            if not cbs:
                return
            # Snapshot only when there are multiple callbacks (rare); single
            # callback (common) path skips the tuple alloc entirely.
            iter_cbs = cbs if len(cbs) == 1 else tuple(cbs)
            for cb in iter_cbs:
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

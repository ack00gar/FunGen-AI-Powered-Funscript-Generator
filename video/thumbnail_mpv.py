"""Dedicated mpv instance for thumbnail rendering.

Replaces ~150-300ms ffmpeg-subprocess spawn per thumbnail with ~20-80ms
seek+decode on a warm decoder. Runs a request queue processed by the
main GL thread so background callers (hover preview, chapter cache) can
submit requests safely.
"""
from __future__ import annotations

import collections
import logging
import platform
import threading
import time
from typing import Callable, Optional

import numpy as np

import OpenGL.GL as gl

from video.mpv_loader import mpv_available

if mpv_available:
    import mpv


log = logging.getLogger(__name__)


def _default_hwdec() -> str:
    # auto-safe lets mpv probe + fall back when a specific backend can't
    # negotiate (8K h264 + videotoolbox failed at init and dropped to SW).
    return "auto-safe"


class ThumbnailMpv:
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger or log
        self._player = None
        self._ctx = None
        self._proc_address_cfunc = None
        self._fbo = 0
        self._texture = 0
        self._fbo_w = 0
        self._fbo_h = 0
        self._video_path = ""
        self._duration_s = 0.0

    @property
    def is_loaded(self) -> bool:
        return self._player is not None and bool(self._video_path)

    def open(self, get_proc_address: Callable[[object, bytes], int]) -> bool:
        if not mpv_available:
            return False
        try:
            self._player = mpv.MPV(
                vo="libmpv",
                audio="no",
                ao="null",
                hwdec=_default_hwdec(),
                keep_open="yes",
                pause="yes",
                profile="fast",
                loglevel="warn",
            )
            self._proc_address_cfunc = mpv.MpvGlGetProcAddressFn(get_proc_address)
            self._ctx = mpv.MpvRenderContext(
                self._player, "opengl",
                opengl_init_params={"get_proc_address": self._proc_address_cfunc},
            )
            self.logger.info("ThumbnailMpv ready")
            return True
        except Exception as e:
            self.logger.warning(f"ThumbnailMpv open failed: {e}")
            self._player = None
            self._ctx = None
            return False

    def load(self, video_path: str) -> bool:
        if self._player is None:
            return False
        try:
            self._player.play(video_path)
            deadline = time.monotonic() + 3.0
            while time.monotonic() < deadline:
                try:
                    d = self._player.duration
                except Exception:
                    d = None
                if d is not None and d > 0:
                    self._duration_s = float(d)
                    self._video_path = video_path
                    try:
                        self._player.pause = True
                        self._player.command("seek", 0.0, "absolute", "keyframes")
                        self._player.command("frame-step")
                    except Exception:
                        pass
                    self.logger.info(
                        f"ThumbnailMpv loaded {video_path} "
                        f"(duration={self._duration_s:.2f}s)")
                    return True
                time.sleep(0.02)
            self.logger.warning(
                f"ThumbnailMpv: duration timeout loading {video_path}")
            return False
        except Exception as e:
            self.logger.warning(f"ThumbnailMpv load failed: {e}")
            return False

    def _alloc_fbo(self, w: int, h: int) -> None:
        if w == self._fbo_w and h == self._fbo_h and self._fbo:
            return
        if self._texture:
            try:
                gl.glDeleteTextures(1, [self._texture])
            except Exception:
                pass
        if self._fbo:
            try:
                gl.glDeleteFramebuffers(1, [self._fbo])
            except Exception:
                pass
        self._texture = int(gl.glGenTextures(1))
        gl.glBindTexture(gl.GL_TEXTURE_2D, self._texture)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MIN_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_MAG_FILTER, gl.GL_LINEAR)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_S, gl.GL_CLAMP_TO_EDGE)
        gl.glTexParameteri(gl.GL_TEXTURE_2D, gl.GL_TEXTURE_WRAP_T, gl.GL_CLAMP_TO_EDGE)
        gl.glTexImage2D(gl.GL_TEXTURE_2D, 0, gl.GL_RGBA8, w, h, 0,
                        gl.GL_RGBA, gl.GL_UNSIGNED_BYTE, None)
        gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
        self._fbo = int(gl.glGenFramebuffers(1))
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
        gl.glFramebufferTexture2D(gl.GL_FRAMEBUFFER, gl.GL_COLOR_ATTACHMENT0,
                                  gl.GL_TEXTURE_2D, self._texture, 0)
        gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
        self._fbo_w, self._fbo_h = w, h

    def _reset_gl_state_before(self) -> None:
        try:
            gl.glDisable(gl.GL_BLEND)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glDisable(gl.GL_CULL_FACE)
            gl.glDisable(gl.GL_STENCIL_TEST)
            gl.glColorMask(True, True, True, True)
            gl.glDepthMask(True)
            gl.glUseProgram(0)
            gl.glBindVertexArray(0)
            gl.glBindBuffer(gl.GL_ARRAY_BUFFER, 0)
            gl.glBindBuffer(gl.GL_ELEMENT_ARRAY_BUFFER, 0)
            gl.glActiveTexture(gl.GL_TEXTURE0)
            gl.glBindTexture(gl.GL_TEXTURE_2D, 0)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 4)
        except Exception:
            pass

    def _reset_gl_state_after(self) -> None:
        try:
            gl.glBindVertexArray(0)
            gl.glUseProgram(0)
            gl.glDisable(gl.GL_SCISSOR_TEST)
            gl.glDisable(gl.GL_DEPTH_TEST)
            gl.glPixelStorei(gl.GL_UNPACK_ALIGNMENT, 4)
        except Exception:
            pass

    def render_frame(self, time_sec: float,
                     w: int = 320, h: int = 320) -> Optional[np.ndarray]:
        """Seek to time_sec and render one frame. Must run on main GL thread."""
        if not self.is_loaded or self._ctx is None:
            return None
        try:
            t0 = time.monotonic()
            t = max(0.0, min(float(time_sec), self._duration_s - 0.01))
            self._player.command("seek", t, "absolute", "keyframes")
            try:
                self._player.command("frame-step")
            except Exception:
                pass
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                try:
                    cur = self._player.time_pos
                except Exception:
                    cur = None
                if cur is not None:
                    break
                time.sleep(0.005)
            self._alloc_fbo(w, h)
            self._reset_gl_state_before()
            try:
                self._ctx.render(
                    opengl_fbo={"w": w, "h": h, "fbo": int(self._fbo)},
                    flip_y=False,
                    block_for_target_time=False,
                )
            finally:
                self._reset_gl_state_after()
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, self._fbo)
            gl.glPixelStorei(gl.GL_PACK_ALIGNMENT, 1)
            buf = gl.glReadPixels(0, 0, w, h, gl.GL_BGR, gl.GL_UNSIGNED_BYTE)
            gl.glBindFramebuffer(gl.GL_FRAMEBUFFER, 0)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3)
            elapsed_ms = (time.monotonic() - t0) * 1000.0
            self.logger.debug(
                f"ThumbnailMpv render {w}x{h} @ {t:.2f}s took {elapsed_ms:.1f}ms")
            return arr
        except Exception as e:
            self.logger.debug(f"ThumbnailMpv render_frame failed: {e}")
            return None

    def close(self) -> None:
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
        if self._texture:
            try:
                gl.glDeleteTextures(1, [self._texture])
            except Exception:
                pass
            self._texture = 0
        if self._fbo:
            try:
                gl.glDeleteFramebuffers(1, [self._fbo])
            except Exception:
                pass
            self._fbo = 0
        self._video_path = ""


class ThumbnailMpvQueue:
    """Serialize thumbnail requests from any thread onto the main GL thread."""

    _MAX_PER_TICK = 1
    _DEFAULT_TIMEOUT_S = 4.0

    def __init__(self, thumb_mpv: ThumbnailMpv):
        self._mpv = thumb_mpv
        self._queue: collections.deque = collections.deque()
        self._lock = threading.Lock()

    def request(self, time_sec: float, w: int = 320, h: int = 320,
                timeout: float = _DEFAULT_TIMEOUT_S) -> Optional[np.ndarray]:
        if not self._mpv.is_loaded:
            return None
        event = threading.Event()
        container: list = [None]
        with self._lock:
            self._queue.append((float(time_sec), int(w), int(h), event, container))
        if not event.wait(timeout=timeout):
            return None
        return container[0]

    def tick(self) -> None:
        """Process up to _MAX_PER_TICK pending requests on the main GL thread."""
        for _ in range(self._MAX_PER_TICK):
            with self._lock:
                if not self._queue:
                    return
                req = self._queue.popleft()
            time_sec, w, h, event, container = req
            try:
                container[0] = self._mpv.render_frame(time_sec, w, h)
            except Exception:
                container[0] = None
            event.set()

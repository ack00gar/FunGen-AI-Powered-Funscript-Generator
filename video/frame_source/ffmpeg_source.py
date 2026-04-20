"""FFmpeg-subprocess video frame source.

One ffmpeg process per open video:

    ffmpeg -hwaccel ... -ss <time> -i <path> -an -sn -vf <filter_chain>
           -pix_fmt bgr24 -f rawvideo pipe:1

Reads exactly ``output_w * output_h * 3`` bytes per frame from the pipe and
advances an integer frame counter. Seek kills the subprocess and restarts
it with a new ``-ss``. Pause freezes the reader thread; ffmpeg blocks on
pipe backpressure naturally.

Invariants:
- input-level ``-ss`` before ``-i`` for fast keyframe-aligned seek
- stderr drained in a background thread so the pipe buffer never blocks
- SIGTERM + timeout + SIGKILL on close
- hwaccel args provided by the integration layer
"""

from __future__ import annotations

import logging
import queue
import subprocess
import threading
import time
from collections import OrderedDict
from typing import List, Optional, Tuple

import numpy as np

from video.ffmpeg_helpers import find_ffmpeg, subprocess_flags
from video.frame_source._types import (
    SourceConfig, SeekCallback, PlaybackStateCallback, PositionCallback,
    _EOS, _FLUSH,
)
from video.frame_source.probe import probe as _probe


class FFmpegFrameSource:
    """Subprocess-backed video frame source."""

    def __init__(self, config: SourceConfig, logger: Optional[logging.Logger] = None,
                 hwaccel_args: Optional[List[str]] = None):
        self.cfg = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        # Hwaccel is passed in from the integration layer so we don't reach
        # into app settings from here. Empty list = CPU decode.
        self._hwaccel_args: List[str] = list(hwaccel_args or [])

        # Stream metadata (populated on open via ffprobe).
        self._fps: float = 0.0
        self._time_base = None  # kept for API parity; not used on this path
        self._total_frames: int = 0
        self._duration_seconds: float = 0.0

        # Byte count per output frame. The filter chain MUST produce frames
        # at output_w x output_h or the pipe reads will desync catastrophically.
        self._frame_bytes: int = config.output_w * config.output_h * 3

        self._current_frame: Optional[np.ndarray] = None
        self._current_frame_index: int = -1
        self._frame_version: int = 0
        self._frame_lock = threading.Lock()

        # Decode thread + frame queue. maxsize=32 balances buffer memory
        # (32 * output_w * output_h * 3 bytes, ~38 MB at 640x640) against
        # hand-off jitter when the consumer is doing heavy work per frame
        # (e.g., YOLO inference that varies 10-40 ms). Smaller queues caused
        # 30-50% throughput loss in jittery-consumer workloads.
        self._decode_thread: Optional[threading.Thread] = None
        self._frame_queue: "queue.Queue" = queue.Queue(maxsize=32)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # SET = paused

        self._seek_lock = threading.Lock()
        self._seek_target: Optional[int] = None
        self._seek_accurate: bool = False
        self._seek_done_event = threading.Event()
        self._seek_done_event.set()

        # Subprocess + stderr drainer.
        self._proc: Optional[subprocess.Popen] = None
        self._proc_lock = threading.Lock()
        self._stderr_thread: Optional[threading.Thread] = None
        self._stderr_tail: List[str] = []

        # Where the current ffmpeg invocation started decoding from. Frame
        # index = _stream_start_frame + frames_read_since_spawn.
        self._stream_start_frame: int = 0
        self._frames_read_in_stream: int = 0

        # Callbacks.
        self._seek_callbacks: List[SeekCallback] = []
        self._playback_state_callbacks: List[PlaybackStateCallback] = []
        self._position_callbacks: List[PositionCallback] = []

        self._eos_reached: bool = False

        # Scrub cache: frame_index -> decoded np.ndarray, LRU-bounded.
        # The hover / arrow-nav scrub path repeatedly seeks to nearby frames;
        # each seek costs an ffmpeg respawn (~50-200 ms). Keeping the last N
        # decoded frames around lets get_frame() short-circuit without killing
        # the subprocess. Cleared on reapply_settings (filter chain change
        # invalidates cached pixel data).
        self._scrub_cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._scrub_cache_max: int = 16
        self._scrub_cache_lock = threading.Lock()
        self._scrub_cache_hits: int = 0
        self._scrub_cache_misses: int = 0

    # --------------------------------------------------------------- lifecycle

    def open(self) -> bool:
        p = _probe(self.cfg.video_path)
        if p is None:
            self.logger.error(f"ffprobe failed for {self.cfg.video_path}")
            return False
        self._fps = p.fps
        self._total_frames = p.total_frames
        self._duration_seconds = p.duration_sec
        self.logger.debug(
            f"FFmpeg source opened {self.cfg.video_path}: "
            f"{p.width}x{p.height} fps={self._fps:.3f} frames={self._total_frames}"
        )
        return True

    def close(self) -> None:
        self.stop()

    def reapply_settings(self, new_config: Optional[SourceConfig] = None) -> bool:
        """Apply a new filter chain / output size without losing position."""
        cur = max(0, self._current_frame_index)
        was_running = self.is_running
        was_paused = self._pause_event.is_set()
        self.stop()
        if new_config is not None:
            self.cfg = new_config
            self._frame_bytes = new_config.output_w * new_config.output_h * 3
        # Filter chain or output dims changed -> cached frames are stale.
        self._cache_clear()
        if was_running:
            self.start(cur)
            if was_paused:
                self._pause_event.set()
        return True

    # ------------------------------------------------------------ scrub cache

    def _cache_put(self, idx: int, frame: np.ndarray) -> None:
        if frame is None or idx < 0:
            return
        with self._scrub_cache_lock:
            if idx in self._scrub_cache:
                self._scrub_cache.move_to_end(idx)
                return
            self._scrub_cache[idx] = frame
            while len(self._scrub_cache) > self._scrub_cache_max:
                self._scrub_cache.popitem(last=False)

    def _cache_get(self, idx: int) -> Optional[np.ndarray]:
        with self._scrub_cache_lock:
            f = self._scrub_cache.get(idx)
            if f is not None:
                self._scrub_cache.move_to_end(idx)
                self._scrub_cache_hits += 1
            else:
                self._scrub_cache_misses += 1
            return f

    def _cache_clear(self) -> None:
        with self._scrub_cache_lock:
            self._scrub_cache.clear()

    def set_scrub_cache_size(self, size: int) -> None:
        """Resize the scrub cache. 0 disables it."""
        with self._scrub_cache_lock:
            self._scrub_cache_max = max(0, int(size))
            while len(self._scrub_cache) > self._scrub_cache_max:
                self._scrub_cache.popitem(last=False)

    @property
    def scrub_cache_stats(self) -> dict:
        with self._scrub_cache_lock:
            total = self._scrub_cache_hits + self._scrub_cache_misses
            hit_rate = (self._scrub_cache_hits / total) if total else 0.0
            return {
                "size": len(self._scrub_cache),
                "max": self._scrub_cache_max,
                "hits": self._scrub_cache_hits,
                "misses": self._scrub_cache_misses,
                "hit_rate": hit_rate,
            }

    # -------------------------------------------------------------- transport

    def start(self, start_frame: int = 0) -> None:
        if self.is_running:
            return
        self._stop_event.clear()
        self._pause_event.clear()
        self._eos_reached = False
        self._drain_queue()

        # Prime the seek so the loop spawns ffmpeg at the right frame.
        with self._seek_lock:
            self._seek_target = max(0, start_frame)
            self._seek_done_event.clear()

        self._decode_thread = threading.Thread(
            target=self._decode_loop, daemon=True, name="FFmpegDecode")
        self._decode_thread.start()
        self._notify_state(is_playing=True)

    def stop(self) -> None:
        if self._decode_thread is None or not self._decode_thread.is_alive():
            self._terminate_proc()
            self._notify_state(is_playing=False)
            return
        self._stop_event.set()
        self._pause_event.clear()
        self._drain_queue()
        try: self._frame_queue.put_nowait(_EOS)
        except queue.Full: pass
        self._terminate_proc()
        self._decode_thread.join(timeout=3.0)
        self._decode_thread = None
        self._notify_state(is_playing=False)

    def pause(self) -> None:
        self._pause_event.set()
        self._notify_state(is_playing=False)

    def resume(self) -> None:
        self._pause_event.clear()
        self._notify_state(is_playing=True)

    @property
    def is_paused(self) -> bool:
        return self._pause_event.is_set()

    @property
    def is_running(self) -> bool:
        return self._decode_thread is not None and self._decode_thread.is_alive()

    # ------------------------------------------------------------------ seek

    def seek(self, frame_index: int, accurate: bool = False) -> None:
        target = max(0, min(int(frame_index), max(0, self._total_frames - 1)))
        with self._seek_lock:
            self._seek_target = target
            self._seek_accurate = accurate
            self._seek_done_event.clear()
        # Previous EOS may have been a bogus one from a transient filter chain
        # failure; give the new seek a fresh chance to deliver frames.
        self._eos_reached = False
        self._drain_queue()
        # Wake the decode loop's read by killing the current subprocess.
        # Must NOT block the caller: a synchronous proc.wait() inside seek()
        # would stall every hover scrub. Fire-and-forget SIGKILL; the decode
        # loop's _spawn() will reap the zombie via its own (short-timeout)
        # wait before spawning the replacement.
        with self._proc_lock:
            proc = self._proc
        if proc is not None and proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        self._notify_seek(target)

    def wait_seek(self, timeout: float = 2.0) -> bool:
        return self._seek_done_event.wait(timeout=timeout)

    def get_frame(self, frame_index: int, timeout: float = 2.0,
                  accurate: bool = True) -> Optional[np.ndarray]:
        """Synchronous random-access fetch (scrub preview, arrow-nav).

        ``timeout`` is the overall budget: wait_seek plus queue.get share it
        so a slow keyframe decode can't double-charge the caller.

        Scrub cache: on a hit we short-circuit and never kill the ffmpeg
        subprocess; that is the main performance win for hover-over-timeline
        preview, which would otherwise respawn ffmpeg on every pixel.
        """
        cached = self._cache_get(frame_index)
        if cached is not None:
            self._publish(frame_index, cached)
            return cached

        if not self.is_running:
            frame = self._oneshot_decode_at(frame_index)
            if frame is not None:
                self._cache_put(frame_index, frame)
            return frame
        deadline = time.monotonic() + max(0.05, timeout)
        self.seek(frame_index, accurate=accurate)
        remaining = max(0.01, deadline - time.monotonic())
        if not self.wait_seek(timeout=remaining):
            return None
        remaining = max(0.01, deadline - time.monotonic())
        try:
            item = self._frame_queue.get(timeout=remaining)
        except queue.Empty:
            return None
        # Decode loop may have queued an EOS sentinel (subprocess died, filter
        # chain failed). Don't try to unpack it.
        if item is _EOS:
            self._eos_reached = True
            return None
        if item is _FLUSH:
            return None
        idx, frame = item
        self._publish(idx, frame)
        # Key the cache by the REQUESTED frame_index (not the decoded idx):
        # ffmpeg keyframe alignment can return a nearby frame whose index
        # differs from the caller's request, and the caller would then miss
        # on a straight-line revisit of their own request. Keying by the
        # request makes the scrub cache actually hit on hover revisits.
        self._cache_put(frame_index, frame)
        if idx != frame_index and idx >= 0:
            self._cache_put(idx, frame)
        return frame

    # ---------------------------------------------------------------- consume

    def next_frame(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        try:
            item = self._frame_queue.get(timeout=timeout)
        except queue.Empty:
            return None
        if item is _EOS:
            self._eos_reached = True
            return None
        if item is _FLUSH:
            return self.next_frame(timeout=timeout)
        idx, frame = item
        self._publish(idx, frame)
        self._cache_put(idx, frame)
        return idx, frame

    @property
    def is_eos(self) -> bool:
        return self._eos_reached

    @property
    def current_frame(self) -> Optional[np.ndarray]:
        with self._frame_lock:
            return self._current_frame

    @property
    def current_frame_index(self) -> int:
        return self._current_frame_index

    @property
    def frame_version(self) -> int:
        return self._frame_version

    @property
    def fps(self) -> float:
        return self._fps

    @property
    def total_frames(self) -> int:
        return self._total_frames

    # -------------------------------------------------------------- callbacks

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

    # ----------------------------------------------------------------- subprocess

    def _filter_chain(self) -> str:
        """The ffmpeg -vf chain, always terminated by format=bgr24."""
        chain = self.cfg.filter_chain.strip().strip(",") if self.cfg.filter_chain else ""
        if "format=" not in chain:
            chain = f"{chain},format=bgr24" if chain else "format=bgr24"
        return chain

    def _build_cmd(self, start_frame: int) -> List[str]:
        """ffmpeg command: hwaccel, input seek, filter chain, raw BGR24 out.

        -ss before -i is input-level seek (fast, keyframe-aligned). This
        matches v0.8.0 and is correct for interactive scrub. Frame-exact seek
        would need -ss after -i, but that decodes from stream start and costs
        seconds on long files.
        """
        t_s = (start_frame / self._fps) if self._fps > 0 else 0.0
        cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "warning", "-nostats"]
        cmd.extend(self._hwaccel_args)
        if t_s > 0.001:
            cmd.extend(["-ss", f"{t_s:.6f}"])
        cmd.extend([
            "-i", self.cfg.video_path,
            "-an", "-sn",
            "-vf", self._filter_chain(),
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "pipe:1",
        ])
        return cmd

    def _spawn(self, start_frame: int) -> bool:
        """Start a fresh ffmpeg subprocess streaming from ``start_frame``."""
        self._terminate_proc()
        cmd = self._build_cmd(start_frame)
        self.logger.debug("ffmpeg spawn: " + " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                bufsize=-1,
                creationflags=subprocess_flags(),
            )
        except FileNotFoundError:
            self.logger.error("ffmpeg binary not found")
            return False
        with self._proc_lock:
            self._proc = proc
            self._stream_start_frame = start_frame
            self._frames_read_in_stream = 0
        # NB: subprocess health is checked post-read, not here. Adding a
        # sleep/poll here would tax every seek with extra latency.

        # Drain stderr to a tail buffer so the pipe never blocks the encoder.
        self._stderr_tail = []
        def _drain():
            if proc.stderr is None:
                return
            try:
                for raw in iter(proc.stderr.readline, b""):
                    line = raw.decode("utf-8", errors="replace").rstrip()
                    if line:
                        self._stderr_tail.append(line)
                        if len(self._stderr_tail) > 200:
                            del self._stderr_tail[:100]
            except Exception:
                pass
        self._stderr_thread = threading.Thread(
            target=_drain, daemon=True, name="FFmpegStderr")
        self._stderr_thread.start()
        return True

    def _terminate_proc(self) -> None:
        """Aggressive shutdown: SIGKILL, short wait, close pipes.

        We kill (not terminate) because we don't care about the subprocess
        finishing cleanly - its output is raw rgb frames that we stopped
        reading, no file being muxed, nothing to lose. ffmpeg's SIGTERM
        handling can be slow (~hundreds of ms to seconds) as it tries to
        flush its output; SIGKILL is instant. Matters for seek latency
        because every seek tears down the current subprocess.
        """
        with self._proc_lock:
            proc = self._proc
            self._proc = None
        if proc is None:
            return
        if proc.poll() is None:
            try:
                proc.kill()
            except Exception:
                pass
        try:
            proc.wait(timeout=0.5)
        except subprocess.TimeoutExpired:
            pass
        self._close_pipes(proc)

    @staticmethod
    def _close_pipes(proc: subprocess.Popen) -> None:
        for pipe in (proc.stdout, proc.stderr, proc.stdin):
            if pipe is None:
                continue
            try: pipe.close()
            except Exception: pass

    def _read_exact(self, n: int) -> Optional[bytes]:
        """Read exactly ``n`` bytes from the current subprocess stdout.

        Returns None on EOS, stop request, or subprocess death. On the
        happy path Python's BufferedReader returns the full ``n`` bytes
        in a single read() call (it loops internally), so we avoid the
        bytearray+extend+bytes copy round-trip from a hand-rolled loop.
        """
        with self._proc_lock:
            proc = self._proc
        if proc is None or proc.stdout is None:
            return None
        try:
            buf = proc.stdout.read(n)
        except (ValueError, OSError):
            return None
        if self._stop_event.is_set() or not buf or len(buf) < n:
            return None
        return buf

    # ----------------------------------------------------------------- decode loop

    def _decode_loop(self) -> None:
        """Background producer: handles seeks then streams filtered frames."""
        try:
            while not self._stop_event.is_set():
                # Pause: block here. Seeks during pause still win because
                # they set _seek_target and terminate the proc (which wakes
                # a blocked read and returns None).
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    if self._seek_target is not None:
                        break
                    time.sleep(0.01)
                if self._stop_event.is_set():
                    break

                # Honor pending seek by spawning a fresh subprocess there.
                with self._seek_lock:
                    target = self._seek_target
                    self._seek_target = None
                if target is not None:
                    self._drain_queue()
                    if not self._spawn(target):
                        self._frame_queue.put(_EOS)
                        self._seek_done_event.set()
                        return
                    # Read one frame before releasing the seek event, so
                    # wait_seek() proves a frame actually made it through.
                    t_spawn = time.perf_counter()
                    raw = self._read_exact(self._frame_bytes)
                    read_ms = (time.perf_counter() - t_spawn) * 1000.0
                    if raw is None:
                        with self._proc_lock:
                            p = self._proc
                        proc_dead = p is None or p.poll() is not None
                        # Another seek arrived while we were reading (rapid
                        # scrubbing): our seek() killed this proc on purpose.
                        # Don't kill the decode loop; let the next iteration
                        # pick up the new seek target.
                        if self._seek_target is not None or self._stop_event.is_set():
                            self._seek_done_event.set()
                            continue
                        if proc_dead:
                            # ffmpeg genuinely exited. Drain stderr and report.
                            if self._stderr_thread is not None:
                                self._stderr_thread.join(timeout=0.2)
                            rc = p.returncode if p is not None else "N/A"
                            tail = "\n  ".join(self._stderr_tail[-10:]) or "(stderr empty)"
                            self.logger.error(
                                f"ffmpeg exited before first frame at seek={target} "
                                f"(rc={rc}, read_wait={read_ms:.0f}ms). stderr:\n  {tail}")
                            self._frame_queue.put(_EOS)
                            self._seek_done_event.set()
                            return
                        # Proc alive but read returned None without another
                        # seek/stop triggering. Unusual; log and recover.
                        self.logger.debug(
                            f"post-seek read None at {target} but proc alive "
                            f"(stop={self._stop_event.is_set()}, "
                            f"new_seek={self._seek_target is not None})")
                    else:
                        try:
                            frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                                (self.cfg.output_h, self.cfg.output_w, 3))
                            self._frame_queue.put((target, frame))
                            self._frames_read_in_stream = 1
                        except ValueError as e:
                            self.logger.error(f"post-seek reshape failed: {e}")
                            self._frame_queue.put(_EOS)
                            self._seek_done_event.set()
                            return
                    self._seek_done_event.set()
                    if self._pause_event.is_set():
                        continue

                # Steady state: read the next frame bytes from stdout.
                raw = self._read_exact(self._frame_bytes)
                if raw is None:
                    # Subprocess ended or we were interrupted. If no seek is
                    # pending and we're not stopping, this is genuine EOS.
                    if (self._seek_target is None
                            and not self._stop_event.is_set()
                            and not self._pause_event.is_set()):
                        self._frame_queue.put(_EOS)
                        return
                    continue

                try:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.cfg.output_h, self.cfg.output_w, 3))
                except ValueError as e:
                    self.logger.error(
                        f"Frame reshape failed (expected {self.cfg.output_w}x"
                        f"{self.cfg.output_h}, got {len(raw)} bytes): {e}")
                    self._frame_queue.put(_EOS)
                    return

                idx = self._stream_start_frame + self._frames_read_in_stream
                self._frames_read_in_stream += 1
                # Queue the frame. Block on backpressure; consumer drains.
                while not self._stop_event.is_set():
                    try:
                        self._frame_queue.put((idx, frame), timeout=0.1)
                        break
                    except queue.Full:
                        if self._seek_target is not None:
                            break
        except Exception as e:
            self.logger.error(f"decode loop crashed: {e}", exc_info=True)
            self._frame_queue.put(_EOS)
        finally:
            self._terminate_proc()

    # ----------------------------------------------------------------- one-shot

    def _oneshot_decode_at(self, frame_index: int,
                           timeout_s: float = 10.0) -> Optional[np.ndarray]:
        """Single-frame fetch without a running decode thread.

        Fresh ffmpeg invocation producing exactly one frame at the target.
        Used by scrub preview / arrow nav when playback isn't running.
        """
        t_s = (frame_index / self._fps) if self._fps > 0 else 0.0
        cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostats"]
        cmd.extend(self._hwaccel_args)
        if t_s > 0.001:
            cmd.extend(["-ss", f"{t_s:.6f}"])
        cmd.extend([
            "-i", self.cfg.video_path,
            "-an", "-sn",
            "-vf", self._filter_chain(),
            "-frames:v", "1",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo", "pipe:1",
        ])
        try:
            result = subprocess.run(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=timeout_s, creationflags=subprocess_flags(),
            )
        except (OSError, subprocess.TimeoutExpired) as e:
            self.logger.debug(f"oneshot ffmpeg failed: {e}")
            return None
        if result.returncode != 0 or len(result.stdout) < self._frame_bytes:
            return None
        try:
            frame = np.frombuffer(result.stdout[:self._frame_bytes],
                                  dtype=np.uint8).reshape(
                (self.cfg.output_h, self.cfg.output_w, 3))
        except ValueError:
            return None
        self._publish(frame_index, frame)
        return frame

    def stream_range(self, start_frame: int, count: int):
        """Disposable ffmpeg that yields ``count`` frames starting at
        ``start_frame`` as (index, BGR ndarray) pairs. Isolated from the
        main frame source; callers must copy frames they want to keep."""
        count = int(max(0, count))
        if count == 0 or self._fps <= 0:
            return
        t_s = max(0.0, start_frame / self._fps)
        cmd = [find_ffmpeg(), "-hide_banner", "-loglevel", "error", "-nostats"]
        cmd.extend(self._hwaccel_args)
        if t_s > 0.001:
            cmd.extend(["-ss", f"{t_s:.6f}"])
        cmd.extend([
            "-i", self.cfg.video_path,
            "-an", "-sn",
            "-vf", self._filter_chain(),
            "-frames:v", str(count),
            "-pix_fmt", "bgr24",
            "-f", "rawvideo", "pipe:1",
        ])
        proc = None
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                bufsize=-1,
                creationflags=subprocess_flags(),
            )
            for i in range(count):
                if proc.stdout is None:
                    return
                try:
                    raw = proc.stdout.read(self._frame_bytes)
                except (ValueError, OSError):
                    return
                if not raw or len(raw) < self._frame_bytes:
                    return
                try:
                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (self.cfg.output_h, self.cfg.output_w, 3))
                except ValueError:
                    return
                yield start_frame + i, frame
        finally:
            if proc is not None:
                try:
                    if proc.poll() is None:
                        proc.kill()
                except Exception:
                    pass
                try:
                    proc.wait(timeout=0.5)
                except Exception:
                    pass
                for pipe in (proc.stdout, proc.stderr):
                    if pipe is not None:
                        try: pipe.close()
                        except Exception: pass

    # --------------------------------------------------------------------- misc

    def _publish(self, idx: int, frame: np.ndarray) -> None:
        with self._frame_lock:
            self._current_frame = frame
            self._current_frame_index = idx
            self._frame_version += 1
        for cb in self._position_callbacks:
            try: cb(idx)
            except Exception as e: self.logger.debug(f"position cb error: {e}")

    def _notify_seek(self, idx: int) -> None:
        for cb in list(self._seek_callbacks):
            try: cb(idx)
            except Exception as e: self.logger.debug(f"seek cb error: {e}")

    def _notify_state(self, is_playing: bool) -> None:
        if not self._playback_state_callbacks:
            return
        ts_ms = (self._current_frame_index / self._fps) * 1000.0 if self._fps > 0 else 0.0
        for cb in list(self._playback_state_callbacks):
            try: cb(is_playing, ts_ms)
            except Exception as e: self.logger.debug(f"state cb error: {e}")

    def _drain_queue(self) -> None:
        while True:
            try: self._frame_queue.get_nowait()
            except queue.Empty: return

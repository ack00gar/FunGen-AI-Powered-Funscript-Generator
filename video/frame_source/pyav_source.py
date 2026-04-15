"""PyAV-backed video frame source.

Replaces the subprocess-ffmpeg + ThumbnailExtractor pair with a single
in-process decoder + persistent filter graph. Same coordinate space as the
existing pipeline (we hand it the same v360/scale/pad filter string the
FFmpegBuildersMixin produces today), so overlays, trackers and stage1/2 data
all see frames in the coordinate system they already expect.

Threading model
---------------
- One decode thread (``_decode_loop``) demuxes packets, decodes them, pushes
  through the filter graph, and pushes filtered frames onto a bounded queue.
- The owner thread (typically the GUI's processing loop) calls ``next_frame``
  / ``current_frame`` to consume.
- Seek is request-driven: the owner sets a target via ``seek(frame_index)``
  and the decode thread coalesces, performs the libavformat seek, drains to
  the target PTS, and resumes producing.

State invariants
----------------
- ``current_frame_index`` always reflects the index of the frame currently
  exposed via ``current_frame``. It only advances when a new frame is
  consumed (drained from the queue), never speculatively.
- ``_frame_version`` is bumped every time ``current_frame`` is replaced.
- All frames emitted are in the post-filter coordinate space (rectilinear
  for VR, padded-and-scaled for 2D), pixel format BGR24.

What this class does NOT do
---------------------------
- It does not run the GUI playback loop or pace frames to wall-clock fps.
  That stays in VideoProcessor's loop / a future PlaybackClock. This source
  decodes as fast as the consumer drains.
- It does not own the audio stream. Audio sync stays separate.
- It does not know about HD vs analysis-frame conversion. The integration
  layer asks for whichever output size it wants by giving the right filter.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import av
import numpy as np


SeekCallback = Callable[[int], None]
PlaybackStateCallback = Callable[[bool, float], None]
PositionCallback = Callable[[int], None]


@dataclass
class SourceConfig:
    """Everything the source needs to open a video at production fidelity."""
    video_path: str
    # Filter chain string in libavfilter syntax, applied after demux/decode.
    # Built by the integration layer (FFmpegBuildersMixin or equivalent) so we
    # match the existing coord space byte-for-byte. e.g.
    #   "crop=4096:4096:0:0,v360=he:in_stereo=0:output=sg:...,format=bgr24"
    # The trailing ``format=bgr24`` is added if missing.
    filter_chain: str
    output_w: int
    output_h: int
    # Decode thread count hint. ``0`` means "AUTO" (let libav decide).
    decoder_threads: int = 0


# Sentinel placed on the queue to signal end-of-stream or seek-flush.
_EOS = object()
_FLUSH = object()


class PyAVFrameSource:
    def __init__(self, config: SourceConfig, logger: Optional[logging.Logger] = None):
        self.cfg = config
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        # Stream metadata (populated on open).
        self._container: Optional[av.container.InputContainer] = None
        self._stream = None
        self._fps: float = 0.0
        self._time_base = None
        self._total_frames: int = 0
        self._duration_seconds: float = 0.0

        # Filter graph — rebuilt on open and on reapply_settings.
        self._graph: Optional[av.filter.Graph] = None

        # State exposed to consumers.
        self._current_frame: Optional[np.ndarray] = None
        self._current_frame_index: int = -1
        self._frame_version: int = 0
        self._frame_lock = threading.Lock()

        # Decode thread + frame queue.
        self._decode_thread: Optional[threading.Thread] = None
        self._frame_queue: "queue.Queue" = queue.Queue(maxsize=8)
        self._stop_event = threading.Event()
        self._pause_event = threading.Event()  # SET = paused (matches VideoProcessor)

        # Seek coordination.
        self._seek_lock = threading.Lock()
        self._seek_target: Optional[int] = None
        self._seek_accurate: bool = False
        self._seek_done_event = threading.Event()
        self._seek_done_event.set()  # No pending seek at start.

        # Callbacks (observers).
        self._seek_callbacks: List[SeekCallback] = []
        self._playback_state_callbacks: List[PlaybackStateCallback] = []
        self._position_callbacks: List[PositionCallback] = []

        # True when the decoder thread has produced the _EOS sentinel on the
        # queue. Lets the consumer differentiate "transient timeout" from
        # "stream is genuinely over" so trackers see every frame.
        self._eos_reached: bool = False

        # High-level decode iterator (container.decode(stream)). Primed after
        # a seek so subsequent +1 arrow-nav steps can fast-path without
        # re-seeking. Also referenced by the decode loop's fast-path check.
        self._decode_iter = None
        self._eos_drain_done: bool = False

    # --------------------------------------------------------------- lifecycle

    def open(self) -> bool:
        if self._container is not None:
            self.close()
        try:
            self._container = av.open(self.cfg.video_path)
        except Exception as e:
            self.logger.error(f"PyAV open failed for {self.cfg.video_path}: {e}")
            return False

        try:
            self._stream = self._container.streams.video[0]
        except (IndexError, AttributeError):
            self.logger.error("No video stream found")
            self._safe_close_container()
            return False

        self._stream.thread_type = "AUTO" if self.cfg.decoder_threads == 0 else "FRAME"
        if self.cfg.decoder_threads > 0:
            try:
                self._stream.codec_context.thread_count = self.cfg.decoder_threads
            except Exception:
                pass

        self._fps = float(self._stream.average_rate or self._stream.guessed_rate or 30.0)
        self._time_base = self._stream.time_base
        self._total_frames = int(self._stream.frames or 0)
        self._duration_seconds = (
            float(self._container.duration / 1_000_000) if self._container.duration else 0.0
        )
        # If frames count is missing (some containers), estimate from duration.
        if self._total_frames <= 0 and self._duration_seconds > 0:
            self._total_frames = int(self._duration_seconds * self._fps)

        if not self._build_graph():
            self.close()
            return False

        self.logger.info(
            f"PyAV opened {self.cfg.video_path}: "
            f"{self._stream.width}x{self._stream.height} "
            f"fps={self._fps:.3f} frames={self._total_frames}"
        )
        return True

    def close(self) -> None:
        self.stop()
        self._graph = None
        self._stream = None
        self._safe_close_container()

    def _safe_close_container(self) -> None:
        if self._container is not None:
            try:
                self._container.close()
            except Exception:
                pass
            self._container = None

    def reapply_settings(self, new_config: Optional[SourceConfig] = None) -> bool:
        """Apply a new filter chain or output size without losing position.

        If ``new_config`` is given it replaces the current config (path stays
        the same — switching video means a fresh open). The decode thread is
        stopped, the graph is rebuilt, and the source is left ready for
        ``start()`` again at the previously held ``current_frame_index``.
        """
        cur = max(0, self._current_frame_index)
        was_running = self._decode_thread is not None and self._decode_thread.is_alive()
        was_paused = self._pause_event.is_set()
        self.stop()
        if new_config is not None:
            self.cfg = new_config
        if not self._build_graph():
            return False
        if was_running:
            self.start(cur)
            if was_paused:
                self._pause_event.set()
        return True

    # ------------------------------------------------------------- filter graph

    def _build_graph(self) -> bool:
        if self._stream is None:
            return False
        try:
            g = av.filter.Graph()
            src = g.add_buffer(template=self._stream)
            chain = self.cfg.filter_chain.strip().strip(",")
            if "format=" not in chain:
                chain = f"{chain},format=bgr24" if chain else "format=bgr24"

            prev = src
            for spec in chain.split(","):
                spec = spec.strip()
                if not spec:
                    continue
                name, _, args = spec.partition("=")
                node = g.add(name.strip(), args.strip())
                prev.link_to(node)
                prev = node

            sink = g.add("buffersink")
            prev.link_to(sink)
            g.configure()
            self._graph = g
            return True
        except Exception as e:
            self.logger.error(f"Filter graph build failed: {e} (chain={self.cfg.filter_chain})")
            self._graph = None
            return False

    # -------------------------------------------------------------- transport

    def start(self, start_frame: int = 0) -> None:
        """Start (or restart) the decode thread at ``start_frame``."""
        if self._decode_thread is not None and self._decode_thread.is_alive():
            return
        if self._container is None or self._graph is None:
            self.logger.error("Cannot start: source not open")
            return

        self._stop_event.clear()
        self._pause_event.clear()
        self._eos_reached = False
        self._drain_queue()

        # Prime the seek so the loop opens at the right place.
        with self._seek_lock:
            self._seek_target = max(0, start_frame)
            self._seek_done_event.clear()

        self._decode_thread = threading.Thread(
            target=self._decode_loop, daemon=True, name="PyAVDecode")
        self._decode_thread.start()
        self._notify_state(is_playing=True)

    def stop(self) -> None:
        if self._decode_thread is None or not self._decode_thread.is_alive():
            self._notify_state(is_playing=False)
            return
        self._stop_event.set()
        self._pause_event.clear()
        # Drain queue so producer can exit; post EOS so any blocked consumer
        # next_frame() returns immediately instead of hitting the timeout.
        self._drain_queue()
        try: self._frame_queue.put_nowait(_EOS)
        except queue.Full: pass
        self._decode_thread.join(timeout=2.0)
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
        """Request a seek. Coalesces — last write wins.

        ``accurate=False`` (default): land on the GOP keyframe at-or-before
        target (~30-300ms on heavy formats). The first delivered frame is
        the keyframe; subsequent frames advance forward toward and past the
        target naturally as decode continues. Best for interactive scrub.

        ``accurate=True``: drain the decoder forward until a frame at-or-after
        target is produced. Frame-exact but expensive on long GOPs (e.g., ~5s
        on 8K 10-bit HEVC for a 5-second-distant keyframe). Use only when the
        exact target frame is required (e.g., funscript point editing).

        The queue is drained so the consumer doesn't pull stale pre-seek
        frames after returning.
        """
        target = max(0, min(int(frame_index), max(0, self._total_frames - 1)))
        with self._seek_lock:
            self._seek_target = target
            self._seek_accurate = accurate
            self._seek_done_event.clear()
        self._drain_queue()
        self._notify_seek(target)

    def wait_seek(self, timeout: float = 2.0) -> bool:
        return self._seek_done_event.wait(timeout=timeout)

    def get_frame(self, frame_index: int, timeout: float = 2.0,
                  accurate: bool = True) -> Optional[np.ndarray]:
        """Synchronous random-access fetch.

        For thumbnail / scrub-preview / arrow-nav use. Blocks until the frame
        is decoded. If a decode thread is running, issues a seek and waits
        for the result (interrupts sequential playback). If not running,
        decodes in-line on the calling thread.

        ``accurate=True`` (default): decoder drains to the exact requested
        frame. Required for arrow nav and point editing; costs one GOP of
        forward decode on a miss but the +1 fast path skips that for
        sequential forward stepping.
        """
        if not self.is_running:
            return self._oneshot_decode_at(frame_index, accurate=accurate)
        self.seek(frame_index, accurate=accurate)
        if not self.wait_seek(timeout=timeout):
            return None
        try:
            idx, frame = self._frame_queue.get(timeout=timeout)
            self._publish(idx, frame)
            return frame
        except queue.Empty:
            return None

    # ---------------------------------------------------------------- consume

    def next_frame(self, timeout: float = 1.0) -> Optional[Tuple[int, np.ndarray]]:
        """Pull the next decoded frame from the queue and publish it.

        Returns ``(frame_index, frame)`` or ``None`` on timeout/EOS. Use the
        ``is_eos`` property to differentiate the two: ``next_frame() is None
        and is_eos`` means stream is actually finished; just None means a
        transient pull timeout (decoder is busy / queue temporarily empty).
        """
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

    # ---------------------------------------------------------------- internals

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

    def _frame_index_from_pts(self, pts: Optional[int]) -> int:
        if pts is None or self._time_base is None or self._fps <= 0:
            return -1
        return int(round(float(pts * self._time_base) * self._fps))

    def _seek_to_target(self, target: int) -> None:
        """Perform a libavformat seek to ``target`` and align the demux
        cursor so the next decode produces frames at-or-after ``target``.

        Also rebuilds the filter graph if it was previously flushed via
        push(None) at EOS — libavfilter marks the graph as EOF after that
        and any subsequent push() raises av.EOFError. Rebuilding is cheap
        (microseconds) and needed to keep seeks working after playback
        reached the end of the video."""
        if self._container is None or self._stream is None or self._time_base is None:
            return
        # If the graph was flushed at a prior EOS, rebuild so future pushes
        # work again. Safe to do on every seek; seeks invalidate filter state
        # anyway (stateless crop/v360/format all keep working either way).
        if getattr(self, '_eos_drain_done', False) or self._graph is None:
            self._build_graph()
            self._eos_drain_done = False
            self._eos_reached = False
        target_pts = int(target / self._fps / self._time_base)
        try:
            self._container.seek(target_pts, backward=True, any_frame=False, stream=self._stream)
        except Exception as e:
            self.logger.warning(f"seek to {target} failed: {e}")

    def _decode_one_after_seek(self, target: int, accurate: bool, max_pump: int = 600) -> Optional[Tuple[int, np.ndarray]]:
        """After a libavformat seek, decode and filter frames until one is
        ready. With ``accurate=True``, drains until idx >= target (slow on
        long GOPs: ~50ms/frame on 8K 10-bit HEVC). With ``accurate=False``,
        returns the first frame after the keyframe (fast, ~30-300ms; lands
        within ~GOP-size of target).
        """
        if self._container is None or self._graph is None:
            return None
        pumped = 0
        for packet in self._container.demux(self._stream):
            for frame in packet.decode():
                pumped += 1
                if pumped > max_pump:
                    return None
                self._graph.push(frame)
                while True:
                    try:
                        out_frame = self._graph.pull()
                    except (av.BlockingIOError, av.FFmpegError):
                        break
                    idx = self._frame_index_from_pts(out_frame.pts)
                    if not accurate or idx < 0 or idx >= target:
                        arr = out_frame.to_ndarray(format="bgr24")
                        return (idx if idx >= 0 else target), arr
        return None

    def _decode_loop(self) -> None:
        """Background producer: handles seeks then streams filtered frames."""
        try:
            while not self._stop_event.is_set():
                # Pause: spin lightly without consuming CPU. New seeks still
                # win because they set _seek_target outside the pause window.
                while self._pause_event.is_set() and not self._stop_event.is_set():
                    if self._seek_target is not None:
                        break
                    time.sleep(0.01)

                if self._stop_event.is_set():
                    break

                # Honor pending seek.
                with self._seek_lock:
                    target = self._seek_target
                    accurate = self._seek_accurate
                    self._seek_target = None
                if target is not None:
                    cur = self._current_frame_index
                    # Fast path: +1 step can just pump the next frame, no seek.
                    # Requires the high-level decode iterator to already be
                    # primed (i.e., a prior seek has landed) and the graph
                    # not to be at EOS. Saves 30-300ms per arrow press.
                    fast_step = (
                        target == cur + 1
                        and self._decode_iter is not None
                        and not getattr(self, '_eos_drain_done', False)
                    )
                    if fast_step:
                        try:
                            self._pump_one_into_queue()
                        except StopIteration:
                            self._frame_queue.put(_EOS)
                            self._seek_done_event.set()
                            return
                    else:
                        self._drain_queue()
                        self._decode_iter = None  # reset so post-seek decode yields fresh frames
                        self._eos_drain_done = False
                        self._seek_to_target(target)
                        result = self._decode_one_after_seek(target, accurate=accurate)
                        if result is not None:
                            idx, arr = result
                            self._frame_queue.put((idx, arr))
                        # Prime the high-level decode iterator so subsequent
                        # +1 arrow presses can take the fast path.
                        if self._container is not None and self._stream is not None:
                            try:
                                self._decode_iter = self._container.decode(self._stream)
                            except Exception:
                                self._decode_iter = None
                    self._seek_done_event.set()
                    if self._pause_event.is_set():
                        # Paused after seek — don't start streaming until
                        # resume or a new seek arrives.
                        continue

                # Steady-state: pump the next frame.
                try:
                    self._pump_one_into_queue()
                except StopIteration:
                    self._frame_queue.put(_EOS)
                    return
        except Exception as e:
            self.logger.error(f"decode loop crashed: {e}", exc_info=True)
            self._frame_queue.put(_EOS)

    def _pump_one_into_queue(self) -> None:
        """Decode and filter exactly one frame, push to queue.

        Uses ``container.decode(stream)`` (high-level API) which handles
        decoder flush internally, plus an explicit graph flush at EOS for
        any frames buffered inside libavfilter. Exact frame-count parity
        with subprocess ffmpeg.
        """
        if self._container is None or self._graph is None:
            raise StopIteration
        if not hasattr(self, '_decode_iter') or self._decode_iter is None:
            self._decode_iter = self._container.decode(self._stream)
            self._eos_drain_done = False

        # Step 1: deliver any frame already buffered in the graph.
        if self._try_pull_into_queue():
            return

        # Step 2: pull next decoded frame, push to graph, try to pull a
        # filtered frame out. A single decoded frame can yield zero or one
        # filtered frame (graph delays match codec delays). If zero, loop.
        for frame in self._decode_iter:
            if self._stop_event.is_set() or self._seek_target is not None:
                return
            self._graph.push(frame)
            if self._try_pull_into_queue():
                return

        # Step 3: decoded iterator exhausted. Flush the filter graph.
        if not self._eos_drain_done:
            self._eos_drain_done = True
            try:
                self._graph.push(None)
            except Exception:
                pass
            # Pull every remaining frame the graph buffered, queue them all.
            queued = 0
            while True:
                try:
                    out_frame = self._graph.pull()
                except (av.BlockingIOError, av.FFmpegError, av.EOFError):
                    break
                idx = self._frame_index_from_pts(out_frame.pts)
                arr = out_frame.to_ndarray(format="bgr24")
                while not self._stop_event.is_set():
                    try:
                        self._frame_queue.put((idx, arr), timeout=0.1)
                        queued += 1
                        break
                    except queue.Full:
                        if self._seek_target is not None:
                            return
            if queued > 0:
                self.logger.debug(f"PyAV graph flush: drained {queued} tail frames")
        raise StopIteration

    def _try_pull_into_queue(self) -> bool:
        """Try to pull one filtered frame from the graph and queue it.
        Returns True if a frame was queued."""
        try:
            out_frame = self._graph.pull()
        except (av.BlockingIOError, av.FFmpegError):
            return False
        idx = self._frame_index_from_pts(out_frame.pts)
        arr = out_frame.to_ndarray(format="bgr24")
        while not self._stop_event.is_set():
            try:
                self._frame_queue.put((idx, arr), timeout=0.1)
                return True
            except queue.Full:
                if self._seek_target is not None:
                    return False
        return False

    def _oneshot_decode_at(self, frame_index: int, accurate: bool = True) -> Optional[np.ndarray]:
        """Decode a single frame without a running decode thread.

        Defaults to accurate=True since one-shot use cases (e.g., extracting
        a specific frame for analysis) typically need the exact frame.
        """
        self._seek_to_target(frame_index)
        result = self._decode_one_after_seek(frame_index, accurate=accurate)
        if result is None:
            return None
        idx, arr = result
        self._publish(idx, arr)
        return arr

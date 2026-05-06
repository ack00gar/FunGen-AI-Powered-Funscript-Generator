"""Nav-anticipation prefetcher + pattern detector.

Watches the sequence of frame indexes the user accesses and, while playback
is paused and no tracker is active, fills the FrameCache around the likely
next target.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from typing import Callable, Deque, Optional, Tuple


_WINDOW_EVENTS = 8
_MIN_TREND_EVENTS = 3
_STOPPED_SILENCE_S = 0.25
_BIDIR_MARGIN = 30
_MAX_LOOKAHEAD = 120
_MIN_LOOKAHEAD = 8
_IDLE_SLEEP_S = 0.4


class NavPatternDetector:
    """Classifies recent nav events into (direction, speed)."""

    def __init__(self, window: int = _WINDOW_EVENTS):
        self._events: Deque[Tuple[int, float]] = deque(maxlen=window)
        self._lock = threading.Lock()

    def record(self, frame_index: int) -> None:
        with self._lock:
            self._events.append((int(frame_index), time.monotonic()))

    def last_event_age_s(self) -> float:
        with self._lock:
            if not self._events:
                return float("inf")
            return time.monotonic() - self._events[-1][1]

    def classify(self) -> Tuple[str, int]:
        """Return (direction, speed_fps). direction: fwd | back | stop | random."""
        with self._lock:
            if len(self._events) < 2:
                return ("stop", 0)
            last_ts = self._events[-1][1]
            if (time.monotonic() - last_ts) > _STOPPED_SILENCE_S:
                return ("stop", 0)

            deltas = []
            for i in range(1, len(self._events)):
                d_idx = self._events[i][0] - self._events[i - 1][0]
                d_t = self._events[i][1] - self._events[i - 1][1]
                deltas.append((d_idx, d_t))
            if len(deltas) < _MIN_TREND_EVENTS:
                return ("stop", 0)

            nonzero = [d for d in deltas if d[0] != 0]
            if not nonzero:
                return ("stop", 0)
            forward = sum(1 for d in nonzero if d[0] > 0)
            backward = sum(1 for d in nonzero if d[0] < 0)
            if forward > 0 and backward == 0:
                direction = "fwd"
            elif backward > 0 and forward == 0:
                direction = "back"
            else:
                return ("random", 0)

            mean_delta = sum(abs(d[0]) for d in nonzero) / len(nonzero)
            total_t = max(1e-6, sum(max(0.0, d[1]) for d in nonzero))
            rate = int(mean_delta * len(nonzero) / total_t)
            return (direction, max(0, rate))


class FramePrefetcher:
    """Daemon thread that keeps the FrameCache warm around the user's likely
    next target. Wakes on notify(); otherwise blocks on the wake event."""

    def __init__(
        self,
        processor,
        cache,
        detector: NavPatternDetector,
        is_idle: Callable[[], bool],
        logger: Optional[logging.Logger] = None,
    ):
        self.proc = processor
        self.cache = cache
        self.detector = detector
        self._is_idle = is_idle
        self.logger = logger or logging.getLogger(self.__class__.__name__)

        self._thread: Optional[threading.Thread] = None
        self._wake = threading.Event()
        self._stop = threading.Event()

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._wake.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="FramePrefetcher")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def notify(self) -> None:
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            self._wake.wait(timeout=_IDLE_SLEEP_S)
            self._wake.clear()
            if self._stop.is_set():
                return
            if not self._is_idle():
                continue
            if self.proc is None:
                continue

            fs = getattr(self.proc, "frame_source", None)
            if fs is None:
                continue
            if getattr(fs, "total_frames", 0) <= 0:
                continue

            direction, rate = self.detector.classify()
            cur = int(getattr(self.proc, "current_frame_index", 0) or 0)
            total = int(getattr(self.proc, "total_frames", 0) or 0)
            if total <= 0:
                continue

            if direction == "fwd":
                lookahead = min(_MAX_LOOKAHEAD, max(_MIN_LOOKAHEAD, rate * 2))
                self._run_plan(cur + 1, lookahead, total)
            elif direction == "back":
                lookahead = min(_MAX_LOOKAHEAD, max(_MIN_LOOKAHEAD, rate * 2))
                start = max(0, cur - lookahead)
                self._run_plan(start, cur - start, total, prefer_tail=True)
            elif direction == "stop":
                self._run_plan(
                    max(0, cur - _BIDIR_MARGIN),
                    min(total - 1, cur + _BIDIR_MARGIN)
                    - max(0, cur - _BIDIR_MARGIN) + 1,
                    total,
                )

    def _run_plan(self, start: int, count: int, total: int,
                  prefer_tail: bool = False) -> None:
        start = max(0, min(int(start), max(0, total - 1)))
        count = int(max(0, count))
        # Cap window to the cache budget. Without this, a window larger than
        # the cache thrashes (decode -> evict -> re-decode) at 800%+ CPU on
        # high-res sources (8K VR at ~19 MB/frame vs default 1 GB cache).
        try:
            budget = int(getattr(self.cache, "max_bytes", 0) or 0)
            w = int(getattr(self.proc, "_display_frame_w", 0) or 0)
            h = int(getattr(self.proc, "_display_frame_h", 0) or 0)
            per_frame = max(1, w * h * 3)
            if budget > 0 and per_frame > 1:
                max_count = max(8, int((budget * 0.7) // per_frame))
                if count > max_count:
                    cur = int(getattr(self.proc, "current_frame_index", 0) or 0)
                    if start <= cur < start + count:
                        start = max(0, cur - max_count // 2)
                    count = max_count
        except Exception:
            pass
        end_exclusive = min(total, start + count)
        if end_exclusive <= start:
            return

        # One lock acquisition for the whole window vs N=count.
        uncached = self.cache.missing(start, end_exclusive)
        if not uncached:
            return
        if prefer_tail:
            uncached.sort(reverse=True)

        fs = self.proc.frame_source
        if fs is None:
            return

        try:
            stream = fs.stream_range(start, end_exclusive - start)
        except Exception as e:
            self.logger.debug(f"prefetch stream_range failed: {e}")
            return

        uncached_set = set(uncached)
        try:
            for idx, frame in stream:
                if self._stop.is_set() or self._wake.is_set():
                    return
                if not self._is_idle():
                    return
                if idx in uncached_set:
                    self.cache.put(idx, frame)
        except Exception as e:
            self.logger.debug(f"prefetch consume error: {e}")

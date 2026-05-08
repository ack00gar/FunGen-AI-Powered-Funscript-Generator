"""Lightweight pipeline metrics for live diagnostics.

Single `PipelineMetrics` instance hung off `VideoProcessor`. Hot paths
call into the metrics object to record timings and state changes; the
GUI reads counters opportunistically. No locks in hot paths; the GIL
makes simple int / float assignments effectively atomic for our reads,
and the deques tolerate transient inconsistency since the GUI just
displays a snapshot.

Goal: stop guessing about the 8K pipeline. Concrete numbers for seeks,
arrow-nav cache hits, prefetcher work, and loop state.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Tuple


@dataclass
class SeekEvent:
    target_frame: int
    total_ms: float
    ffmpeg_ms: float
    mpv_ms: float
    tracker_needs: bool
    accurate: bool
    timestamp: float  # time.monotonic()


class _RollingCounter:
    """Hit/miss ratio over the last `window_s` seconds."""
    __slots__ = ("_events", "_window_s")

    def __init__(self, window_s: float = 5.0) -> None:
        self._events: Deque[Tuple[float, bool]] = deque()
        self._window_s = window_s

    def record(self, hit: bool) -> None:
        now = time.monotonic()
        self._events.append((now, hit))
        cutoff = now - self._window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()

    def stats(self) -> Tuple[int, float]:
        now = time.monotonic()
        cutoff = now - self._window_s
        while self._events and self._events[0][0] < cutoff:
            self._events.popleft()
        n = len(self._events)
        if n == 0:
            return (0, 0.0)
        hits = sum(1 for _, h in self._events if h)
        return (n, hits / n)


class PipelineMetrics:
    """Live counters + recent-event ring for video pipeline diagnostics."""

    def __init__(self, history: int = 16) -> None:
        self._seeks: Deque[SeekEvent] = deque(maxlen=history)

        # Pause / resume single-slot timing.
        self.last_pause_ms: float = 0.0
        self.last_resume_ms: float = 0.0
        self.last_pause_t: float = 0.0
        self.last_resume_t: float = 0.0

        # Loop state observed on its last iteration.
        self.loop_tracker_needs: bool = False
        self.loop_mpv_drives: bool = False
        self.loop_src_paused: bool = True
        self.loop_iter_t: float = 0.0

    # ---- recording ----

    def record_seek(self, target_frame: int, total_ms: float, ffmpeg_ms: float,
                    mpv_ms: float, tracker_needs: bool, accurate: bool) -> None:
        self._seeks.append(SeekEvent(
            target_frame=target_frame,
            total_ms=total_ms,
            ffmpeg_ms=ffmpeg_ms,
            mpv_ms=mpv_ms,
            tracker_needs=tracker_needs,
            accurate=accurate,
            timestamp=time.monotonic(),
        ))

    def record_pause_ms(self, ms: float) -> None:
        self.last_pause_ms = float(ms)
        self.last_pause_t = time.monotonic()

    def record_resume_ms(self, ms: float) -> None:
        self.last_resume_ms = float(ms)
        self.last_resume_t = time.monotonic()

    def record_loop_state(self, tracker_needs: bool, mpv_drives: bool,
                          src_paused: bool) -> None:
        self.loop_tracker_needs = bool(tracker_needs)
        self.loop_mpv_drives = bool(mpv_drives)
        self.loop_src_paused = bool(src_paused)
        self.loop_iter_t = time.monotonic()

    # ---- read-side ----

    def recent_seeks(self) -> List[SeekEvent]:
        return list(self._seeks)

    def loop_state_age_s(self) -> float:
        if self.loop_iter_t <= 0:
            return float("inf")
        return time.monotonic() - self.loop_iter_t

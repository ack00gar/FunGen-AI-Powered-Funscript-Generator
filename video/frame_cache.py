"""LRU frame cache, byte-budgeted."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict, deque
from typing import Deque, Optional, Tuple

import numpy as np


class FrameCache:
    """Byte-budgeted LRU cache of frame_index -> BGR ndarray."""

    def __init__(self, max_bytes: int):
        self._cache: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._max_bytes: int = max(64 * 1024 * 1024, int(max_bytes))
        self._bytes_used: int = 0
        self._lock = threading.Lock()
        self._events: Deque[Tuple[float, bool]] = deque(maxlen=4096)

    def get(self, frame_index: int) -> Optional[np.ndarray]:
        with self._lock:
            arr = self._cache.get(frame_index)
            if arr is None:
                self._events.append((time.monotonic(), False))
                return None
            self._cache.move_to_end(frame_index)
            self._events.append((time.monotonic(), True))
            return arr

    def put(self, frame_index: int, frame: np.ndarray) -> None:
        if frame is None or not isinstance(frame, np.ndarray):
            return
        # No copy: all callers hand off frames that are already owned
        # (np.frombuffer views over the immutable bytes from ffmpeg stdout).
        # A .copy() here cost ~300us per 4K frame at 60fps (~18ms/sec) and
        # ~900us per 8K VR frame (~54ms/sec) for no safety gain.
        if frame.flags['C_CONTIGUOUS']:
            stored = frame
        else:
            stored = np.ascontiguousarray(frame)
        nbytes = int(stored.nbytes)
        with self._lock:
            if frame_index in self._cache:
                prior = self._cache[frame_index]
                self._bytes_used -= int(prior.nbytes)
                self._cache[frame_index] = stored
                self._bytes_used += nbytes
                self._cache.move_to_end(frame_index)
            else:
                self._cache[frame_index] = stored
                self._bytes_used += nbytes
            while self._bytes_used > self._max_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._bytes_used -= int(evicted.nbytes)

    def contains(self, frame_index: int) -> bool:
        with self._lock:
            return frame_index in self._cache

    def missing(self, start: int, end_exclusive: int) -> list:
        """Return indices in [start, end) not in the cache, single lock."""
        with self._lock:
            cached = self._cache
            return [i for i in range(start, end_exclusive) if i not in cached]

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._bytes_used = 0
            self._events.clear()

    def stats(self) -> dict:
        now = time.monotonic()
        with self._lock:
            n = len(self._cache)
            bytes_used = self._bytes_used
            budget = self._max_bytes
            first_idx = next(iter(self._cache), None) if self._cache else None
            last_idx = next(reversed(self._cache), None) if self._cache else None
            window_start = now - 5.0
            hits = 0
            total = 0
            for ts, was_hit in self._events:
                if ts < window_start:
                    continue
                total += 1
                if was_hit:
                    hits += 1
            hit_rate = (hits / total) if total > 0 else 0.0
        return {
            "frames": n,
            "bytes": bytes_used,
            "budget": budget,
            "fill_pct": (bytes_used / budget) if budget > 0 else 0.0,
            "oldest_idx": first_idx,
            "newest_idx": last_idx,
            "hit_rate_5s": hit_rate,
            "sample_count_5s": total,
        }

    @property
    def bytes_used(self) -> int:
        with self._lock:
            return self._bytes_used

    @property
    def max_bytes(self) -> int:
        return self._max_bytes

    def resize(self, max_bytes: int) -> None:
        with self._lock:
            self._max_bytes = max(64 * 1024 * 1024, int(max_bytes))
            while self._bytes_used > self._max_bytes and len(self._cache) > 1:
                _, evicted = self._cache.popitem(last=False)
                self._bytes_used -= int(evicted.nbytes)

"""VideoProcessor nav-buffer mixin: LRU cache + anticipatory prefetcher."""

from __future__ import annotations

import threading
import time as _time
from typing import Optional

import numpy as np

from video.frame_cache import FrameCache
from video.nav_prefetcher import NavPatternDetector, FramePrefetcher


POINT_NAV_PREFETCH_MARGIN = 15


class NavBufferMixin:

    def _compute_cache_bytes_budget(self) -> int:
        budget = 1024 * 1024 * 1024
        app = getattr(self, "app", None)
        if app is not None and hasattr(app, "app_settings"):
            try:
                raw = app.app_settings.config.navigation.nav_cache_bytes
                if raw:
                    budget = int(raw)
            except Exception:
                pass
        budget = max(128 * 1024 * 1024, int(budget))
        # Cap at 25% of system RAM so a misconfigured setting can't OOM.
        try:
            import psutil
            ram_cap = int(psutil.virtual_memory().total * 0.25)
            if ram_cap > 0:
                budget = min(budget, ram_cap)
        except Exception:
            pass
        try:
            self.logger.debug(f"Nav cache byte budget: {budget / (1024*1024):.0f} MB")
        except Exception:
            pass
        return budget

    def _compute_nav_buffer_size(self) -> int:
        # Legacy: some external callers still expect a frame count.
        try:
            w = getattr(self, "_display_frame_w", None) or getattr(self, "yolo_input_size", 640)
            h = getattr(self, "_display_frame_h", None) or getattr(self, "yolo_input_size", 640)
        except Exception:
            w = h = 640
        per_frame = max(1, int(w) * int(h) * 3)
        return max(60, self._compute_cache_bytes_budget() // per_frame)

    def _init_nav_cache(self) -> None:
        self._nav_cache = FrameCache(self._compute_cache_bytes_budget())
        self._nav_detector = NavPatternDetector()
        self._nav_prefetcher = FramePrefetcher(
            processor=self,
            cache=self._nav_cache,
            detector=self._nav_detector,
            is_idle=self._nav_prefetcher_can_run,
            logger=self.logger,
        )

    def _start_nav_prefetcher(self) -> None:
        if not hasattr(self, "_nav_prefetcher"):
            return
        self._nav_prefetcher.start()

    def _stop_nav_prefetcher(self) -> None:
        if not hasattr(self, "_nav_prefetcher"):
            return
        try:
            self._nav_prefetcher.stop()
        except Exception:
            pass

    # async arrow-nav fetch: dedicated worker for cache-miss targets so the
    # imgui main thread never blocks on get_frame.
    def _init_arrow_async(self) -> None:
        self._arrow_target: Optional[int] = None
        self._arrow_target_lock = threading.Lock()
        self._arrow_wake = threading.Event()
        self._arrow_stop = threading.Event()
        self._arrow_epoch = 0
        self._arrow_thread = threading.Thread(
            target=self._arrow_async_loop, daemon=True, name="ArrowNavFetch")
        self._arrow_thread.start()

    def _stop_arrow_async(self) -> None:
        ev = getattr(self, "_arrow_stop", None)
        if ev is None:
            return
        ev.set()
        self._arrow_wake.set()
        try:
            self._arrow_thread.join(timeout=1.0)
        except Exception:
            pass

    def _enqueue_arrow_fetch(self, target: int) -> int:
        with self._arrow_target_lock:
            self._arrow_epoch += 1
            self._arrow_target = int(target)
            my_epoch = self._arrow_epoch
        self._arrow_wake.set()
        return my_epoch

    def _arrow_async_loop(self) -> None:
        while not self._arrow_stop.is_set():
            if not self._arrow_wake.wait(timeout=0.5):
                continue
            self._arrow_wake.clear()
            with self._arrow_target_lock:
                target = self._arrow_target
                my_epoch = self._arrow_epoch
            if target is None or self.frame_source is None:
                continue
            t0 = _time.perf_counter()
            frame = self.frame_source.get_frame(int(target), timeout=2.0)
            dur_ms = (_time.perf_counter() - t0) * 1000.0
            if frame is None:
                if self._nav_dbg_enabled():
                    self.logger.info(
                        f"NAV arrow_async    target={target} path=miss      dur={dur_ms:.1f}ms")
                continue
            self._nav_cache.put(int(target), frame)
            with self._arrow_target_lock:
                stale = (self._arrow_epoch != my_epoch)
            if stale:
                if self._nav_dbg_enabled():
                    self.logger.info(
                        f"NAV arrow_async    target={target} path=stale     dur={dur_ms:.1f}ms")
                continue
            with self.frame_lock:
                self.current_frame = frame
                self._frame_version += 1
            if self._nav_dbg_enabled():
                self.logger.info(
                    f"NAV arrow_async    target={target} path=commit    dur={dur_ms:.1f}ms")

    def _nav_prefetcher_can_run(self) -> bool:
        # Only when paused and tracker idle. Playback pumps the cache itself;
        # tracker owns the decoder.
        if getattr(self, "is_processing", False):
            pev = getattr(self, "pause_event", None)
            if pev is None or not pev.is_set():
                return False
        tracker = getattr(self, "tracker", None)
        if tracker is not None and getattr(tracker, "tracking_active", False):
            return False
        return True

    def _buffer_lookup(self, target_frame: int) -> Optional[np.ndarray]:
        return self._nav_cache.get(int(target_frame))

    def _buffer_append(self, frame_index: int, frame_data: np.ndarray) -> None:
        if frame_data is None:
            return
        self._nav_cache.put(int(frame_index), frame_data)

    def _clear_nav_state(self) -> None:
        self._nav_cache.clear()

    @property
    def buffer_info(self) -> dict:
        stats = self._nav_cache.stats()
        return {
            "size": stats["frames"],
            "bytes": stats["bytes"],
            "budget": stats["budget"],
            "fill_pct": stats["fill_pct"],
            "hit_rate_5s": stats["hit_rate_5s"],
            "capacity": stats["frames"],
            "start": stats["oldest_idx"] if stats["oldest_idx"] is not None else -1,
            "end": stats["newest_idx"] if stats["newest_idx"] is not None else -1,
            "current": getattr(self, "current_frame_index", -1),
        }

    def get_cached_frame(self, frame_index: int) -> Optional[np.ndarray]:
        return self._nav_cache.get(int(frame_index))

    def _clear_cache(self) -> None:
        self._nav_cache.clear()

    def _nav_dbg_enabled(self) -> bool:
        try:
            return self.app.app_settings.config.navigation.debug_logging
        except Exception:
            return False

    def _nav_to_target(self, target_frame: int) -> Optional[np.ndarray]:
        import time as _time
        target_frame = int(target_frame)
        prev = getattr(self, 'current_frame_index', -1)
        if hasattr(self, "_nav_detector"):
            self._nav_detector.record(target_frame)

        t0 = _time.perf_counter()
        frame = self._nav_cache.get(target_frame)
        if frame is not None:
            self.current_frame_index = target_frame
            if hasattr(self, "_nav_prefetcher"):
                self._nav_prefetcher.notify()
            if self._nav_dbg_enabled():
                self.logger.info(
                    f"NAV nav_to         from={prev} to={target_frame} "
                    f"delta={target_frame-prev:+d} path=cache      "
                    f"dur={(_time.perf_counter()-t0)*1000:.1f}ms hit=1")
            return frame

        if self.frame_source is None:
            self.logger.warning(f"Nav miss and no frame source for frame {target_frame}")
            return None
        # Cache miss: advance cursor sync, enqueue async fetch, return None.
        # Background worker commits the frame + bumps _frame_version under lock.
        self.current_frame_index = target_frame
        if hasattr(self, "_enqueue_arrow_fetch"):
            self._enqueue_arrow_fetch(target_frame)
        if hasattr(self, "_nav_prefetcher"):
            self._nav_prefetcher.notify()
        if self._nav_dbg_enabled():
            dur_ms = (_time.perf_counter() - t0) * 1000.0
            self.logger.info(
                f"NAV nav_to         from={prev} to={target_frame} "
                f"delta={target_frame-prev:+d} path=async_queue "
                f"dur={dur_ms:.1f}ms hit=0")
        return None

    def arrow_nav_forward(self, target_frame: int) -> Optional[np.ndarray]:
        if getattr(self, "arrow_nav_in_progress", False):
            return None
        self.arrow_nav_in_progress = True
        try:
            return self._nav_to_target(target_frame)
        finally:
            self.arrow_nav_in_progress = False

    def arrow_nav_backward(self, target_frame: int) -> Optional[np.ndarray]:
        return self._nav_to_target(target_frame)

    def prefetch_around(self, center_frame: int, margin: int = POINT_NAV_PREFETCH_MARGIN):
        total = getattr(self, "total_frames", 0) or 0
        if total <= 0 or not self.video_path or self.frame_source is None:
            return
        if hasattr(self, "_nav_detector"):
            self._nav_detector.record(int(center_frame))
        if hasattr(self, "_nav_prefetcher"):
            self._nav_prefetcher.notify()

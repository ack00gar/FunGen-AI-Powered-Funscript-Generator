"""VideoProcessor nav-buffer mixin: LRU cache + anticipatory prefetcher."""

from __future__ import annotations

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
        # Short budget: UI thread is blocked while this runs. Cache miss
        # on a slow VR keyframe can still exceed this; better to drop the
        # frame than freeze the UI for seconds.
        frame = self.frame_source.get_frame(target_frame, timeout=0.8)
        dur_ms = (_time.perf_counter() - t0) * 1000.0
        if frame is None:
            if self._nav_dbg_enabled():
                self.logger.info(
                    f"NAV nav_to         from={prev} to={target_frame} "
                    f"delta={target_frame-prev:+d} path=source     "
                    f"dur={dur_ms:.1f}ms hit=0 miss=1")
            else:
                self.logger.debug(f"frame_source.get_frame({target_frame}) missed within budget")
            return None
        self._nav_cache.put(target_frame, frame)
        self.current_frame_index = target_frame
        if hasattr(self, "_nav_prefetcher"):
            self._nav_prefetcher.notify()
        if self._nav_dbg_enabled():
            self.logger.info(
                f"NAV nav_to         from={prev} to={target_frame} "
                f"delta={target_frame-prev:+d} path=source     "
                f"dur={dur_ms:.1f}ms hit=0 fetched=1")
        return frame

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

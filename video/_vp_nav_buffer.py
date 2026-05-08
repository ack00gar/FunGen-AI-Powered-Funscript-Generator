"""Arrow nav driver. Drives libmpv (frame-step / exact seek) for the
forward / backward arrow keys, with a ring-buffer hit path for backward
on long-GOP HEVC where mpv frame-back-step is unreliable.

The legacy LRU cache + prefetcher this file used to host are gone in
phase C. Display, thumbnails, and arrow-nav all read from libmpv now;
the cache stored numpy frames nothing read.
"""
from __future__ import annotations

import time as _time
from typing import Optional

import numpy as np


class NavBufferMixin:

    def _init_nav_cache(self) -> None:
        # Phase C: the cache is gone; nothing to initialize. Method kept
        # because video_processor.__init__ calls it; will be removed in the
        # next mechanical cleanup pass once all hot paths are confirmed.
        return None

    def _start_nav_prefetcher(self) -> None:
        return None

    def _stop_nav_prefetcher(self) -> None:
        return None

    def _init_arrow_async(self) -> None:
        # Arrow nav is fully libmpv-driven now (frame-step + ring + seek).
        # No worker thread needed.
        return None

    def _stop_arrow_async(self) -> None:
        return None

    def _clear_cache(self) -> None:
        return None

    def _clear_nav_state(self) -> None:
        return None

    @property
    def buffer_info(self) -> dict:
        return {"size": 0, "bytes": 0, "budget": 0, "fill_pct": 0.0,
                "hit_rate_5s": 0.0, "capacity": 0, "start": -1, "end": -1,
                "current": getattr(self, "current_frame_index", -1)}

    def get_cached_frame(self, frame_index: int) -> Optional[np.ndarray]:
        return None

    def prefetch_around(self, center_frame: int, margin: int = 15) -> None:
        return None

    def _nav_dbg_enabled(self) -> bool:
        try:
            return self.app.app_settings.config.navigation.debug_logging
        except Exception:
            return False

    def _nav_to_target(self, target_frame: int) -> Optional[np.ndarray]:
        target_frame = int(target_frame)
        prev = getattr(self, 'current_frame_index', -1)
        with self.trace.span("arrow_nav", frame=target_frame, prev=prev) as _root:
            # Logical cursor latch on arrow nav: the cursor jumps to
            # target_frame instantly; mpv catches up async.
            fps = getattr(self, 'fps', 0) or 0
            if fps and fps > 0:
                self.playhead_override_ms = target_frame * 1000.0 / fps
            else:
                self.playhead_override_ms = None

            # Drive libmpv. delta=+1 -> frame-step (instant, no decode from
            # keyframe). delta=-1 -> ring replay if cached, else exact seek
            # (frame-back-step is unreliable on long-GOP HEVC). Larger
            # deltas fall back to exact seek. command_async coalesces
            # rapid bursts inside libmpv.
            disp_getter = getattr(self, "_get_mpv_display", None)
            disp = disp_getter() if disp_getter else None
            mpv_alive = disp is not None and getattr(disp, "is_alive", False)
            delta = target_frame - prev
            ring_hit = False
            if delta == -1:
                gui = getattr(getattr(self, 'app', None), 'gui_instance', None)
                ring = getattr(gui, 'back_frame_ring', None) if gui else None
                if ring is not None:
                    tex = ring.get(target_frame)
                    if tex is not None:
                        gui._ring_override = (int(target_frame), int(tex))
                        ring_hit = True
            with self.trace.span("arrow_nav.mpv_drive", delta=delta,
                                  ring_hit=ring_hit):
                try:
                    if mpv_alive and delta == 1:
                        disp.step_forward()
                    elif mpv_alive and (ring_hit or delta == -1):
                        if hasattr(self, "_mpv_seek_to_frame_ex"):
                            self._mpv_seek_to_frame_ex(target_frame, exact=True)
                    elif hasattr(self, "_mpv_seek_to_frame_ex"):
                        self._mpv_seek_to_frame_ex(target_frame, exact=True)
                except Exception as e:
                    self.logger.debug(f"arrow nav mpv drive failed: {e}")

            self.current_frame_index = target_frame
            _root.add(hit=ring_hit, mpv_alive=mpv_alive)
        if self._nav_dbg_enabled():
            t_ms = (_time.perf_counter()) * 0.0  # debug log only on enable path
            self.logger.info(
                f"NAV nav_to from={prev} to={target_frame} "
                f"delta={target_frame-prev:+d} ring_hit={ring_hit}")
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

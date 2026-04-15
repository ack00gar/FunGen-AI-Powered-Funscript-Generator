"""VideoProcessor NavBufferMixin — arrow-key navigation backed by a deque
of recently-played frames and the PyAV source for misses.

The buffer is populated by the PyAV processing loop as frames are decoded,
so sequential forward/backward arrow-key presses land on instant O(1)
lookups. On miss, we fall through to ``pyav_source.get_frame(target)``
which is ~300ms on 8K VR — no subprocess respawn.
"""

import threading
from typing import Optional

import numpy as np


POINT_NAV_PREFETCH_MARGIN = 15


class NavBufferMixin:
    """Mixin fragment for VideoProcessor."""

    def _compute_nav_buffer_size(self) -> int:
        """Compute arrow-nav buffer size based on available RAM.

        Each frame is yolo_input_size^2 * 3 bytes. Cap at ~10% of free RAM,
        floor 120 frames, ceiling 1800 (~1 minute at 30fps).
        """
        frame_bytes = self.yolo_input_size * self.yolo_input_size * 3
        try:
            import psutil
            avail = psutil.virtual_memory().available
            budget = int(avail * 0.10)
            max_frames = max(120, budget // frame_bytes)
            max_frames = min(max_frames, 1800)
        except (ImportError, OSError):
            max_frames = 300
        self.logger.debug(
            f"Arrow-nav buffer: {max_frames} frames "
            f"({max_frames * frame_bytes / (1024*1024):.0f} MB)"
        )
        return max_frames

    # ------------------------------------------------------------------ buffer

    def _buffer_lookup(self, target_frame: int) -> Optional[np.ndarray]:
        """O(1) lookup in the frame buffer by offset from first entry."""
        with self._frame_buffer_lock:
            if not self._frame_buffer:
                return None
            first_idx = self._frame_buffer[0][0]
            offset = target_frame - first_idx
            if 0 <= offset < len(self._frame_buffer):
                stored_idx, frame_data = self._frame_buffer[offset]
                if stored_idx == target_frame:
                    return frame_data
            return None

    def _buffer_append(self, frame_index: int, frame_data: np.ndarray):
        """Append a frame to the buffer; clear on non-contiguous append."""
        with self._frame_buffer_lock:
            if self._frame_buffer:
                last_idx = self._frame_buffer[-1][0]
                if frame_index != last_idx + 1:
                    self._frame_buffer.clear()
            self._frame_buffer.append((frame_index, frame_data))

    def _clear_nav_state(self):
        """Clear buffered nav frames. Called on seeks and teardown."""
        with self._frame_buffer_lock:
            self._frame_buffer.clear()

    @property
    def buffer_info(self) -> dict:
        """Return buffer stats for status strip display."""
        with self._frame_buffer_lock:
            size = len(self._frame_buffer)
            capacity = self._frame_buffer.maxlen or 0
            start = self._frame_buffer[0][0] if self._frame_buffer else -1
            end = self._frame_buffer[-1][0] if self._frame_buffer else -1
        return {'size': size, 'capacity': capacity, 'start': start, 'end': end,
                'current': self.current_frame_index}

    def _clear_cache(self):
        with self.frame_cache_lock:
            if self.frame_cache is not None:
                cache_len = 0
                try:
                    cache_len = len(self.frame_cache)
                except (TypeError, AttributeError):
                    pass
                if cache_len > 0:
                    self.logger.debug(f"Clearing frame cache (had {cache_len} items).")
                    self.frame_cache.clear()
        self._clear_nav_state()

    # --------------------------------------------------------------- arrow nav

    def _nav_to_target(self, target_frame: int) -> Optional[np.ndarray]:
        """Shared path for both forward and backward arrow nav: buffer hit
        first, fall through to the PyAV source."""
        frame = self._buffer_lookup(target_frame)
        if frame is not None:
            self.current_frame_index = target_frame
            return frame
        if self.pyav_source is None:
            self.logger.warning(f"Nav miss and no PyAV source for frame {target_frame}")
            return None
        frame = self.pyav_source.get_frame(target_frame, timeout=2.0)
        if frame is None:
            self.logger.warning(f"PyAV get_frame({target_frame}) returned None")
            return None
        self._buffer_append(target_frame, frame)
        self.current_frame_index = target_frame
        return frame

    def arrow_nav_forward(self, target_frame: int) -> Optional[np.ndarray]:
        """Navigate forward: buffer hit, otherwise PyAV get_frame."""
        if self.arrow_nav_in_progress:
            return None
        self.arrow_nav_in_progress = True
        try:
            return self._nav_to_target(target_frame)
        finally:
            self.arrow_nav_in_progress = False

    def arrow_nav_backward(self, target_frame: int) -> Optional[np.ndarray]:
        """Navigate backward: buffer hit, otherwise PyAV get_frame."""
        return self._nav_to_target(target_frame)

    def prefetch_around(self, center_frame: int, margin: int = POINT_NAV_PREFETCH_MARGIN):
        """Warm ±margin frames around ``center_frame`` into the nav buffer
        using the PyAV source. Runs in a background thread so the caller
        (usually a UI click) returns immediately."""
        total = getattr(self, 'total_frames', 0) or 0
        if total <= 0 or not self.video_path or self.pyav_source is None:
            return

        target_end = min(total - 1, center_frame + margin)
        with self._frame_buffer_lock:
            if self._frame_buffer:
                buf_end = self._frame_buffer[-1][0]
                if buf_end >= target_end:
                    return

        def _fill():
            src = self.pyav_source
            if src is None: return
            read = 0
            for idx in range(max(0, center_frame), target_end + 1):
                frame = self._buffer_lookup(idx)
                if frame is not None:
                    continue
                frame = src.get_frame(idx, timeout=2.0)
                if frame is None:
                    break
                self._buffer_append(idx, frame)
                read += 1
            if read > 0:
                self.logger.debug(f"Prefetch: warmed {read} frames near {center_frame}")

        t = threading.Thread(target=_fill, daemon=True, name="NavPrefetch")
        t.start()

"""Recording mode capture for mouse-to-funscript recording.

Captures mouse Y position mapped to 0-100 funscript values while
video plays, then simplifies the result using Ramer-Douglas-Peucker.
"""
from typing import List, Dict, Optional


class RecordingCapture:
    """Captures mouse position samples during video playback.

    Usage::

        cap = RecordingCapture()
        cap.start_recording()
        # Each frame while recording:
        cap.capture_frame(time_ms, mouse_y_normalized)
        # When done:
        simplified = cap.stop_recording(epsilon=2.0)
    """

    def __init__(self, capture_fps: int = 60):
        self._capture_fps = capture_fps
        self._min_interval_ms = 1000.0 / max(1, capture_fps)
        self._samples: List[Dict] = []
        self._is_recording = False
        self._last_capture_ms = 0.0
        self.input_delay_ms: int = 0  # Subtracted from time_ms (controller lag compensation)

    @property
    def is_recording(self) -> bool:
        return self._is_recording

    @property
    def sample_count(self) -> int:
        return len(self._samples)

    def start_recording(self):
        """Begin a new recording session."""
        self._samples.clear()
        self._is_recording = True
        self._last_capture_ms = 0.0

    def capture_frame(self, time_ms: float, position_0_100: float):
        """Record one sample.

        Args:
            time_ms: Current video time in milliseconds
            position_0_100: Mouse Y mapped to 0-100 funscript range
        """
        if not self._is_recording:
            return

        # Apply input delay compensation (controller lag offset)
        corrected_time = time_ms - self.input_delay_ms

        # Rate-limit captures
        if self._samples and (corrected_time - self._last_capture_ms) < self._min_interval_ms:
            return

        pos = max(0, min(100, int(round(position_0_100))))
        self._samples.append({'at': int(corrected_time), 'pos': pos})
        self._last_capture_ms = corrected_time

    def stop_recording(self, epsilon: float = 2.0) -> List[Dict]:
        """Stop recording and return simplified actions.

        Args:
            epsilon: RDP simplification tolerance (higher = fewer points)

        Returns:
            Simplified list of funscript actions
        """
        self._is_recording = False

        if len(self._samples) < 2:
            return list(self._samples)

        # Apply RDP simplification
        return self._simplify_rdp(self._samples, epsilon)

    def _simplify_rdp(self, actions: List[Dict], epsilon: float) -> List[Dict]:
        """Ramer-Douglas-Peucker simplification of recorded actions.

        Reduces point count while preserving the shape of the motion.
        """
        if len(actions) <= 2:
            return list(actions)

        try:
            import numpy as np

            # Convert to 2D points (time, position)
            points = np.array([[a['at'], a['pos']] for a in actions], dtype=np.float64)

            # Normalize dimensions for distance calculation
            time_range = points[-1, 0] - points[0, 0]
            if time_range > 0:
                points_norm = points.copy()
                points_norm[:, 0] = (points[:, 0] - points[0, 0]) / time_range * 100.0
            else:
                points_norm = points.copy()

            # Run RDP
            mask = self._rdp_mask(points_norm, epsilon)

            return [actions[i] for i in range(len(actions)) if mask[i]]
        except Exception:
            # Fallback: return as-is
            return list(actions)

    def _rdp_mask(self, points, epsilon: float) -> List[bool]:
        """Compute RDP keep/discard mask."""
        import numpy as np

        n = len(points)
        mask = [False] * n
        mask[0] = True
        mask[-1] = True

        stack = [(0, n - 1)]
        while stack:
            start, end = stack.pop()
            if end - start <= 1:
                continue

            # Find point with maximum distance from line segment
            line_start = points[start]
            line_end = points[end]
            line_vec = line_end - line_start
            line_len = np.linalg.norm(line_vec)

            if line_len == 0:
                # All points on same location, keep first and last only
                continue

            line_unit = line_vec / line_len

            max_dist = 0
            max_idx = start

            for i in range(start + 1, end):
                vec = points[i] - line_start
                proj = np.dot(vec, line_unit)
                proj = max(0, min(line_len, proj))
                closest = line_start + proj * line_unit
                dist = np.linalg.norm(points[i] - closest)

                if dist > max_dist:
                    max_dist = dist
                    max_idx = i

            if max_dist > epsilon:
                mask[max_idx] = True
                stack.append((start, max_idx))
                stack.append((max_idx, end))

        return mask

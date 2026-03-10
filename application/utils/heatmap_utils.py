"""Heatmap color mapping for funscript speed visualization.

Maps funscript segment speeds to a gradient optimised for dark backgrounds:
Steel-blue -> DodgerBlue -> Cyan -> Green -> Yellow -> Red

Normalized at 400 units/second (the community-accepted device speed limit).
"""
import numpy as np
from typing import Tuple, List, Dict

# Gradient stops (normalized 0.0-1.0 positions mapped to RGBA)
# Position thresholds as fraction of max speed (400 u/s default)
# Lowest stop uses a visible steel-blue instead of black so flat/slow
# segments remain clearly visible on the dark canvas background.
_SPEED_GRADIENT = [
    (0.00, (0.30, 0.35, 0.50, 1.0)),     # Steel-blue — stationary / flat
    (0.05, (0.20, 0.45, 0.90, 1.0)),     # DodgerBlue — slow
    (0.20, (0.0, 0.9, 0.9, 1.0)),        # Cyan — moderate
    (0.40, (0.0, 0.8, 0.2, 1.0)),        # Green — medium
    (0.65, (0.9, 0.9, 0.1, 1.0)),        # Yellow — fast
    (1.00, (0.9, 0.1, 0.1, 1.0)),        # Red — device limit
]


def _lerp_color(c1: Tuple, c2: Tuple, t: float) -> Tuple[float, float, float, float]:
    """Linear interpolation between two RGBA colors."""
    return (
        c1[0] + (c2[0] - c1[0]) * t,
        c1[1] + (c2[1] - c1[1]) * t,
        c1[2] + (c2[2] - c1[2]) * t,
        c1[3] + (c2[3] - c1[3]) * t,
    )


class HeatmapColorMapper:
    """Maps speeds to heatmap colors.

    Usage::

        mapper = HeatmapColorMapper()
        rgba = mapper.speed_to_color_rgba(250.0)  # moderate speed -> greenish
    """

    def __init__(self, max_speed: float = 400.0):
        self.max_speed = max(1.0, max_speed)
        self._gradient = _SPEED_GRADIENT

    def speed_to_color_rgba(self, speed: float) -> Tuple[float, float, float, float]:
        """Convert a single speed value (units/sec) to an RGBA color tuple."""
        t = min(1.0, max(0.0, abs(speed) / self.max_speed))

        # Find the two gradient stops that bracket t
        for i in range(len(self._gradient) - 1):
            pos0, col0 = self._gradient[i]
            pos1, col1 = self._gradient[i + 1]
            if t <= pos1:
                seg_t = (t - pos0) / max(0.001, pos1 - pos0)
                return _lerp_color(col0, col1, seg_t)

        # Clamp to final stop
        return self._gradient[-1][1]

    def speeds_to_colors_rgba(self, speeds: np.ndarray) -> np.ndarray:
        """Vectorized: convert array of speeds to Nx4 RGBA float array."""
        t = np.clip(np.abs(speeds) / self.max_speed, 0.0, 1.0)
        result = np.zeros((len(t), 4), dtype=np.float32)

        for i in range(len(self._gradient) - 1):
            pos0, col0 = self._gradient[i]
            pos1, col1 = self._gradient[i + 1]
            mask = (t >= pos0) & (t <= pos1)
            if not np.any(mask):
                continue
            seg_t = (t[mask] - pos0) / max(0.001, pos1 - pos0)
            for ch in range(4):
                result[mask, ch] = col0[ch] + (col1[ch] - col0[ch]) * seg_t

        # Handle values at exactly 1.0 (already covered by last segment)
        return result

    def speeds_to_colors_u32(self, speeds: np.ndarray) -> np.ndarray:
        """Vectorized: convert speeds to packed imgui u32 color values.

        Avoids per-segment get_color_u32_rgba calls in the draw loop.
        Packing: R | (G<<8) | (B<<16) | (A<<24)  (imgui's ABGR layout).
        """
        rgba = self.speeds_to_colors_rgba(speeds)
        # Convert float [0,1] to uint8 [0,255] and pack
        rgba_u8 = (rgba * 255.0 + 0.5).astype(np.uint32)
        return rgba_u8[:, 0] | (rgba_u8[:, 1] << 8) | (rgba_u8[:, 2] << 16) | (rgba_u8[:, 3] << 24)

    @staticmethod
    def compute_segment_speeds(actions: List[Dict], ats_np=None, poss_np=None) -> np.ndarray:
        """Compute speed (units/sec) for each segment between consecutive actions.

        Args:
            actions: List of action dicts (used if ats_np/poss_np not provided).
            ats_np: Optional pre-built float array of timestamps.
            poss_np: Optional pre-built float array of positions.

        Returns array of length len(actions)-1 with speed for each segment.
        """
        if len(actions) < 2:
            return np.array([], dtype=np.float32)

        if ats_np is not None and poss_np is not None:
            # Guard against cached numpy arrays being out of sync length-wise
            min_len = min(ats_np.shape[0], poss_np.shape[0])
            if min_len == 0:
                return np.array([], dtype=np.float32)
            ats = ats_np[:min_len].astype(np.float64)
            poss = poss_np[:min_len].astype(np.float64)
        else:
            ats = np.array([a['at'] for a in actions], dtype=np.float64)
            poss = np.array([a['pos'] for a in actions], dtype=np.float64)

        if len(ats) != len(poss):
            min_len = min(len(ats), len(poss))
            ats = ats[:min_len]
            poss = poss[:min_len]

        if len(ats) < 2:
            return np.array([], dtype=np.float32)

        dt = np.diff(ats)  # ms
        dp = np.abs(np.diff(poss))  # position units

        # Avoid division by zero for simultaneous points
        dt_safe = np.where(dt > 0, dt, 1.0)
        speeds = (dp / dt_safe) * 1000.0  # units/sec

        return speeds.astype(np.float32)

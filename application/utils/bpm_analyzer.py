"""BPM (Beats Per Minute) analysis and tempo overlay utilities.

Provides tap-tempo BPM detection and beat grid configuration for
timeline visualization and beat-snapping.
"""
import time
from dataclasses import dataclass, field
from typing import List, Optional


SUBDIVISION_LABELS = [
    "whole measures", "2nd measures", "4th measures", "8th measures",
    "12th measures", "16th measures", "24th measures", "32nd measures",
]
SUBDIVISION_VALUES = [1, 2, 4, 8, 12, 16, 24, 32]


@dataclass
class BPMOverlayConfig:
    """Configuration for BPM/beat grid overlay."""
    bpm: float = 120.0
    offset_ms: float = 0.0
    subdivision: int = 1  # index into SUBDIVISION_VALUES
    snap_to_beat: bool = False

    @property
    def subdivision_value(self) -> int:
        """Actual subdivision multiplier from the dropdown index."""
        if 0 <= self.subdivision < len(SUBDIVISION_VALUES):
            return SUBDIVISION_VALUES[self.subdivision]
        return 1

    @property
    def beat_interval_ms(self) -> float:
        """Time between beats at current BPM and subdivision."""
        if self.bpm <= 0:
            return 1000.0
        return 60000.0 / (self.bpm * self.subdivision_value)


class TapTempo:
    """Accumulate taps to compute average BPM.

    Resets if more than `timeout_s` passes between taps.
    """

    def __init__(self, timeout_s: float = 3.0, max_taps: int = 16):
        self._taps: List[float] = []
        self._timeout_s = timeout_s
        self._max_taps = max_taps

    def tap(self) -> Optional[float]:
        """Record a tap and return current BPM estimate, or None if insufficient taps."""
        now = time.monotonic()

        # Reset if too much time has passed since last tap
        if self._taps and (now - self._taps[-1]) > self._timeout_s:
            self._taps.clear()

        self._taps.append(now)

        # Limit tap history
        if len(self._taps) > self._max_taps:
            self._taps = self._taps[-self._max_taps:]

        # Need at least 2 taps to compute BPM
        if len(self._taps) < 2:
            return None

        # Average interval between taps
        intervals = [self._taps[i + 1] - self._taps[i]
                     for i in range(len(self._taps) - 1)]
        avg_interval = sum(intervals) / len(intervals)

        if avg_interval <= 0:
            return None

        return 60.0 / avg_interval

    def reset(self):
        """Clear all recorded taps."""
        self._taps.clear()

    @property
    def tap_count(self) -> int:
        return len(self._taps)

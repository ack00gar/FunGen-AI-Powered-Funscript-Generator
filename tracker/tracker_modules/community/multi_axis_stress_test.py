"""
Multi-Axis Stress Test Tracker — generates random funscript data on all axes.

Purpose:
  - Stress test the multi-axis pipeline end-to-end
  - Reference implementation for community devs building multi-axis trackers
  - Demonstrates TrackerResult.multi_axis_data usage with arbitrary axis names

Generates sine-wave-based patterns (not pure random) so output is visually
verifiable: each axis uses a different frequency and phase offset.

Author: FunGen / walton (requested)
Version: 1.0.0
"""

import math
import numpy as np
from typing import Dict, Any, Optional, List

try:
    from ..core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
except ImportError:
    from tracker.tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult


# Axis configs: (name, frequency_hz, phase_offset_rad, amplitude 0-50)
_AXIS_CONFIGS = [
    ("stroke",  0.5,   0.0,          50),   # Slow full-range
    ("roll",    0.3,   math.pi / 4,  35),   # Slower, offset
    ("pitch",   0.25,  math.pi / 2,  30),
    ("twist",   0.4,   math.pi,      25),
    ("sway",    0.15,  math.pi / 3,  20),
    ("surge",   0.2,   math.pi / 6,  15),
]


class MultiAxisStressTestTracker(BaseTracker):
    """Generates deterministic multi-axis funscript data for pipeline testing.

    Each axis produces a sine wave at a unique frequency and phase,
    making output easy to visually verify in the timeline.
    """

    def __init__(self):
        super().__init__()
        self._action_interval_ms = 100  # One point every 100ms (10 Hz)
        self._last_action_time = -9999
        self._noise_scale = 5.0  # Small random jitter added to sine

    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="multi_axis_stress_test",
            display_name="Multi-Axis Stress Test",
            description=(
                "Generates sine-wave funscript data on all 6 axes simultaneously. "
                "For stress testing the multi-axis pipeline and as a reference "
                "implementation for community tracker developers."
            ),
            category="community",
            version="1.0.0",
            author="FunGen Community",
            tags=["stress-test", "multi-axis", "example", "debug"],
            requires_roi=False,
            supports_dual_axis=True,
            primary_axis="stroke",
            secondary_axis="roll",
            additional_axes=["pitch", "twist", "sway", "surge"],
        )

    def initialize(self, app_instance, **kwargs) -> bool:
        self.app = app_instance
        if hasattr(app_instance, 'app_settings'):
            settings = app_instance.app_settings
            self._action_interval_ms = settings.get(
                'stress_test_interval_ms', 100)
            self._noise_scale = settings.get(
                'stress_test_noise_scale', 5.0)
        self._initialized = True
        self.logger.info("Multi-Axis Stress Test tracker initialized")
        return True

    def start_tracking(self) -> bool:
        self.tracking_active = True
        self._last_action_time = -9999
        self.logger.info("Stress test tracking started")
        return True

    def stop_tracking(self) -> bool:
        self.tracking_active = False
        self.logger.info("Stress test tracking stopped")
        return True

    def process_frame(self, frame: np.ndarray, frame_time_ms: int,
                      frame_index: Optional[int] = None) -> TrackerResult:
        action_log = None
        secondary_action_log = None
        multi_axis = {}

        if self.tracking_active:
            if frame_time_ms - self._last_action_time >= self._action_interval_ms:
                self._last_action_time = frame_time_ms
                t_sec = frame_time_ms / 1000.0

                for axis_name, freq, phase, amp in _AXIS_CONFIGS:
                    # Deterministic sine + small noise for realism
                    base = math.sin(2 * math.pi * freq * t_sec + phase)
                    noise = np.random.uniform(-1, 1) * self._noise_scale
                    pos = int(round(50 + base * amp + noise))
                    pos = max(0, min(100, pos))

                    action = {'at': frame_time_ms, 'pos': pos}

                    if axis_name == "stroke":
                        action_log = [action]
                    elif axis_name == "roll":
                        secondary_action_log = [action]
                    else:
                        multi_axis.setdefault(axis_name, []).append(action)

        # Minimal overlay: stamp text on frame
        display_frame = frame.copy() if frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)
        if self.tracking_active:
            import cv2
            cv2.putText(display_frame, "STRESS TEST - ALL AXES",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 0), 2)
            if frame_time_ms > 0:
                cv2.putText(display_frame, f"t={frame_time_ms}ms",
                            (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (200, 200, 200), 1)

        return TrackerResult(
            processed_frame=display_frame,
            action_log=action_log,
            secondary_action_log=secondary_action_log,
            multi_axis_data=multi_axis if multi_axis else None,
            status_message=f"Stress test: {len(_AXIS_CONFIGS)} axes @ {1000 // self._action_interval_ms} Hz"
        )

    def get_settings_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "stress_test_interval_ms": {
                    "type": "integer",
                    "title": "Action Interval (ms)",
                    "description": "Time between generated actions per axis",
                    "minimum": 20,
                    "maximum": 1000,
                    "default": 100
                },
                "stress_test_noise_scale": {
                    "type": "number",
                    "title": "Noise Scale",
                    "description": "Random jitter amplitude (0 = pure sine)",
                    "minimum": 0.0,
                    "maximum": 50.0,
                    "default": 5.0
                }
            }
        }

    def get_status_info(self) -> Dict[str, Any]:
        return {
            "tracking": self.tracking_active,
            "axes": len(_AXIS_CONFIGS),
            "interval_ms": self._action_interval_ms,
        }

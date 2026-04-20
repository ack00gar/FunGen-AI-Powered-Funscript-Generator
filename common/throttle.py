"""
Rate and speed limiting helpers used by sync loops, device output paths, and
broadcast workers.

  RateLimiter   -- "max N events per second" gate (Hz throttle).
  SpeedLimiter  -- "value cannot move faster than X units per second" smoother.

Both are stateful and not thread-safe; create one per axis / per producer.
"""

import time
from typing import Optional


class RateLimiter:
    """
    Hz-based throttle. Returns True when an event is allowed and False to drop.

        rl = RateLimiter(60)            # 60 Hz
        if rl.allow():                  # call inside the work loop
            send_packet()
    """

    def __init__(self, hz: float):
        self._interval = 1.0 / hz if hz > 0 else 0.0
        self._next_allowed = 0.0

    def allow(self) -> bool:
        if self._interval <= 0:
            return True
        now = time.monotonic()
        if now >= self._next_allowed:
            # Anchor to the current tick so a slow caller does not drift.
            self._next_allowed = now + self._interval
            return True
        return False

    def reset(self) -> None:
        self._next_allowed = 0.0


class SpeedLimiter:
    """
    Per-axis units/sec clamp.

    Use case: an axis just jumped 80 units in one tick (because a script
    keyframe fired) which would slam the device. The limiter caps the per-call
    delta to `max_units_per_second * dt_since_last_call`, returning the
    clamped value.

        sl = SpeedLimiter(max_units_per_second=400)
        smoothed = sl.step(target_value)
    """

    def __init__(self, max_units_per_second: float):
        self.max_ups = max_units_per_second
        self._last_value: Optional[float] = None
        self._last_time: float = 0.0

    def step(self, target: float, now: Optional[float] = None) -> float:
        if now is None:
            now = time.monotonic()
        if self._last_value is None or self.max_ups <= 0:
            self._last_value = target
            self._last_time = now
            return target

        dt = max(0.0, now - self._last_time)
        max_delta = self.max_ups * dt
        delta = target - self._last_value
        if abs(delta) <= max_delta:
            new_value = target
        elif delta > 0:
            new_value = self._last_value + max_delta
        else:
            new_value = self._last_value - max_delta

        self._last_value = new_value
        self._last_time = now
        return new_value

    def reset(self) -> None:
        """Forget last value -- next step() seeds with the supplied target."""
        self._last_value = None
        self._last_time = 0.0

    @property
    def last_value(self) -> Optional[float]:
        return self._last_value

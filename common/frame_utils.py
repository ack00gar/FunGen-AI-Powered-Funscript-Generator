"""
Frame/time conversion utilities.

Centralizes ms-to-frame and frame-to-ms conversions so every call site
uses round() (nearest frame) instead of int() (truncation).  The maximum
error with round() is 0.5 ms, which is always less than half a frame at
any standard fps, guaranteeing perfect round-trip: frame -> ms -> frame.
"""


def ms_to_frame(ms: float, fps: float) -> int:
    """Convert milliseconds to the nearest frame index."""
    if fps <= 0 or ms < 0:
        return 0
    return int(round(ms * fps / 1000.0))


def frame_to_ms(frame: int, fps: float) -> int:
    """Convert a frame index to the nearest whole millisecond."""
    if fps <= 0 or frame < 0:
        return 0
    return int(round(frame / fps * 1000.0))

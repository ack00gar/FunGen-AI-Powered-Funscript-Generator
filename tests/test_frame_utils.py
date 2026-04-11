"""Tests for common.frame_utils round-trip correctness."""

import pytest
from common.frame_utils import ms_to_frame, frame_to_ms


@pytest.mark.parametrize("fps", [23.976, 24.0, 25.0, 29.97, 30.0, 59.94, 60.0, 120.0])
def test_round_trip_frame_to_ms_to_frame(fps):
    """frame -> ms -> frame must be identity for all frames."""
    for frame in range(0, 10000):
        ms = frame_to_ms(frame, fps)
        recovered = ms_to_frame(ms, fps)
        assert recovered == frame, (
            f"fps={fps}, frame={frame}, ms={ms}, recovered={recovered}"
        )


def test_zero_fps():
    assert ms_to_frame(1000, 0) == 0
    assert frame_to_ms(30, 0) == 0


def test_negative_fps():
    assert ms_to_frame(1000, -30) == 0
    assert frame_to_ms(30, -30) == 0


def test_negative_inputs():
    assert ms_to_frame(-100, 30) == 0
    assert frame_to_ms(-5, 30) == 0


def test_60fps_frame_1_regression():
    """Old code: int(16.667) = 16, ms_to_frame(16, 60) = 0. Off by 1."""
    assert frame_to_ms(1, 60.0) == 17
    assert ms_to_frame(17, 60.0) == 1


def test_59_94fps_known_regression():
    """The original bug report: frame 73830 at 59.94fps."""
    ms = frame_to_ms(73830, 59.94)
    assert ms_to_frame(ms, 59.94) == 73830

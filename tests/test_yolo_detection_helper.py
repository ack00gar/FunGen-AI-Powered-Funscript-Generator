"""Tests for the YOLO load_model warmup semantics shipped in v0.9.0.

Rationale: the warmup fix pays cold-start cost at load time instead of on
the user's first real frame. We mock ultralytics.YOLO so these tests run
without any model file or GPU.
"""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def yolo_mock():
    """Patch ultralytics.YOLO so load_model returns a MagicMock instance."""
    with patch("ultralytics.YOLO") as mock_yolo:
        instance = MagicMock()
        mock_yolo.return_value = instance
        yield mock_yolo, instance


def test_load_model_without_warmup_does_not_run_forward(yolo_mock):
    mock_yolo, instance = yolo_mock
    from tracker.tracker_modules.helpers.yolo_detection_helper import load_model

    returned = load_model("fake.pt", task="detect")

    mock_yolo.assert_called_once_with("fake.pt", task="detect")
    instance.assert_not_called()
    assert returned is instance


def test_load_model_with_warmup_runs_one_forward_with_zeros(yolo_mock):
    mock_yolo, instance = yolo_mock
    from tracker.tracker_modules.helpers.yolo_detection_helper import load_model

    load_model("fake.pt", task="detect", warmup_device="cpu", warmup_imgsz=320)

    assert instance.call_count == 1
    call = instance.call_args
    frame = call.args[0]
    assert isinstance(frame, np.ndarray)
    assert frame.shape == (320, 320, 3)
    assert frame.dtype == np.uint8
    assert frame.sum() == 0  # zero-filled dummy
    assert call.kwargs["device"] == "cpu"
    assert call.kwargs["imgsz"] == 320
    assert call.kwargs["verbose"] is False


def test_load_model_warmup_failure_is_silently_swallowed():
    """Dummy forward raising must not break load_model."""
    from tracker.tracker_modules.helpers import yolo_detection_helper as helper

    with patch.object(helper, "YOLO", create=True) as _:
        pass  # sanity: symbol exists or is lazy-imported

    with patch("ultralytics.YOLO") as mock_yolo:
        instance = MagicMock(side_effect=RuntimeError("simulated device OOM"))
        mock_yolo.return_value = instance
        from tracker.tracker_modules.helpers.yolo_detection_helper import load_model

        returned = load_model("fake.pt", warmup_device="mps", warmup_imgsz=640)
        assert returned is instance


def test_load_model_default_task_is_detect(yolo_mock):
    mock_yolo, _ = yolo_mock
    from tracker.tracker_modules.helpers.yolo_detection_helper import load_model

    load_model("fake.pt")
    assert mock_yolo.call_args.kwargs.get("task") == "detect"


def test_load_model_warmup_only_when_device_provided(yolo_mock):
    """Passing only imgsz (no device) still skips the warmup forward."""
    _, instance = yolo_mock
    from tracker.tracker_modules.helpers.yolo_detection_helper import load_model

    load_model("fake.pt", warmup_imgsz=640)  # no warmup_device
    instance.assert_not_called()

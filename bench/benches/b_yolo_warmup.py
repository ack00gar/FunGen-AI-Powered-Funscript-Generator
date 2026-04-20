"""Cold first-forward vs warm steady-state forward (win #6).

Ultralytics YOLO(path) constructor loads weights but defers device placement
and graph build until first forward. A post-load dummy forward eliminates the
first-frame stutter visible to users opening a video.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np

from ..harness import Report, Sample, _sync
from ..registry import register


@register(
    "yolo_warmup",
    "First forward (cold, after load) vs steady-state forward (win #6).",
)
def run(device: str, imgsz: int, model: str, **_) -> Report:
    if not Path(model).exists():
        raise FileNotFoundError(f"model not found: {model}")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    from ultralytics import YOLO

    rng = np.random.default_rng(0)
    f = rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # BEFORE: plain YOLO(path) then first forward = cold.
    mdl = YOLO(model, task="detect")
    t0 = time.perf_counter()
    mdl(f, device=device, verbose=False, conf=0.25, imgsz=imgsz)
    _sync(device)
    cold_before = time.perf_counter() - t0

    warm = []
    for _ in range(10):
        t0 = time.perf_counter()
        mdl(f, device=device, verbose=False, conf=0.25, imgsz=imgsz)
        _sync(device)
        warm.append(time.perf_counter() - t0)

    # AFTER: load_model with warmup kwarg (the shipped fix). First real forward
    # should now match steady-state.
    from tracker.tracker_modules.helpers.yolo_detection_helper import load_model as _load_model
    mdl2 = _load_model(model, task='detect',
                       warmup_device=device, warmup_imgsz=imgsz)
    _sync(device)
    t0 = time.perf_counter()
    mdl2(f, device=device, verbose=False, conf=0.25, imgsz=imgsz)
    _sync(device)
    cold_after = time.perf_counter() - t0

    r = Report(
        name="yolo_warmup",
        description="Cold vs warm forward, with and without load-time warmup kwarg.",
        device=device,
    )
    r.add(Sample(label="cold: no warmup kwarg", samples_s=[cold_before], device=device))
    r.add(Sample(label="warm: steady state", samples_s=warm, device=device))
    r.add(Sample(label="cold: with warmup kwarg", samples_s=[cold_after], device=device))
    warm_mean = sum(warm) / len(warm) if warm else 0.0
    r.extra["first_frame_penalty_ms_before"] = round((cold_before - warm_mean) * 1000, 2)
    r.extra["first_frame_penalty_ms_after"] = round((cold_after - warm_mean) * 1000, 2)
    return r

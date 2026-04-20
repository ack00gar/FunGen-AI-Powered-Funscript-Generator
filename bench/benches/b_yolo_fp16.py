"""YOLO FP32 vs FP16 inference (win #1).

Passes half=True to ultralytics. On CUDA expect ~2-3x, MPS ~1.5x, CPU neutral
(or worse -- FP16 on CPU often falls back to FP32 with conversion overhead).
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from ..harness import Report, Sample, measure
from ..registry import register


@register(
    "yolo_fp16",
    "FP32 (half=False) vs FP16 (half=True) YOLO forward pass.",
)
def run(device: str, iters: int, warmup: int, imgsz: int, model: str, **_) -> Report:
    if not Path(model).exists():
        raise FileNotFoundError(
            f"model not found: {model}. pass --model <path> or drop one into models/."
        )

    os.environ.setdefault("YOLO_VERBOSE", "False")
    from ultralytics import YOLO
    mdl = YOLO(model, task="detect")

    rng = np.random.default_rng(42)
    pool = [rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(16)]
    idx = {"i": 0}

    def step(half: bool):
        f = pool[idx["i"] % len(pool)]
        idx["i"] += 1
        mdl(f, device=device, verbose=False, conf=0.25, imgsz=imgsz, half=half)

    r = Report(
        name="yolo_fp16",
        description=f"ultralytics YOLO forward (imgsz={imgsz}, iters={iters}).",
        device=device,
    )
    try:
        fp32 = measure(lambda: step(False), iters=iters, warmup=warmup, device=device)
        r.add(Sample(label="fp32 (half=False)", samples_s=fp32, device=device))
    except Exception as e:
        r.extra["fp32_error"] = repr(e)
    try:
        fp16 = measure(lambda: step(True), iters=iters, warmup=warmup, device=device)
        r.add(Sample(label="fp16 (half=True)", samples_s=fp16, device=device))
    except Exception as e:
        r.extra["fp16_error"] = repr(e)
    r.extra["model"] = model
    return r

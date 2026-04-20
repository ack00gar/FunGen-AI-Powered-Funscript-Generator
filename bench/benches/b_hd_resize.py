"""End-to-end _make_processing_frame: HD -> imgsz resize + zero + paste.

Simulates 8K-VR-per-eye HD frame going into 640x640 YOLO input buffer,
comparing current (buf[:]=0) vs pad-only-zero.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..harness import Report, Sample, measure
from ..registry import register


@register(
    "hd_resize",
    "HD (2160x2160) -> imgsz resize + paste; current vs pad-only zero.",
)
def run(iters: int, warmup: int, imgsz: int, **_) -> Report:
    hd_w, hd_h = 2160, 2160
    size = imgsz
    scale = size / max(hd_w, hd_h)
    new_w = int(hd_w * scale) & ~1
    new_h = int(hd_h * scale) & ~1
    new_w = min(new_w, size)
    new_h = min(new_h, size)
    pad_top = (size - new_h) // 2
    pad_bot = size - new_h - pad_top
    pad_left = (size - new_w) // 2
    pad_right = size - new_w - pad_left

    rng = np.random.default_rng(0)
    hd = rng.integers(0, 255, (hd_h, hd_w, 3), dtype=np.uint8)
    buf = np.zeros((size, size, 3), dtype=np.uint8)

    def current():
        resized = cv2.resize(hd, (new_w, new_h), interpolation=cv2.INTER_AREA)
        buf[:] = 0
        buf[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    def pad_only():
        resized = cv2.resize(hd, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if pad_top:
            buf[:pad_top, :, :] = 0
        if pad_bot:
            buf[pad_top + new_h:, :, :] = 0
        if pad_left:
            buf[pad_top:pad_top + new_h, :pad_left, :] = 0
        if pad_right:
            buf[pad_top:pad_top + new_h, pad_left + new_w:, :] = 0
        buf[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    a = measure(current, iters=iters, warmup=warmup)
    b = measure(pad_only, iters=iters, warmup=warmup)

    r = Report(
        name="hd_resize",
        description=f"HD {hd_w}x{hd_h} -> {size}x{size} (INTER_AREA + zero + paste).",
        device="cpu",
    )
    r.add(Sample(label="current (buf[:]=0)", samples_s=a))
    r.add(Sample(label="pad-only zero", samples_s=b))
    return r

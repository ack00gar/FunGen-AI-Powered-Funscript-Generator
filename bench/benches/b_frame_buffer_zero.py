"""Micro: `buf[:] = 0 + paste` vs pad-only zero + paste.

Corresponds to win #4 in the perf audit. Replicates the final paste step of
video_processor.py:_make_processing_frame in isolation. Measures only the
zero + paste cost; no resize.
"""
from __future__ import annotations

import numpy as np

from ..harness import Report, Sample, measure
from ..registry import register


@register(
    "frame_buffer_zero",
    "Full buf[:]=0 + paste vs pad-only zero + paste (video_processor.py:798-800).",
)
def run(iters: int, warmup: int, imgsz: int, **_) -> Report:
    size = imgsz
    new_w = size
    new_h = int(size * 9 / 16) & ~1
    pad_top = (size - new_h) // 2
    pad_bot = size - new_h - pad_top

    buf = np.zeros((size, size, 3), dtype=np.uint8)
    content = np.random.default_rng(0).integers(0, 255, (new_h, new_w, 3), dtype=np.uint8)

    def full_zero():
        buf[:] = 0
        buf[pad_top:pad_top + new_h, 0:new_w] = content

    def pad_only():
        if pad_top:
            buf[:pad_top, :, :] = 0
        if pad_bot:
            buf[pad_top + new_h:, :, :] = 0
        buf[pad_top:pad_top + new_h, 0:new_w] = content

    a = measure(full_zero, iters=iters, warmup=warmup)
    b = measure(pad_only, iters=iters, warmup=warmup)

    r = Report(
        name="frame_buffer_zero",
        description=f"{size}x{size}x3 buffer; content {new_w}x{new_h}; pad {pad_top}+{pad_bot} rows.",
        device="cpu",
    )
    r.add(Sample(label="full buf[:]=0", samples_s=a))
    r.add(Sample(label="pad-only zero", samples_s=b))
    r.extra["note"] = "Pad-only skips zeroing the content region (overwritten by paste anyway)."
    return r

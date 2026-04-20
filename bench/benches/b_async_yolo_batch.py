"""AsyncYoloWorker batch_size=1 vs 4, with tight vs loose timeout (win #5).

Runs a real FunGen AsyncYoloWorker end-to-end with synthetic frames. Measures
total wall time for N frames submit -> all results received.
"""
from __future__ import annotations

import os
import threading
import time
from pathlib import Path

import numpy as np

from ..harness import Report, Sample
from ..registry import register


def _run_worker(mdl, device: str, imgsz: int, frames: list, batch_size: int,
                timeout_s: float) -> tuple:
    from tracker.tracker_modules.helpers.async_yolo_worker import AsyncYoloWorker

    done = threading.Event()
    counter = {"n": 0}
    target = len(frames)

    def on_result(idx, h, w, det_objs, payload):
        counter["n"] += 1
        if counter["n"] >= target:
            done.set()

    w = AsyncYoloWorker(
        mdl, conf=0.25, imgsz=imgsz, device=device,
        on_result=on_result,
        input_queue_size=16,
        batch_size=batch_size,
        batch_timeout_s=timeout_s,
    )
    w.start()
    t0 = time.perf_counter()
    for i, f in enumerate(frames):
        while not w.submit(i, f, payload=None, submit_timeout_s=1.0):
            w.drain()
            if done.is_set():
                break
    # Drain until all frames are accounted for. AsyncYoloWorker posts results
    # to a queue; on_result fires only when the main thread calls drain().
    deadline = time.perf_counter() + 120.0
    while not done.is_set() and time.perf_counter() < deadline:
        w.drain()
        if done.is_set():
            break
        time.sleep(0.001)
    w.drain()
    elapsed = time.perf_counter() - t0
    w.stop()
    return elapsed, w.dropped


@register(
    "async_yolo_batch",
    "AsyncYoloWorker batch=1 vs batch=4; tight (20ms) vs loose (50ms) timeout.",
)
def run(device: str, imgsz: int, frames: int, model: str, warmup: int, **_) -> Report:
    if not Path(model).exists():
        raise FileNotFoundError(f"model not found: {model}")
    os.environ.setdefault("YOLO_VERBOSE", "False")
    from ultralytics import YOLO

    mdl = YOLO(model, task="detect")
    rng = np.random.default_rng(7)
    pool = [rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(frames)]

    # Warmup the model on this device (GPU graph compile, allocator warmup, etc.)
    for _ in range(max(3, warmup)):
        mdl(pool[0], device=device, verbose=False, conf=0.25, imgsz=imgsz)

    r = Report(
        name="async_yolo_batch",
        description=f"{frames} frames through AsyncYoloWorker on {device}.",
        device=device,
    )

    configs = [
        ("batch=1 timeout=20ms", 1, 0.020),
        ("batch=4 timeout=20ms", 4, 0.020),
        ("batch=4 timeout=50ms", 4, 0.050),
        ("batch=8 timeout=50ms", 8, 0.050),
    ]
    for label, bs, to in configs:
        try:
            elapsed, dropped = _run_worker(mdl, device, imgsz, pool, bs, to)
            per_frame = elapsed / frames if frames > 0 else 0.0
            r.add(Sample(
                label=label,
                samples_s=[per_frame],
                device=device,
                meta={"wall_s": elapsed, "fps": frames / elapsed if elapsed else 0, "dropped": dropped},
            ))
        except Exception as e:
            r.extra[f"{label}_error"] = repr(e)

    r.extra["note"] = (
        "Sample values are per-frame amortized. meta.fps is throughput. "
        "If batch>1 rows silently degrade to single-frame, batch is unsupported on this device."
    )
    return r

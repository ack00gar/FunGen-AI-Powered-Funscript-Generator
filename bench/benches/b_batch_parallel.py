"""1 vs 2 vs 4 YOLO processes sharing the device (win #2).

Approximates the parallel-batch scenario for application/batch/batch_worker.py
where N videos are processed concurrently. Each subprocess loads its own model
and runs inference on a private frame pool. Wall clock + aggregate FPS reveals
whether the GPU saturates at 1, 2, or 4 concurrent processes.

This bench spawns processes; on macOS the default start method is 'spawn',
which is what we want (clean Python, no fork-safety traps with ultralytics).
"""
from __future__ import annotations

import multiprocessing as mp
import time
from pathlib import Path

from ..harness import Report, Sample
from ..registry import register


def _worker(model_path: str, device: str, imgsz: int, n_frames: int, seed: int, q):
    import os
    os.environ["YOLO_VERBOSE"] = "False"
    import numpy as np
    from ultralytics import YOLO

    rng = np.random.default_rng(seed)
    mdl = YOLO(model_path, task="detect")
    # warmup once
    mdl(rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8),
        device=device, verbose=False, conf=0.25, imgsz=imgsz)

    frames = [rng.integers(0, 255, (imgsz, imgsz, 3), dtype=np.uint8) for _ in range(n_frames)]
    t0 = time.perf_counter()
    for f in frames:
        mdl(f, device=device, verbose=False, conf=0.25, imgsz=imgsz)
    q.put(time.perf_counter() - t0)


@register(
    "batch_parallel",
    "1 vs 2 vs 4 YOLO processes sharing device; aggregate FPS reveals saturation.",
)
def run(device: str, frames: int, imgsz: int, model: str, **_) -> Report:
    if not Path(model).exists():
        raise FileNotFoundError(f"model not found: {model}")
    ctx = mp.get_context("spawn")
    r = Report(
        name="batch_parallel",
        description=f"{frames} frames/proc, 1..4 parallel procs on {device}.",
        device=device,
    )
    for n_procs in (1, 2, 4):
        q = ctx.Queue()
        procs = [
            ctx.Process(target=_worker, args=(model, device, imgsz, frames, 100 + i, q))
            for i in range(n_procs)
        ]
        t0 = time.perf_counter()
        for p in procs:
            p.start()
        times = [q.get() for _ in range(n_procs)]
        for p in procs:
            p.join()
        wall = time.perf_counter() - t0
        total_frames = frames * n_procs
        fps = total_frames / wall if wall else 0.0
        per_frame = wall / total_frames if total_frames else 0.0
        r.add(Sample(
            label=f"{n_procs} proc",
            samples_s=[per_frame],
            device=device,
            meta={
                "wall_s": round(wall, 3),
                "total_frames": total_frames,
                "agg_fps": round(fps, 1),
                "per_proc_s": [round(t, 3) for t in times],
            },
        ))
    r.extra["how_to_read"] = (
        "If 4-proc agg_fps >= ~3.5x 1-proc fps, the device has headroom; "
        "parallel batch (win #2) is worth shipping. If <1.5x, GPU is already saturated."
    )
    return r

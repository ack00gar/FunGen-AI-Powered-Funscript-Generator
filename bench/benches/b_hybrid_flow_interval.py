"""Compare hybrid_flow signal quality at different YOLO intervals.

Replays the same decoded BGR frames through two fresh tracker
instances (interval=5 vs 10 vs 20 ...), collects the position
streams, and prints timing + signal-correlation metrics.

Deterministic: no live-timing, no threading. YOLO inference still
happens but is awaited synchronously per-frame via the worker
thread so timings reflect real CPU cost without frame drops.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np


def _decode_bgr_frames(video: str, width: int, height: int,
                       frames: int, start_sec: float) -> np.ndarray:
    vf = (
        f"crop=iw/2:ih:0:0,scale={width}:{height}:flags=fast_bilinear,"
        f"format=bgr24"
    )
    cmd = [
        "ffmpeg", "-v", "error", "-an",
        "-ss", str(start_sec),
        "-i", video,
        "-vf", vf,
        "-frames:v", str(frames),
        "-f", "rawvideo", "-pix_fmt", "bgr24", "-",
    ]
    raw = subprocess.check_output(cmd)
    arr = np.frombuffer(raw, dtype=np.uint8)
    need = frames * width * height * 3
    if arr.size < need:
        frames = arr.size // (width * height * 3)
        arr = arr[: frames * width * height * 3]
    return arr.reshape(frames, height, width, 3)


def _run_with_interval(frames: np.ndarray, interval: int, fps: float,
                       warmup: int = 30) -> dict:
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from tracker.tracker_modules.live.hybrid_flow import HybridFlowTracker
    import config.constants as config_constants

    class _App:
        def __init__(self):
            self.yolo_det_model_path = "models/FunGen_YOLO26_VR_POV_n_finetuned.mlpackage"
            self.yolo_input_size = 640
            self.tracking_axis_mode = "both"
            self.single_axis_output_target = "primary"
            self.funscript = None

    app = _App()
    tr = HybridFlowTracker()
    tr._yolo_interval = interval
    if not tr.initialize(app):
        raise RuntimeError("tracker init failed")
    tr.start_tracking()
    tr._yolo_interval = interval

    primary: List[int] = []
    secondary: List[int] = []
    per_frame_ms: List[float] = []

    frame_period_ms = int(round(1000.0 / fps))
    n = frames.shape[0]

    try:
        for i in range(n):
            t0 = time.perf_counter()
            ts_ms = i * frame_period_ms
            result = tr.process_frame(frames[i], ts_ms, frame_index=i)
            dt = (time.perf_counter() - t0) * 1000.0
            per_frame_ms.append(dt)
            dbg = getattr(result, 'debug_info', None) or {}
            primary.append(int(dbg.get('position', 50)))
            secondary.append(int(dbg.get('secondary_position', 50)))
            # Wait for any in-flight YOLO call so interval=10 sees the
            # result by the time the next YOLO cycle is due; otherwise
            # a slow worker would effectively raise the interval.
            while getattr(tr, '_yolo_inflight', False):
                time.sleep(0.001)
    finally:
        tr.stop_tracking()

    arr = np.asarray(per_frame_ms[warmup:], dtype=np.float32)
    return {
        "interval": interval,
        "n_frames": n,
        "primary": np.asarray(primary, dtype=np.int16),
        "secondary": np.asarray(secondary, dtype=np.int16),
        "mean_ms": float(arr.mean()),
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "p99_ms": float(np.percentile(arr, 99)),
    }


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    den = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
    return float((a * b).sum() / den)


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.float32) - b.astype(np.float32)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/Volumes/Crucial/pn/VR/SLR_SLR Originals_Legal Cast_ Agatha_4000p_55127_FISHEYE190.mp4")
    ap.add_argument("--frames", type=int, default=600, help="~10s at 60fps")
    ap.add_argument("--start", type=float, default=300.0)
    ap.add_argument("--fps", type=float, default=60.0)
    ap.add_argument("--w", type=int, default=640)
    ap.add_argument("--h", type=int, default=640)
    ap.add_argument("--intervals", type=int, nargs="+", default=[5, 10, 20])
    ap.add_argument("--warmup", type=int, default=30)
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"video not found: {args.video}", file=sys.stderr)
        return 1

    print(f"decoding {args.frames} frames @ {args.w}x{args.h} from t={args.start}s...")
    frames = _decode_bgr_frames(args.video, args.w, args.h, args.frames, args.start)
    print(f"decoded {frames.shape[0]} frames, {frames.nbytes / 1e6:.1f} MB")

    results = []
    for iv in args.intervals:
        print(f"\n--- interval={iv} ---")
        r = _run_with_interval(frames, iv, args.fps, warmup=args.warmup)
        results.append(r)
        print(f"  per-frame: mean={r['mean_ms']:.2f}ms p50={r['p50_ms']:.2f}ms "
              f"p95={r['p95_ms']:.2f}ms p99={r['p99_ms']:.2f}ms "
              f"theoretical fps = {1000.0 / r['mean_ms']:.0f}")

    print("\nsignal comparison vs interval=" + str(args.intervals[0]) + ":")
    ref = results[0]
    print(f"{'interval':>8} {'p_corr':>8} {'s_corr':>8} "
          f"{'p_mae':>8} {'s_mae':>8} {'speedup':>8}")
    rows = []
    for r in results:
        pc = _corr(ref['primary'], r['primary'])
        sc = _corr(ref['secondary'], r['secondary'])
        pm = _mae(ref['primary'], r['primary'])
        sm = _mae(ref['secondary'], r['secondary'])
        speedup = ref['mean_ms'] / r['mean_ms']
        print(f"{r['interval']:>8} {pc:8.4f} {sc:8.4f} "
              f"{pm:8.3f} {sm:8.3f} {speedup:7.2f}x")
        rows.append({
            "interval": r['interval'],
            "mean_ms": r['mean_ms'], "p50_ms": r['p50_ms'],
            "p95_ms": r['p95_ms'], "p99_ms": r['p99_ms'],
            "p_corr": pc, "s_corr": sc,
            "p_mae": pm, "s_mae": sm,
            "speedup": speedup,
        })

    out_dir = Path(__file__).resolve().parents[2] / "bench_results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_hybrid_flow_interval.json"
    with open(out_path, "w") as f:
        json.dump({"frames": args.frames, "w": args.w, "h": args.h,
                   "fps": args.fps, "rows": rows}, f, indent=2)
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

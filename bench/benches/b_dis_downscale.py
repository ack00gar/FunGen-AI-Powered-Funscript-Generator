"""Compare DIS ULTRAFAST at native ROI vs pre-downscaled ROI.

Decodes a VR clip, crops a centered ROI matching the live-tracking
working size, then runs DIS at native / 1/2 / 1/3 / 1/4 scales.

Reports per-call ms and the drift of magnitude-weighted (dy, dx)
output vs the native reference. dy, dx drives the funscript so
that's what actually matters -- not the full flow field error.

Usage:
    python -m bench.benches.b_dis_downscale [--frames 300] [--roi 400]
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np

DEFAULT_VIDEO = "/Volumes/Crucial/pn/VR/SLR_SLR Originals_Legal Cast_ Agatha_4000p_55127_FISHEYE190.mp4"


def _dis_ultrafast() -> "cv2.DISOpticalFlow":
    return cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)


def _gaussian_spatial(h: int, w: int) -> np.ndarray:
    cx, sx = w * 0.5, w * 0.25
    cy, sy = h * 0.5, h * 0.25
    wx = np.exp(-((np.arange(w, dtype=np.float32) - cx) ** 2) / (2 * sx * sx))
    wy = np.exp(-((np.arange(h, dtype=np.float32) - cy) ** 2) / (2 * sy * sy))
    return np.outer(wy, wx)


def _mag_weighted(flow: np.ndarray, spatial: np.ndarray) -> Tuple[float, float]:
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag2 = fx * fx + fy * fy
    combined = mag2 * spatial
    total = combined.sum()
    if total > 0:
        dy = float((fy * combined).sum() / total)
        dx = float((fx * combined).sum() / total)
    else:
        dy = float(np.median(fy))
        dx = float(np.median(fx))
    return dy, dx


def _decode_rois(video: str, frames: int, roi_px: int) -> np.ndarray:
    """Decode `frames` grayscale ROIs centered in the SBS left half.

    Uses ffmpeg scale -> crop to pull roi_px x roi_px from the centre of
    the left eye (matches live-tracking ROI geometry closely enough for
    the DIS comparison; we just need real motion).
    """
    proc_side = max(roi_px, 640)
    # Crop the left eye, downsample to proc_side x proc_side, then center-crop.
    vf = (
        f"crop=iw/2:ih:0:0,scale={proc_side}:{proc_side}:flags=fast_bilinear,"
        f"crop={roi_px}:{roi_px}:(iw-ow)/2:(ih-oh)/2,format=gray"
    )
    cmd = [
        "ffmpeg",
        "-v", "error",
        "-an",
        "-ss", "300",
        "-i", video,
        "-vf", vf,
        "-frames:v", str(frames),
        "-f", "rawvideo",
        "-pix_fmt", "gray",
        "-",
    ]
    raw = subprocess.check_output(cmd)
    arr = np.frombuffer(raw, dtype=np.uint8)
    need = frames * roi_px * roi_px
    if arr.size < need:
        frames = arr.size // (roi_px * roi_px)
        arr = arr[: frames * roi_px * roi_px]
    return arr.reshape(frames, roi_px, roi_px)


def _run_bench(rois: np.ndarray, scale: float) -> dict:
    dis = _dis_ultrafast()
    n, h0, w0 = rois.shape
    if scale == 1.0:
        pyr = rois
        h, w = h0, w0
    else:
        h = max(64, int(round(h0 * scale)))
        w = max(64, int(round(w0 * scale)))
        pyr = np.empty((n, h, w), dtype=np.uint8)
        for i in range(n):
            cv2.resize(rois[i], (w, h),
                       dst=pyr[i], interpolation=cv2.INTER_AREA)
    spatial = _gaussian_spatial(h, w).astype(np.float32)

    # Warmup: DIS builds internal buffers on the first call.
    for _ in range(3):
        dis.calc(pyr[0], pyr[1], None)

    times_ms = []
    outputs = np.zeros((n - 1, 2), dtype=np.float32)  # dy, dx per pair
    for i in range(1, n):
        t0 = time.perf_counter()
        flow = dis.calc(pyr[i - 1], pyr[i], None)
        dt = (time.perf_counter() - t0) * 1000.0
        times_ms.append(dt)
        dy, dx = _mag_weighted(flow, spatial)
        # Scale output back to native-pixel units so results compare directly.
        outputs[i - 1, 0] = dy / scale
        outputs[i - 1, 1] = dx / scale

    times = np.asarray(times_ms, dtype=np.float32)
    return {
        "scale": scale,
        "w": w,
        "h": h,
        "mean_ms": float(times.mean()),
        "p50_ms": float(np.percentile(times, 50)),
        "p95_ms": float(np.percentile(times, 95)),
        "p99_ms": float(np.percentile(times, 99)),
        "max_ms": float(times.max()),
        "outputs": outputs,
    }


def _error_metrics(ref: np.ndarray, test: np.ndarray) -> dict:
    """Compare test (dy, dx) vs reference. Both are (n, 2)."""
    diff = test - ref
    dy_err = diff[:, 0]
    dx_err = diff[:, 1]
    # Correlation per axis.
    def _corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-9
        return float((a * b).sum() / denom)
    ref_scale = float(np.sqrt((ref ** 2).sum(axis=1).mean()) + 1e-9)
    return {
        "dy_mae": float(np.abs(dy_err).mean()),
        "dx_mae": float(np.abs(dx_err).mean()),
        "dy_rmse": float(np.sqrt((dy_err ** 2).mean())),
        "dx_rmse": float(np.sqrt((dx_err ** 2).mean())),
        "dy_corr": _corr(ref[:, 0], test[:, 0]),
        "dx_corr": _corr(ref[:, 1], test[:, 1]),
        "rel_mae_vs_ref_norm": float(np.abs(diff).mean() / ref_scale),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default=DEFAULT_VIDEO)
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--roi", type=int, default=400,
                    help="Native ROI side length in pixels.")
    ap.add_argument("--scales", type=float, nargs="+",
                    default=[1.0, 0.75, 0.5, 0.33, 0.25])
    args = ap.parse_args()

    if not os.path.exists(args.video):
        print(f"video not found: {args.video}", file=sys.stderr)
        return 1

    print(f"decoding {args.frames} frames at {args.roi}x{args.roi}...")
    rois = _decode_rois(args.video, args.frames, args.roi)
    n = rois.shape[0]
    print(f"decoded {n} frames")

    results = []
    for s in args.scales:
        r = _run_bench(rois, s)
        results.append(r)
        print(f"scale={s:.2f}  {r['w']:4d}x{r['h']:4d}  "
              f"mean={r['mean_ms']:5.2f}ms p50={r['p50_ms']:5.2f}ms "
              f"p95={r['p95_ms']:5.2f}ms p99={r['p99_ms']:5.2f}ms")

    ref = results[0]["outputs"]
    print()
    print("quality vs native reference (scale=1.0):")
    print(f"{'scale':>6} {'mean_ms':>8} {'speedup':>8} {'dy_mae':>8} "
          f"{'dx_mae':>8} {'dy_corr':>8} {'dx_corr':>8} {'rel_mae':>8}")
    base_ms = results[0]["mean_ms"]
    rows = []
    for r in results:
        err = _error_metrics(ref, r["outputs"])
        speedup = base_ms / r["mean_ms"]
        print(f"{r['scale']:6.2f} {r['mean_ms']:8.2f} {speedup:7.2f}x "
              f"{err['dy_mae']:8.4f} {err['dx_mae']:8.4f} "
              f"{err['dy_corr']:8.4f} {err['dx_corr']:8.4f} "
              f"{err['rel_mae_vs_ref_norm']:8.4f}")
        rows.append({
            "scale": r["scale"], "w": r["w"], "h": r["h"],
            "mean_ms": r["mean_ms"], "p50_ms": r["p50_ms"],
            "p95_ms": r["p95_ms"], "p99_ms": r["p99_ms"],
            "speedup": speedup, **err,
        })

    out_dir = Path(__file__).resolve().parents[2] / "bench_results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_dis_downscale_roi{args.roi}_n{n}.json"
    with open(out_path, "w") as f:
        json.dump({"roi": args.roi, "frames": n, "rows": rows}, f, indent=2)
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

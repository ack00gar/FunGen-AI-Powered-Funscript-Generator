"""Measure ffmpeg decode throughput at the tracker input resolution.

Tests several filter chains and thread configurations against a
real VR file to find where we are actually losing FPS.

Goal: isolate whether the live-tracker 200-250 fps cap is
ffmpeg decode itself or the consumer downstream.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List

import numpy as np


def _cmd(ffmpeg: str, video: str, vf: str, frames: int,
         threads: int = 0, start_sec: float = 300.0,
         extra_in: List[str] | None = None) -> List[str]:
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostats"]
    if threads > 0:
        cmd += ["-threads", str(threads)]
    if extra_in:
        cmd += list(extra_in)
    if start_sec > 0.001:
        cmd += ["-ss", f"{start_sec:.6f}"]
    cmd += [
        "-i", video,
        "-an", "-sn",
        "-vf", vf,
        "-frames:v", str(frames),
        "-pix_fmt", "bgr24",
        "-f", "rawvideo", "pipe:1",
    ]
    return cmd


def _time_run(cmd: List[str], frame_bytes: int, frames: int) -> dict:
    t0 = time.perf_counter()
    try:
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            timeout=300,
        )
    except subprocess.TimeoutExpired:
        return {"error": "timeout", "elapsed_s": 300}
    elapsed = time.perf_counter() - t0
    if result.returncode != 0:
        tail = (result.stderr or b"").decode("utf-8", "replace")[-500:]
        return {"error": f"rc={result.returncode}", "stderr": tail,
                "elapsed_s": elapsed}
    got_bytes = len(result.stdout)
    expected = frame_bytes * frames
    got_frames = got_bytes // frame_bytes
    return {
        "elapsed_s": elapsed,
        "frames": got_frames,
        "fps": got_frames / elapsed if elapsed > 0 else 0,
        "bytes_per_s": got_bytes / elapsed if elapsed > 0 else 0,
        "completeness": got_bytes / expected if expected else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", default="/Volumes/Crucial/pn/VR/SLR_SLR Originals_Legal Cast_ Agatha_4000p_55127_FISHEYE190.mp4")
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--start", type=float, default=300.0)
    ap.add_argument("--out_size", type=int, default=640)
    args = ap.parse_args()

    ffmpeg = "ffmpeg"
    w = h = args.out_size
    frame_bytes = w * h * 3

    # Live-tracking filter chain: v360 fisheye 190 crop + dewarp.
    live_vr_vf = (
        "crop=iw/2:ih:0:0,"
        f"v360=fisheye:in_stereo=0:output=sg:"
        f"iv_fov=190:ih_fov=190:d_fov=190:"
        f"v_fov=90:h_fov=90:pitch=0:yaw=0:roll=0:"
        f"w={w}:h={h}:interp=linear,format=bgr24"
    )

    # Baseline without v360 dewarp (crop + scale).
    simple_vr_vf = (
        f"crop=iw/2:ih:0:0,scale={w}:{h}:flags=fast_bilinear,format=bgr24"
    )

    # Scale only, no crop.
    scale_only_vf = (
        f"scale={w}:{h}:flags=fast_bilinear,format=bgr24"
    )

    scenarios = [
        ("live_vr_v360 (current live-tracking chain)", live_vr_vf, 0),
        ("simple_vr_crop_scale (no v360)", simple_vr_vf, 0),
        ("scale_only_no_crop", scale_only_vf, 0),
        ("live_vr_v360 threads=1", live_vr_vf, 1),
        ("live_vr_v360 threads=4", live_vr_vf, 4),
        ("live_vr_v360 threads=8", live_vr_vf, 8),
        ("live_vr_v360 threads=14", live_vr_vf, 14),
        ("simple_vr threads=1", simple_vr_vf, 1),
        ("simple_vr threads=4", simple_vr_vf, 4),
        ("simple_vr threads=8", simple_vr_vf, 8),
        ("simple_vr threads=14", simple_vr_vf, 14),
    ]

    print(f"video: {args.video}")
    print(f"frames: {args.frames}, out: {w}x{h}\n")

    rows = []
    for name, vf, threads in scenarios:
        cmd = _cmd(ffmpeg, args.video, vf, args.frames,
                   threads=threads, start_sec=args.start)
        result = _time_run(cmd, frame_bytes, args.frames)
        if "error" in result:
            print(f"{name:55s}  ERROR: {result['error']}")
            continue
        print(f"{name:55s}  {result['fps']:7.1f} fps "
              f"{result['bytes_per_s']/1e6:6.1f} MB/s  "
              f"{result['elapsed_s']:5.2f}s")
        rows.append({"name": name, "threads": threads, **result})

    import json
    out_dir = Path(__file__).resolve().parents[2] / "bench_results"
    out_dir.mkdir(exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"{stamp}_ffmpeg_decode.json"
    with open(out_path, "w") as f:
        json.dump({"video": args.video, "frames": args.frames,
                   "out_size": args.out_size, "rows": rows}, f, indent=2)
    print(f"\nsaved: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

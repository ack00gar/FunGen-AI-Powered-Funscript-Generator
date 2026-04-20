"""Deeper ffmpeg decode exploration.

Goes beyond the first bench: output formats (BGR vs GRAY vs YUV),
-filter_threads vs -threads, -skip_frame, and PyAV in-process.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import List


def _time_ffmpeg(cmd: List[str], expected_bytes_per_frame: int,
                 frames: int, timeout: float = 120.0) -> dict:
    t0 = time.perf_counter()
    try:
        r = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"error": "timeout"}
    elapsed = time.perf_counter() - t0
    if r.returncode != 0:
        return {"error": f"rc={r.returncode}",
                "stderr": (r.stderr or b"").decode("utf-8", "replace")[-400:]}
    got_bytes = len(r.stdout)
    got_frames = got_bytes // expected_bytes_per_frame
    return {
        "elapsed_s": elapsed,
        "frames": got_frames,
        "fps": got_frames / elapsed if elapsed > 0 else 0,
        "mb_s": (got_bytes / 1e6) / elapsed if elapsed > 0 else 0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video",
                    default="/Volumes/Crucial/pn/VR/SLR_SLR Originals_Legal Cast_ Agatha_4000p_55127_FISHEYE190.mp4")
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--start", type=float, default=300.0)
    ap.add_argument("--out_size", type=int, default=640)
    args = ap.parse_args()

    ffmpeg = "ffmpeg"
    w = h = args.out_size
    bgr_bpf = w * h * 3
    gray_bpf = w * h * 1
    yuv420p_bpf = int(w * h * 1.5)

    VF_V360_BGR = (
        "crop=iw/2:ih:0:0,"
        f"v360=fisheye:in_stereo=0:output=sg:iv_fov=190:ih_fov=190:"
        f"d_fov=190:v_fov=90:h_fov=90:pitch=0:yaw=0:roll=0:"
        f"w={w}:h={h}:interp=linear,format=bgr24")
    VF_V360_GRAY = VF_V360_BGR.replace("format=bgr24", "format=gray")
    VF_V360_YUV = VF_V360_BGR.replace("format=bgr24", "format=yuv420p")
    VF_SIMPLE_BGR = f"crop=iw/2:ih:0:0,scale={w}:{h}:flags=fast_bilinear,format=bgr24"
    VF_SIMPLE_GRAY = f"crop=iw/2:ih:0:0,scale={w}:{h}:flags=fast_bilinear,format=gray"

    def _base(vf: str, pix_fmt: str, threads: int = 0, filter_threads: int = 0,
              skip_frame: str = "", extra_in: List[str] | None = None):
        c = [ffmpeg, "-hide_banner", "-loglevel", "error", "-nostats"]
        if threads > 0:
            c += ["-threads", str(threads)]
        if filter_threads > 0:
            c += ["-filter_threads", str(filter_threads)]
        if skip_frame:
            c += ["-skip_frame", skip_frame]
        if extra_in:
            c += extra_in
        if args.start > 0.001:
            c += ["-ss", f"{args.start:.6f}"]
        c += [
            "-i", args.video, "-an", "-sn", "-vf", vf,
            "-frames:v", str(args.frames),
            "-pix_fmt", pix_fmt, "-f", "rawvideo", "pipe:1",
        ]
        return c

    scenarios = [
        ("v360 BGR (baseline)", _base(VF_V360_BGR, "bgr24"), bgr_bpf),
        ("v360 GRAY (saves 2/3 pipe)", _base(VF_V360_GRAY, "gray"), gray_bpf),
        ("v360 YUV420P (native)", _base(VF_V360_YUV, "yuv420p"), yuv420p_bpf),
        ("simple_crop+scale BGR", _base(VF_SIMPLE_BGR, "bgr24"), bgr_bpf),
        ("simple_crop+scale GRAY", _base(VF_SIMPLE_GRAY, "gray"), gray_bpf),
        ("v360 GRAY filter_threads=4", _base(VF_V360_GRAY, "gray", filter_threads=4), gray_bpf),
        ("v360 GRAY filter_threads=8", _base(VF_V360_GRAY, "gray", filter_threads=8), gray_bpf),
        ("v360 GRAY threads=8 ft=4", _base(VF_V360_GRAY, "gray", threads=8, filter_threads=4), gray_bpf),
        ("v360 GRAY threads=14 ft=4", _base(VF_V360_GRAY, "gray", threads=14, filter_threads=4), gray_bpf),
        ("v360 GRAY skip_frame=nokey", _base(VF_V360_GRAY, "gray", skip_frame="nokey"), gray_bpf),
        ("v360 GRAY skip_frame=noref", _base(VF_V360_GRAY, "gray", skip_frame="noref"), gray_bpf),
    ]

    print(f"video: {args.video}")
    print(f"frames: {args.frames}, out: {w}x{h}\n")

    for name, cmd, bpf in scenarios:
        r = _time_ffmpeg(cmd, bpf, args.frames)
        if "error" in r:
            print(f"{name:48s}  ERROR: {r['error']}")
            if r.get("stderr"):
                print(f"    {r['stderr']}")
            continue
        print(f"{name:48s}  {r['fps']:7.1f} fps  {r['mb_s']:6.1f} MB/s  {r['elapsed_s']:5.2f}s")

    # PyAV: in-process decode (no subprocess spawn).
    print()
    try:
        import av
    except ImportError:
        print("PyAV not installed; skipping in-process decode test.")
        return 0

    print("PyAV in-process (crop+scale manual, gray output)...")
    t0 = time.perf_counter()
    container = av.open(args.video)
    stream = container.streams.video[0]
    stream.thread_type = 'AUTO'
    container.seek(int(args.start * av.time_base))
    n = 0
    for packet in container.demux(stream):
        if n >= args.frames:
            break
        for frame in packet.decode():
            if n >= args.frames:
                break
            # Force decode and conversion
            _ = frame.to_ndarray(format='gray')
            n += 1
    container.close()
    elapsed = time.perf_counter() - t0
    fps = n / elapsed if elapsed > 0 else 0
    print(f"{'PyAV (native size, gray ndarray)':48s}  {fps:7.1f} fps  (decode only, no crop/v360)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

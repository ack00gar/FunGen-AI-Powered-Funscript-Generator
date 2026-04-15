"""v360 playback throughput: subprocess vs PyAV (CPU) vs PyAV (videotoolbox).

Measures sustained post-warmup decode rate of the user's 8K 10-bit VR file
with the production v360 filter chain. Compares:

  1. Subprocess ffmpeg, no hwaccel
  2. PyAV software decode (current integration)
  3. PyAV with videotoolbox hwaccel + hwdownload for the filter chain

The third requires hwdownload before v360 because libavfilter runs on CPU
frames. On Apple Silicon software HEVC decode is already fast; the gain from
videotoolbox (if any) is narrow and often paid back by the hw→cpu transfer.

    KMP_DUPLICATE_LIB_OK=TRUE python -u -m video.frame_source._hwaccel_bench [PATH] [N]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import Optional, Tuple

import av
import av.filter
import numpy as np


DEFAULT_PATH = (
    "/Volumes/Crucial/pn/VR/"
    "VRCONK_marvel_rivals_luna_snow_a_porn_parody_8K_180x180_3dh.mp4"
)
DEFAULT_N = 500

CROP = "crop=4096:4096:0:0"
V360 = (
    "v360=he:in_stereo=0:output=sg:iv_fov=190:ih_fov=190:d_fov=190:"
    "v_fov=90:h_fov=90:pitch=-21:yaw=0:roll=0:w=640:h=640:interp=linear"
)
OUT_W, OUT_H = 640, 640
FRAME_BYTES = OUT_W * OUT_H * 3


def out(*args) -> None:
    print(*args, flush=True)


# --------------------------------------------------------------- subprocess

def subprocess_decode(path: str, n: int) -> Tuple[int, float]:
    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", path, "-an", "-sn",
        "-vf", f"{CROP},{V360}",
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=-1)
    # warmup
    for _ in range(50):
        if len(proc.stdout.read(FRAME_BYTES)) < FRAME_BYTES:
            break
    t0 = time.perf_counter()
    count = 0
    for _ in range(n):
        if len(proc.stdout.read(FRAME_BYTES)) < FRAME_BYTES:
            break
        count += 1
    el = time.perf_counter() - t0
    try: proc.terminate(); proc.wait(timeout=5)
    except Exception: proc.kill()
    return count, el


# --------------------------------------------------------------- PyAV CPU

def _build_cpu_graph(stream) -> av.filter.Graph:
    g = av.filter.Graph()
    src = g.add_buffer(template=stream)
    crop = g.add("crop", "4096:4096:0:0")
    v360 = g.add("v360", V360.split("=", 1)[1])
    fmt = g.add("format", "bgr24")
    sink = g.add("buffersink")
    src.link_to(crop); crop.link_to(v360); v360.link_to(fmt); fmt.link_to(sink)
    g.configure()
    return g


def pyav_cpu_decode(path: str, n: int) -> Tuple[int, float]:
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"
    graph = _build_cpu_graph(stream)

    count = 0
    warmup = 50
    t0: Optional[float] = None
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                graph.push(frame)
                while True:
                    try: out_frame = graph.pull()
                    except (av.BlockingIOError, av.FFmpegError): break
                    _ = out_frame.to_ndarray(format="bgr24")
                    if count == warmup:
                        t0 = time.perf_counter()
                    if count >= warmup:
                        if count - warmup >= n:
                            container.close()
                            return n, time.perf_counter() - t0
                    count += 1
    finally:
        container.close()
    return max(0, count - warmup), (time.perf_counter() - t0) if t0 else 0.0


# -------------------------------------------------------- PyAV videotoolbox

def pyav_vtb_decode(path: str, n: int) -> Tuple[int, float, str]:
    """Try videotoolbox hwaccel. Returns (count, elapsed, note).

    Note string describes the mode actually used (hw+hwdownload, or fallback).
    """
    # Open with hwaccel hint. libavformat/libavcodec will use the
    # videotoolbox decoder if the codec supports it.
    try:
        container = av.open(path, hwaccel='videotoolbox')
    except TypeError:
        # Older PyAV: hwaccel is passed via options dict.
        try:
            container = av.open(path, options={'hwaccel': 'videotoolbox'})
        except Exception as e:
            return 0, 0.0, f"failed to open with vtb: {e}"
    except Exception as e:
        return 0, 0.0, f"failed to open with vtb: {e}"

    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    # Filter graph must handle hw frames. Insert hwdownload,format=nv12 before
    # the software filters, then the normal crop + v360 + bgr24.
    note = "videotoolbox + hwdownload -> cpu filters"
    try:
        g = av.filter.Graph()
        src = g.add_buffer(template=stream)
        hwd = g.add("hwdownload")
        fmt_nv12 = g.add("format", "nv12")
        crop = g.add("crop", "4096:4096:0:0")
        v360 = g.add("v360", V360.split("=", 1)[1])
        fmt_bgr = g.add("format", "bgr24")
        sink = g.add("buffersink")
        src.link_to(hwd); hwd.link_to(fmt_nv12); fmt_nv12.link_to(crop)
        crop.link_to(v360); v360.link_to(fmt_bgr); fmt_bgr.link_to(sink)
        g.configure()
    except Exception as e:
        container.close()
        return 0, 0.0, f"graph build failed: {e}"

    count = 0
    warmup = 50
    t0: Optional[float] = None
    try:
        for packet in container.demux(stream):
            for frame in packet.decode():
                try:
                    g.push(frame)
                except av.FFmpegError as e:
                    # Frame not in hw context (decoder fell back to sw?)
                    container.close()
                    return 0, 0.0, f"graph push failed: {e}"
                while True:
                    try: out_frame = g.pull()
                    except (av.BlockingIOError, av.FFmpegError): break
                    _ = out_frame.to_ndarray(format="bgr24")
                    if count == warmup:
                        t0 = time.perf_counter()
                    if count >= warmup:
                        if count - warmup >= n:
                            container.close()
                            return n, time.perf_counter() - t0, note
                    count += 1
    finally:
        container.close()
    return max(0, count - warmup), (time.perf_counter() - t0) if t0 else 0.0, note


# ============================================================== main

def report(label: str, count: int, elapsed: float, note: str = "") -> None:
    if elapsed <= 0 or count <= 0:
        out(f"  {label}: incomplete  ({note})")
        return
    fps = count / elapsed
    out(f"  {label}: {count} frames in {elapsed:.1f}s = {fps:.1f} fps  {note}")


def main(path: str, n: int) -> None:
    if not os.path.exists(path):
        out(f"missing: {path}"); sys.exit(2)

    container = av.open(path); stream = container.streams.video[0]
    fps_source = float(stream.average_rate or stream.guessed_rate)
    w, h = stream.width, stream.height
    codec = stream.codec_context.name
    container.close()
    out(f"File: {path}")
    out(f"Stream: {w}x{h} codec={codec} fps={fps_source:.2f}")
    out(f"Sample: {n} frames after 50-frame warmup\n")

    out("[subprocess, no hwaccel]")
    c, e = subprocess_decode(path, n)
    report("subprocess", c, e)

    out("\n[PyAV, CPU software decode]")
    c, e = pyav_cpu_decode(path, n)
    report("PyAV CPU", c, e)

    out("\n[PyAV, videotoolbox hwaccel]")
    c, e, note = pyav_vtb_decode(path, n)
    report("PyAV VTB", c, e, note=f"({note})")

    out("\nDone.")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    p = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    main(p, n)

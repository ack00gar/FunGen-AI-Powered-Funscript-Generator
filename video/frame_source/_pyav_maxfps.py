"""Maximum sustained decode throughput: PyAV vs subprocess.

Measures the upper bound on output FPS for both decoders, on the user's 8K
10-bit VR file. Two configurations:

  - WITH v360 filter (production: what the GUI and trackers actually consume)
  - WITHOUT v360 filter (raw decode: shows the upper ceiling, isolates filter cost)

Also runs PyAV in a "background prefetch" mode to show the headroom available
when decode and consumer run on separate threads (which a real FrameSource
implementation will exploit).

    KMP_DUPLICATE_LIB_OK=TRUE python -u -m video.frame_source._pyav_maxfps [PATH] [N]
"""

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from typing import Optional

import numpy as np

import av


DEFAULT_PATH = (
    "/Volumes/Crucial/pn/VR/"
    "VRCONK_marvel_rivals_luna_snow_a_porn_parody_8K_180x180_3dh.mp4"
)
DEFAULT_N = 1000

CROP_FILTER = "4096:4096:0:0"
V360_ARGS = (
    "he:in_stereo=0:output=sg:iv_fov=190:ih_fov=190:d_fov=190:"
    "v_fov=90:h_fov=90:pitch=-21:yaw=0:roll=0:w=640:h=640:interp=linear"
)
OUT_W, OUT_H = 640, 640
FRAME_BYTES = OUT_W * OUT_H * 3
RAW_FRAME_BYTES_8K = 8192 * 4096 * 3 // 4  # rough — depends on stream resolution


def out(*args) -> None:
    print(*args, flush=True)


# ---------------------------------------------------------------- PyAV

def pyav_decode_loop(path: str, n: int, with_v360: bool) -> tuple[int, float]:
    """Decode n frames, return (count, elapsed_sec). Discards 50 warm-up frames."""
    container = av.open(path)
    stream = container.streams.video[0]
    stream.thread_type = "AUTO"

    if with_v360:
        g = av.filter.Graph()
        src = g.add_buffer(template=stream)
        crop = g.add("crop", CROP_FILTER)
        v360 = g.add("v360", V360_ARGS)
        fmt = g.add("format", "bgr24")
        sink = g.add("buffersink")
        src.link_to(crop); crop.link_to(v360); v360.link_to(fmt); fmt.link_to(sink)
        g.configure()

    decoded = 0
    warmup = 50
    t0: Optional[float] = None
    for packet in container.demux(stream):
        for frame in packet.decode():
            if with_v360:
                g.push(frame)
                while True:
                    try:
                        out_frame = g.pull()
                    except (av.BlockingIOError, av.FFmpegError):
                        break
                    arr = out_frame.to_ndarray(format="bgr24")
                    if decoded == warmup:
                        t0 = time.perf_counter()
                    if decoded >= warmup:
                        if decoded - warmup >= n:
                            container.close()
                            return n, time.perf_counter() - t0
                    decoded += 1
            else:
                # Raw decode: simulate consuming the frame (force materialization)
                arr = frame.to_ndarray(format="bgr24")
                if decoded == warmup:
                    t0 = time.perf_counter()
                if decoded >= warmup:
                    if decoded - warmup >= n:
                        container.close()
                        return n, time.perf_counter() - t0
                decoded += 1
    container.close()
    return decoded - warmup, (time.perf_counter() - t0) if t0 else 0.0


def pyav_prefetch_loop(path: str, n: int) -> tuple[int, float]:
    """PyAV with v360, decode in background thread, consumer pulls from queue."""
    q: "queue.Queue[Optional[np.ndarray]]" = queue.Queue(maxsize=8)
    stop = threading.Event()

    def producer() -> None:
        container = av.open(path)
        stream = container.streams.video[0]
        stream.thread_type = "AUTO"
        g = av.filter.Graph()
        src = g.add_buffer(template=stream)
        crop = g.add("crop", CROP_FILTER)
        v360 = g.add("v360", V360_ARGS)
        fmt = g.add("format", "bgr24")
        sink = g.add("buffersink")
        src.link_to(crop); crop.link_to(v360); v360.link_to(fmt); fmt.link_to(sink)
        g.configure()
        try:
            for packet in container.demux(stream):
                if stop.is_set(): break
                for frame in packet.decode():
                    g.push(frame)
                    while True:
                        try: out_frame = g.pull()
                        except (av.BlockingIOError, av.FFmpegError): break
                        q.put(out_frame.to_ndarray(format="bgr24"))
                        if stop.is_set(): break
        finally:
            q.put(None)
            container.close()

    th = threading.Thread(target=producer, daemon=True); th.start()

    decoded = 0; warmup = 50; t0: Optional[float] = None
    while True:
        item = q.get()
        if item is None: break
        if decoded == warmup:
            t0 = time.perf_counter()
        if decoded >= warmup:
            if decoded - warmup >= n:
                stop.set()
                return n, time.perf_counter() - t0
        decoded += 1
    return decoded - warmup, (time.perf_counter() - t0) if t0 else 0.0


# ---------------------------------------------------------------- subprocess

def subprocess_decode_loop(path: str, n: int, with_v360: bool) -> tuple[int, float]:
    """Spawn ffmpeg with same filter chain (or none), read raw frames, count."""
    if with_v360:
        vf_args = ["-vf", f"crop={CROP_FILTER},v360={V360_ARGS}"]
        out_w, out_h = OUT_W, OUT_H
    else:
        # Force decode + copy pixels but no v360. Use scale to a reasonable size
        # so we're comparing apples-to-apples on output bytes.
        vf_args = ["-vf", f"crop={CROP_FILTER},scale={OUT_W}:{OUT_H}"]
        out_w, out_h = OUT_W, OUT_H

    cmd = [
        "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error",
        "-i", path, "-an", "-sn",
        *vf_args,
        "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=-1)
    fb = out_w * out_h * 3
    decoded = 0; warmup = 50; t0: Optional[float] = None
    try:
        while True:
            raw = proc.stdout.read(fb)
            if len(raw) < fb: break
            if decoded == warmup:
                t0 = time.perf_counter()
            if decoded >= warmup:
                if decoded - warmup >= n:
                    return n, time.perf_counter() - t0
            decoded += 1
    finally:
        try: proc.terminate(); proc.wait(timeout=10)
        except Exception:
            try: proc.kill()
            except Exception: pass
    return decoded - warmup, (time.perf_counter() - t0) if t0 else 0.0


# ---------------------------------------------------------------- main

def report(label: str, n: int, elapsed: float, source_fps: float) -> None:
    if elapsed <= 0:
        out(f"  {label}: incomplete"); return
    fps = n / elapsed
    out(f"  {label}: {n} frames in {elapsed:.1f}s = {fps:.1f} fps "
        f"(realtime ratio: {fps / source_fps:.2f}x)")


def main(path: str, n: int) -> None:
    out(f"File: {path}")
    if not os.path.exists(path):
        out("  MISSING"); sys.exit(2)

    # Probe
    container = av.open(path); stream = container.streams.video[0]
    fps = float(stream.average_rate or stream.guessed_rate)
    total = int(stream.frames or 0)
    width, height = stream.width, stream.height
    container.close()
    out(f"Stream: {width}x{height} fps={fps:.2f} total_frames={total}")
    out(f"Sample size: {n} frames after 50-frame warmup")
    out("(higher = more headroom for tracker work / faster offline analysis)")

    out("\n=== WITH v360 filter (what the GUI consumes) ===")
    decoded, el = pyav_decode_loop(path, n, with_v360=True)
    report("PyAV         (single-threaded consumer)", decoded, el, fps)

    decoded, el = pyav_prefetch_loop(path, n)
    report("PyAV + prefetch (decode in background thread)", decoded, el, fps)

    decoded, el = subprocess_decode_loop(path, n, with_v360=True)
    report("subprocess   (current MAX_SPEED batch path)", decoded, el, fps)

    out("\n=== WITHOUT v360 filter (decode-only ceiling) ===")
    decoded, el = pyav_decode_loop(path, n, with_v360=False)
    report("PyAV         (decode + bgr24 conversion)", decoded, el, fps)

    decoded, el = subprocess_decode_loop(path, n, with_v360=False)
    report("subprocess   (decode + scale, no v360)", decoded, el, fps)

    out("\nDone.")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    p = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH
    n = int(sys.argv[2]) if len(sys.argv) > 2 else DEFAULT_N
    main(p, n)

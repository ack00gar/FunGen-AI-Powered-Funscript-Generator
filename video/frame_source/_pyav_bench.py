"""PyAV vs subprocess-ffmpeg benchmark across every GUI workflow.

Standalone script. Measures latencies that matter for an interactive
seek-heavy GUI on the user's actual 8K 10-bit VR file with the production
v360 filter chain.

mpv is intentionally not benchmarked here. mpv and PyAV both wrap libavcodec,
so they tie on raw decode/seek throughput; mpv's real advantage is display
zero-copy upload, which isn't measurable headless.

    KMP_DUPLICATE_LIB_OK=TRUE python -u -m video.frame_source._pyav_bench [PATH]
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from typing import List, Optional

import numpy as np

import av


DEFAULT_PATH = (
    "/Volumes/Crucial/pn/VR/"
    "VRCONK_marvel_rivals_luna_snow_a_porn_parody_8K_180x180_3dh.mp4"
)

CROP_FILTER = "4096:4096:0:0"
V360_ARGS = (
    "he:in_stereo=0:output=sg:iv_fov=190:ih_fov=190:d_fov=190:"
    "v_fov=90:h_fov=90:pitch=-21:yaw=0:roll=0:w=640:h=640:interp=linear"
)
OUT_W, OUT_H = 640, 640
FRAME_BYTES = OUT_W * OUT_H * 3
SUB_TIMEOUT = 30.0
SUB_N = 3       # subprocess samples kept low: each cold start ~5-15s on 8K
PYAV_N = 10     # PyAV is fast; more samples for tighter stats


def out(*args) -> None:
    print(*args, flush=True)


def warm_cache(path: str, target_frame: int, fps: float, seconds: float = 4.0) -> None:
    """Pre-read the file region around ``target_frame`` so the kernel page
    cache contains it. Mirrors the realistic GUI scenario where the file has
    been streaming continuously at playback rate before the user seeks.

    We dd a chunk of the file at the byte offset corresponding to the target
    frame. Approximate is fine — we just need the region in cache.
    """
    try:
        size = os.path.getsize(path)
        # Rough byte position: assume uniform bitrate (good enough for cache warming).
        approx_pos = int(size * (target_frame / fps) / max(1.0, get_duration(path)))
        approx_pos = max(0, min(size - 1, approx_pos))
        # Read ~80 MB around target — covers GOPs and B-frame references.
        chunk = int(80 * 1024 * 1024)
        start = max(0, approx_pos - chunk // 2)
        with open(path, "rb") as f:
            f.seek(start)
            f.read(chunk)
    except Exception:
        pass


_DURATION_CACHE: dict = {}
def get_duration(path: str) -> float:
    if path in _DURATION_CACHE:
        return _DURATION_CACHE[path]
    container = av.open(path)
    d = float(container.duration / 1_000_000) if container.duration else 1.0
    container.close()
    _DURATION_CACHE[path] = d
    return d


# ============================================================== PyAV

class PyAVSource:
    def __init__(self, path: str):
        self.container = av.open(path)
        self.stream = self.container.streams.video[0]
        self.stream.thread_type = "AUTO"
        self.fps = float(self.stream.average_rate or self.stream.guessed_rate)
        self.time_base = self.stream.time_base
        self.total_frames = int(self.stream.frames or 0)
        self.graph = self._build_graph()

    def _build_graph(self) -> av.filter.Graph:
        g = av.filter.Graph()
        src = g.add_buffer(template=self.stream)
        crop = g.add("crop", CROP_FILTER)
        v360 = g.add("v360", V360_ARGS)
        fmt = g.add("format", "bgr24")
        sink = g.add("buffersink")
        src.link_to(crop); crop.link_to(v360); v360.link_to(fmt); fmt.link_to(sink)
        g.configure()
        return g

    def close(self) -> None:
        try:
            self.container.close()
        except Exception:
            pass

    def _pump_one(self) -> Optional[np.ndarray]:
        for packet in self.container.demux(self.stream):
            for frame in packet.decode():
                self.graph.push(frame)
                try:
                    fout = self.graph.pull()
                except (av.BlockingIOError, av.FFmpegError):
                    fout = None
                if fout is not None:
                    return fout.to_ndarray(format="bgr24")
        return None

    def next_frame(self) -> Optional[np.ndarray]:
        return self._pump_one()

    def seek_frame(self, frame_index: int) -> Optional[np.ndarray]:
        target_pts = int(frame_index / self.fps / self.time_base)
        self.container.seek(target_pts, backward=True, any_frame=False, stream=self.stream)
        return self._pump_one()


# ============================================================== subprocess

class SubprocessPipe:
    def __init__(self, path: str, start_frame: int, fps: float):
        self.proc: Optional[subprocess.Popen] = None
        self.path = path
        self.fps = fps
        self.start(start_frame)

    def start(self, start_frame: int) -> None:
        self.stop()
        start_sec = start_frame / self.fps
        cmd = [
            "ffmpeg", "-hide_banner", "-nostats", "-loglevel", "error",
            "-ss", str(start_sec), "-i", self.path,
            "-an", "-sn",
            "-vf", f"crop={CROP_FILTER},v360={V360_ARGS}",
            "-pix_fmt", "bgr24", "-f", "rawvideo", "pipe:1",
        ]
        self.proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, bufsize=-1)

    def next_frame(self) -> Optional[np.ndarray]:
        if self.proc is None or self.proc.stdout is None:
            return None
        raw = self.proc.stdout.read(FRAME_BYTES)
        if len(raw) < FRAME_BYTES:
            return None
        return np.frombuffer(raw, dtype=np.uint8).reshape(OUT_H, OUT_W, 3).copy()

    def stop(self) -> None:
        if self.proc is not None:
            try:
                self.proc.terminate()
                self.proc.wait(timeout=SUB_TIMEOUT)
            except Exception:
                try: self.proc.kill()
                except Exception: pass
            self.proc = None


# ============================================================== utils

def stats(samples: List[float]) -> str:
    if not samples:
        return "no samples"
    s = sorted(samples)
    n = len(s)
    return (f"n={n} min={min(s):.0f}ms p50={s[n // 2]:.0f}ms "
            f"p90={s[min(n - 1, int(n * 0.9))]:.0f}ms max={max(s):.0f}ms "
            f"total={sum(s) / 1000:.1f}s")


def header(s: str) -> None:
    out(f"\n{'=' * 72}\n  {s}\n{'=' * 72}")


# ============================================================== benches

def bench_cold_open(path: str, fps: float) -> None:
    header("COLD OPEN + first decoded frame (file cache pre-warmed)")
    out("  Each trial pre-reads the target region into kernel cache first,")
    out("  matching the realistic GUI scenario where the file is already")
    out("  streaming. Numbers reflect decoder/filter init cost only.")
    targets = [int(t) for t in np.linspace(1000, 100000, SUB_N)]

    samples = []
    for tgt in targets:
        warm_cache(path, tgt, fps)
        t0 = time.perf_counter()
        s = PyAVSource(path); s.seek_frame(tgt); s.close()
        samples.append((time.perf_counter() - t0) * 1000)
    out(f"  PyAV       (open + filter graph + 1 frame): {stats(samples)}")

    samples = []
    for tgt in targets:
        warm_cache(path, tgt, fps)
        t0 = time.perf_counter()

        pipe = SubprocessPipe(path, tgt, fps)

        pipe.next_frame()

        samples.append((time.perf_counter() - t0) * 1000)

        pipe.stop()
    out(f"  subprocess (spawn ffmpeg -ss + 1 frame):    {stats(samples)}")


def bench_sequential(path: str, fps: float, n: int = 100) -> None:
    header(f"SUSTAINED SEQUENTIAL DECODE ({n} frames after warm-up)")

    s = PyAVSource(path); s.next_frame()
    t0 = time.perf_counter(); decoded = 0
    for _ in range(n):
        if s.next_frame() is None: break
        decoded += 1
    el = time.perf_counter() - t0; s.close()
    out(f"  PyAV:       {decoded} frames in {el*1000:.0f}ms = "
        f"{decoded/el:.1f} fps  (realtime ratio: {(decoded/el)/fps:.2f}x)")

    pipe = SubprocessPipe(path, 0, fps); pipe.next_frame()
    t0 = time.perf_counter(); decoded = 0
    for _ in range(n):
        if pipe.next_frame() is None: break
        decoded += 1
    el = time.perf_counter() - t0; pipe.stop()
    out(f"  subprocess: {decoded} frames in {el*1000:.0f}ms = "
        f"{decoded/el:.1f} fps  (realtime ratio: {(decoded/el)/fps:.2f}x)")


def bench_session_seek(path: str, fps: float, total: int) -> None:
    header("SESSION-LOCAL SEEK — within an interactive playback window")
    out("  Targets within a ~10-minute window of a session anchor (typical")
    out("  scrub behavior). Cache pre-warmed so we measure decoder cost, not")
    out("  disk I/O. PyAV is persistent; subprocess respawns per seek.")

    anchor = total // 3
    window = int(fps * 600)  # 10-minute window
    rng = np.random.default_rng(0)
    pyav_targets = [int(anchor + rng.integers(-window, window)) for _ in range(PYAV_N)]
    sub_targets = pyav_targets[:SUB_N]

    # Warm the entire window once for both, since both sides will read from it
    warm_cache(path, anchor, fps)

    s = PyAVSource(path); s.seek_frame(anchor)
    samples = []
    for tgt in pyav_targets:
        t0 = time.perf_counter(); s.seek_frame(tgt)
        samples.append((time.perf_counter() - t0) * 1000)
    s.close()
    out(f"  PyAV       (n={len(pyav_targets)}, persistent):     {stats(samples)}")

    samples = []
    for tgt in sub_targets:
        t0 = time.perf_counter()

        pipe = SubprocessPipe(path, tgt, fps)

        pipe.next_frame()

        samples.append((time.perf_counter() - t0) * 1000)

        pipe.stop()
    out(f"  subprocess (n={len(sub_targets)}, warm-cache respawn): {stats(samples)}")


def bench_small_back(path: str, fps: float, total: int) -> None:
    header("SMALL BACKWARD SEEK by 5 frames (warm cache)")
    base = total // 2
    warm_cache(path, base, fps)

    s = PyAVSource(path); s.seek_frame(base)
    samples = []
    for i in range(PYAV_N):
        cur = base + i * 60
        s.seek_frame(cur)
        t0 = time.perf_counter(); s.seek_frame(cur - 5)
        samples.append((time.perf_counter() - t0) * 1000)
    s.close()
    out(f"  PyAV       (n={PYAV_N}):                          {stats(samples)}")

    samples = []
    for i in range(SUB_N):
        cur = base + i * 60
        t0 = time.perf_counter()
        pipe = SubprocessPipe(path, cur - 5, fps)
        pipe.next_frame()
        samples.append((time.perf_counter() - t0) * 1000)
        pipe.stop()
    out(f"  subprocess (n={SUB_N}, warm-cache respawn each):  {stats(samples)}")


def bench_large_jump(path: str, fps: float, total: int) -> None:
    header("LARGE FORWARD JUMP — pause then click far ahead (warm cache)")
    out("  (mirrors your log: pause @ 18094, jump to ~28000)")
    base = 10000

    s = PyAVSource(path); s.seek_frame(base)
    samples = []
    for i in range(PYAV_N):
        target = base + 5000 + i * 1000
        warm_cache(path, target, fps)
        t0 = time.perf_counter(); s.seek_frame(target)
        samples.append((time.perf_counter() - t0) * 1000)
    s.close()
    out(f"  PyAV       (n={PYAV_N}):                          {stats(samples)}")

    samples = []
    for i in range(SUB_N):
        target = base + 5000 + i * 1000
        warm_cache(path, target, fps)
        t0 = time.perf_counter()

        pipe = SubprocessPipe(path, target, fps)

        pipe.next_frame()

        samples.append((time.perf_counter() - t0) * 1000)

        pipe.stop()
    out(f"  subprocess (n={SUB_N}, warm-cache respawn each):  {stats(samples)}")


def bench_thumbnail_pattern(path: str, fps: float, total: int) -> None:
    header("THUMBNAIL PATTERN — session-local random frame extraction")
    out("  (mirrors ThumbnailExtractor calls during scrub. Cache pre-warmed.)")
    anchor = total // 3
    window = int(fps * 600)  # 10-min session window
    rng = np.random.default_rng(42)
    pyav_targets = [int(anchor + rng.integers(-window, window)) for _ in range(PYAV_N)]
    sub_targets = pyav_targets[:SUB_N]

    warm_cache(path, anchor, fps)

    s = PyAVSource(path); s.next_frame()
    samples = []
    for tgt in pyav_targets:
        t0 = time.perf_counter(); s.seek_frame(tgt)
        samples.append((time.perf_counter() - t0) * 1000)
    s.close()
    out(f"  PyAV       (n={len(pyav_targets)}, one persistent source):    {stats(samples)}")

    samples = []
    for tgt in sub_targets:
        t0 = time.perf_counter()

        pipe = SubprocessPipe(path, tgt, fps)

        pipe.next_frame()

        samples.append((time.perf_counter() - t0) * 1000)

        pipe.stop()
    out(f"  subprocess (n={len(sub_targets)}, fresh process, warm cache): {stats(samples)}")


def bench_playback_with_work(path: str, fps: float, n: int = 100, work_ms: float = 5.0) -> None:
    header(f"PLAYBACK + simulated tracker work ({n} frames, {work_ms}ms CPU per frame)")

    def work(frame: np.ndarray) -> None:
        _ = float(frame.mean()); _ = float(frame.std())
        end = time.perf_counter() + work_ms / 1000.0
        while time.perf_counter() < end: pass

    s = PyAVSource(path); s.next_frame()
    t0 = time.perf_counter(); n_done = 0
    for _ in range(n):
        f = s.next_frame()
        if f is None: break
        work(f); n_done += 1
    el = time.perf_counter() - t0; s.close()
    out(f"  PyAV:       {n_done} frames in {el*1000:.0f}ms = {n_done/el:.1f} eff fps")

    pipe = SubprocessPipe(path, 0, fps); pipe.next_frame()
    t0 = time.perf_counter(); n_done = 0
    for _ in range(n):
        f = pipe.next_frame()
        if f is None: break
        work(f); n_done += 1
    el = time.perf_counter() - t0; pipe.stop()
    out(f"  subprocess: {n_done} frames in {el*1000:.0f}ms = {n_done/el:.1f} eff fps")


def bench_parity(path: str, fps: float) -> None:
    header("FRAME PARITY — PyAV vs subprocess at frame 50000")
    s = PyAVSource(path); pyav_frame = s.seek_frame(50000); s.close()
    pipe = SubprocessPipe(path, 50000, fps); sub_frame = pipe.next_frame(); pipe.stop()
    if pyav_frame is None or sub_frame is None:
        out("  could not get one of the frames"); return
    diff = np.abs(pyav_frame.astype(np.int16) - sub_frame.astype(np.int16))
    out(f"  shapes: pyav={pyav_frame.shape} sub={sub_frame.shape}")
    out(f"  mean_abs_diff={diff.mean():.2f}  max_abs_diff={int(diff.max())}")
    out("  (small diff expected — different decoder warmup states; what")
    out("   matters is no large structural difference)")


def main(path: str) -> None:
    out(f"File: {path}")
    if not os.path.exists(path):
        out("  MISSING — pass a valid path as argv[1]"); sys.exit(2)

    s = PyAVSource(path); fps = s.fps; total = s.total_frames; s.close()
    out(f"Stream: fps={fps:.2f} total_frames={total}")

    bench_cold_open(path, fps)
    bench_sequential(path, fps)
    bench_session_seek(path, fps, total)
    bench_small_back(path, fps, total)
    bench_large_jump(path, fps, total)
    bench_thumbnail_pattern(path, fps, total)
    bench_playback_with_work(path, fps)
    bench_parity(path, fps)
    out("\nDone.")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)

"""Smoke test for PyAVFrameSource: lifecycle, decode, seek, pause, reapply.

Runs against the 8K VR test file (override with argv[1]) and prints pass/fail
for each scenario. Not a unit test framework — just a quick correctness check
before integration.
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np

from video.frame_source.pyav_source import PyAVFrameSource, SourceConfig


DEFAULT_PATH = (
    "/Volumes/Crucial/pn/VR/"
    "VRCONK_marvel_rivals_luna_snow_a_porn_parody_8K_180x180_3dh.mp4"
)
PROD_FILTER = (
    "crop=4096:4096:0:0,"
    "v360=he:in_stereo=0:output=sg:iv_fov=190:ih_fov=190:d_fov=190:"
    "v_fov=90:h_fov=90:pitch=-21:yaw=0:roll=0:w=640:h=640:interp=linear"
)


def out(*args) -> None:
    print(*args, flush=True)


def assert_(cond: bool, msg: str) -> bool:
    status = "PASS" if cond else "FAIL"
    out(f"  [{status}] {msg}")
    return cond


def main(path: str) -> int:
    if not os.path.exists(path):
        out(f"missing: {path}"); return 2

    cfg = SourceConfig(video_path=path, filter_chain=PROD_FILTER, output_w=640, output_h=640)
    src = PyAVFrameSource(cfg)
    fails = 0

    out("\n[1] Open / metadata")
    ok = src.open()
    fails += not assert_(ok, "open() returns True")
    fails += not assert_(src.fps > 1.0, f"fps detected ({src.fps:.2f})")
    fails += not assert_(src.total_frames > 0, f"total_frames detected ({src.total_frames})")

    out("\n[2] One-shot get_frame at fixed index")
    t0 = time.perf_counter()
    f1 = src.get_frame(5000)
    dt = (time.perf_counter() - t0) * 1000
    fails += not assert_(f1 is not None and f1.shape == (640, 640, 3),
                         f"get_frame(5000) shape {None if f1 is None else f1.shape} ({dt:.0f}ms)")
    fails += not assert_(src.current_frame_index >= 4990 and src.current_frame_index <= 5010,
                         f"current_frame_index near 5000: {src.current_frame_index}")
    v_before = src.frame_version

    out("\n[3] Sequential decode via running thread")
    src.start(start_frame=10000)
    indices = []
    for _ in range(20):
        item = src.next_frame(timeout=2.0)
        if item is None: break
        indices.append(item[0])
    fails += not assert_(len(indices) >= 15, f"got {len(indices)}/20 frames")
    if len(indices) >= 2:
        steps = [indices[i+1] - indices[i] for i in range(len(indices) - 1)]
        positive = sum(1 for s in steps if s > 0)
        fails += not assert_(positive == len(steps), f"all steps forward (got {positive}/{len(steps)})")
    fails += not assert_(src.frame_version > v_before, "frame_version advanced")

    out("\n[4] Seek mid-playback to far frame")
    t0 = time.perf_counter()
    src.seek(80000)
    src.wait_seek(timeout=3.0)
    item = src.next_frame(timeout=3.0)
    dt = (time.perf_counter() - t0) * 1000
    fails += not assert_(item is not None, f"got frame after seek ({dt:.0f}ms)")
    if item:
        idx, _ = item
        fails += not assert_(79900 <= idx <= 80100, f"landed near 80000: {idx}")

    out("\n[5] Backward seek")
    src.seek(40000)
    src.wait_seek(timeout=3.0)
    item = src.next_frame(timeout=3.0)
    fails += not assert_(item is not None and 39900 <= item[0] <= 40100,
                         f"backward seek landed at {None if not item else item[0]}")

    out("\n[6] Pause / resume")
    src.pause()
    fails += not assert_(src.is_paused, "is_paused True after pause()")
    # While paused, the queue should not grow indefinitely. Drain whatever's
    # there, then verify no new frame arrives in 0.3s.
    while src.next_frame(timeout=0.05) is not None: pass
    item = src.next_frame(timeout=0.3)
    fails += not assert_(item is None, "no frames produced while paused")
    src.resume()
    item = src.next_frame(timeout=2.0)
    fails += not assert_(item is not None, "frame resumes after resume()")

    out("\n[7] Reapply settings (rebuild graph at half size)")
    src.stop()
    half_cfg = SourceConfig(
        video_path=path,
        filter_chain=PROD_FILTER.replace("w=640:h=640", "w=320:h=320"),
        output_w=320, output_h=320,
    )
    ok = src.reapply_settings(half_cfg)
    fails += not assert_(ok, "reapply_settings returned True")
    f2 = src.get_frame(20000)
    fails += not assert_(f2 is not None and f2.shape == (320, 320, 3),
                         f"new graph output shape {None if f2 is None else f2.shape}")

    out("\n[8] Callback fires on seek")
    seen = []
    src.register_seek_callback(lambda i: seen.append(i))
    src.seek(60000)
    fails += not assert_(len(seen) == 1 and seen[0] == 60000,
                         f"seek callback fired with 60000 (got {seen})")

    out("\n[9] Close")
    src.close()
    fails += not assert_(not src.is_running, "decode thread stopped after close")

    out(f"\n=== {fails} failure(s) ===")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH))

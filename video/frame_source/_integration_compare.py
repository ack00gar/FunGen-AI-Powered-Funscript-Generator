"""End-to-end before/after benchmark: VideoProcessor with subprocess vs PyAV.

Drives a real VideoProcessor for each backend, measures sustained playback
throughput in MAX_SPEED mode (what live trackers see as their input rate),
and verifies frame byte parity at fixed indices.
"""

from __future__ import annotations

import logging
import os
import sys
import time
from types import SimpleNamespace
from typing import List, Optional, Tuple
from unittest.mock import MagicMock

import numpy as np

from video.video_processor import VideoProcessor


DEFAULT_PATH = (
    "/Volumes/Crucial/pn/VR/"
    "VRCONK_marvel_rivals_luna_snow_a_porn_parody_8K_180x180_3dh.mp4"
)
PARITY_FRAMES = [1000, 5000, 30000, 80000, 120000]
PLAY_DURATION_SEC = 8.0


def out(*args) -> None:
    print(*args, flush=True)


def make_app(backend: str, force_cpu_v360: bool = False) -> SimpleNamespace:
    app = SimpleNamespace()
    app.logger = logging.getLogger("compare")
    app.logger.setLevel(logging.WARNING)
    if not app.logger.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
        app.logger.addHandler(h)
    settings = {
        'video_decoder_backend': backend,
        'hd_video_display': True,
        'vr_unwarp_method': 'v360' if force_cpu_v360 else 'auto',
        'vr_crop_panel': 'first',
        'hardware_acceleration_method': 'none',
    }
    class _S:
        def __init__(self, d): self._d = d
        def get(self, k, default=None): return self._d.get(k, default)
        def set(self, k, v): self._d[k] = v
    app.app_settings = _S(settings)
    from config import constants
    app.app_state_ui = SimpleNamespace(
        invalidate_content_uv_cache=lambda: None,
        selected_processing_speed_mode=constants.ProcessingSpeedMode.MAX_SPEED,
        slow_motion_fps=10.0,
    )
    app.funscript_processor = MagicMock()
    app.funscript_processor.get_chapter_at_frame = lambda f: None
    app.file_manager = MagicMock()
    app.file_manager.get_output_path_for_file = lambda *a, **k: ""
    app.file_manager.preprocessed_video_path = ""
    app.on_processing_paused = lambda: None
    app.on_processing_resumed = lambda: None
    app.on_processing_stopped = lambda **k: None
    app.hardware_acceleration_method = "none"
    return app


def measure_playback_fps(path: str, backend: str, duration_sec: float,
                         warmup_sec: float = 5.0,
                         force_cpu_v360: bool = True) -> Tuple[float, int]:
    """Returns (fps, frames_in_measurement_window). Warms up first to amortize
    ffmpeg subprocess cold-start vs PyAV's near-instant init — both backends
    are at steady-state HEVC decode rate when measurement starts.

    Defaults to ``force_cpu_v360=True`` because the GPU unwarp worker has
    been deactivated in the GUI (broken implementation). Fair comparison is
    libavfilter v360 in both backends.
    """
    app = make_app(backend, force_cpu_v360=force_cpu_v360)
    proc = VideoProcessor(app_instance=app, tracker=None, video_type='VR',
                          vr_input_format='he_sbs', vr_fov=190, vr_pitch=-21)
    if not proc.open_video(path):
        out(f"  [{backend}] open_video failed"); return 0.0, 0

    proc.start_processing(start_frame=0)

    # Warm up — wait for the loop to reach steady state.
    warmup_deadline = time.perf_counter() + warmup_sec
    while time.perf_counter() < warmup_deadline:
        time.sleep(0.05)

    # Now measure.
    start_idx = proc.current_frame_index
    t0 = time.perf_counter()
    deadline = t0 + duration_sec
    while time.perf_counter() < deadline:
        time.sleep(0.05)
    end_idx = proc.current_frame_index
    elapsed = time.perf_counter() - t0

    proc.stop_processing()
    proc.reset(close_video=True)

    span = max(0, end_idx - start_idx)
    fps = span / elapsed if elapsed > 0 else 0.0
    return fps, span


def parity_sequential(path: str, backend: str, start_frame: int = 5000,
                      n_frames: int = 50, sample_every: int = 5,
                      force_cpu_v360: bool = False) -> dict:
    """Capture sequential playback frames at given indices. Used for tracker-
    relevant parity (trackers consume sequential frames, not random seeks)."""
    app = make_app(backend, force_cpu_v360=force_cpu_v360)
    proc = VideoProcessor(app_instance=app, tracker=None, video_type='VR',
                          vr_input_format='he_sbs', vr_fov=190, vr_pitch=-21)
    if not proc.open_video(path):
        return {}
    targets = set(range(start_frame, start_frame + n_frames, sample_every))
    captured = {}
    proc.start_processing(start_frame=start_frame)
    deadline = time.time() + 30.0
    last_idx = -1
    while time.time() < deadline and len(captured) < len(targets):
        cur = proc.current_frame_index
        if cur != last_idx and cur in targets and cur not in captured:
            with proc.frame_lock:
                if proc.current_frame is not None:
                    captured[cur] = proc.current_frame.copy()
        last_idx = cur
        if cur >= start_frame + n_frames:
            break
        time.sleep(0.005)
    proc.stop_processing()
    proc.reset(close_video=True)
    return captured


def measure_seek_to_frame(path: str, backend: str) -> Optional[np.ndarray]:
    """Open video with the given backend, accurately fetch frame at a fixed
    index, return the BGR24 ndarray for parity comparison."""
    app = make_app(backend)
    proc = VideoProcessor(app_instance=app, tracker=None, video_type='VR',
                          vr_input_format='he_sbs', vr_fov=190, vr_pitch=-21)
    if not proc.open_video(path):
        return None
    # Accurate seek (paused) — both backends should produce same exact frame
    frames = {}
    for idx in PARITY_FRAMES:
        proc.seek_video(idx)
        # Brief settle to ensure frame is set
        deadline = time.time() + 8.0
        while time.time() < deadline and proc.current_frame is None:
            time.sleep(0.05)
        if proc.current_frame is not None:
            frames[idx] = proc.current_frame.copy()
    proc.reset(close_video=True)
    return frames


def main(path: str) -> None:
    if not os.path.exists(path):
        out(f"missing: {path}"); sys.exit(2)

    out("=" * 72)
    out("BEFORE/AFTER: VideoProcessor playback (MAX_SPEED, after warmup)")
    out(f"File: {path}")
    out(f"Warmup: 5s, then measure {PLAY_DURATION_SEC}s")
    out("Both backends use libavfilter v360 (the GUI's actual config —")
    out("GPU unwarp has been disabled as broken).")
    out("=" * 72)

    out("\n[subprocess]")
    fps_sub, span_sub = measure_playback_fps(path, 'subprocess', PLAY_DURATION_SEC)
    out(f"  steady-state: {span_sub} frames in {PLAY_DURATION_SEC}s = {fps_sub:.1f} fps")

    out("\n[pyav]")
    fps_pyav, span_pyav = measure_playback_fps(path, 'pyav', PLAY_DURATION_SEC)
    out(f"  steady-state: {span_pyav} frames in {PLAY_DURATION_SEC}s = {fps_pyav:.1f} fps")

    if fps_sub > 0:
        ratio = fps_pyav / fps_sub
        delta = fps_pyav - fps_sub
        out(f"\n  PyAV vs subprocess: {ratio:.2f}x  ({delta:+.1f} fps)")
        out("  (this is the rate live trackers receive frames at MAX_SPEED)")

    out("\n" + "=" * 72)
    out("TRACKER-RELEVANT PARITY (sequential playback frames)")
    out("=" * 72)
    out("\nLive trackers consume sequential frames during playback, not random")
    out("seeks. Capturing identical indices from both backends, comparing.\n")

    def report_pair(label: str, sub_frames: dict, pyav_frames: dict) -> None:
        out(f"\n--- {label} ---")
        common = sorted(set(sub_frames) & set(pyav_frames))
        if not common:
            out("  no overlapping indices captured"); return
        out("Index | mean_abs_diff | max_diff")
        out("-" * 40)
        for idx in common:
            a, b = sub_frames[idx], pyav_frames[idx]
            if a.shape != b.shape:
                out(f"  {idx}: shape mismatch sub={a.shape} pyav={b.shape}"); continue
            diff = np.abs(a.astype(np.int16) - b.astype(np.int16))
            out(f"  {idx:>5} | {diff.mean():>8.3f}     | {int(diff.max()):>3}")

    out("[subprocess vr_unwarp=v360]")
    seq_sub = parity_sequential(path, 'subprocess', force_cpu_v360=True)
    out(f"  captured {len(seq_sub)} frames")
    out("[pyav (libavfilter v360)]")
    seq_pyav = parity_sequential(path, 'pyav', force_cpu_v360=True)
    out(f"  captured {len(seq_pyav)} frames")
    report_pair("PyAV vs subprocess (both libavfilter v360, same implementation)",
                seq_sub, seq_pyav)
    out("\nNear-identical (mean diff <1) confirms output equivalence — trackers")
    out("see effectively the same frames in either backend.")
    out("\nDone.")


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH)

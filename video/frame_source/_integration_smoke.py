"""End-to-end smoke test for VideoProcessor + PyAV backend.

Drives a real VideoProcessor with app_settings['video_decoder_backend'] = 'pyav'
against the user's 8K VR file. Verifies open, seek, play, pause, resume, stop.
"""

from __future__ import annotations

import os
import sys
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from video.video_processor import VideoProcessor


DEFAULT_PATH = (
    "/Volumes/Crucial/pn/VR/"
    "VRCONK_marvel_rivals_luna_snow_a_porn_parody_8K_180x180_3dh.mp4"
)


def out(*args) -> None:
    print(*args, flush=True)


def assert_(cond: bool, msg: str) -> bool:
    status = "PASS" if cond else "FAIL"
    out(f"  [{status}] {msg}")
    return cond


def make_app() -> SimpleNamespace:
    """Minimal app stand-in matching the surface VideoProcessor reads."""
    import logging
    app = SimpleNamespace()
    app.logger = logging.getLogger("smoke")
    app.logger.setLevel(logging.WARNING)
    h = logging.StreamHandler()
    h.setFormatter(logging.Formatter("%(name)s %(levelname)s: %(message)s"))
    app.logger.addHandler(h)
    settings = {
        'video_decoder_backend': 'pyav',
        'hd_video_display': True,
        'vr_unwarp_method': 'auto',
        'vr_crop_panel': 'first',
        'hardware_acceleration_method': 'none',
    }
    class _Settings:
        def __init__(self, d): self._d = d
        def get(self, k, default=None): return self._d.get(k, default)
        def set(self, k, v): self._d[k] = v
    app.app_settings = _Settings(settings)
    app.app_state_ui = SimpleNamespace(
        invalidate_content_uv_cache=lambda: None,
        selected_processing_speed_mode=None,  # set below
        slow_motion_fps=10.0,
    )
    # ProcessingSpeedMode is an enum imported in video_processor.py.
    from config import constants
    app.app_state_ui.selected_processing_speed_mode = constants.ProcessingSpeedMode.MAX_SPEED
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


def main(path: str) -> int:
    if not os.path.exists(path):
        out(f"missing: {path}"); return 2

    app = make_app()
    proc = VideoProcessor(app_instance=app, tracker=None, video_type='VR',
                          vr_input_format='he_sbs', vr_fov=190, vr_pitch=-21)
    fails = 0

    out("\n[1] open_video with PyAV backend")
    ok = proc.open_video(path)
    fails += not assert_(ok, "open_video() returns True")
    fails += not assert_(proc.pyav_source is not None, "pyav_source created")
    fails += not assert_(proc._use_pyav(), "_use_pyav() is True")
    fails += not assert_(proc.fps > 1.0, f"fps={proc.fps:.2f}")
    fails += not assert_(proc.total_frames > 0, f"total_frames={proc.total_frames}")
    fails += not assert_(proc.current_frame is not None, "initial frame loaded")

    out("\n[2] start playback from frame 0 (MAX_SPEED) and verify advance")
    start_idx = proc.current_frame_index
    proc.start_processing(start_frame=0)
    fails += not assert_(proc.is_processing, "is_processing True")
    time.sleep(1.5)
    after_play = proc.current_frame_index
    fails += not assert_(after_play > start_idx + 5, f"index advanced ({start_idx} → {after_play})")

    out("\n[3] pause and verify index frozen")
    proc.pause_processing()
    time.sleep(0.3)
    pause_idx = proc.current_frame_index
    time.sleep(0.5)
    after_pause_wait = proc.current_frame_index
    fails += not assert_(after_pause_wait == pause_idx,
                         f"index frozen while paused ({pause_idx} → {after_pause_wait})")

    out("\n[4] resume and verify advance")
    proc.start_processing()
    time.sleep(0.7)
    after_resume = proc.current_frame_index
    fails += not assert_(after_resume > pause_idx + 3,
                         f"index advanced after resume ({pause_idx} → {after_resume})")

    out("\n[5] mid-playback seek to far frame (fast/keyframe)")
    t0 = time.perf_counter()
    proc.seek_video(80000)
    dt = (time.perf_counter() - t0) * 1000
    out(f"    seek_video returned in {dt:.0f}ms")
    fails += not assert_(proc.current_frame_index == 80000,
                         f"current_frame_index pinned to target 80000 (got {proc.current_frame_index})")
    # While catching up from the GOP keyframe, current_frame_index stays at
    # 80000 and current_frame advances visually. We verify that the source
    # is actually decoding past target by waiting and checking that the loop
    # exits the catch-up state (pending_seek_target becomes None).
    deadline = time.time() + 8.0
    while time.time() < deadline and proc._pyav_pending_seek_target is not None:
        time.sleep(0.05)
    fails += not assert_(proc._pyav_pending_seek_target is None,
                         f"seek caught up to target (pending={proc._pyav_pending_seek_target})")
    time.sleep(0.5)
    fails += not assert_(proc.current_frame_index >= 80000,
                         f"index past 80000 after catch-up ({proc.current_frame_index})")

    out("\n[6] backward seek (paused — accurate)")
    proc.pause_processing()
    time.sleep(0.2)
    t0 = time.perf_counter()
    proc.seek_video(40000)
    dt = (time.perf_counter() - t0) * 1000
    out(f"    accurate seek_video returned in {dt:.0f}ms")
    fails += not assert_(proc.current_frame_index == 40000,
                         f"current_frame_index=40000 (got {proc.current_frame_index})")
    fails += not assert_(proc.current_frame is not None, "frame loaded after accurate seek")

    out("\n[7] stop_processing")
    proc.stop_processing()
    fails += not assert_(not proc.is_processing, "is_processing False after stop")

    out("\n[6] reset / close")
    proc.reset(close_video=True)
    fails += not assert_(proc.pyav_source is None, "pyav_source closed on close")

    out(f"\n=== {fails} failure(s) ===")
    return 0 if fails == 0 else 1


if __name__ == "__main__":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_PATH))

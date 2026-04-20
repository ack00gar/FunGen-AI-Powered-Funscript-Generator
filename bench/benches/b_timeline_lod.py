"""Measure interactive-timeline draw-call count with LOD on vs off.

Stubs out imgui draw primitives as counters, replays _draw_curve at
various zoom levels on a synthetic funscript, and reports the actual
draw-call + per-call Python time delta between LOD and no-LOD.

Direct cost model: per-call imgui overhead scales linearly with the
number of add_* invocations (pyimgui boundary crossing dominates over
the native imgui cost per primitive).
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np


class _FakeDL:
    """Counts draw calls. Used to isolate draw-call cost from GL."""

    def __init__(self):
        self.counts = {"circle_filled": 0, "circle": 0, "polyline": 0,
                       "line": 0, "rect_filled": 0, "rect": 0, "text": 0}

    def add_circle_filled(self, *a, **k): self.counts["circle_filled"] += 1
    def add_circle(self, *a, **k): self.counts["circle"] += 1
    def add_polyline(self, *a, **k): self.counts["polyline"] += 1
    def add_line(self, *a, **k): self.counts["line"] += 1
    def add_rect_filled(self, *a, **k): self.counts["rect_filled"] += 1
    def add_rect(self, *a, **k): self.counts["rect"] += 1
    def add_text(self, *a, **k): self.counts["text"] += 1


def _make_actions(n_points: int, duration_ms: int = 3600_000):
    """Synthetic funscript with n points over duration."""
    ts = np.linspace(0, duration_ms, n_points).astype(np.int64)
    phase = np.linspace(0, 120 * np.pi, n_points)
    pos = (50 + 45 * np.sin(phase)).astype(np.int64)
    return [{"at": int(ts[i]), "pos": int(pos[i])} for i in range(n_points)]


def _build_tf(zoom_ms_per_px: float, canvas_w: int, canvas_h: int,
              center_ms: int, duration_ms: int):
    """Build a TimelineTransformer-lookalike."""
    tf = SimpleNamespace()
    tf.zoom = zoom_ms_per_px
    tf.width = canvas_w
    tf.height = canvas_h
    tf.x_offset = 0.0
    tf.y_offset = 0.0
    visible_ms = zoom_ms_per_px * canvas_w
    tf.visible_start_ms = max(0, center_ms - visible_ms / 2)
    tf.visible_end_ms = min(duration_ms, center_ms + visible_ms / 2)
    def v2x(ats):
        return (np.asarray(ats, dtype=np.float32) - tf.visible_start_ms) * (
            canvas_w / max(1e-3, (tf.visible_end_ms - tf.visible_start_ms)))
    def v2y(poss):
        return canvas_h - (np.asarray(poss, dtype=np.float32) / 100.0) * canvas_h
    tf.vec_time_to_x = v2x
    tf.vec_val_to_y = v2y
    return tf


def _run_draw(tl, tf, actions, n_reps: int) -> dict:
    dl = _FakeDL()
    t0 = time.perf_counter()
    for _ in range(n_reps):
        dl.counts = {k: 0 for k in dl.counts}
        tl._draw_curve(dl, tf, actions)
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {"ms_per_frame": total_ms / n_reps, "counts": dict(dl.counts)}


def _setup_timeline(no_lod: bool):
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    if no_lod:
        os.environ["FUNGEN_TIMELINE_NO_LOD"] = "1"
    else:
        os.environ.pop("FUNGEN_TIMELINE_NO_LOD", None)
    # Import inside the function so env-var takes effect.
    import imgui
    imgui.create_context()
    from application.classes.interactive_timeline import InteractiveFunscriptTimeline

    app = SimpleNamespace()
    app.app_state_ui = SimpleNamespace(
        timeline_point_radius=3.5, timeline_base_height=180,
    )
    app.funscript_processor = SimpleNamespace(
        get_funscript_obj=lambda: None, video_chapters=[])
    app.logger = None
    tl = InteractiveFunscriptTimeline.__new__(InteractiveFunscriptTimeline)
    tl.app = app
    tl.timeline_num = 1
    tl.logger = None
    tl.multi_selected_action_indices = set()
    tl.dragging_action_idx = -1
    tl._hovered_point_idx = -1
    tl._show_smooth_curve = False
    tl._heatmap_colors_cache = None
    tl._heatmap_cache_np_id = None
    tl._waveform_cache_key = None
    tl._heatmap_speeds_cache = None
    tl._no_lod_override = no_lod
    # DrawingMixin expects these caches; implement minimal getters.
    def _get_cached_timestamps():
        return tl._timestamps
    def _get_cached_numpy_arrays():
        return tl._ats_np, tl._poss_np
    tl._get_cached_timestamps = _get_cached_timestamps
    tl._get_cached_numpy_arrays = _get_cached_numpy_arrays
    return tl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--points", type=int, nargs="+",
                    default=[3000, 10000, 30000])
    ap.add_argument("--canvas_w", type=int, default=2000)
    ap.add_argument("--canvas_h", type=int, default=200)
    ap.add_argument("--reps", type=int, default=100)
    args = ap.parse_args()

    # Zoom levels in ms/px. 0.5 = 30s in 60px (very wide, dense).
    # 200 = 24min in 2000px (whole video overview).
    zoom_levels_ms_per_px = [1.0, 5.0, 20.0, 100.0, 500.0]

    for n in args.points:
        print(f"\n=== points={n} ===")
        actions = _make_actions(n, duration_ms=3600_000)
        print(f"{'zoom_ms/px':>10}  {'lod':>5}  {'ms/frame':>10}  "
              f"{'circles':>8}  {'polylines':>10}  {'total_calls':>12}")

        for lod_off in (False, True):
            tl = _setup_timeline(no_lod=lod_off)
            tl._timestamps = [a["at"] for a in actions]
            tl._ats_np = np.array([a["at"] for a in actions], dtype=np.float32)
            tl._poss_np = np.array([a["pos"] for a in actions], dtype=np.float32)

            for zoom in zoom_levels_ms_per_px:
                tf = _build_tf(zoom, args.canvas_w, args.canvas_h,
                               1800_000, 3600_000)
                r = _run_draw(tl, tf, actions, n_reps=args.reps)
                c = r["counts"]
                total = sum(c.values())
                tag = "OFF" if lod_off else "ON"
                print(f"{zoom:10.1f}  {tag:>5}  {r['ms_per_frame']:10.3f}  "
                      f"{c['circle_filled']+c['circle']:8d}  "
                      f"{c['polyline']:10d}  {total:12d}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

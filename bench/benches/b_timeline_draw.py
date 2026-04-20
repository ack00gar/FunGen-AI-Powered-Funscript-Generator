"""Measure the CPU cost of interactive_timeline curve drawing at various
action counts and LOD modes.

We bypass imgui entirely by passing a mock draw_list that accepts every
add_* call as a no-op, and a mock TimelineTransformer. This isolates the
Python-side per-frame work: comprehensions, bisect, LOD branching,
numpy transforms, point-loop bookkeeping. The GPU cost of the real imgui
draw calls is separate; this bench answers "how expensive is the LOD
decision + Python traversal itself?" -- which is what we can prune or
simplify without touching rendering.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np

from ..harness import Report, Sample, measure
from ..registry import register


class _FakeDrawList:
    """imgui DrawList stand-in -- records call counts, does no work."""
    __slots__ = ("counts",)

    def __init__(self):
        self.counts: Dict[str, int] = {}

    def _tick(self, name: str):
        self.counts[name] = self.counts.get(name, 0) + 1

    def add_rect_filled(self, *a, **kw):   self._tick("rect_filled")
    def add_rect(self, *a, **kw):          self._tick("rect")
    def add_line(self, *a, **kw):          self._tick("line")
    def add_text(self, *a, **kw):          self._tick("text")
    def add_circle_filled(self, *a, **kw): self._tick("circle_filled")
    def add_circle(self, *a, **kw):        self._tick("circle")
    def add_polyline(self, *a, **kw):      self._tick("polyline")
    def add_convex_poly_filled(self, *a, **kw): self._tick("convex_poly")
    def add_image(self, *a, **kw):         self._tick("image")


class _FakeTF:
    """Simplified TimelineTransformer: maps (ms, value) -> (x, y) linearly."""
    __slots__ = ("x_offset", "y_offset", "width", "height", "zoom",
                 "visible_start_ms", "visible_end_ms")

    def __init__(self, width: float, zoom_ms_per_px: float, actions: list):
        self.x_offset = 0.0
        self.y_offset = 0.0
        self.width = float(width)
        self.height = 180.0
        self.zoom = float(zoom_ms_per_px)
        # Center viewport on the middle of the action range so the visible
        # slice is large.
        if actions:
            t0, t1 = actions[0]["at"], actions[-1]["at"]
            mid = 0.5 * (t0 + t1)
            half = width * zoom_ms_per_px * 0.5
            self.visible_start_ms = mid - half
            self.visible_end_ms = mid + half
        else:
            self.visible_start_ms = 0.0
            self.visible_end_ms = width * zoom_ms_per_px

    def time_to_x(self, t_ms: float) -> float:
        return self.x_offset + (t_ms - self.visible_start_ms) / self.zoom

    def x_to_time(self, x: float) -> float:
        return self.visible_start_ms + (x - self.x_offset) * self.zoom

    def val_to_y(self, v: float) -> float:
        return self.y_offset + (100.0 - v) * (self.height / 100.0)

    def vec_time_to_x(self, t_arr: np.ndarray) -> np.ndarray:
        return self.x_offset + (t_arr - self.visible_start_ms) / self.zoom

    def vec_val_to_y(self, v_arr: np.ndarray) -> np.ndarray:
        return self.y_offset + (100.0 - v_arr) * (self.height / 100.0)


def _synth_actions(n: int, duration_ms: float = 3600_000.0) -> List[Dict[str, int]]:
    """n actions over `duration_ms`, positions on a damped sine."""
    rng = np.random.default_rng(17)
    ts = np.linspace(0, duration_ms, n, dtype=np.int64)
    # Smooth curve plus small noise so RDP-like code has something to see
    phase = ts * (2.0 * np.pi / 4000.0)
    pos = 50 + 40 * np.sin(phase) * np.exp(-ts / duration_ms * 0.3)
    pos += rng.normal(0, 2.0, size=n)
    pos = np.clip(pos, 0, 100).astype(np.int64)
    return [{"at": int(t), "pos": int(p)} for t, p in zip(ts, pos)]


@register(
    "timeline_draw",
    "CPU cost of _draw_curve against a fake DrawList; compares LOD paths.",
)
def run(iters: int, warmup: int, **_) -> Report:
    # imgui.get_color_u32_rgba segfaults without a live GL context. Patch to
    # a pure-python stub before the drawing_mixin module loads so the module-
    # level cache never touches the real bind.
    import imgui as _imgui
    _imgui.get_color_u32_rgba = lambda *args: 0xFF_00_00_00
    _imgui.calc_text_size = lambda s: (len(s) * 7.0, 13.0)

    from application.classes.timeline.drawing_mixin import DrawingMixin

    class _TL(DrawingMixin):
        """Minimal instance with just the attributes DrawingMixin touches."""

        def __init__(self):
            class _App: pass
            class _AppStateUI:
                timeline_point_radius = 3
                timeline_base_height = 180
            self.app = _App()
            self.app.app_state_ui = _AppStateUI()
            self.dragging_action_idx = -1
            self._hovered_point_idx = -1
            self.multi_selected_action_indices = set()
            self._show_smooth_curve = False
            self._point_fade_opacity_val = 1.0
            self._waveform_cache_key = None
            self._waveform_cache_xs = None
            self._waveform_cache_ys_top = None
            self._waveform_cache_ys_bot = None
            self._waveform_cache_step = 0

        # DrawingMixin expects these helpers on the host class.
        def _get_cached_timestamps(self):  return None
        def _get_cached_numpy_arrays(self): return (None, None)
        def _point_fade_opacity(self, tf): return 1.0

    tl = _TL()
    r = Report(
        name="timeline_draw",
        description="_draw_curve vs action count and viewport zoom (fake DrawList).",
        device="cpu",
    )

    # (n_actions, width_px, zoom_ms_per_px, label)
    scenarios = [
        (500,   2000,  10.0,  "500 pts, pixels/pt~40 (zoomed in)"),
        (500,   2000, 500.0,  "500 pts, pixels/pt~2 (zoomed out)"),
        (5000,  2000,  10.0,  "5k pts, pixels/pt~0.4"),
        (5000,  2000,   1.0,  "5k pts, pixels/pt~4"),
        (20000, 2000,   1.0,  "20k pts, pixels/pt~1 (LOD A threshold)"),
        (50000, 2000,   1.0,  "50k pts, pixels/pt~0.4 (LOD A active)"),
    ]

    for n, w, zoom, label in scenarios:
        actions = _synth_actions(n)
        tf = _FakeTF(w, zoom, actions)
        dl = _FakeDrawList()

        def _call():
            dl.counts.clear()
            tl._draw_curve(dl, tf, actions)

        try:
            samples = measure(_call, iters=iters, warmup=warmup)
            counts = dict(dl.counts)
            r.add(Sample(
                label=label, samples_s=samples,
                device="cpu", meta={"draw_calls": counts},
            ))
        except Exception as e:
            r.extra[label + "_error"] = str(e).splitlines()[0][:200]

    # --- "full detail" variant: bypass LOD A and draw everything. ---
    # Monkey-patch to force the non-LOD-A path regardless of pixels/pt.
    import application.classes.timeline.drawing_mixin as _dm
    orig_draw_curve = _dm.DrawingMixin._draw_curve

    def _full_detail_draw_curve(self, dl, tf, actions, is_preview=False,
                                 color_override=None, alpha=1.0,
                                 force_lines_only=False):
        # is_preview=True bypasses LOD A (pixels_per_point < 2 branch).
        return orig_draw_curve(self, dl, tf, actions,
                               is_preview=True,
                               color_override=color_override,
                               alpha=alpha,
                               force_lines_only=force_lines_only)

    _dm.DrawingMixin._draw_curve = _full_detail_draw_curve
    try:
        for n, w, zoom, label in [
            (20000, 2000, 1.0, "[NO LOD A] 20k pts"),
            (50000, 2000, 1.0, "[NO LOD A] 50k pts"),
        ]:
            actions = _synth_actions(n)
            tf = _FakeTF(w, zoom, actions)
            dl = _FakeDrawList()
            def _call():
                dl.counts.clear()
                tl._draw_curve(dl, tf, actions)
            samples = measure(_call, iters=iters, warmup=warmup)
            r.add(Sample(label=label, samples_s=samples, device="cpu",
                         meta={"draw_calls": dict(dl.counts)}))
    finally:
        _dm.DrawingMixin._draw_curve = orig_draw_curve

    return r

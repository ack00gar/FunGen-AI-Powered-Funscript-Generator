"""Drawing methods for InteractiveFunscriptTimeline.

Pure-presentation code: reads timeline / funscript / app state and draws
primitives via imgui's DrawList. No state mutation. Broken out of the
3300-line interactive_timeline.py so the core class stays focused on
orchestration.

Consumed as a mixin: ``InteractiveFunscriptTimeline(DrawingMixin, ...)``.
"""

import time
from bisect import bisect_left, bisect_right
from typing import Dict, List

import imgui
import numpy as np

from application.utils import _format_time
from application.utils.heatmap_utils import HeatmapColorMapper
from application.utils.imgui_helpers import center_next_window_pivot
from common.frame_utils import ms_to_frame
from config.element_group_colors import TimelineColors


_GRID_VALUES = (0, 25, 50, 75, 100)
_GRID_VALUE_LABELS = ("0", "25", "50", "75", "100")


def _u32_from_const(color):
    """Return the imgui u32 for a theme-constant color tuple; cached."""
    cache = _u32_from_const.__dict__.setdefault("_cache", {})
    key = tuple(color)
    u = cache.get(key)
    if u is None:
        u = imgui.get_color_u32_rgba(*key)
        cache[key] = u
    return u


def _u32_alpha_blend(color, alpha_mult):
    """u32 for ``color`` with its alpha channel multiplied by ``alpha_mult``.

    ``alpha_mult`` is quantized to 0.025 so viewport fades only ever sample
    ~40 distinct values per RGB tuple -- far less than one new u32 per frame.
    """
    cache = _u32_alpha_blend.__dict__.setdefault("_cache", {})
    qa = round(alpha_mult * 40) / 40.0
    key = (tuple(color), qa)
    u = cache.get(key)
    if u is None:
        u = imgui.get_color_u32_rgba(color[0], color[1], color[2], color[3] * qa)
        cache[key] = u
    return u


class DrawingMixin:
    def _draw_background_grid(self, dl, tf: 'TimelineTransformer'):
        # Theme-static u32 colors + label text extents cached once per instance.
        grid_cache = getattr(self, "_grid_cache", None)
        if grid_cache is None:
            grid_cache = {
                "canvas_bg": _u32_from_const(TimelineColors.CANVAS_BACKGROUND),
                "grid_major": _u32_from_const(TimelineColors.GRID_MAJOR_LINES),
                "grid_minor": _u32_from_const(TimelineColors.GRID_LINES),
                "grid_labels": _u32_from_const(TimelineColors.GRID_LABELS),
                "midline": imgui.get_color_u32_rgba(0.75, 0.75, 0.78, 0.55),
                "label_sizes": tuple(imgui.calc_text_size(s) for s in _GRID_VALUE_LABELS),
            }
            self._grid_cache = grid_cache
        canvas_bg_u32 = grid_cache["canvas_bg"]
        grid_major_u32 = grid_cache["grid_major"]
        grid_minor_u32 = grid_cache["grid_minor"]
        grid_labels_u32 = grid_cache["grid_labels"]
        midline_u32 = grid_cache["midline"]
        label_sizes = grid_cache["label_sizes"]

        # 1. Background
        dl.add_rect_filled(tf.x_offset, tf.y_offset, tf.x_offset + tf.width, tf.y_offset + tf.height,
                           canvas_bg_u32)

        # 2. Horizontal Lines (0, 25, 50, 75, 100)
        for label_idx, val in enumerate(_GRID_VALUES):
            y = tf.val_to_y(val)
            if val == 50:
                col_u32 = midline_u32
                thick = 1.5
            else:
                col_u32 = grid_major_u32
                thick = 1.0
            dl.add_line(tf.x_offset, y, tf.x_offset + tf.width, y, col_u32, thick)

            # Position labels
            label_text = _GRID_VALUE_LABELS[label_idx]
            text_size = label_sizes[label_idx]

            if val == 100:
                # Place below the line
                label_y = y + 2
            elif val == 25 or val == 50 or val == 75:
                # Center on the line with background for readability
                label_y = y - text_size[1] / 2
                # Draw background rectangle for readability
                padding = 2
                dl.add_rect_filled(
                    tf.x_offset + 2 - padding,
                    label_y - padding,
                    tf.x_offset + 2 + text_size[0] + padding,
                    label_y + text_size[1] + padding,
                    canvas_bg_u32
                )
            else:
                # 0: above the line
                label_y = y - 12

            dl.add_text(tf.x_offset + 2, label_y, grid_labels_u32, label_text)

        # 3. Vertical Lines (Adaptive Time Steps)
        pixels_per_sec = 1000.0 / tf.zoom
        # Determine grid interval based on visual density
        if pixels_per_sec > 200: step_ms = 100
        elif pixels_per_sec > 50: step_ms = 1000
        elif pixels_per_sec > 10: step_ms = 5000
        else: step_ms = 30000

        # Snap start time to step
        start_ms = (tf.visible_start_ms // step_ms) * step_ms
        curr_ms = start_ms
        
        while curr_ms <= tf.visible_end_ms:
            x = tf.time_to_x(curr_ms)
            if x >= tf.x_offset:
                is_major = (curr_ms % (step_ms * 5) == 0)
                dl.add_line(x, tf.y_offset, x, tf.y_offset + tf.height,
                            grid_major_u32 if is_major else grid_minor_u32)
                if is_major and curr_ms >= 0:
                     dl.add_text(x + 3, tf.y_offset + tf.height - 15, grid_labels_u32, f"{curr_ms/1000:.1f}s")
            curr_ms += step_ms


    def _draw_audio_waveform(self, dl, tf: 'TimelineTransformer'):
        data = self.app.get_waveform_data()
        if not self.app.app_state_ui.show_audio_waveform or data is None: return
        total_frames = self.app.processor.total_frames
        fps = self.app.processor.fps
        if total_frames <= 0 or fps <= 0: return

        duration_ms = (total_frames / fps) * 1000.0

        # Map visible range to data indices
        idx_start = max(0, int((tf.visible_start_ms / duration_ms) * len(data)))
        idx_end = min(len(data), int((tf.visible_end_ms / duration_ms) * len(data)))
        if idx_end <= idx_start: return

        # Decimate for performance (Max 1 sample per pixel)
        step = max(1, (idx_end - idx_start) // int(tf.width))

        # Cache key: viewport range + canvas geometry (rounded to avoid float jitter)
        cache_key = (round(tf.visible_start_ms, 1), round(tf.visible_end_ms, 1),
                     int(tf.width), int(tf.height), round(tf.y_offset, 1))
        if self._waveform_cache_key == cache_key and self._waveform_cache_xs is not None:
            xs = self._waveform_cache_xs
            ys_top = self._waveform_cache_ys_top
            ys_bot = self._waveform_cache_ys_bot
            step = self._waveform_cache_step
        else:
            subset = data[idx_start:idx_end:step]
            times = np.linspace(tf.visible_start_ms, tf.visible_end_ms, len(subset))
            xs = tf.vec_time_to_x(times)
            center_y = tf.y_offset + tf.height / 2
            ys_top = center_y - (subset * tf.height / 2)
            ys_bot = center_y + (subset * tf.height / 2)
            self._waveform_cache_key = cache_key
            self._waveform_cache_xs = xs
            self._waveform_cache_ys_top = ys_top
            self._waveform_cache_ys_bot = ys_bot
            self._waveform_cache_step = step

        col = _u32_from_const(TimelineColors.AUDIO_WAVEFORM)

        # LOD: Lines vs Polylines
        if step > 10:
            xs_l = xs.tolist()
            yt_l = ys_top.tolist()
            yb_l = ys_bot.tolist()
            _add_line = dl.add_line
            for i in range(len(xs_l)):
                _add_line(xs_l[i], yt_l[i], xs_l[i], yb_l[i], col)
        else:
            pts_top = np.column_stack((xs, ys_top)).tolist()
            pts_bot = np.column_stack((xs, ys_bot)).tolist()
            dl.add_polyline(pts_top, col, False, 1.0)
            dl.add_polyline(pts_bot, col, False, 1.0)


    def _line_thickness_for_height(self) -> float:
        """Curve line thickness derived from the current timeline row height.
        Baseline: height=180 → thickness=2.5. Scales linearly, clamped to a
        sensible range so lines neither vanish on tall rows nor fatten too
        much on short ones."""
        h = float(getattr(self.app.app_state_ui, 'timeline_base_height', 180))
        return max(1.0, min(6.0, h / 72.0))


    def _spline_samples_for_view(self, canvas_width_px: float, n_visible_segments: int) -> int:
        """Pixel-aware sample count per catmull-rom segment.

        Budget ~150 samples per 2000px of canvas, spread over visible segments
        (so dense scripts don't explode and sparse scripts stay smooth). If the
        per-segment budget drops below 3, caller should fall back to a straight
        line, subsampling with <3 points per segment produces visible kinks
        worse than the straight interpolation.
        """
        if n_visible_segments <= 0 or canvas_width_px <= 0:
            return 0
        total_budget = int(150.0 * canvas_width_px / 2000.0)
        return max(0, total_budget // n_visible_segments)


    def _point_fade_opacity(self, tf: 'TimelineTransformer') -> float:
        """No fade. Every visible point always renders at full opacity."""
        return 1.0


    def _expand_catmull(self, ats: np.ndarray, poss: np.ndarray, k: int):
        """Subsample each segment with catmull-rom spline. Returns (xs_a, ys_p, seg_idx).
        seg_idx maps each dense sample (excluding the appended final point) to its
        source segment index in [0, n-2]."""
        n = len(ats)
        if n < 2 or k < 2:
            return ats, poss, np.arange(max(0, n - 1), dtype=np.int32)
        i = np.arange(n - 1)
        i0 = np.maximum(0, i - 1)
        i3 = np.minimum(n - 1, i + 2)
        p0 = poss[i0]; p1 = poss[i]; p2 = poss[i + 1]; p3 = poss[i3]
        a1 = ats[i]; a2 = ats[i + 1]
        t = np.linspace(0.0, 1.0, k, endpoint=False, dtype=np.float32)
        T = t[None, :]
        P0 = p0[:, None]; P1 = p1[:, None]; P2 = p2[:, None]; P3 = p3[:, None]
        P = 0.5 * (2.0 * P1 + (-P0 + P2) * T
                   + (2.0 * P0 - 5.0 * P1 + 4.0 * P2 - P3) * T * T
                   + (-P0 + 3.0 * P1 - 3.0 * P2 + P3) * T * T * T)
        A = a1[:, None] + T * (a2[:, None] - a1[:, None])
        P = np.clip(P, 0.0, 100.0)
        out_a = np.concatenate([A.reshape(-1), ats[-1:]])
        out_p = np.concatenate([P.reshape(-1), poss[-1:]])
        seg_idx = np.repeat(i, k)
        return out_a.astype(np.float32), out_p.astype(np.float32), seg_idx.astype(np.int32)


    def _draw_curve(self, dl, tf: 'TimelineTransformer', actions: List[Dict],
                    is_preview=False, color_override=None, force_lines_only=False, alpha=1.0):
        if not actions or len(actions) < 2: return

        # 1. Culling: Identify visible slice using cached timestamps
        margin_ms = tf.zoom * 100
        # For main curves, prefer the funscript's cached timestamp list (avoids O(n) rebuild)
        if not is_preview and not color_override:
            timestamps = self._get_cached_timestamps()
            if not timestamps or len(timestamps) != len(actions):
                timestamps = [a['at'] for a in actions]
        else:
            timestamps = [a['at'] for a in actions]
        s_idx = bisect_left(timestamps, tf.visible_start_ms - margin_ms)
        e_idx = bisect_right(timestamps, tf.visible_end_ms + margin_ms)
        
        s_idx = max(0, s_idx - 1)
        e_idx = min(len(actions), e_idx + 1)
        
        if e_idx - s_idx < 2: return

        visible_actions = actions[s_idx:e_idx]

        # 2. Vectorized Transform, use cached numpy arrays, slice instead of rebuild
        all_ats, all_poss = self._get_cached_numpy_arrays()
        if all_ats is not None and len(all_ats) == len(actions):
            ats = all_ats[s_idx:e_idx]
            poss = all_poss[s_idx:e_idx]
        else:
            ats = np.array([a['at'] for a in visible_actions], dtype=np.float32)
            poss = np.array([a['pos'] for a in visible_actions], dtype=np.float32)

        xs = tf.vec_time_to_x(ats)
        ys = tf.vec_val_to_y(poss)

        # CLAMP COORDINATES: Fix invisible lines when zoomed in on sparse data
        # ImGui rendering can glitch if coordinates exceed +/- 32k (integer overflow in vertex buffer)
        # We clamp x coordinates to a safe range slightly outside the viewport
        safe_min_x = tf.x_offset - 5000
        safe_max_x = tf.x_offset + tf.width + 5000
        xs = np.clip(xs, safe_min_x, safe_max_x)

        # No point LOD: always draw every visible action point. Bench
        # showed ~2-3ms worst-case on 30k-point / full-zoom frames, and
        # zero saving at normal zoom levels.
        points_on_screen = len(xs)
        pixels_per_point = tf.width / points_on_screen if points_on_screen > 0 else 0
        base_col = color_override or (TimelineColors.PREVIEW_LINES if is_preview else (0.8, 0.8, 0.8, 1.0))
        col_u32 = imgui.get_color_u32_rgba(base_col[0], base_col[1], base_col[2], base_col[3] * alpha)
        base_col = color_override or (TimelineColors.PREVIEW_LINES if is_preview else (0.8, 0.8, 0.8, 1.0))
        col_u32 = imgui.get_color_u32_rgba(base_col[0], base_col[1], base_col[2], base_col[3] * alpha)
        base_thick = self._line_thickness_for_height()
        thick = max(1.0, base_thick - 0.5) if is_preview else base_thick

        if self._show_smooth_curve and not is_preview and len(ats) >= 2:
            k = self._spline_samples_for_view(tf.width, len(ats) - 1)
            if k >= 3:
                d_ats, d_poss, _ = self._expand_catmull(ats, poss, k)
                d_xs = np.clip(tf.vec_time_to_x(d_ats), safe_min_x, safe_max_x)
                d_ys = tf.vec_val_to_y(d_poss)
                pts = np.column_stack((d_xs, d_ys)).tolist()
            else:
                # Per-segment budget too low to render a smooth curve without
                # visible kinks, fall back to straight segments, which at
                # this density are indistinguishable from the spline anyway.
                pts = np.column_stack((xs, ys)).tolist()
        else:
            pts = np.column_stack((xs, ys)).tolist()
        dl.add_polyline(pts, col_u32, False, thick)

        # -- Points: always render every visible action point.
        if not force_lines_only:
            radius = self.app.app_state_ui.timeline_point_radius

            _default_c = TimelineColors.POINT_DEFAULT if not is_preview else TimelineColors.PREVIEW_POINTS
            col_default = _u32_alpha_blend(_default_c, alpha)
            col_drag = _u32_alpha_blend(TimelineColors.POINT_DRAGGING, alpha)
            col_sel = _u32_alpha_blend(TimelineColors.POINT_SELECTED, alpha)
            col_hover = _u32_alpha_blend(TimelineColors.POINT_HOVER, alpha)
            col_sel_border = _u32_from_const(TimelineColors.SELECTED_POINT_BORDER)
            r_drag = radius + 2
            r_sel = radius + 1
            r_hover = radius + 1
            r_sel_border = r_sel + 1

            _sel_set = self.multi_selected_action_indices
            _drag_idx = self.dragging_action_idx
            _hover_idx = self._hovered_point_idx
            xs_l = xs.tolist()
            ys_l = ys.tolist()
            n_actions = len(visible_actions)
            _add_circle_filled = dl.add_circle_filled
            _add_circle = dl.add_circle
            sel_empty = not _sel_set
            if sel_empty:
                is_sel_list = [False] * n_actions
            else:
                is_sel_list = [(a['at'], a['pos']) in _sel_set for a in visible_actions]

            for i in range(n_actions):
                real_idx = s_idx + i
                is_sel = is_sel_list[i]
                is_drag = (real_idx == _drag_idx)
                is_hover = (real_idx == _hover_idx)

                px, py = xs_l[i], ys_l[i]

                if is_drag:
                    _add_circle_filled(px, py, r_drag, col_drag)
                elif is_sel:
                    _add_circle_filled(px, py, r_sel, col_sel)
                    _add_circle(px, py, r_sel_border, col_sel_border)
                elif is_hover:
                    _add_circle_filled(px, py, r_hover, col_hover)
                else:
                    _add_circle_filled(px, py, radius, col_default)

    # ==================================================================================
    # VISUALIZATION DRAWING METHODS
    # ==================================================================================


    def _draw_curve_heatmap(self, dl, tf: 'TimelineTransformer', actions: List[Dict]):
        """Draw the main curve with per-segment heatmap coloring."""
        if not actions or len(actions) < 2:
            return

        # Culling
        margin_ms = tf.zoom * 100
        timestamps = self._get_cached_timestamps()
        if not timestamps or len(timestamps) != len(actions):
            timestamps = [a['at'] for a in actions]
        s_idx = bisect_left(timestamps, tf.visible_start_ms - margin_ms)
        e_idx = bisect_right(timestamps, tf.visible_end_ms + margin_ms)
        s_idx = max(0, s_idx - 1)
        e_idx = min(len(actions), e_idx + 1)
        if e_idx - s_idx < 2:
            return

        visible_actions = actions[s_idx:e_idx]

        # Vectorized transform, use cached numpy arrays when available
        all_ats, all_poss = self._get_cached_numpy_arrays()
        if all_ats is not None and len(all_ats) == len(actions):
            ats = all_ats[s_idx:e_idx]
            poss = all_poss[s_idx:e_idx]
        else:
            ats = np.array([a['at'] for a in visible_actions], dtype=np.float32)
            poss = np.array([a['pos'] for a in visible_actions], dtype=np.float32)
        xs = tf.vec_time_to_x(ats)
        ys = tf.vec_val_to_y(poss)

        # Clamp coordinates
        safe_min_x = tf.x_offset - 5000
        safe_max_x = tf.x_offset + tf.width + 5000
        xs = np.clip(xs, safe_min_x, safe_max_x)

        # Heatmap colors: cache for full script, slice visible range
        all_ats_full, all_poss_full = self._get_cached_numpy_arrays()
        np_id = id(all_ats_full) if all_ats_full is not None else 0
        if self._heatmap_colors_cache is None or self._heatmap_cache_np_id != np_id:
            # Rebuild cache for entire funscript
            if all_ats_full is not None and all_poss_full is not None and len(all_ats_full) >= 2:
                self._heatmap_speeds_cache = HeatmapColorMapper.compute_segment_speeds(
                    actions, ats_np=all_ats_full, poss_np=all_poss_full)
                self._heatmap_colors_cache = self._heatmap_mapper.speeds_to_colors_u32(
                    self._heatmap_speeds_cache)
            else:
                self._heatmap_speeds_cache = HeatmapColorMapper.compute_segment_speeds(actions)
                self._heatmap_colors_cache = self._heatmap_mapper.speeds_to_colors_u32(
                    self._heatmap_speeds_cache)
            self._heatmap_cache_np_id = np_id

        # Slice cached colors for visible range (segments = points - 1)
        seg_start = max(0, s_idx)
        seg_end = min(len(self._heatmap_colors_cache), e_idx - 1)
        colors_u32 = self._heatmap_colors_cache[seg_start:seg_end]

        # Draw per-segment colored lines. Pre-convert to Python lists to avoid
        # per-element numpy indexing overhead in the loop.
        n_segs = len(colors_u32)
        xs_list = xs.tolist()
        ys_list = ys.tolist()
        hm_thick = self._line_thickness_for_height()
        # Adaptive LOD for the smooth curve (see _spline_samples_for_view).
        lod_k = self._spline_samples_for_view(tf.width, n_segs) if n_segs > 0 else 0

        def _draw_runs(point_xs, point_ys, sample_colors, add_polyline):
            """Emit one polyline per run of consecutive same-color samples.
            Cuts Python-side draw calls from O(N) to O(color_runs), which for
            physically continuous motion is typically 5-20x fewer. Run
            boundaries are located via numpy comparison, avoids the Python
            while-loop overhead for large N."""
            n = len(sample_colors)
            if n <= 0 or len(point_xs) < 2:
                return
            arr = np.asarray(sample_colors, dtype=np.uint32)
            # Positions where the color changes (0-indexed boundary of next run).
            change_idx = np.flatnonzero(arr[:-1] != arr[1:]) + 1
            # Build list of run starts: [0, change_1, change_2, ..., n]
            starts = [0, *change_idx.tolist(), n]
            # Precompute the full polyline points once; slicing is O(run_len).
            pts_all = list(zip(point_xs, point_ys))
            for run_i in range(len(starts) - 1):
                i, j = starts[run_i], starts[run_i + 1]
                add_polyline(pts_all[i:j + 1], int(arr[i]), False, hm_thick)

        if self._show_smooth_curve and len(ats) >= 2 and n_segs > 0 and lod_k >= 3:
            d_ats, d_poss, seg_idx = self._expand_catmull(ats, poss, lod_k)
            d_xs = np.clip(tf.vec_time_to_x(d_ats), safe_min_x, safe_max_x).tolist()
            d_ys = tf.vec_val_to_y(d_poss).tolist()
            # Map each dense sample to its segment's color via numpy fancy
            # indexing (was a Python list comp with per-element min() + bounds
            # check). Pads seg_idx to the sample count if needed.
            n_dense = len(d_xs) - 1
            seg_clamped = np.minimum(seg_idx[:n_dense], n_segs - 1)
            if seg_clamped.shape[0] < n_dense:
                seg_clamped = np.concatenate([
                    seg_clamped,
                    np.full(n_dense - seg_clamped.shape[0], n_segs - 1, dtype=seg_clamped.dtype),
                ])
            dense_colors = colors_u32[seg_clamped].tolist()
            _draw_runs(d_xs, d_ys, dense_colors, dl.add_polyline)
        else:
            _draw_runs(xs_list, ys_list, colors_u32.tolist(), dl.add_polyline)

        # Always draw every visible point in heatmap view.
        radius = self.app.app_state_ui.timeline_point_radius
        col_default = _u32_from_const(TimelineColors.POINT_DEFAULT)
        col_drag = _u32_from_const(TimelineColors.POINT_DRAGGING)
        col_sel = _u32_from_const(TimelineColors.POINT_SELECTED)
        col_hover = _u32_from_const(TimelineColors.POINT_HOVER)
        col_sel_border = _u32_from_const(TimelineColors.SELECTED_POINT_BORDER)
        r_drag, r_sel, r_hover = radius + 2, radius + 1, radius + 1
        r_sel_border = r_sel + 1
        _sel_set = self.multi_selected_action_indices
        _drag_idx = self.dragging_action_idx
        _hover_idx = self._hovered_point_idx
        _add_circle_filled = dl.add_circle_filled
        _add_circle = dl.add_circle
        n_actions = len(visible_actions)
        if _sel_set:
            is_sel_list = [(a['at'], a['pos']) in _sel_set for a in visible_actions]
        else:
            is_sel_list = [False] * n_actions

        for i in range(n_actions):
            real_idx = s_idx + i
            is_sel = is_sel_list[i]
            is_drag = (real_idx == _drag_idx)
            is_hover = (real_idx == _hover_idx)

            px, py = xs_list[i], ys_list[i]
            if is_drag:
                _add_circle_filled(px, py, r_drag, col_drag)
            elif is_sel:
                _add_circle_filled(px, py, r_sel, col_sel)
                _add_circle(px, py, r_sel_border, col_sel_border)
            elif is_hover:
                _add_circle_filled(px, py, r_hover, col_hover)
            else:
                _add_circle_filled(px, py, radius, col_default)


    def _draw_speed_limit_overlay(self, dl, tf: 'TimelineTransformer', actions: List[Dict]):
        """Draw red semi-transparent bands for speed limit violations."""
        if not actions or len(actions) < 2:
            return

        # Culling
        margin_ms = tf.zoom * 100
        timestamps = self._get_cached_timestamps()
        if not timestamps or len(timestamps) != len(actions):
            timestamps = [a['at'] for a in actions]
        s_idx = bisect_left(timestamps, tf.visible_start_ms - margin_ms)
        e_idx = bisect_right(timestamps, tf.visible_end_ms + margin_ms)
        s_idx = max(0, s_idx - 1)
        e_idx = min(len(actions), e_idx + 1)
        if e_idx - s_idx < 2:
            return

        visible_actions = actions[s_idx:e_idx]
        # Use cached speeds if available (built by heatmap renderer)
        if self._heatmap_speeds_cache is not None and len(self._heatmap_speeds_cache) == len(actions) - 1:
            seg_start = max(0, s_idx)
            seg_end = min(len(self._heatmap_speeds_cache), e_idx - 1)
            speeds = self._heatmap_speeds_cache[seg_start:seg_end]
        else:
            speeds = HeatmapColorMapper.compute_segment_speeds(visible_actions)
        threshold = self._speed_limit_threshold

        all_ats, _ = self._get_cached_numpy_arrays()
        if all_ats is not None and len(all_ats) == len(actions):
            ats = all_ats[s_idx:e_idx]
        else:
            ats = np.array([a['at'] for a in visible_actions], dtype=np.float32)
        xs = tf.vec_time_to_x(ats)
        xs = np.clip(xs, tf.x_offset - 100, tf.x_offset + tf.width + 100)

        violation_col = _u32_from_const(TimelineColors.SPEED_VIOLATION)
        for i in range(len(speeds)):
            if speeds[i] > threshold:
                x1 = float(xs[i])
                x2 = float(xs[i + 1])
                dl.add_rect_filled(x1, tf.y_offset, x2, tf.y_offset + tf.height, violation_col)


    def _draw_reference_peak_markers(self, dl, tf: 'TimelineTransformer'):
        """Draw color-coded squares on matched peaks and hollow squares on unmatched peaks."""
        if self._reference_metrics_dirty:
            # Throttle: wait 0.3s after last edit before recomputing (avoids per-frame recompute during drag)
            dirty_time = getattr(self, '_reference_metrics_dirty_time', 0)
            if time.monotonic() - dirty_time >= 0.3:
                self._recompute_reference_data()

        color_map = {
            'gold': TimelineColors.REFERENCE_MATCH_GOLD,
            'green': TimelineColors.REFERENCE_MATCH_GREEN,
            'yellow': TimelineColors.REFERENCE_MATCH_YELLOW,
            'red': TimelineColors.REFERENCE_MATCH_RED,
        }
        unmatched_col = TimelineColors.REFERENCE_UNMATCHED
        sq = 4  # Square half-size (pixels)

        # Draw matched peaks, filled square at the main peak position
        if self._reference_peak_matches:
            for mp, rp, offset, classification in self._reference_peak_matches:
                px = tf.time_to_x(mp['at'])
                if px < tf.x_offset - 20 or px > tf.x_offset + tf.width + 20:
                    continue
                py = tf.val_to_y(mp['pos'])
                col = color_map.get(classification, unmatched_col)
                col_u32 = imgui.get_color_u32_rgba(*col)
                dl.add_rect_filled(px - sq, py - sq, px + sq, py + sq, col_u32)

        # Draw unmatched main peaks, hollow square
        if self._reference_unmatched_main:
            col_u32 = imgui.get_color_u32_rgba(*unmatched_col)
            for p in self._reference_unmatched_main:
                px = tf.time_to_x(p['at'])
                if px < tf.x_offset - 20 or px > tf.x_offset + tf.width + 20:
                    continue
                py = tf.val_to_y(p['pos'])
                dl.add_rect(px - sq, py - sq, px + sq, py + sq, col_u32, 0, 0, 1.5)

        # Draw unmatched ref peaks, hollow square on the reference curve
        if self._reference_unmatched_ref:
            col_u32 = imgui.get_color_u32_rgba(*unmatched_col)
            for p in self._reference_unmatched_ref:
                px = tf.time_to_x(p['at'])
                if px < tf.x_offset - 20 or px > tf.x_offset + tf.width + 20:
                    continue
                py = tf.val_to_y(p['pos'])
                dl.add_rect(px - sq, py - sq, px + sq, py + sq, col_u32, 0, 0, 1.5)


    def _draw_reference_problem_bands(self, dl, tf: 'TimelineTransformer'):
        """Draw semi-transparent red bands over detected problem sections."""
        if not self._reference_problem_sections:
            return
        band_col = _u32_from_const(TimelineColors.REFERENCE_PROBLEM_FILL)
        border_col = _u32_from_const(TimelineColors.REFERENCE_PROBLEM_BORDER)
        for sec in self._reference_problem_sections:
            x1 = tf.time_to_x(sec['start_ms'])
            x2 = tf.time_to_x(sec['end_ms'])
            # Skip if entirely off-screen
            if x2 < tf.x_offset or x1 > tf.x_offset + tf.width:
                continue
            dl.add_rect_filled(x1, tf.y_offset, x2, tf.y_offset + tf.height, band_col)
            dl.add_line(x1, tf.y_offset, x1, tf.y_offset + tf.height, border_col, 1.0)
            dl.add_line(x2, tf.y_offset, x2, tf.y_offset + tf.height, border_col, 1.0)


    def _draw_chapter_highlight_overlay(self, dl, tf: 'TimelineTransformer'):
        """Draw gold highlight band for context-selected chapters."""
        nav_ui = None
        if self.app.gui_instance and hasattr(self.app.gui_instance, 'video_navigation_ui'):
            nav_ui = self.app.gui_instance.video_navigation_ui
        if not nav_ui or not nav_ui.context_selected_chapters:
            return

        processor = self.app.processor
        if not processor or not processor.video_info:
            return
        fps = processor.fps
        if fps <= 0:
            return

        fill_col = _u32_from_const(TimelineColors.CHAPTER_HIGHLIGHT_FILL)
        edge_col = _u32_from_const(TimelineColors.CHAPTER_HIGHLIGHT_EDGE)

        for chapter in nav_ui.context_selected_chapters:
            start_ms = (chapter.start_frame_id / fps) * 1000.0
            end_ms = (chapter.end_frame_id / fps) * 1000.0

            # Cull offscreen chapters
            if end_ms < tf.visible_start_ms or start_ms > tf.visible_end_ms:
                continue

            x1 = max(tf.x_offset, tf.time_to_x(start_ms))
            x2 = min(tf.x_offset + tf.width, tf.time_to_x(end_ms))

            dl.add_rect_filled(x1, tf.y_offset, x2, tf.y_offset + tf.height, fill_col)
            dl.add_line(x1, tf.y_offset, x1, tf.y_offset + tf.height, edge_col, 1.5)
            dl.add_line(x2, tf.y_offset, x2, tf.y_offset + tf.height, edge_col, 1.5)


    def _draw_bpm_grid(self, dl, tf: 'TimelineTransformer'):
        """Draw BPM beat grid lines on the timeline with visual hierarchy."""
        cfg = self._bpm_config
        if not cfg or cfg.bpm <= 0:
            return

        interval_ms = cfg.beat_interval_ms
        if interval_ms <= 0:
            return

        # Calculate visible beat positions
        start_beat = int((tf.visible_start_ms - cfg.offset_ms) / interval_ms)
        end_beat = int((tf.visible_end_ms - cfg.offset_ms) / interval_ms) + 1

        # Base beat interval (whole note / downbeat)
        base_interval = 60000.0 / cfg.bpm
        # Quarter beat interval
        quarter_interval = base_interval / 4.0 if base_interval > 0 else 0

        # 3-tier colors: downbeat (bright), quarter (medium), subdivision (faint)
        downbeat_col = _u32_from_const(TimelineColors.BPM_DOWNBEAT)
        quarter_col = _u32_from_const(TimelineColors.BPM_QUARTER)
        sub_col = _u32_from_const(TimelineColors.BPM_SUB)

        for beat_num in range(start_beat, end_beat + 1):
            t_ms = cfg.offset_ms + beat_num * interval_ms
            if t_ms < tf.visible_start_ms or t_ms > tf.visible_end_ms:
                continue
            x = tf.time_to_x(t_ms)

            # Classify line tier
            rel = t_ms - cfg.offset_ms
            if base_interval > 0 and abs(rel % base_interval) < 1.0:
                # Downbeat (measure start)
                col, thick = downbeat_col, 2.0
            elif quarter_interval > 0 and abs(rel % quarter_interval) < 1.0:
                # Quarter beat
                col, thick = quarter_col, 1.2
            else:
                # Subdivision
                col, thick = sub_col, 0.7

            dl.add_line(x, tf.y_offset, x, tf.y_offset + tf.height, col, thick)


    def _draw_bookmarks(self, dl, tf: 'TimelineTransformer'):
        """Draw bookmark markers on the timeline."""
        visible = self._bookmark_manager.get_in_range(tf.visible_start_ms, tf.visible_end_ms)
        if not visible:
            return

        for bm in visible:
            x = tf.time_to_x(bm.time_ms)
            col = imgui.get_color_u32_rgba(*bm.color)

            # Vertical line
            dl.add_line(x, tf.y_offset, x, tf.y_offset + tf.height, col, 1.5)

            # Triangle marker at top
            tri_size = 6
            dl.add_triangle_filled(
                x, tf.y_offset,
                x - tri_size, tf.y_offset - tri_size,
                x + tri_size, tf.y_offset - tri_size,
                col
            )

            # Label
            if bm.name:
                dl.add_text(x + 4, tf.y_offset + 2, col, bm.name[:20])

    # ==================================================================================
    # MODE-SPECIFIC INPUT HANDLERS
    # ==================================================================================


    def _draw_controller_settings(self):
        """Draw collapsible controller settings panel below the recording toolbar."""
        gp = self._gamepad_input
        if not gp:
            return

        imgui.indent(10)

        # Axis mapping
        axis_options = ["left_y", "left_x", "right_y", "right_x"]
        axis_labels = ["Left Y", "Left X", "Right Y", "Right X"]
        cur = axis_options.index(gp.axis_mapping) if gp.axis_mapping in axis_options else 0
        imgui.push_item_width(70)
        changed, new_idx = imgui.combo(f"Axis##{self.timeline_num}_gp", cur, axis_labels)
        if changed:
            gp.axis_mapping = axis_options[new_idx]
        imgui.pop_item_width()

        imgui.same_line()
        changed, inv = imgui.checkbox(f"Inv##{self.timeline_num}_inv", gp.invert_primary)
        if changed:
            gp.invert_primary = inv

        imgui.same_line()
        changed, cm = imgui.checkbox(f"Center##{self.timeline_num}_cm", gp.center_mode)
        if changed:
            gp.center_mode = cm
            self.app.app_settings.config.recording.gamepad_center_mode = cm
        if imgui.is_item_hovered():
            imgui.set_tooltip(
                "On: full stick travel maps to 0..100 (rest = 50).\n"
                "Off: stick deflection magnitude maps to 0..100 (rest = 0)."
            )

        # Deadzone
        imgui.same_line()
        imgui.push_item_width(60)
        changed, dz = imgui.slider_float(f"DZ##{self.timeline_num}", gp.deadzone, 0.05, 0.40, "%.2f")
        if changed:
            gp.deadzone = dz
        imgui.pop_item_width()

        # Input delay
        imgui.push_item_width(80)
        changed, delay = imgui.slider_int(
            f"Delay##{self.timeline_num}", self._controller_input_delay_ms, 0, 200, "%d ms")
        if changed:
            self._controller_input_delay_ms = delay
        imgui.pop_item_width()

        imgui.same_line()
        if imgui.small_button(f"Cal##{self.timeline_num}"):
            self._start_calibration()
        if self._calibration and self._calibration.result_ms is not None and not self._calibration.is_running:
            imgui.same_line()
            imgui.text(f"({self._calibration.result_ms:.0f}ms)")

        # Device preview toggle (only if device_control addon loaded)
        dm = self._get_device_manager()
        if dm is not None:
            changed, preview = imgui.checkbox(
                f"Device Preview##{self.timeline_num}", self._controller_device_preview)
            if changed:
                self._controller_device_preview = preview

        # Live position bar
        state = gp.poll()
        if state:
            imgui.progress_bar(state.primary / 100.0, (-1, 14), f"{state.primary:.0f}")

        imgui.unindent(10)


    def _draw_ui_overlays(self, dl, tf: 'TimelineTransformer'):
        # 1. Playhead (Center), line + inverted triangle at top
        # Round to nearest pixel + 0.5 so the 1px-wide visual center of the
        # 2px line lands exactly on a pixel boundary, and the triangle tip aligns.
        center_x = round(tf.x_offset + tf.width / 2) + 0.5
        marker_color = _u32_from_const(TimelineColors.CENTER_MARKER)
        tri_top = tf.y_offset
        dl.add_triangle_filled(
            center_x - 6, tri_top,
            center_x + 6, tri_top,
            center_x, tri_top + 8,
            marker_color)
        dl.add_line(center_x, tri_top, center_x, tf.y_offset + tf.height,
                    marker_color, 1.0)

        # Optional video-time sync line: thin red line at the actual frame's
        # ms (current_frame_index/fps), distinct from the white playhead which
        # follows playhead_override_ms when set. Reveals sub-frame drift
        # between video position and the timeline marker.
        if self.app.app_settings.config.ui.timeline_show_video_sync_line:
            proc = self.app.processor
            if proc and proc.fps and proc.fps > 0:
                video_ms = (proc.current_frame_index / proc.fps) * 1000.0
                video_x = tf.time_to_x(video_ms)
                if tf.x_offset <= video_x <= tf.x_offset + tf.width:
                    dl.add_line(video_x, tri_top, video_x,
                                tf.y_offset + tf.height,
                                imgui.get_color_u32_rgba(0.95, 0.2, 0.2, 0.85), 1.0)

        # Playhead Time Info (timecode + frame number)
        time_ms = tf.x_to_time(center_x)
        txt = _format_time(self.app, time_ms/1000.0)
        proc = self.app.processor
        if proc and proc.fps and proc.fps > 0:
            frame_num = ms_to_frame(time_ms, proc.fps)
            txt = f"{txt}  ({frame_num})"
        dl.add_text(center_x + 6, tf.y_offset + 6, _u32_from_const(TimelineColors.TIME_DISPLAY_TEXT), txt)
        
        # 2. Marquee Box
        if self.is_marqueeing and self.marquee_start and self.marquee_end:
            p1 = self.marquee_start
            p2 = self.marquee_end
            x_min, x_max = min(p1[0], p2[0]), max(p1[0], p2[0])
            y_min, y_max = min(p1[1], p2[1]), max(p1[1], p2[1])
            
            dl.add_rect_filled(x_min, y_min, x_max, y_max, _u32_from_const(TimelineColors.MARQUEE_SELECTION_FILL))
            dl.add_rect(x_min, y_min, x_max, y_max, _u32_from_const(TimelineColors.MARQUEE_SELECTION_BORDER))

        # 3. Range Selection Highlight
        if self.range_selecting:
            t1, t2 = sorted([self.range_start_time, self.range_end_time])
            x1 = tf.time_to_x(t1)
            x2 = tf.time_to_x(t2)
            dl.add_rect_filled(x1, tf.y_offset, x2, tf.y_offset + tf.height, _u32_from_const(TimelineColors.SELECTION_RANGE_FILL))
            dl.add_line(x1, tf.y_offset, x1, tf.y_offset+tf.height, _u32_from_const(TimelineColors.SELECTION_RANGE_BORDER))
            dl.add_line(x2, tf.y_offset, x2, tf.y_offset+tf.height, _u32_from_const(TimelineColors.SELECTION_RANGE_BORDER))

        # 4. Bookmarks
        self._draw_bookmarks(dl, tf)

        # 5. Recording indicator
        if self._recording_capture and self._recording_capture.is_recording:
            rec_col = _u32_from_const(TimelineColors.RECORDING)
            dl.add_circle_filled(tf.x_offset + 12, tf.y_offset + 12, 5, rec_col)
            dl.add_text(tf.x_offset + 20, tf.y_offset + 5,
                        _u32_from_const(TimelineColors.RECORDING), "REC")

        # 6. Calibration modal
        if self._calibration and self._calibration.is_running:
            self._draw_calibration_modal()


    def _draw_calibration_modal(self):
        """Render the controller input-delay calibration popup."""
        cal = self._calibration
        modal_id = f"Calibrate Input Delay##{self.timeline_num}_cal"
        imgui.open_popup(modal_id)
        center_next_window_pivot()
        opened, _ = imgui.begin_popup_modal(modal_id, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE)
        if opened:
            beat_active = cal.get_current_beat_active()
            progress = cal.current_beat / cal.NUM_BEATS
            imgui.text(f"Beat {cal.current_beat}/{cal.NUM_BEATS}")
            imgui.progress_bar(progress, (-1, 0))
            if beat_active:
                imgui.text_colored(">>> PRESS A <<<", 0.2, 1.0, 0.2, 1.0)
            else:
                imgui.text("   Wait...")
            done = cal.update()
            if done and cal.result_ms is not None:
                imgui.separator()
                imgui.text(f"Result: {cal.result_ms:.0f}ms")
                if imgui.button("Accept"):
                    self._controller_input_delay_ms = int(cal.result_ms)
                    self._calibration = None
                    imgui.close_current_popup()
                imgui.same_line()
                if imgui.button("Retry"):
                    cal.start()
                imgui.same_line()
                if imgui.button("Cancel"):
                    self._calibration = None
                    imgui.close_current_popup()
            imgui.end_popup()


    def _draw_state_border(self, dl, canvas_pos, canvas_size, app_state):
        """
        Draw a colored border indicating timeline state:
        - Green: Active and editable (shortcuts will work)
        - Red: Active but read-only (during playback, text input, etc.)
        - Gray: Inactive (another timeline is active)
        """
        is_active = app_state.active_timeline_num == self.timeline_num

        is_read_only = self._is_timeline_read_only(app_state) if is_active else False

        if not is_active:
            border_color = _u32_from_const(TimelineColors.STATE_BORDER_NORMAL)
        else:
            if is_read_only:
                # Red border for active but read-only
                border_color = _u32_from_const(TimelineColors.STATE_BORDER_LOCKED)
            else:
                # Green border for active and editable
                border_color = _u32_from_const(TimelineColors.STATE_BORDER_ACTIVE)

        # Draw border around canvas area
        x1, y1 = canvas_pos[0], canvas_pos[1]
        x2, y2 = x1 + canvas_size[0], y1 + canvas_size[1]
        border_thickness = 2.0 if is_active else 1.0
        dl.add_rect(x1, y1, x2, y2, border_color, 0.0, 0, border_thickness)

        # Tooltip on border hover (bottom 6px strip)
        mouse = imgui.get_mouse_pos()
        if (x1 <= mouse[0] <= x2 and y2 - 6 <= mouse[1] <= y2):
            imgui.begin_tooltip()
            if not is_active:
                imgui.text("Inactive (click to activate)")
            elif is_read_only:
                imgui.text_colored("Read-only (stop playback to edit)", 0.9, 0.3, 0.3, 1.0)
            else:
                imgui.text_colored("Active (editable)", 0.3, 0.8, 0.3, 1.0)
            imgui.end_tooltip()


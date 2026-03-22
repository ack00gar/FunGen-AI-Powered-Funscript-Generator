"""
Gauge Widget — sleek vertical position indicator (0-100) for funscript playback.

Features: pill-shaped bar with inner depth, glowing fill with bright edge at
current level, value-dependent color gradient, position indicator line,
multiple independent instances, semi-transparent overlay-friendly.
"""

import imgui
import math


def _lerp(a, b, t):
    return a + (b - a) * t


def _lerp_col(c1, c2, t):
    return tuple(_lerp(a, b, t) for a, b in zip(c1, c2))


# Color stops: value -> (r, g, b)
_C_LO = (0.20, 0.55, 0.95)   # blue at 0
_C_MID = (0.15, 0.82, 0.45)  # green at 50
_C_HI = (1.00, 0.30, 0.18)   # red-orange at 100


def _color_rgb(val):
    """Value-dependent color: blue -> green -> red."""
    val = max(0.0, min(100.0, val))
    if val <= 50:
        return _lerp_col(_C_LO, _C_MID, val / 50.0)
    return _lerp_col(_C_MID, _C_HI, (val - 50) / 50.0)


_SCALE_MARKS = [0, 25, 50, 75, 100]
_next_gauge_id = 1


_AXIS_LABELS = {1: "Stroke", 2: "Roll"}


class GaugeInstance:
    __slots__ = ('gauge_id', 'selected_timeline', '_prev_pos')

    def __init__(self, gauge_id: int, timeline: int = 1):
        self.gauge_id = gauge_id
        self.selected_timeline = timeline
        self._prev_pos = 0.0

    def window_title(self):
        axis = _AXIS_LABELS.get(self.selected_timeline, f"Axis {self.selected_timeline}")
        return f"T{self.selected_timeline} - {axis}##{self.gauge_id}"


class ScriptGaugeWindow:
    """Manages one or more floating gauge windows."""

    def __init__(self, app_instance):
        global _next_gauge_id
        self.app = app_instance
        self.gauges = [GaugeInstance(_next_gauge_id)]
        _next_gauge_id += 1

    def render(self):
        app_state = self.app.app_state_ui
        if not getattr(app_state, 'show_script_gauge', False):
            return
        closed_ids = []
        for g in self.gauges:
            if not self._render_gauge(g, app_state):
                closed_ids.append(g.gauge_id)
        if closed_ids:
            self.gauges = [g for g in self.gauges if g.gauge_id not in closed_ids]
            if not self.gauges:
                app_state.show_script_gauge = False

    def add_gauge(self, timeline: int = 1):
        global _next_gauge_id
        self.gauges.append(GaugeInstance(_next_gauge_id, timeline))
        _next_gauge_id += 1

    # ------------------------------------------------------------------

    def _render_gauge(self, g, app_state):
        imgui.set_next_window_size(64, 340, condition=imgui.ONCE)
        imgui.push_style_color(imgui.COLOR_WINDOW_BACKGROUND, 0.06, 0.06, 0.08, 0.50)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND, 0.10, 0.10, 0.13, 0.60)
        imgui.push_style_color(imgui.COLOR_TITLE_BACKGROUND_ACTIVE, 0.12, 0.12, 0.16, 0.70)
        imgui.push_style_var(imgui.STYLE_WINDOW_ROUNDING, 8.0)
        imgui.push_style_var(imgui.STYLE_WINDOW_PADDING, (4, 4))

        flags = (imgui.WINDOW_NO_SCROLLBAR | imgui.WINDOW_NO_SCROLL_WITH_MOUSE |
                 imgui.WINDOW_NO_COLLAPSE)
        visible, opened = imgui.begin(g.window_title(), closable=True, flags=flags)

        if not opened:
            imgui.end()
            imgui.pop_style_var(2)
            imgui.pop_style_color(3)
            return False

        if visible:
            self._draw(g, app_state)

        imgui.end()
        imgui.pop_style_var(2)
        imgui.pop_style_color(3)
        return True

    def _draw(self, g, app_state):
        visible_tl = self._get_visible_timelines(app_state)

        # Auto-select / validate
        if len(visible_tl) == 1:
            g.selected_timeline = visible_tl[0][0]
        elif visible_tl:
            nums = [t[0] for t in visible_tl]
            if g.selected_timeline not in nums:
                g.selected_timeline = nums[0]

        if not visible_tl:
            return

        # Get position
        pos = max(0.0, min(100.0, self._get_pos(g, app_state)))

        # Smooth for visual (simple exponential)
        g._prev_pos = _lerp(g._prev_pos, pos, 0.35)

        self._draw_bar(g, g._prev_pos, pos)

        # Double-click anywhere in gauge to cycle timeline
        if len(visible_tl) > 1 and imgui.is_window_hovered() and imgui.is_mouse_double_clicked(0):
            nums = [t[0] for t in visible_tl]
            idx = nums.index(g.selected_timeline) if g.selected_timeline in nums else 0
            g.selected_timeline = nums[(idx + 1) % len(nums)]

        # Right-click context menu: switch timeline, add gauge
        if imgui.begin_popup_context_window(f"##gctx{g.gauge_id}"):
            if len(visible_tl) > 1:
                for t_num, t_label in visible_tl:
                    axis_name = _AXIS_LABELS.get(t_num, f"T{t_num}")
                    selected = (t_num == g.selected_timeline)
                    if imgui.menu_item(axis_name, selected=selected)[0]:
                        g.selected_timeline = t_num
                imgui.separator()
            if imgui.menu_item("Add Gauge")[0]:
                self.add_gauge(g.selected_timeline)
            imgui.end_popup()

    def _draw_bar(self, g, smooth_pos, exact_pos):
        dl = imgui.get_window_draw_list()
        avail_w = imgui.get_content_region_available_width()
        avail_h = imgui.get_content_region_available().y
        if avail_h < 50:
            avail_h = 50

        # Reserve bottom for value readout
        value_h = 22
        bar_h = avail_h - value_h
        if bar_h < 40:
            bar_h = 40

        # Bar dimensions — centered pill
        pad_x = 2
        bar_w = max(20, avail_w - pad_x * 2)
        cur = imgui.get_cursor_screen_position()
        bx = cur.x + pad_x
        by = cur.y
        rounding = min(bar_w * 0.35, 10)

        frac = smooth_pos / 100.0

        # ---- Outer shell (dark inset) ----
        dl.add_rect_filled(bx, by, bx + bar_w, by + bar_h,
                           imgui.get_color_u32_rgba(0.08, 0.08, 0.10, 0.85), rounding)
        # Inner border for depth
        dl.add_rect(bx, by, bx + bar_w, by + bar_h,
                    imgui.get_color_u32_rgba(0.18, 0.18, 0.22, 0.6), rounding, thickness=1.0)

        # ---- Fill ----
        if frac > 0.005:
            fill_h = bar_h * frac
            fy_top = by + bar_h - fill_h
            fy_bot = by + bar_h

            # Main gradient fill — darker at bottom, brighter at fill level
            c_top = _color_rgb(smooth_pos)
            c_bot = _color_rgb(max(0, smooth_pos * 0.3))

            # Inset fill by 2px for depth illusion
            inset = 2
            fbx = bx + inset
            fbw = bar_w - inset * 2
            fr = max(0, rounding - 1)

            ct = imgui.get_color_u32_rgba(c_top[0], c_top[1], c_top[2], 0.92)
            cb = imgui.get_color_u32_rgba(c_bot[0], c_bot[1], c_bot[2], 0.55)
            dl.add_rect_filled_multicolor(fbx, fy_top, fbx + fbw, fy_bot, ct, ct, cb, cb)

            # Glow band at fill level (bright horizontal line with falloff)
            glow_c = _color_rgb(smooth_pos)
            glow_h = min(6, fill_h * 0.15)
            for i in range(int(glow_h) + 1):
                alpha = 0.7 * (1.0 - i / max(glow_h, 1))
                y = fy_top + i
                dl.add_line(fbx + 1, y, fbx + fbw - 1, y,
                            imgui.get_color_u32_rgba(glow_c[0], glow_c[1], glow_c[2], alpha))

            # Bright indicator line at exact fill level
            dl.add_line(bx + 1, fy_top, bx + bar_w - 1, fy_top,
                        imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.85), 2.0)

        # ---- Scale marks ----
        for mv in _SCALE_MARKS:
            f = mv / 100.0
            y = by + bar_h * (1.0 - f)
            # Tick on the left edge
            tick_a = 0.45 if mv == 50 else 0.25
            dl.add_line(bx, y, bx + 6, y,
                        imgui.get_color_u32_rgba(1.0, 1.0, 1.0, tick_a), 1.0)
            # Label
            label = str(mv)
            ts = imgui.calc_text_size(label)
            # Right-align labels inside the bar at the right edge
            lx = bx + bar_w - ts.x - 3
            ly = y - ts.y * 0.5
            # Clamp within bar vertically
            ly = max(by + 1, min(by + bar_h - ts.y - 1, ly))
            dl.add_text(lx, ly, imgui.get_color_u32_rgba(0.65, 0.65, 0.70, 0.45), label)

        # Advance cursor
        imgui.dummy(avail_w, bar_h + 2)

        # ---- Value readout ----
        val_text = str(int(round(exact_pos)))
        c = _color_rgb(exact_pos)
        ts = imgui.calc_text_size(val_text)
        # Center under bar
        cx = bx + (bar_w - ts.x) * 0.5
        imgui.set_cursor_screen_position((cx, imgui.get_cursor_screen_position().y + 2))
        imgui.text_colored(val_text, c[0], c[1], c[2], 1.0)

    # ------------------------------------------------------------------

    @staticmethod
    def _get_visible_timelines(app_state):
        tl = []
        if getattr(app_state, 'show_funscript_interactive_timeline', True):
            tl.append((1, "T1"))
        if getattr(app_state, 'show_funscript_interactive_timeline2', False):
            tl.append((2, "T2"))
        from application.utils.timeline_constants import EXTRA_TIMELINE_RANGE
        for n in EXTRA_TIMELINE_RANGE:
            if getattr(app_state, f'show_funscript_interactive_timeline{n}', False):
                tl.append((n, f"T{n}"))
        return tl

    @staticmethod
    def _get_pos(g, app_state):
        return getattr(app_state, f'script_position_t{g.selected_timeline}', 0.0)

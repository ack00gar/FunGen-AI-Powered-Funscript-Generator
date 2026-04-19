"""Tron-style perspective grid floor with neon glow."""

import random

import imgui


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height
    horizon_y = h * 0.42
    cx = w / 2
    a = alpha * 0.7
    cyan_glow = imgui.get_color_u32_rgba(0.0, 0.7, 1.0, a * 0.15)
    cyan_line = imgui.get_color_u32_rgba(0.0, 0.85, 1.0, a * 0.55)

    # Horizontal lines (get closer together near horizon)
    n_hlines = 20
    scroll = (current_time * 0.3) % 1.0
    for i in range(n_hlines):
        t = (i + scroll) / n_hlines
        y = horizon_y + t * t * (h - horizon_y)  # Quadratic spacing
        if y > h:
            continue
        brightness = t  # Brighter closer to viewer
        draw_list.add_line(0, y, w, y, imgui.get_color_u32_rgba(0.0, 0.7, 1.0, a * 0.1 * brightness), 3.0)
        draw_list.add_line(0, y, w, y, imgui.get_color_u32_rgba(0.0, 0.85, 1.0, a * 0.4 * brightness), 1.0)

    # Vertical lines converging to vanishing point
    n_vlines = 16
    for i in range(n_vlines):
        vx = (i / (n_vlines - 1)) * w
        # Line from bottom to horizon, converging at center
        bx = vx
        tx = cx + (vx - cx) * 0.05  # Converge toward center at horizon
        draw_list.add_line(bx, h, tx, horizon_y, cyan_glow, 3.0)
        draw_list.add_line(bx, h, tx, horizon_y, cyan_line, 1.0)

    # Light cycle trails (a few horizontal streaks)
    if not hasattr(splash, '_tron_trails'):
        splash._tron_trails = [(random.uniform(horizon_y + 30, h - 30),
                                random.uniform(0.3, 0.8)) for _ in range(3)]
    for ty, speed in splash._tron_trails:
        trail_x = ((current_time * speed * w) % (w * 1.5)) - w * 0.25
        trail_len = 120
        draw_list.add_line(trail_x, ty, trail_x + trail_len, ty,
                           imgui.get_color_u32_rgba(0.0, 0.9, 1.0, a * 0.4), 2.5)
        draw_list.add_line(trail_x + trail_len, ty, trail_x + trail_len + 20, ty,
                           imgui.get_color_u32_rgba(0.0, 0.9, 1.0, a * 0.8), 2.0)

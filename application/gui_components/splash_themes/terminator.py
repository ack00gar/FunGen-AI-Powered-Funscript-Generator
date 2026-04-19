"""Terminator T-800 POV: red HUD overlay with corner brackets and scrolling data."""

import math

import imgui


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    a = alpha * 0.7
    red = imgui.get_color_u32_rgba(1.0, 0.1, 0.05, a * 0.4)
    red_bright = imgui.get_color_u32_rgba(1.0, 0.15, 0.1, a * 0.8)
    w, h = window_width, window_height

    # Screen border
    draw_list.add_rect(4, 4, w - 4, h - 4, red, thickness=1.5)

    # Corner brackets (L-shapes, 60px arms)
    arm = 60
    t = 2.5
    for cx, cy, dx, dy in [(0, 0, 1, 1), (w, 0, -1, 1), (0, h, 1, -1), (w, h, -1, -1)]:
        draw_list.add_line(cx, cy, cx + dx * arm, cy, red_bright, t)
        draw_list.add_line(cx, cy, cx, cy + dy * arm, red_bright, t)

    # Crosshair at center
    cx, cy = w / 2, h / 2
    gap = 12
    size = 30
    for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
        draw_list.add_line(cx + dx * gap, cy + dy * gap,
                           cx + dx * size, cy + dy * size, red_bright, 1.5)

    # Scrolling data readouts on left edge
    data_lines = [
        "SCAN MODE 439", "MATCH: 0.97", "THREAT LEVEL: NONE", "CPU TEMP: 47C",
        "TARGET: LOCKED", "FPS: UNLIMITED", "YOLO: ONLINE", "STATUS: NOMINAL",
        "TRACKING: ACTIVE", "BUFFER: 98%", "NEURAL NET: OK",
    ]
    scroll_offset = current_time * 18
    imgui.set_window_font_scale(0.85)
    for i, txt in enumerate(data_lines):
        ty = ((i * 22 - scroll_offset) % (len(data_lines) * 22 + h)) - 50
        if 0 < ty < h:
            line_a = a * 0.5 * (0.5 + 0.5 * math.sin(i + current_time * 2))
            draw_list.add_text(12, ty, imgui.get_color_u32_rgba(1.0, 0.2, 0.1, line_a), txt)
    imgui.set_window_font_scale(1.0)

    # Subtle red tint overlay
    draw_list.add_rect_filled(0, 0, w, h, imgui.get_color_u32_rgba(0.3, 0.0, 0.0, a * 0.08))

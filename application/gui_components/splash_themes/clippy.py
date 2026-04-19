"""Office-document background with a giant animated paperclip (Clippy)
in a speech bubble. Pure nostalgia bait."""

import math

import imgui


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height
    t = current_time

    # Office document background
    draw_list.add_rect_filled(
        0, 0, w, h,
        imgui.get_color_u32_rgba(0.93, 0.93, 0.88, alpha))
    for ly in range(80, h, 28):
        draw_list.add_line(
            60, ly, w - 60, ly,
            imgui.get_color_u32_rgba(0.7, 0.75, 0.85, 0.4 * alpha), 1.0)

    # Speech bubble on the right
    bubble_x, bubble_y = w * 0.40, h * 0.30
    bw, bh = w * 0.50, h * 0.35
    draw_list.add_rect_filled(
        bubble_x, bubble_y, bubble_x + bw, bubble_y + bh,
        imgui.get_color_u32_rgba(1, 1, 0.88, alpha), 12)
    draw_list.add_rect(
        bubble_x, bubble_y, bubble_x + bw, bubble_y + bh,
        imgui.get_color_u32_rgba(0.1, 0.1, 0.2, alpha), 12, thickness=2.5)
    # Speech tail
    draw_list.add_triangle_filled(
        bubble_x, bubble_y + bh * 0.55,
        bubble_x - 28, bubble_y + bh * 0.70,
        bubble_x, bubble_y + bh * 0.80,
        imgui.get_color_u32_rgba(1, 1, 0.88, alpha))
    draw_list.add_line(
        bubble_x, bubble_y + bh * 0.55,
        bubble_x - 28, bubble_y + bh * 0.70,
        imgui.get_color_u32_rgba(0.1, 0.1, 0.2, alpha), 2.5)
    draw_list.add_line(
        bubble_x - 28, bubble_y + bh * 0.70,
        bubble_x, bubble_y + bh * 0.80,
        imgui.get_color_u32_rgba(0.1, 0.1, 0.2, alpha), 2.5)

    # Bubble text with typewriter reveal (version + paypal baked in)
    from config.constants import APP_VERSION
    imgui.set_window_font_scale(1.7)
    full = ("It looks like you're trying\n"
            "to generate a funscript.\n\n"
            "Would you like help?\n\n"
            f"FunGen v{APP_VERSION}\n"
            "paypal.me/k00gar")
    chars_vis = max(0, int(t * 45))
    lines = full[:chars_vis].split('\n')
    for i, ln in enumerate(lines):
        draw_list.add_text(
            bubble_x + 22, bubble_y + 24 + i * 34,
            imgui.get_color_u32_rgba(0.1, 0.1, 0.2, alpha), ln)
    imgui.set_window_font_scale(1.0)

    # Buttons below the text (after full text shown)
    if t > 1.4:
        btns = [("Yes", 0), ("No", 1), ("Maybe later", 2)]
        bby = bubble_y + bh - 58
        bbx = bubble_x + 22
        imgui.set_window_font_scale(1.2)
        for lbl, i in btns:
            ts = imgui.calc_text_size(lbl)
            pad = 12
            x0 = bbx + i * 130
            draw_list.add_rect_filled(
                x0, bby, x0 + ts[0] + pad * 2, bby + ts[1] + pad,
                imgui.get_color_u32_rgba(0.88, 0.88, 0.92, alpha), 4)
            draw_list.add_rect(
                x0, bby, x0 + ts[0] + pad * 2, bby + ts[1] + pad,
                imgui.get_color_u32_rgba(0.1, 0.1, 0.2, alpha), 4,
                thickness=1.5)
            draw_list.add_text(
                x0 + pad, bby + pad / 2,
                imgui.get_color_u32_rgba(0.1, 0.1, 0.2, alpha), lbl)
        imgui.set_window_font_scale(1.0)

    # Clippy: 3 rounded ovals forming a paperclip + eyes + brows
    cx, cy = w * 0.22, h * 0.55
    bob = math.sin(t * 5.5) * 8
    tilt = math.sin(t * 3.2) * 6
    cy += bob
    cx += tilt
    clip_col = imgui.get_color_u32_rgba(0.82, 0.82, 0.86, alpha)
    draw_list.add_rect(
        cx - 50, cy - 90, cx + 50, cy + 110,
        clip_col, 40, thickness=8)
    draw_list.add_rect(
        cx - 30, cy - 70, cx + 30, cy + 90,
        clip_col, 28, thickness=8)
    draw_list.add_line(
        cx - 30, cy - 10, cx + 10, cy - 10, clip_col, 7)
    # Eyes
    eye_y = cy - 55
    for ex in (cx - 18, cx + 18):
        draw_list.add_circle_filled(
            ex, eye_y, 10,
            imgui.get_color_u32_rgba(1, 1, 1, alpha))
        draw_list.add_circle(
            ex, eye_y, 10,
            imgui.get_color_u32_rgba(0, 0, 0, alpha), thickness=2)
        px_off = 4 * math.cos(t * 4.0)
        py_off = 3 * math.sin(t * 4.7)
        draw_list.add_circle_filled(
            ex + px_off + 2, eye_y + py_off, 4,
            imgui.get_color_u32_rgba(0, 0, 0, alpha))
    # Eyebrows
    brow = math.sin(t * 4.2) * 8
    draw_list.add_line(
        cx - 30, eye_y - 22 + brow, cx - 6, eye_y - 18 + brow,
        imgui.get_color_u32_rgba(0, 0, 0, alpha), 4)
    draw_list.add_line(
        cx + 6, eye_y - 18 - brow, cx + 30, eye_y - 22 - brow,
        imgui.get_color_u32_rgba(0, 0, 0, alpha), 4)

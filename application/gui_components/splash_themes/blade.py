"""Rainy neo-noir NYC: smoggy sky, neon haze, flickering windows, drifting
flying-car lights, rain streaks, orange+blue neon halo behind the logo."""

import math
import random

import imgui


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height
    t = current_time

    state = getattr(splash, '_blade_state', None)
    if state is None:
        rng = random.Random(77)
        buildings = []
        x = 0
        while x < w:
            bw = rng.randint(20, 70)
            bh_ = rng.randint(80, 260)
            buildings.append((x, bw, bh_, rng.random()))
            x += bw
        rng2 = random.Random(3)
        rain = [(rng2.uniform(0, 1), rng2.uniform(0, 1),
                 rng2.uniform(0.8, 1.4)) for _ in range(220)]
        state = {'skyline': buildings, 'rain': rain}
        splash._blade_state = state

    # Smoggy sky gradient
    for i in range(40):
        f = i / 40
        y0 = f * h
        y1 = (f + 1 / 40) * h + 1
        r_ = 0.06 + 0.19 * f
        g_ = 0.02 + 0.08 * f
        b_ = 0.05 + 0.01 * f
        draw_list.add_rect_filled(
            0, y0, w, y1, imgui.get_color_u32_rgba(r_, g_, b_, alpha))

    # Neon haze blobs
    for i, (cx, cy, cc) in enumerate([
        (w * 0.25, h * 0.40, (1.0, 0.45, 0.1)),
        (w * 0.70, h * 0.30, (1.0, 0.25, 0.5)),
        (w * 0.55, h * 0.55, (0.3, 0.6, 1.0)),
    ]):
        pulse = 0.7 + 0.3 * math.sin(t * 1.5 + i)
        for rad, aa in ((260, 0.05), (180, 0.08), (100, 0.12)):
            draw_list.add_circle_filled(
                cx, cy, rad * pulse,
                imgui.get_color_u32_rgba(*cc, aa * pulse * alpha))

    # Skyline
    skyline_y = h * 0.70
    for (bx, bw, bh_, seed) in state['skyline']:
        by = skyline_y - bh_
        draw_list.add_rect_filled(
            bx, by, bx + bw, skyline_y,
            imgui.get_color_u32_rgba(0.03, 0.01, 0.04, alpha))
        for wi in range(int(bh_ / 18)):
            for wj in range(int(bw / 14)):
                lit = ((int(seed * 1000) + wi * 7 + wj * 11
                        + int(t * 0.5)) % 11) < 3
                if lit:
                    wx = bx + 3 + wj * 14
                    wy = by + 6 + wi * 18
                    draw_list.add_rect_filled(
                        wx, wy, wx + 4, wy + 4,
                        imgui.get_color_u32_rgba(1.0, 0.65, 0.2,
                                                 0.7 * alpha))

    # FunGen logos as Blade Runner "spinner" aircraft
    if splash.logo_texture:
        spinners = [
            (h * 0.18, 42,  0.85, 0.0),
            (h * 0.34, -32, 0.65, 2.1),
            (h * 0.12, 25,  0.95, 4.3),
            (h * 0.45, -52, 0.55, 1.4),
            (h * 0.28, 60,  0.48, 3.6),
        ]
        for sy_base, sp, sz_f, phase in spinners:
            sx = (t * sp + phase * 300) % (w + 400) - 200
            bob = math.sin(t * 1.8 + phase) * 6
            sy = sy_base + bob
            logo_sz = min(w, h) * 0.065 * sz_f + 28
            cone_sway = math.sin(t * 0.7 + phase * 2) * 60
            cone_ground_x = sx + cone_sway
            cone_ground_y = min(sy + logo_sz * 5, skyline_y - 5)
            cone_half_w = (cone_ground_y - sy) * 0.18
            draw_list.add_triangle_filled(
                sx, sy + logo_sz * 0.5,
                cone_ground_x - cone_half_w, cone_ground_y,
                cone_ground_x + cone_half_w, cone_ground_y,
                imgui.get_color_u32_rgba(1.0, 0.9, 0.7, 0.06 * alpha))
            draw_list.add_circle_filled(
                sx - logo_sz * 0.6, sy, 3.5,
                imgui.get_color_u32_rgba(0.2, 1.0, 0.3, 0.95 * alpha))
            draw_list.add_circle_filled(
                sx + logo_sz * 0.6, sy, 3.5,
                imgui.get_color_u32_rgba(1.0, 0.2, 0.2, 0.95 * alpha))
            if int(t * 3 + phase) % 2 == 0:
                draw_list.add_circle_filled(
                    sx, sy - logo_sz * 0.4, 2.5,
                    imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.95 * alpha))
            draw_list.add_image(
                splash.logo_texture,
                (sx - logo_sz / 2, sy - logo_sz / 2),
                (sx + logo_sz / 2, sy + logo_sz / 2))
    else:
        for i, (y, sp) in enumerate([(h * 0.25, 60),
                                     (h * 0.35, -45),
                                     (h * 0.45, 30)]):
            cx_c = (t * sp + i * 400) % (w + 200) - 100
            draw_list.add_circle_filled(
                cx_c, y, 3,
                imgui.get_color_u32_rgba(1.0, 0.8, 0.3, 0.9 * alpha))

    # Rain streaks
    for rx, ry, sp in state['rain']:
        scroll = (t * sp * 400) % h
        y0 = (ry * h + scroll) % h
        x0 = rx * w - y0 * 0.15
        draw_list.add_line(
            x0, y0, x0 - 8, y0 + 18,
            imgui.get_color_u32_rgba(0.8, 0.9, 1.0, 0.25 * alpha), 1.2)

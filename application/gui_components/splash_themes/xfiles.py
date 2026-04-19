"""UFO over a dark pine forest: starfield, the FunGen logo IS the saucer
disc (seen from slightly below) with a dome on top, pulsing rim lights,
eerie green abduction beam dropping to the forest, fog + TV scanlines."""

import math
import random

import imgui


def _ufo_pose(t, w, h):
    """Saucer center + size. Shared by place_logo and render_bg so the
    logo (disc body) lines up with the rim / dome / beam drawn around it."""
    ufo_cx = w * 0.55 + math.sin(t * 0.3) * 40
    ufo_cy = h * 0.28 + math.sin(t * 0.5) * 8
    ufo_size = min(w, h) * 0.26
    return ufo_cx, ufo_cy, ufo_size


def place_logo(splash, width, height, current_time):
    """Logo = the saucer's underside disc. The dome, rim, perimeter
    lights, and abduction beam are drawn around it by render_bg."""
    ufo_cx, ufo_cy, ufo_size = _ufo_pose(current_time, width, height)
    logo_size = ufo_size
    logo_x = ufo_cx - logo_size / 2
    logo_y = ufo_cy - logo_size / 2
    return logo_x, logo_y, logo_size, 0.0


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height
    t = current_time
    state = getattr(splash, '_xfiles_state', None)
    if state is None:
        rng = random.Random(1013)  # X-Files debut year: 1993
        stars = [(rng.uniform(0, 1), rng.uniform(0, 0.65),
                  rng.uniform(0.5, 1.2)) for _ in range(140)]
        trees = []
        x = 0
        while x < w:
            tw = rng.randint(22, 55)
            th_ = rng.randint(60, 150)
            trees.append((x, tw, th_))
            x += tw - 4
        state = {'stars': stars, 'trees': trees}
        splash._xfiles_state = state
    # Dark night sky
    for i in range(30):
        f = i / 30
        y0 = f * h
        y1 = (f + 1 / 30) * h + 1
        r_ = 0.02 + 0.02 * f
        g_ = 0.02 + 0.05 * f
        b_ = 0.06 + 0.08 * f
        draw_list.add_rect_filled(
            0, y0, w, y1, imgui.get_color_u32_rgba(r_, g_, b_, alpha))
    # Stars (with gentle twinkle)
    for sx, sy, sp in state['stars']:
        twi = 0.55 + 0.45 * math.sin(t * sp * 3.0 + sx * 50)
        draw_list.add_circle_filled(
            sx * w, sy * h, 1.2,
            imgui.get_color_u32_rgba(0.85, 0.90, 1.0,
                                      twi * 0.85 * alpha))
    # UFO — the FunGen logo is the disc (painted on top by the main loop).
    # Here we draw: a dark rim ellipse BEHIND the logo (so the saucer
    # silhouette reads against the sky), a small dome on top with a white
    # window highlight, a ring of pulsing perimeter lights around the
    # disc bottom, and the green abduction beam dropping from under it.
    ufo_cx, ufo_cy, ufo_size = _ufo_pose(t, w, h)
    ufo_w = ufo_size  # full width of the disc (= logo_size)
    ufo_r = ufo_w * 0.5
    # Rim shadow — a slightly wider dark disc so the UFO has an edge
    draw_list.add_circle_filled(
        ufo_cx, ufo_cy + ufo_r * 0.05, ufo_r * 1.04,
        imgui.get_color_u32_rgba(0.08, 0.09, 0.12, 0.85 * alpha), 40)
    # Hull ring just inside the rim (subtle metallic band)
    draw_list.add_circle(
        ufo_cx, ufo_cy, ufo_r * 1.00,
        imgui.get_color_u32_rgba(0.30, 0.33, 0.38, 0.75 * alpha),
        40, 3.0)

    # Dome on top of the disc
    dome_cx = ufo_cx
    dome_cy = ufo_cy - ufo_r * 0.20
    dome_r = ufo_r * 0.40
    draw_list.add_circle_filled(
        dome_cx, dome_cy - dome_r * 0.5, dome_r,
        imgui.get_color_u32_rgba(0.46, 0.52, 0.60, 0.92 * alpha), 24)
    # Cover the lower half of the dome with the rim so it reads as a
    # hemisphere resting on top of the disc.
    draw_list.add_rect_filled(
        dome_cx - dome_r - 2, dome_cy - 1,
        dome_cx + dome_r + 2, dome_cy + dome_r,
        imgui.get_color_u32_rgba(0.08, 0.09, 0.12, 0.0))  # placeholder
    # Window highlight on the dome
    draw_list.add_circle_filled(
        dome_cx - dome_r * 0.25, dome_cy - dome_r * 0.85,
        dome_r * 0.30,
        imgui.get_color_u32_rgba(0.85, 0.90, 1.0, 0.65 * alpha), 16)

    # Perimeter pulsing lights around the disc bottom edge
    n_lights = 5
    for i in range(n_lights):
        lcol = [(1.0, 0.25, 0.25),
                (1.0, 0.80, 0.20),
                (0.25, 1.0, 0.55),
                (0.25, 0.55, 1.0),
                (0.90, 0.35, 1.0)][i]
        ang_frac = (i + 0.5) / n_lights  # 0 .. 1 around lower arc
        # Spread across the bottom semicircle, angles from 0.15π to 0.85π
        ang = math.pi * (0.15 + 0.70 * ang_frac)
        lx = ufo_cx + math.cos(ang) * ufo_r * 0.92
        ly = ufo_cy + math.sin(ang) * ufo_r * 0.45 + ufo_r * 0.02
        lp = 0.55 + 0.45 * math.sin(t * 4 + i * 1.7)
        draw_list.add_circle_filled(
            lx, ly, 7 * lp,
            imgui.get_color_u32_rgba(*lcol, 0.95 * lp * alpha))
        draw_list.add_circle_filled(
            lx, ly, 22 * lp,
            imgui.get_color_u32_rgba(*lcol, 0.18 * lp * alpha))

    # Green abduction beam — origin at the disc's underside (bottom of logo)
    beam_bot_y = h * 0.95
    beam_top_y = ufo_cy + ufo_r * 0.35
    beam_top_hw = ufo_r * 0.30
    beam_bot_hw = ufo_r * 1.25
    for layer, aa in ((1.35, 0.05), (1.10, 0.10),
                       (0.85, 0.18), (0.60, 0.30)):
        draw_list.add_triangle_filled(
            ufo_cx - beam_top_hw * layer, beam_top_y,
            ufo_cx + beam_top_hw * layer, beam_top_y,
            ufo_cx + beam_bot_hw * layer, beam_bot_y,
            imgui.get_color_u32_rgba(0.40, 1.0, 0.55, aa * alpha))
        draw_list.add_triangle_filled(
            ufo_cx - beam_top_hw * layer, beam_top_y,
            ufo_cx - beam_bot_hw * layer, beam_bot_y,
            ufo_cx + beam_bot_hw * layer, beam_bot_y,
            imgui.get_color_u32_rgba(0.40, 1.0, 0.55, aa * alpha))

    # Pine forest silhouette
    ground_y = h * 0.82
    for (tx, tw, th_) in state['trees']:
        draw_list.add_rect_filled(
            tx + tw * 0.40, ground_y - 8,
            tx + tw * 0.60, ground_y + 4,
            imgui.get_color_u32_rgba(0.02, 0.02, 0.03, alpha))
        draw_list.add_triangle_filled(
            tx, ground_y,
            tx + tw, ground_y,
            tx + tw * 0.5, ground_y - th_ * 0.55,
            imgui.get_color_u32_rgba(0.02, 0.03, 0.02, alpha))
        draw_list.add_triangle_filled(
            tx + tw * 0.1, ground_y - th_ * 0.35,
            tx + tw * 0.9, ground_y - th_ * 0.35,
            tx + tw * 0.5, ground_y - th_,
            imgui.get_color_u32_rgba(0.02, 0.03, 0.02, alpha))
    draw_list.add_rect_filled(
        0, ground_y, w, h,
        imgui.get_color_u32_rgba(0.02, 0.02, 0.03, alpha))

    # Ground fog
    for i in range(8):
        fy = ground_y - i * 10 - 5
        fa = (1.0 - i / 8.0) * 0.15
        draw_list.add_rect_filled(
            0, fy, w, fy + 10,
            imgui.get_color_u32_rgba(0.45, 0.55, 0.55, fa * alpha))

    # Scanline overlay
    for ys in range(0, int(h), 4):
        draw_list.add_rect_filled(
            0, ys, w, ys + 1,
            imgui.get_color_u32_rgba(0, 0, 0, 0.10 * alpha))

    # TV static pulse
    stat_cycle = 5.0
    stat_t = (t % stat_cycle) / stat_cycle
    if stat_t < 0.08:
        flash_a = (1.0 - stat_t / 0.08) * 0.22
        draw_list.add_rect_filled(
            0, 0, w, h,
            imgui.get_color_u32_rgba(0.60, 0.65, 0.70, flash_a * alpha))

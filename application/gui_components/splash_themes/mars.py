"""Mars Attacks!: saucer fleet firing red death rays on a silhouetted city.
One of the saucers is the hero FunGen saucer — the logo is the disc body,
with dome, rim lights and death ray drawn around it."""

import math
import random

import imgui


def _hero_saucer_pose(t, w, h):
    """Hero FunGen saucer position + size. Hovers in the upper-center of
    the screen with a gentle bob and horizontal drift."""
    cx = w * 0.50 + math.sin(t * 0.75) * (w * 0.08)
    cy = h * 0.20 + math.sin(t * 1.6) * 18
    sz = min(w, h) * 0.19
    return cx, cy, sz


def place_logo(splash, width, height, current_time):
    """Logo = the hero saucer's disc body. Dome + rim + lights + beam are
    drawn around it by render_bg."""
    cx, cy, sz = _hero_saucer_pose(current_time, width, height)
    logo_size = sz
    logo_x = cx - logo_size / 2
    logo_y = cy - logo_size / 2
    return logo_x, logo_y, logo_size, 0.0


_SAUCER = (
    [
        ".....#######.....",
        "....#########....",
        "...##.#####.##...",
        "#################",
        "#.#.#.#.#.#.#.#.#",
        "#################",
        "..#############..",
    ],
    [
        ".....#######.....",
        "....#########....",
        "...#####.#####...",
        "#################",
        ".#.#.#.#.#.#.#.#.",
        "#################",
        "..#############..",
    ],
)

_MARTIAN = [
    "...#####...",
    "..#######..",
    ".#.#####.#.",
    ".#########.",
    "#.##...##.#",
    "###########",
    "##..###..##",
    ".####.####.",
    "...#####...",
]


def _blit(draw_list, sprite, x, y, sz, r, g, b, aa):
    col = imgui.get_color_u32_rgba(r, g, b, aa)
    for pr, row_str in enumerate(sprite):
        py = y + pr * sz
        for pc, ch in enumerate(row_str):
            if ch == '#':
                sx2 = x + pc * sz
                draw_list.add_rect_filled(sx2, py, sx2 + sz, py + sz, col)


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height

    if not hasattr(splash, '_mars_stars'):
        rng = random.Random(42)
        splash._mars_stars = [(rng.uniform(0, 1), rng.uniform(0, 0.7),
                               rng.uniform(0.3, 1.0), rng.uniform(0.5, 2.0))
                              for _ in range(80)]
        splash._mars_saucers = []
        y_bands = [0.08, 0.18, 0.12, 0.26]
        rng.shuffle(y_bands)
        for i in range(4):
            splash._mars_saucers.append({
                'x_frac': 0.14 + i * 0.23 + rng.uniform(-0.02, 0.02),
                'y_frac': y_bands[i] + rng.uniform(-0.025, 0.025),
                'bob_amp': rng.uniform(16.0, 28.0),
                'bob_speed': rng.uniform(1.6, 2.6),
                'drift_amp': rng.uniform(0.06, 0.10),
                'drift_speed': rng.uniform(0.7, 1.2),
                'fire_phase': rng.uniform(0.0, 6.0),
                'fire_period': rng.uniform(4.5, 6.5),
            })
        splash._mars_city = []
        x = 0.0
        while x < 1.0:
            bw = rng.uniform(0.05, 0.11)
            bh = rng.uniform(0.09, 0.21)
            splash._mars_city.append((x, bw, bh, rng.random()))
            x += bw
        splash._mars_moon = (rng.uniform(0.78, 0.92), rng.uniform(0.08, 0.16))

    px = max(3, int(min(w, h) / 300))

    # 1. Starfield
    for sx, sy, brightness, twinkle in splash._mars_stars:
        a = alpha * brightness * (0.5 + 0.5 * math.sin(current_time * twinkle + sx * 12))
        size = 1.0 + brightness * 0.9
        draw_list.add_circle_filled(sx * w, sy * h, size,
                                    imgui.get_color_u32_rgba(1.0, 1.0, 1.0, a))

    # 2. Crescent moon
    mx, my = splash._mars_moon
    mcx, mcy = mx * w, my * h
    moon_r = min(w, h) * 0.032
    draw_list.add_circle_filled(mcx, mcy, moon_r,
                                imgui.get_color_u32_rgba(1.0, 0.95, 0.78, alpha * 0.9))
    draw_list.add_circle_filled(mcx - moon_r * 0.38, mcy - moon_r * 0.12, moon_r * 0.94,
                                imgui.get_color_u32_rgba(0.01, 0.0, 0.04, 1.0))

    # 3. City skyline
    ground_y = h - 28
    for (bx_frac, bw_frac, bh_frac, seed) in splash._mars_city:
        bx = bx_frac * w
        bw = bw_frac * w
        bh = bh_frac * h
        by = ground_y - bh
        draw_list.add_rect_filled(bx, by, bx + bw, ground_y,
                                  imgui.get_color_u32_rgba(0.12, 0.08, 0.18, alpha))
        # Lit windows
        win_px = max(2, int(px * 0.9))
        step = win_px * 3
        rows_w = max(2, int(bh / step))
        cols_w = max(1, int((bw - 6) / step))
        for wr in range(rows_w):
            for wc in range(cols_w):
                on = ((int(seed * 1000) + wr * 7 + wc * 11) % 7) < 3
                flick = ((int(current_time * 2) + wr * 3 + wc) % 29) > 2
                if on and flick:
                    wx = bx + 4 + wc * step
                    wy = by + 4 + wr * step
                    if wx + win_px < bx + bw - 2 and wy + win_px < ground_y - 2:
                        draw_list.add_rect_filled(wx, wy, wx + win_px, wy + win_px,
                                                  imgui.get_color_u32_rgba(1.0, 0.65, 0.2, alpha * 0.85))
    draw_list.add_line(0, ground_y, w, ground_y,
                       imgui.get_color_u32_rgba(0.25, 0.15, 0.3, alpha), 2.0)

    saucer_px = max(6, int(px * 2.6))
    saucer_w_px = 17 * saucer_px
    saucer_h_px = 7 * saucer_px

    # 4. Saucer fleet
    for sau in splash._mars_saucers:
        drift = math.sin(current_time * sau['drift_speed']) * sau['drift_amp']
        bob = math.sin(current_time * sau['bob_speed']) * sau['bob_amp']
        sx = (sau['x_frac'] + drift) * w - saucer_w_px / 2
        sy = sau['y_frac'] * h + bob
        frame = int(current_time * 3 + sau['fire_phase']) % 2

        fire_t = (current_time + sau['fire_phase']) % sau['fire_period']
        firing = fire_t < 0.5
        charging = 0.5 <= fire_t < 1.3

        ray_x = sx + saucer_w_px / 2
        ray_y0 = sy + saucer_h_px - saucer_px
        ray_y1 = ground_y - 2

        if charging:
            prog = (fire_t - 0.5) / 0.8
            gr = 6 + prog * 20
            draw_list.add_circle_filled(ray_x, ray_y0, gr,
                                        imgui.get_color_u32_rgba(1.0, 0.3, 0.2, alpha * 0.4 * prog))
            draw_list.add_circle_filled(ray_x, ray_y0, gr * 0.45,
                                        imgui.get_color_u32_rgba(1.0, 0.9, 0.6, alpha * 0.75 * prog))
        if firing:
            flicker = 0.85 + 0.15 * math.sin(current_time * 42 + sau['fire_phase'] * 7)
            draw_list.add_line(ray_x, ray_y0, ray_x, ray_y1,
                               imgui.get_color_u32_rgba(1.0, 0.3, 0.25, alpha * 0.35 * flicker),
                               saucer_px * 5)
            draw_list.add_line(ray_x, ray_y0, ray_x, ray_y1,
                               imgui.get_color_u32_rgba(1.0, 0.5, 0.4, alpha * 0.75 * flicker),
                               saucer_px * 2.5)
            draw_list.add_line(ray_x, ray_y0, ray_x, ray_y1,
                               imgui.get_color_u32_rgba(1.0, 1.0, 0.95, alpha * 0.95 * flicker),
                               max(1.0, saucer_px * 0.7))
            fa = alpha * flicker
            draw_list.add_circle_filled(ray_x, ray_y1, saucer_px * 7,
                                        imgui.get_color_u32_rgba(1.0, 0.4, 0.2, fa * 0.28))
            draw_list.add_circle_filled(ray_x, ray_y1, saucer_px * 4,
                                        imgui.get_color_u32_rgba(1.0, 0.8, 0.5, fa * 0.55))
            draw_list.add_circle_filled(ray_x, ray_y1, saucer_px * 1.8,
                                        imgui.get_color_u32_rgba(1.0, 1.0, 0.9, fa * 0.9))

        _blit(draw_list, _SAUCER[frame], sx, sy, saucer_px, 0.72, 0.75, 0.82, alpha)
        dome_pulse = 0.55 + 0.30 * math.sin(current_time * 3.5 + sau['fire_phase'])
        dome_y1 = sy + saucer_px
        dome_x0 = sx + 4 * saucer_px
        dome_x1 = sx + 13 * saucer_px
        draw_list.add_rect_filled(dome_x0, dome_y1,
                                  dome_x1, dome_y1 + 2 * saucer_px,
                                  imgui.get_color_u32_rgba(0.35, 1.0, 0.45,
                                                           alpha * 0.55 * dome_pulse))

    # 5. Lone martian peeking above the skyline (original, small)
    mart_px = max(3, int(px * 1.3))
    mart_x = 0.14 * w
    mart_h_px = 9 * mart_px
    mart_y = ground_y - mart_h_px - 4
    mart_bob = abs(math.sin(current_time * 1.2)) * mart_px * 0.7
    _blit(draw_list, _MARTIAN, mart_x, mart_y - mart_bob, mart_px,
          0.4, 1.0, 0.35, alpha)
    dome_cx_m = mart_x + 5.5 * mart_px
    dome_cy_m = mart_y + 4.5 * mart_px - mart_bob
    dome_r_m = 7 * mart_px
    draw_list.add_circle(
        dome_cx_m, dome_cy_m, dome_r_m,
        imgui.get_color_u32_rgba(0.7, 0.9, 1.0, alpha * 0.55), thickness=2.0)
    draw_list.add_circle(
        dome_cx_m, dome_cy_m, dome_r_m,
        imgui.get_color_u32_rgba(0.4, 0.7, 0.9, alpha * 0.18), thickness=6.0)

    # 6. Hero FunGen saucer — logo is the disc body. Dome on top, dark
    # rim behind (so the silhouette shows through the logo's transparent
    # margins), pulsing perimeter lights, and its own death ray cycle.
    hx, hy, hsz = _hero_saucer_pose(current_time, w, h)
    hr = hsz * 0.5
    # Rim shadow behind the logo
    draw_list.add_circle_filled(
        hx, hy + hr * 0.05, hr * 1.06,
        imgui.get_color_u32_rgba(0.08, 0.10, 0.13, 0.82 * alpha), 48)
    # Metallic hull band
    draw_list.add_circle(
        hx, hy, hr, imgui.get_color_u32_rgba(0.55, 0.60, 0.70,
                                              0.75 * alpha), 48, 3.0)
    # Green dome (same martian green as the fleet's domes) on top
    dome_cx_h = hx
    dome_cy_h = hy - hr * 0.18
    dome_r_h = hr * 0.40
    dome_pulse = 0.55 + 0.30 * math.sin(current_time * 3.5)
    draw_list.add_circle_filled(
        dome_cx_h, dome_cy_h - dome_r_h * 0.5, dome_r_h,
        imgui.get_color_u32_rgba(0.35, 1.0, 0.45,
                                  0.70 * alpha * dome_pulse), 24)
    draw_list.add_circle(
        dome_cx_h, dome_cy_h - dome_r_h * 0.5, dome_r_h,
        imgui.get_color_u32_rgba(0.85, 1.0, 0.90, 0.85 * alpha),
        24, 2.0)
    # Pulsing perimeter lights around the lower arc
    n_lights = 5
    for i in range(n_lights):
        lcol = [(1.0, 0.25, 0.25), (1.0, 0.82, 0.20),
                (0.25, 1.0, 0.55), (0.25, 0.55, 1.0),
                (0.90, 0.35, 1.0)][i]
        ang_frac = (i + 0.5) / n_lights
        ang = math.pi * (0.15 + 0.70 * ang_frac)
        lx = hx + math.cos(ang) * hr * 0.92
        ly = hy + math.sin(ang) * hr * 0.45 + hr * 0.02
        lp = 0.55 + 0.45 * math.sin(current_time * 4 + i * 1.7)
        draw_list.add_circle_filled(
            lx, ly, 7 * lp,
            imgui.get_color_u32_rgba(*lcol, 0.95 * lp * alpha))
        draw_list.add_circle_filled(
            lx, ly, 22 * lp,
            imgui.get_color_u32_rgba(*lcol, 0.18 * lp * alpha))
    # Death ray — fires on a 5s cycle with the same style as the fleet
    hero_fire_t = (current_time + 2.5) % 5.0
    if hero_fire_t < 0.6:
        ray_x = hx
        ray_y0 = hy + hr * 0.35
        ray_y1 = ground_y - 2
        flicker = 0.85 + 0.15 * math.sin(current_time * 42)
        fa = alpha * flicker
        draw_list.add_line(ray_x, ray_y0, ray_x, ray_y1,
                            imgui.get_color_u32_rgba(1.0, 0.30, 0.25,
                                                      0.35 * fa),
                            28.0)
        draw_list.add_line(ray_x, ray_y0, ray_x, ray_y1,
                            imgui.get_color_u32_rgba(1.0, 0.50, 0.40,
                                                      0.75 * fa),
                            14.0)
        draw_list.add_line(ray_x, ray_y0, ray_x, ray_y1,
                            imgui.get_color_u32_rgba(1.0, 1.0, 0.95,
                                                      0.95 * fa),
                            5.0)
        draw_list.add_circle_filled(
            ray_x, ray_y1, 55,
            imgui.get_color_u32_rgba(1.0, 0.4, 0.2, fa * 0.28))
        draw_list.add_circle_filled(
            ray_x, ray_y1, 30,
            imgui.get_color_u32_rgba(1.0, 0.8, 0.5, fa * 0.55))
        draw_list.add_circle_filled(
            ray_x, ray_y1, 14,
            imgui.get_color_u32_rgba(1.0, 1.0, 0.9, fa * 0.9))

    # 6. "ACK! ACK!" speech bubble rotating through saucers
    ack_period = 2.6
    ack_phase = current_time % ack_period
    if ack_phase < 1.3 and splash._mars_saucers:
        idx = int(current_time / ack_period) % len(splash._mars_saucers)
        sau = splash._mars_saucers[idx]
        drift = math.sin(current_time * sau['drift_speed']) * sau['drift_amp']
        bob = math.sin(current_time * sau['bob_speed']) * sau['bob_amp']
        sx = (sau['x_frac'] + drift) * w - saucer_w_px / 2
        sy = sau['y_frac'] * h + bob
        fade = 1.0 - ack_phase / 1.3
        fade = fade * fade
        ack_a = alpha * fade
        pop = min(1.0, ack_phase / 0.15)
        pop = pop * (2.0 - pop)
        imgui.set_window_font_scale(3.2 * (0.6 + 0.4 * pop))
        text = "ACK! ACK!"
        ts = imgui.calc_text_size(text)
        bx = sx + saucer_w_px + 14
        by = sy - ts[1] * 0.4
        pad = int(ts[1] * 0.35)
        if bx + ts[0] + pad > w - 14:
            bx = sx - ts[0] - pad * 2 - 14
        draw_list.add_rect_filled(bx - pad, by - pad,
                                  bx + ts[0] + pad, by + ts[1] + pad,
                                  imgui.get_color_u32_rgba(0.98, 0.98, 1.0, ack_a * 0.94),
                                  8.0)
        draw_list.add_rect(bx - pad, by - pad,
                           bx + ts[0] + pad, by + ts[1] + pad,
                           imgui.get_color_u32_rgba(0.1, 0.1, 0.15, ack_a),
                           8.0, thickness=2.5)
        tail_x = bx - pad if bx > sx else bx + ts[0] + pad
        tail_sign = -1 if tail_x == bx - pad else 1
        draw_list.add_triangle_filled(
            tail_x, by + ts[1] * 0.3,
            tail_x + tail_sign * 18, by + ts[1] * 0.55,
            tail_x, by + ts[1] * 0.8,
            imgui.get_color_u32_rgba(0.98, 0.98, 1.0, ack_a * 0.94))
        draw_list.add_text(bx, by,
                           imgui.get_color_u32_rgba(0.9, 0.1, 0.2, ack_a), text)
        imgui.set_window_font_scale(1.0)

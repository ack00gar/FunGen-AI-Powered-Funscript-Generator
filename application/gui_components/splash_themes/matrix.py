"""Matrix-style falling character columns, with the FunGen logo acting as
the source — a green halo pulses behind it and four brighter 'hero' code
streams cascade downward from its underside."""

import math
import random

import imgui


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    if not hasattr(splash, '_matrix_columns'):
        n_cols = max(8, int(window_width / 28))
        splash._matrix_columns = []
        chars = "01{}[]<>|/\\=+-*&%$#@!?abcdefghijklmnopqrstuvwxyz"
        for i in range(n_cols):
            x = i * (window_width / n_cols) + random.uniform(0, 12)
            speed = random.uniform(80, 220)
            offset = random.uniform(0, window_height * 2)
            trail_len = random.randint(8, 25)
            col_chars = [random.choice(chars) for _ in range(trail_len)]
            splash._matrix_columns.append((x, speed, offset, col_chars))

    char_h = 22
    imgui.set_window_font_scale(1.3)
    for x, speed, offset, col_chars in splash._matrix_columns:
        total_h = len(col_chars) * char_h
        head_y = ((current_time * speed + offset) % (window_height + total_h)) - total_h
        for ci, ch in enumerate(col_chars):
            cy = head_y + ci * char_h
            if cy < -char_h or cy > window_height:
                continue
            # Head character is bright white-green, trail fades to dark green
            t = ci / max(1, len(col_chars) - 1)
            if ci == 0:
                r, g, b = 0.7, 1.0, 0.7
                a = alpha
            else:
                brightness = 1.0 - t * 0.9
                r = 0.0
                g = 0.85 * brightness
                b = 0.15 * brightness
                a = alpha * brightness
            if a < 0.03:
                continue
            draw_list.add_text(x, cy, imgui.get_color_u32_rgba(r, g, b, a), ch)

    # --- Logo-as-source decoration ---
    # Track the default-placed logo's animated center so the halo + hero
    # streams stay locked to the floating logo. 250 and the sin float
    # offset come from splash_screen.py's default placement branch.
    logo_size = 250
    cx = window_width / 2
    cy = window_height / 2 + math.sin(current_time * 2.0) * 8.0

    # Pulsing green halo behind the logo — three stacked discs, outermost
    # faintest. Softly modulated so the whole halo breathes.
    pulse = 0.80 + 0.20 * math.sin(current_time * 1.8)
    for rr, aa in ((logo_size * 0.95, 0.05),
                   (logo_size * 0.72, 0.11),
                   (logo_size * 0.54, 0.20),
                   (logo_size * 0.40, 0.30)):
        draw_list.add_circle_filled(
            cx, cy, rr * pulse,
            imgui.get_color_u32_rgba(0.0, 0.95, 0.22,
                                      aa * pulse * alpha))

    # Four hero code streams pouring out from the logo's underside —
    # brighter than the ambient rain, with a white head character.
    chars_rain = "01{}[]<>|/\\=+-*&%$#@!?"
    if not hasattr(splash, '_matrix_hero_cols'):
        splash._matrix_hero_cols = []
        for offs in (-70, -25, 25, 70):
            splash._matrix_hero_cols.append(
                (offs, [random.choice(chars_rain) for _ in range(28)],
                 random.uniform(0.0, 100.0)))
    stream_top_y = cy + logo_size * 0.48
    loop_h = window_height - stream_top_y + 28 * char_h
    for (offs, col_chars, phase) in splash._matrix_hero_cols:
        hx = cx + offs
        head_y = stream_top_y + ((current_time * 190 + phase) % loop_h)
        for ci, ch in enumerate(col_chars):
            ccy = head_y - ci * char_h
            if ccy < stream_top_y - char_h or ccy > window_height:
                continue
            t_frac = ci / max(1, len(col_chars) - 1)
            if ci == 0:
                r, g, b = 0.95, 1.0, 0.95
                a = 1.0
            else:
                brightness = (1.0 - t_frac) ** 1.3
                r = 0.25 * brightness
                g = 1.0 * brightness
                b = 0.40 * brightness
                a = brightness
            if a < 0.05:
                continue
            draw_list.add_text(
                hx, ccy,
                imgui.get_color_u32_rgba(r, g, b, a * alpha), ch)

    imgui.set_window_font_scale(1.0)

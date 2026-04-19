"""Star Wars background starfield (foreground lightsabers kept in splash_screen.py)."""

import math
import random

import imgui


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height
    if not hasattr(splash, '_sw_stars'):
        splash._sw_stars = [(random.uniform(0, 1), random.uniform(0, 1),
                             random.uniform(0.3, 1.0), random.uniform(0.5, 3.0))
                            for _ in range(150)]
    for sx, sy, brightness, twinkle_speed in splash._sw_stars:
        twinkle = 0.4 + 0.6 * (0.5 + 0.5 * math.sin(current_time * twinkle_speed + sx * 20))
        a = alpha * brightness * twinkle
        size = 1.0 + brightness * 1.5
        draw_list.add_circle_filled(sx * w, sy * h, size,
                                    imgui.get_color_u32_rgba(1.0, 1.0, 1.0, a))

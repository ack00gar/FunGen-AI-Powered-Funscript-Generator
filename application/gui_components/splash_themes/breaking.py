"""Breaking Bad: big pulsing periodic-table element boxes + a simple pulsing
chemical formula at the bottom."""

import math
import random

import imgui


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height

    if not hasattr(splash, '_bb_positions'):
        # Constrain range so the 110px boxes fully fit on-screen
        splash._bb_positions = [(random.uniform(0.03, 0.82), random.uniform(0.03, 0.70))
                                for _ in range(14)]

    elements = [
        ("Br", "35"), ("Ba", "56"), ("H", "1"), ("He", "2"),
        ("C", "6"), ("N", "7"), ("O", "8"), ("F", "9"),
        ("Na", "11"), ("Cl", "17"), ("K", "19"), ("Fe", "26"),
        ("Cu", "29"), ("Zn", "30"),
    ]

    # Big periodic-table element boxes
    box_size = 110
    for i, ((sym, num), (px, py)) in enumerate(zip(elements, splash._bb_positions)):
        bx = px * w
        by = py * h
        pulse = 0.3 + 0.3 * math.sin(current_time * 1.5 + i * 0.8)
        box_a = alpha * pulse
        draw_list.add_rect_filled(
            bx, by, bx + box_size, by + box_size,
            imgui.get_color_u32_rgba(0.05, 0.15, 0.05, box_a * 0.6), 6.0)
        draw_list.add_rect(
            bx, by, bx + box_size, by + box_size,
            imgui.get_color_u32_rgba(0.1, 0.7, 0.2, box_a * 0.85),
            6.0, thickness=2.5)
        # Atomic number (top-left, small)
        imgui.set_window_font_scale(1.4)
        draw_list.add_text(
            bx + 8, by + 6,
            imgui.get_color_u32_rgba(0.3, 0.8, 0.3, box_a * 0.7), num)
        # Element symbol (large, centered)
        imgui.set_window_font_scale(3.8)
        sym_size = imgui.calc_text_size(sym)
        draw_list.add_text(
            bx + (box_size - sym_size[0]) / 2, by + 28,
            imgui.get_color_u32_rgba(0.1, 0.9, 0.2, box_a), sym)
        imgui.set_window_font_scale(1.0)

    # Chemical formula at the bottom (simple sine-pulse)
    imgui.set_window_font_scale(1.5)
    formula = "C10H15N  +  AI  =  FunScript"
    f_size = imgui.calc_text_size(formula)
    f_a = alpha * 0.25 * (0.5 + 0.5 * math.sin(current_time * 2))
    draw_list.add_text((w - f_size[0]) / 2, h - 60,
                       imgui.get_color_u32_rgba(0.1, 0.8, 0.2, f_a), formula)
    imgui.set_window_font_scale(1.0)

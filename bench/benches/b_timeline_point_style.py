"""Measure imgui add_rect_filled (quad) vs add_circle_filled (tessellated).

Creates an offscreen imgui context, builds a DrawList and fires N
primitives of each type, timing wall-clock. Captures whether the
cffi boundary cost or the tessellation cost dominates.
"""
from __future__ import annotations

import time

import imgui


def _measure(n: int):
    imgui.create_context()
    io = imgui.get_io()
    io.display_size = (1920, 1080)
    io.fonts.get_tex_data_as_rgba32()

    imgui.new_frame()
    imgui.begin("bench", flags=imgui.WINDOW_NO_DECORATION)
    dl = imgui.get_window_draw_list()

    col = imgui.get_color_u32_rgba(1, 0, 0, 1)

    # Circle filled.
    t0 = time.perf_counter()
    for i in range(n):
        x = 10.0 + (i % 1900)
        y = 10.0 + ((i // 1900) % 1000)
        dl.add_circle_filled(x, y, 3.5, col)
    t_circle = time.perf_counter() - t0

    # Rect filled.
    t0 = time.perf_counter()
    for i in range(n):
        x = 10.0 + (i % 1900)
        y = 10.0 + ((i // 1900) % 1000)
        dl.add_rect_filled(x - 3.5, y - 3.5, x + 3.5, y + 3.5, col)
    t_rect = time.perf_counter() - t0

    # Ngon filled at n=4 (diamond).
    t0 = time.perf_counter()
    for i in range(n):
        x = 10.0 + (i % 1900)
        y = 10.0 + ((i // 1900) % 1000)
        dl.add_ngon_filled(x, y, 3.5, col, 4)
    t_ngon4 = time.perf_counter() - t0

    imgui.end()
    imgui.end_frame()

    return {
        "n": n,
        "add_circle_filled_ms": t_circle * 1000.0,
        "add_rect_filled_ms": t_rect * 1000.0,
        "add_ngon_filled_4_ms": t_ngon4 * 1000.0,
        "per_call_circle_us": t_circle * 1e6 / n,
        "per_call_rect_us": t_rect * 1e6 / n,
        "per_call_ngon4_us": t_ngon4 * 1e6 / n,
    }


def main():
    for n in (1000, 5000, 15000, 30000):
        r = _measure(n)
        print(f"n={r['n']:6d}  "
              f"circle {r['add_circle_filled_ms']:7.2f}ms ({r['per_call_circle_us']:.2f} us/call)  "
              f"rect {r['add_rect_filled_ms']:7.2f}ms ({r['per_call_rect_us']:.2f} us/call)  "
              f"ngon4 {r['add_ngon_filled_4_ms']:7.2f}ms ({r['per_call_ngon4_us']:.2f} us/call)")


if __name__ == "__main__":
    main()

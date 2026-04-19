"""Windows 98 Blue Screen of Death with properly centered text block and
ScanDisk progress bar."""

import imgui


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    from config.constants import APP_VERSION
    w, h = window_width, window_height
    t = current_time

    draw_list.add_rect_filled(
        0, 0, w, h, imgui.get_color_u32_rgba(0.0, 0.04, 0.67, alpha))

    block_w = min(w * 0.72, 920)
    block_x = (w - block_w) / 2
    # Vertically center the text block between the top of the screen
    # and the ScanDisk bar at h - 90. Title bar (44) + 13 body lines * 30
    # + cursor line (42) ~ 476 px total.
    content_h = 44 + 13 * 30 + 42
    available_h = (h - 90) - 0
    block_y = max(20, (available_h - content_h) / 2)

    # Title bar (gray, centered)
    bar_h = 44
    draw_list.add_rect_filled(
        block_x, block_y, block_x + block_w, block_y + bar_h,
        imgui.get_color_u32_rgba(0.85, 0.85, 0.85, alpha))
    imgui.set_window_font_scale(2.2)
    draw_list.add_text(
        block_x + 20, block_y + 6,
        imgui.get_color_u32_rgba(0.0, 0.04, 0.67, alpha), "FunGen")
    imgui.set_window_font_scale(1.0)

    # Body text
    imgui.set_window_font_scale(1.8)
    tx = block_x + 20
    ty = block_y + bar_h + 28
    line_h = 30
    white = imgui.get_color_u32_rgba(1, 1, 1, alpha)
    body = [
        f"FunGen v{APP_VERSION}",
        "",
        "A fatal amount of fun has occurred at 0069:0xDEADBEEF",
        "in FUNGEN.EXE:VR_SHUTTER_MODULE.",
        "",
        "The current script will be terminated.",
        "",
        "*  Press any key to terminate the current script.",
        "*  Press CTRL+ALT+DEL to restart your computer.",
        "   You will lose any unsaved information in all",
        "   scripts.",
        "",
        "Support the project: paypal.me/k00gar",
    ]
    for ln in body:
        draw_list.add_text(tx, ty, white, ln)
        ty += line_h

    # Blinking cursor
    ty += 12
    cursor_char = "_" if int(t * 2) % 2 == 0 else " "
    draw_list.add_text(tx, ty, white,
                       f"Press any key to continue {cursor_char}")
    imgui.set_window_font_scale(1.0)

    # ScanDisk bar (centered, matched to block width)
    sd_w = min(w * 0.50, block_w)
    sd_x = (w - sd_w) / 2
    sd_y = h - 90
    draw_list.add_rect(
        sd_x, sd_y, sd_x + sd_w, sd_y + 36,
        white, 0, thickness=2)
    prog = (t * 0.12) % 1.0
    seg_w = 12
    n_seg = max(1, int(sd_w / (seg_w + 2)))
    fill = int(n_seg * prog)
    for i in range(fill):
        x0 = sd_x + 3 + i * (seg_w + 2)
        draw_list.add_rect_filled(
            x0, sd_y + 4, x0 + seg_w, sd_y + 32,
            imgui.get_color_u32_rgba(0.3, 0.85, 1.0, alpha))
    imgui.set_window_font_scale(1.5)
    draw_list.add_text(
        sd_x, sd_y - 26, white,
        f"Checking filesystem... {int(prog * 100):3d}%")
    imgui.set_window_font_scale(1.0)

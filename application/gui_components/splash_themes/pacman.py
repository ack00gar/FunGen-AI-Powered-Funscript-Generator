"""Pac-Man chasing pellets around the screen perimeter, 4 ghosts trailing.
The FunGen logo IS Pac-Man: a yellow backdrop disc sits behind the logo so
the Pac-Man silhouette reads through the logo's transparent margins, and a
rotating black mouth wedge is drawn over both in render_fg."""

import math

import imgui

# World scale — matches mario / sonic / ET. Everything in the arcade scene
# (pellets, Pac-Man, ghosts, perimeter margin) is multiplied by this.
S = 1.8

PAC_R = int(38 * S)      # Pac-Man body radius
GHOST_R = int(32 * S)    # Ghost body radius
PELLET_R = 4.0 * S
PELLET_SPACING = 28 * S
MARGIN = 80 * S
PAC_SPEED = 260 * S      # px/sec along the perimeter
GHOST_GAP = 110 * S      # perimeter spacing between Pac-Man and each ghost


def _perim_fn(w, h):
    """Closure returning (point_on_perim, perim_length). Kept close to the
    original implementation but pulled out so place_logo and render_bg can
    both project onto the same perimeter path."""
    side_w = w - 2 * MARGIN
    side_h = h - 2 * MARGIN
    perim = 2 * (side_w + side_h)

    def on_perim(s):
        s = s % perim
        if s < side_w:
            return (MARGIN + s, MARGIN)
        s -= side_w
        if s < side_h:
            return (w - MARGIN, MARGIN + s)
        s -= side_h
        if s < side_w:
            return (w - MARGIN - s, h - MARGIN)
        s -= side_w
        return (MARGIN, h - MARGIN - s)

    return on_perim, perim


def _pacman_pose(t, w, h):
    """Current Pac-Man position + facing angle + mouth opening. Shared by
    place_logo (runs before render_bg) and render_bg so the yellow body,
    the logo on top of it, and the mouth wedge all line up."""
    on_perim, perim = _perim_fn(w, h)
    pac_s = (t * PAC_SPEED) % perim
    nx, ny = on_perim(pac_s)
    fx, fy = on_perim(pac_s + 10)
    ang = math.atan2(fy - ny, fx - nx)
    mouth = (0.5 + 0.5 * math.sin(t * 18)) * 0.9
    return nx, ny, ang, mouth, pac_s, on_perim, perim


def place_logo(splash, width, height, current_time):
    """Logo IS Pac-Man's face. Travels the perimeter with him; stays upright
    (no rotation) so the FunGen mark is always legible. The yellow body
    backdrop + rotating mouth wedge are drawn by render_bg / render_fg."""
    nx, ny, _ang, _mouth, _s, _fn, _pm = _pacman_pose(
        current_time, width, height)
    logo_size = PAC_R * 2.0
    logo_x = nx - logo_size / 2
    logo_y = ny - logo_size / 2
    return logo_x, logo_y, logo_size, 0.0


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height
    t = current_time

    draw_list.add_rect_filled(
        0, 0, w, h, imgui.get_color_u32_rgba(0, 0, 0, alpha))

    nx, ny, ang, mouth, pac_s, on_perim, perim = _pacman_pose(t, w, h)

    # Pellets on the perimeter, eaten out from under Pac-Man as he passes.
    n_pellets = int(perim / PELLET_SPACING)
    for i in range(n_pellets):
        ps = i * PELLET_SPACING
        rel = (ps - pac_s) % perim
        if rel < PELLET_SPACING * 1.2:
            continue
        x, y = on_perim(ps)
        draw_list.add_circle_filled(
            x, y, PELLET_R,
            imgui.get_color_u32_rgba(1.0, 0.9, 0.6, alpha))

    # Yellow Pac-Man body — drawn BEHIND the logo so the classic yellow
    # silhouette shows through the logo's transparent margins. The logo
    # texture is painted by the main splash loop right after render_bg.
    pac_col = imgui.get_color_u32_rgba(1.0, 0.95, 0.1, alpha)
    draw_list.add_circle_filled(nx, ny, PAC_R, pac_col, 40)
    # Subtle outer ring for definition against the black arcade sky
    draw_list.add_circle(
        nx, ny, PAC_R,
        imgui.get_color_u32_rgba(0.55, 0.50, 0.05, 0.8 * alpha),
        40, 2.5)

    # Ghosts trailing at spaced intervals behind Pac-Man
    ghost_cols = [
        (1.0, 0.2, 0.2),    # Blinky
        (1.0, 0.6, 0.85),   # Pinky
        (0.2, 0.9, 1.0),    # Inky
        (1.0, 0.7, 0.2),    # Clyde
    ]
    for i, gc in enumerate(ghost_cols):
        gs = (pac_s - (i + 1) * GHOST_GAP) % perim
        gx, gy = on_perim(gs)
        draw_list.add_circle_filled(
            gx, gy, GHOST_R, imgui.get_color_u32_rgba(*gc, alpha))
        draw_list.add_rect_filled(
            gx - GHOST_R, gy, gx + GHOST_R, gy + GHOST_R,
            imgui.get_color_u32_rgba(*gc, alpha))
        # Wavy skirt: 4 triangular bumps along the bottom
        bump_w = (GHOST_R * 2) / 4
        for k in range(4):
            wx = gx - GHOST_R + k * bump_w
            draw_list.add_triangle_filled(
                wx, gy + GHOST_R,
                wx + bump_w * 0.5, gy + GHOST_R * 0.65,
                wx + bump_w, gy + GHOST_R,
                imgui.get_color_u32_rgba(0, 0, 0, alpha))
        # Eyes (white + blue pupil), scaled with ghost size
        eye_off = GHOST_R * 0.35
        eye_r = GHOST_R * 0.22
        pup_r = GHOST_R * 0.11
        for ex in (gx - eye_off, gx + eye_off):
            draw_list.add_circle_filled(
                ex, gy - GHOST_R * 0.12, eye_r,
                imgui.get_color_u32_rgba(1, 1, 1, alpha))
            draw_list.add_circle_filled(
                ex + eye_r * 0.25, gy - GHOST_R * 0.05, pup_r,
                imgui.get_color_u32_rgba(0.1, 0.1, 1.0, alpha))


def render_fg(splash, logo_x, logo_y, logo_size, laser_time):
    """Pac-Man's mouth wedge — drawn OVER the logo so the logo appears to
    be chomping. A black triangle from Pac-Man's center outward along the
    direction of travel, with the jaw opening and closing at t*18 rad/s."""
    # Reconstruct pose from the splash window's dims + the current time.
    # laser_time is "current_time - 0.3" per the dispatcher signature, so
    # rebuild the same value we'd compute inside _pacman_pose.
    import glfw
    try:
        win = splash.window
        if win is None:
            return
        w, h = glfw.get_window_size(win)
    except Exception:
        return
    import time as _time
    t = _time.time() - getattr(splash, '_splash_start_time', _time.time())
    nx, ny, ang, mouth, _s, _fn, _pm = _pacman_pose(t, w, h)
    if mouth <= 0.05:
        return
    draw_list = imgui.get_window_draw_list()
    a1 = ang - mouth
    a2 = ang + mouth
    reach = PAC_R * 1.5
    p1x = nx + math.cos(a1) * reach
    p1y = ny + math.sin(a1) * reach
    p2x = nx + math.cos(a2) * reach
    p2y = ny + math.sin(a2) * reach
    draw_list.add_triangle_filled(
        nx, ny, p1x, p1y, p2x, p2y,
        imgui.get_color_u32_rgba(0, 0, 0, 1.0))

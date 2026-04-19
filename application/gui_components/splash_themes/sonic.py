"""Green Hill Zone: sky gradient, checkerboard hill silhouettes, background
loop-de-loop, spinning gold rings, Sonic-blue dash trail with Tails and
Eggman trailing, RINGS counter HUD."""

import math

import imgui


# World scale — matches mario theme. Bumps pipes/rings/Tails/Eggman/ball up
# to chunky NES-scale so the spin-ball doesn't read as a small badge on a
# huge canvas.
S = 1.6

SONIC_CYCLE = 5.0  # seconds per full traversal (dash → loop → dash)


def _loop_geom(w, h):
    """Loop-de-loop center + radius. Shared between the background
    visualization and the actual trajectory so Sonic physically rolls
    around the circle you see on screen."""
    loop_cx = w * 0.55
    loop_cy = h * 0.48
    loop_r = min(w, h) * 0.22
    return loop_cx, loop_cy, loop_r


def _sonic_pose(t, w, h):
    """Three-phase trajectory: dash-in, full revolution around the
    loop-de-loop, dash-out. Returns (sx, sy, sr, is_looping) so callers
    (render_bg) can switch motion-trail direction during the loop."""
    horizon_y = h * 0.62
    sph = (t % SONIC_CYCLE) / SONIC_CYCLE
    sr = 85  # was 52 — scaled 1.6x so the ball fits the rest of the world
    loop_cx, loop_cy, loop_r = _loop_geom(w, h)

    # Phase boundaries
    P1 = 0.38   # dash-in ends
    P2 = 0.78   # loop ends

    if sph < P1:
        # Dash from off-screen left to the BOTTOM of the loop.
        p = sph / P1
        start_x = -100 * S
        end_x = loop_cx
        end_y = loop_cy + loop_r
        ground_y = horizon_y - 48 * S + math.sin(p * math.pi * 4) * 22 * S
        sx = start_x + p * (end_x - start_x)
        # Smoothly ramp from bouncing ground height up to the loop bottom.
        blend = p * p  # ease-in
        sy = ground_y * (1 - blend) + end_y * blend
        is_looping = False
    elif sph < P2:
        # One full revolution around the loop, clockwise from the bottom.
        # Parametric form (screen coords, y grows downward):
        #   sx = cx + r * sin(theta)
        #   sy = cy + r * cos(theta)
        # θ=0 → bottom; θ=π/2 → right; θ=π → top; θ=3π/2 → left.
        p = (sph - P1) / (P2 - P1)
        theta = p * 2 * math.pi
        sx = loop_cx + loop_r * math.sin(theta)
        sy = loop_cy + loop_r * math.cos(theta)
        is_looping = True
    else:
        # Dash from the loop bottom off-screen to the right.
        p = (sph - P2) / (1.0 - P2)
        start_x = loop_cx
        start_y = loop_cy + loop_r
        end_x = w + 100 * S
        ground_y = horizon_y - 48 * S + math.sin((1 + p) * math.pi * 4) * 22 * S
        sx = start_x + p * (end_x - start_x)
        # Ease out of loop bottom back to ground bouncing.
        blend = p * p
        sy = start_y * (1 - blend) + ground_y * blend
        is_looping = False
    return sx, sy, sr, is_looping


def place_logo(splash, width, height, current_time):
    """Logo rides inside Sonic's spin-ball. Upright (not rotated) so the
    FunGen mark stays legible; the blue ball around it carries the spin
    motion via quill streaks in render_bg."""
    sx, sy, sr, _looping = _sonic_pose(current_time, width, height)
    logo_size = sr * 1.55
    logo_x = sx - logo_size / 2
    logo_y = sy - logo_size / 2
    return logo_x, logo_y, logo_size, 0.0


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height
    t = current_time
    # Sky gradient (Sonic blue)
    for i in range(30):
        f = i / 30
        y0 = f * h
        y1 = (f + 1 / 30) * h + 1
        r_ = 0.30 - 0.15 * f
        g_ = 0.65 - 0.10 * f
        b_ = 1.00 - 0.20 * f
        draw_list.add_rect_filled(
            0, y0, w, y1, imgui.get_color_u32_rgba(r_, g_, b_, alpha))
    # Clouds (white blobs drifting) — scaled to match the world
    for i in range(5):
        cx_c = ((i * 0.23 + t * 0.015) % 1.2) * w - w * 0.1
        cy_c = h * (0.12 + 0.06 * (i % 2))
        for offset, rad in ((-25 * S, 30 * S), (0, 40 * S),
                            (25 * S, 30 * S), (-50 * S, 22 * S),
                            (50 * S, 22 * S)):
            draw_list.add_circle_filled(
                cx_c + offset, cy_c, rad,
                imgui.get_color_u32_rgba(1, 1, 1, 0.78 * alpha))
    # Background loop-de-loop visualization — same geometry as the
    # trajectory in _sonic_pose, so the ring you see is the ring he rolls.
    loop_cx, loop_cy, loop_r = _loop_geom(w, h)
    for thick in (18 * S, 12 * S, 6 * S):
        draw_list.add_circle(
            loop_cx, loop_cy, loop_r,
            imgui.get_color_u32_rgba(0.1, 0.55, 0.85,
                                      0.30 * alpha), 64, thick)
    # Inner rail so the loop track reads as solid
    draw_list.add_circle(
        loop_cx, loop_cy, loop_r,
        imgui.get_color_u32_rgba(0.0, 0.15, 0.30, 0.35 * alpha),
        64, 2.5 * S)
    # Checker ground pattern (striped hills bottom)
    horizon_y = h * 0.62
    hills_color_base = (0.15, 0.58, 0.18)
    hills_color_dark = (0.09, 0.40, 0.12)
    num_scallops = 6
    for i in range(num_scallops + 1):
        cx_sc = (i - 0.5) * (w / num_scallops)
        draw_list.add_circle_filled(
            cx_sc, horizon_y, w * 0.12,
            imgui.get_color_u32_rgba(*hills_color_base, alpha))
    draw_list.add_rect_filled(
        0, horizon_y, w, h,
        imgui.get_color_u32_rgba(*hills_color_base, alpha))
    # Checker dirt band below horizon
    stripe_h = 18 * S
    checker_top = horizon_y + w * 0.10
    for row in range(int((h - checker_top) / stripe_h) + 1):
        y0 = checker_top + row * stripe_h
        y1 = y0 + stripe_h
        for col in range(int(w / stripe_h) + 2):
            x0 = col * stripe_h - (t * 40) % (stripe_h * 2)
            x1 = x0 + stripe_h
            dark = (row + col) % 2
            col_c = hills_color_dark if dark else hills_color_base
            draw_list.add_rect_filled(
                x0, y0, x1, y1,
                imgui.get_color_u32_rgba(*col_c, alpha))
    # Spinning gold rings
    rings = [(0.15, 0.38, 0.0), (0.35, 0.28, 0.8),
             (0.55, 0.45, 1.5), (0.42, 0.52, 2.2),
             (0.72, 0.32, 3.0), (0.88, 0.48, 3.7)]
    for rx, ry, ph in rings:
        ring_cx = rx * w
        ring_cy = ry * h
        spin = abs(math.cos(t * 5.0 + ph))
        ring_w = 18 * S * (0.2 + 0.8 * spin)
        ring_h_ = 18 * S
        draw_list.add_circle(
            ring_cx, ring_cy, ring_h_,
            imgui.get_color_u32_rgba(1.0, 0.85, 0.15, 0.95 * alpha),
            16, max(2.0, 3.0 * spin))
        if spin > 0.6:
            draw_list.add_circle_filled(
                ring_cx - ring_w * 0.3, ring_cy - 3, 2.5,
                imgui.get_color_u32_rgba(1.0, 1.0, 0.7, 0.9 * alpha))

    # Sonic is balled up into a spin-dash — a blue disc with quill streaks
    # rotating around it. The FunGen logo sits centered inside via the
    # ``place_logo`` hook; the ball ring + quills are drawn here.
    sx, sy, sr, is_looping = _sonic_pose(t, w, h)
    blue = imgui.get_color_u32_rgba(0.10, 0.40, 0.92, 0.98 * alpha)
    blue_dk = imgui.get_color_u32_rgba(0.05, 0.25, 0.70, 0.98 * alpha)
    blue_lt = imgui.get_color_u32_rgba(0.30, 0.60, 1.0, 0.98 * alpha)
    white = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.97 * alpha)
    # Shared below by Tails + Eggman (previously scoped to the removed
    # Sonic body block).
    black_eye = imgui.get_color_u32_rgba(0.05, 0.05, 0.08, 0.98 * alpha)

    # Motion trail — only during ground dash phases. Looks wrong during the
    # loop-de-loop since the ball isn't moving horizontally there.
    if not is_looping:
        for i in range(12):
            tx_ = sx - sr * 0.6 - i * 28 * S
            ta = (1.0 - i / 12.0) * 0.55
            draw_list.add_rect_filled(
                tx_, sy - 3 * S, tx_ + 22 * S, sy + 3 * S,
                imgui.get_color_u32_rgba(1.0, 1.0, 0.85, ta * alpha))
            draw_list.add_rect_filled(
                tx_, sy - 14 * S + (i % 3) * 5 * S,
                tx_ + 18 * S, sy - 10 * S + (i % 3) * 5 * S,
                imgui.get_color_u32_rgba(0.20, 0.55, 1.0,
                                          ta * 0.7 * alpha))

    # Dark blue outer ring — the ball's shell beyond the logo
    draw_list.add_circle_filled(sx, sy, sr * 1.10, blue_dk, 36)
    draw_list.add_circle_filled(sx, sy, sr * 0.95, blue, 36)

    # Spinning quill streaks radiating outward. Six quills, rotating around
    # the ball at a speed that reads as "rolling fast". Each quill is a
    # triangle with its base at the ball edge and tip a few radii out.
    spin = t * 9.0  # radians/sec — fast enough to feel like a spin-dash
    n_quills = 6
    for i in range(n_quills):
        ang = spin + i * (2 * math.pi / n_quills)
        base_r = sr * 1.02
        tip_r = sr * 1.55
        base_x = sx + math.cos(ang) * base_r
        base_y = sy + math.sin(ang) * base_r
        tip_x = sx + math.cos(ang) * tip_r
        tip_y = sy + math.sin(ang) * tip_r
        # Perpendicular at the base for triangle width
        perp = ang + math.pi / 2
        half_w = sr * 0.14
        bx1 = base_x + math.cos(perp) * half_w
        by1 = base_y + math.sin(perp) * half_w
        bx2 = base_x - math.cos(perp) * half_w
        by2 = base_y - math.sin(perp) * half_w
        draw_list.add_triangle_filled(
            bx1, by1, bx2, by2, tip_x, tip_y, blue_dk)

    # Faint spin-blur ring (dashed arc, segments rotating the other way to
    # sell the motion) — drawn as short overlapping arcs.
    blur_col = imgui.get_color_u32_rgba(0.55, 0.80, 1.0, 0.35 * alpha)
    n_dashes = 14
    for i in range(n_dashes):
        d_ang = -spin * 0.8 + i * (2 * math.pi / n_dashes)
        ax = sx + math.cos(d_ang) * sr * 1.22
        ay = sy + math.sin(d_ang) * sr * 1.22
        draw_list.add_circle_filled(ax, ay, sr * 0.06, blur_col, 8)

    # Little white sparkles spinning off the ball — "speed sparks"
    for i in range(3):
        s_ang = -spin * 1.4 + i * (2 * math.pi / 3)
        sr_ = sr * 1.70 + math.sin(t * 8 + i) * 6 * S
        sxp = sx + math.cos(s_ang) * sr_
        syp = sy + math.sin(s_ang) * sr_
        draw_list.add_circle_filled(
            sxp, syp, 3.5 * S, white, 10)
        draw_list.add_circle_filled(
            sxp, syp, 2.0 * S, blue_lt, 8)

    # Tails trails Sonic along the same path (including into the loop) via
    # a time-lagged sample of the trajectory. Gives the fun "conga line
    # through the loop" effect.
    tails_lag = SONIC_CYCLE * 0.08
    tx_pos, ty_pos, _srT, _T_loop = _sonic_pose(t - tails_lag, w, h)
    ty_pos += math.sin(t * 5) * 6 * S - sr * 0.20
    tr = sr * 0.75
    orange = imgui.get_color_u32_rgba(1.0, 0.70, 0.15, 0.98 * alpha)
    orange_dk = imgui.get_color_u32_rgba(0.85, 0.50, 0.10,
                                          0.98 * alpha)
    white_tail = imgui.get_color_u32_rgba(1.0, 0.96, 0.90,
                                           0.97 * alpha)
    tail_angle = t * 18.0
    for tail_i in range(2):
        ta = tail_angle + tail_i * math.pi
        tail_cx = tx_pos - tr * 1.1
        tail_cy = ty_pos
        tail_len = tr * 1.5
        tail_hw = tr * 0.30
        tip_x = tail_cx + math.cos(ta) * tail_len
        tip_y = tail_cy + math.sin(ta) * tail_len * 0.45
        perp_a = ta + math.pi / 2
        pxoff = math.cos(perp_a) * tail_hw
        pyoff = math.sin(perp_a) * tail_hw * 0.45
        draw_list.add_quad_filled(
            tail_cx, tail_cy,
            tail_cx + pxoff, tail_cy + pyoff,
            tip_x, tip_y,
            tail_cx - pxoff, tail_cy - pyoff,
            orange)
        draw_list.add_circle_filled(
            tip_x, tip_y, tail_hw * 0.9, white_tail, 14)
    draw_list.add_circle_filled(tx_pos, ty_pos, tr, orange, 24)
    draw_list.add_circle_filled(
        tx_pos + tr * 0.22, ty_pos + tr * 0.15,
        tr * 0.50, white_tail, 20)
    draw_list.add_circle_filled(
        tx_pos + tr * 0.50, ty_pos + tr * 0.05,
        tr * 0.30, white_tail, 18)
    draw_list.add_circle_filled(
        tx_pos + tr * 0.70, ty_pos - tr * 0.08,
        tr * 0.07, black_eye, 10)
    draw_list.add_triangle_filled(
        tx_pos - tr * 0.25, ty_pos - tr * 0.55,
        tx_pos - tr * 0.05, ty_pos - tr * 0.55,
        tx_pos - tr * 0.15, ty_pos - tr * 1.05,
        orange_dk)
    draw_list.add_triangle_filled(
        tx_pos + tr * 0.20, ty_pos - tr * 0.60,
        tx_pos + tr * 0.40, ty_pos - tr * 0.60,
        tx_pos + tr * 0.30, ty_pos - tr * 1.10,
        orange_dk)
    for (ear_tip_x, ear_tip_y) in [
        (tx_pos - tr * 0.15, ty_pos - tr * 0.85),
        (tx_pos + tr * 0.30, ty_pos - tr * 0.90),
    ]:
        draw_list.add_circle_filled(
            ear_tip_x, ear_tip_y, tr * 0.08,
            imgui.get_color_u32_rgba(1.0, 0.70, 0.55,
                                      0.95 * alpha), 10)
    draw_list.add_circle_filled(
        tx_pos + tr * 0.25, ty_pos - tr * 0.30,
        tr * 0.20, white, 16)
    draw_list.add_circle_filled(
        tx_pos + tr * 0.30, ty_pos - tr * 0.28,
        tr * 0.08, black_eye, 10)

    # Dr. Robotnik / Eggman chasing on the same path with a longer lag
    eggman_lag = SONIC_CYCLE * 0.16
    ex_pos, ey_pos, _srE, _E_loop = _sonic_pose(t - eggman_lag, w, h)
    ey_pos += math.sin(t * 2.3) * 8 * S - sr * 0.45
    er = sr * 0.95
    eggman_red = imgui.get_color_u32_rgba(0.85, 0.10, 0.12, 0.98 * alpha)
    eggman_yellow = imgui.get_color_u32_rgba(1.0, 0.85, 0.20, 0.98 * alpha)
    eggman_skin = imgui.get_color_u32_rgba(0.98, 0.78, 0.60, 0.98 * alpha)
    dark_goggle = imgui.get_color_u32_rgba(0.05, 0.08, 0.15, 0.98 * alpha)
    draw_list.add_rect_filled(
        ex_pos - er * 1.1, ey_pos + er * 0.65,
        ex_pos + er * 1.1, ey_pos + er * 1.15,
        eggman_yellow, er * 0.25)
    jet_pulse = 0.6 + 0.4 * math.sin(t * 18)
    for jx_off in (-er * 0.7, er * 0.7):
        draw_list.add_circle_filled(
            ex_pos + jx_off, ey_pos + er * 1.20,
            er * 0.18 * jet_pulse,
            imgui.get_color_u32_rgba(0.35, 0.70, 1.0,
                                      0.85 * jet_pulse * alpha), 14)
        draw_list.add_circle_filled(
            ex_pos + jx_off, ey_pos + er * 1.35,
            er * 0.10 * jet_pulse,
            imgui.get_color_u32_rgba(0.85, 0.95, 1.0,
                                      0.95 * jet_pulse * alpha), 12)
    draw_list.add_rect_filled(
        ex_pos - er * 0.75, ey_pos + er * 0.10,
        ex_pos + er * 0.75, ey_pos + er * 0.70,
        eggman_red, er * 0.40)
    draw_list.add_circle_filled(
        ex_pos, ey_pos - er * 0.20, er * 0.75,
        eggman_skin, 28)
    draw_list.add_circle_filled(
        ex_pos - er * 0.20, ey_pos - er * 0.55,
        er * 0.20,
        imgui.get_color_u32_rgba(1.0, 0.95, 0.85,
                                  0.65 * alpha), 14)
    for gx_off in (-er * 0.30, er * 0.30):
        draw_list.add_circle_filled(
            ex_pos + gx_off, ey_pos - er * 0.22,
            er * 0.26, dark_goggle, 18)
        draw_list.add_circle_filled(
            ex_pos + gx_off - er * 0.08,
            ey_pos - er * 0.30,
            er * 0.08,
            imgui.get_color_u32_rgba(1.0, 1.0, 1.0,
                                      0.75 * alpha), 12)
    draw_list.add_circle_filled(
        ex_pos, ey_pos - er * 0.12, er * 0.12,
        imgui.get_color_u32_rgba(0.95, 0.55, 0.55,
                                  0.95 * alpha), 14)
    for sgn in (-1, +1):
        mx0 = ex_pos
        my0 = ey_pos + er * 0.05
        mx1 = ex_pos + sgn * er * 0.55
        my1 = ey_pos + er * 0.20
        draw_list.add_triangle_filled(
            mx0, my0,
            mx1, my1,
            ex_pos + sgn * er * 0.25, ey_pos + er * 0.22,
            imgui.get_color_u32_rgba(1.0, 0.55, 0.10,
                                      0.97 * alpha))
        draw_list.add_circle_filled(
            mx1, my1, er * 0.12,
            imgui.get_color_u32_rgba(1.0, 0.55, 0.10,
                                      0.97 * alpha), 14)

    # RINGS counter in top-left
    imgui.set_window_font_scale(1.8)
    rings_txt = f"RINGS {int((t * 12) % 100):02d}"
    rt_sz = imgui.calc_text_size(rings_txt)
    rt_x = 18
    rt_y = 16
    draw_list.add_rect_filled(
        rt_x - 6, rt_y - 4,
        rt_x + rt_sz[0] + 6, rt_y + rt_sz[1] + 4,
        imgui.get_color_u32_rgba(0, 0, 0, 0.55 * alpha))
    draw_list.add_text(
        rt_x + 2, rt_y + 2,
        imgui.get_color_u32_rgba(0, 0, 0, 0.85 * alpha), rings_txt)
    draw_list.add_text(
        rt_x, rt_y,
        imgui.get_color_u32_rgba(1.0, 0.85, 0.15, 0.95 * alpha),
        rings_txt)
    imgui.set_window_font_scale(1.0)

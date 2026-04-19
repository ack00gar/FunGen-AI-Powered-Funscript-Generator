"""The iconic moon shot, but lit: huge warm moon, dark forest, Elliott on a
visible (not silhouetted) BMX with the FunGen mark riding in the basket
where E.T. used to sit, plus a ``FunGen phone home`` speech bubble. Pulsing
finger-glow and Reese's Pieces unchanged."""

import math
import random

import imgui

BIKE_CYCLE = 14.0


def _et_pose(t, w, h):
    """Bike position + scale. Shared between ``place_logo`` (runs before
    render_bg) and ``render_bg`` so the basket cradle and the logo line up."""
    bp = (t % BIKE_CYCLE) / BIKE_CYCLE
    bx = -120 + bp * (w + 240)
    arc_y = h * 0.95 - (h * 0.70) * (1 - (1 - bp) ** 1.6)
    by = arc_y
    moon_r = min(w, h) * 0.32
    # Bigger bike than the classic silhouette so the basket can actually hold
    # a legible logo (was 0.25 — tiny BMX).
    scale = moon_r * 0.45
    return bx, by, scale


def _basket_center(bx, by, scale):
    """Screen position of the basket slot (where the logo goes). Derived
    from the same handlebar geometry the bike itself uses."""
    wheel_cy = by + scale * 0.45
    wr_cx = bx + scale * 0.55
    hdlbar_x = wr_cx + scale * 0.10
    hdlbar_y = wheel_cy - scale * 0.35
    basket_cx = hdlbar_x + scale * 0.18
    basket_cy = hdlbar_y - scale * 0.22
    return basket_cx, basket_cy


def place_logo(splash, width, height, current_time):
    """Logo = the FunGen-as-E.T. passenger. The basket is only half the
    logo's height, so the lower half of the logo sits INSIDE the basket
    and the upper half pokes out above the rim — same composition as
    E.T.'s head and shoulders sticking out over the wicker."""
    bx, by, scale = _et_pose(current_time, width, height)
    basket_cx, basket_cy = _basket_center(bx, by, scale)
    logo_size = scale * 0.60
    basket_h = logo_size * 0.50
    logo_x = basket_cx - logo_size / 2
    # Logo vertical center sits at the basket's TOP rim, so exactly the
    # bottom half of the logo fits inside the basket (basket_h = logo/2).
    logo_y = basket_cy - basket_h / 2 - logo_size / 2
    return logo_x, logo_y, logo_size, 0.0


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height
    t = current_time
    state = getattr(splash, '_et_state', None)
    if state is None:
        rng = random.Random(1982)  # E.T. release year
        stars = [(rng.uniform(0, 1), rng.uniform(0, 0.6),
                  rng.uniform(0.4, 1.2)) for _ in range(90)]
        trees = []
        x = 0
        while x < w:
            tw = rng.randint(30, 70)
            th_ = rng.randint(50, 140)
            trees.append((x, tw, th_))
            x += tw - 4
        state = {'stars': stars, 'trees': trees}
        splash._et_state = state

    # Twilight sky gradient
    horizon_y = h * 0.70
    for i in range(30):
        f = i / 30
        y0 = f * horizon_y
        y1 = (f + 1 / 30) * horizon_y + 1
        r_ = 0.05 + 0.45 * f
        g_ = 0.05 + 0.20 * f
        b_ = 0.15 - 0.05 * f
        draw_list.add_rect_filled(
            0, y0, w, y1, imgui.get_color_u32_rgba(r_, g_, b_, alpha))

    # Stars
    for sx, sy, sp in state['stars']:
        twi = 0.55 + 0.45 * math.sin(t * sp * 2.5 + sx * 30)
        draw_list.add_circle_filled(
            sx * w, sy * horizon_y, 1.5,
            imgui.get_color_u32_rgba(1.0, 0.95, 0.85,
                                      0.85 * twi * alpha))

    # BIG warm moon
    moon_cx = w * 0.5
    moon_cy = h * 0.40
    moon_r = min(w, h) * 0.32
    # Outer glow halos
    for rad, aa in ((moon_r * 1.6, 0.06),
                     (moon_r * 1.3, 0.12),
                     (moon_r * 1.1, 0.22)):
        draw_list.add_circle_filled(
            moon_cx, moon_cy, rad,
            imgui.get_color_u32_rgba(1.0, 0.92, 0.70, aa * alpha))
    # Main disc
    draw_list.add_circle_filled(
        moon_cx, moon_cy, moon_r,
        imgui.get_color_u32_rgba(0.98, 0.92, 0.70, 0.98 * alpha), 64)
    # Subtle terminator shadow — a dimmer crescent on the lower-right so
    # the moon reads as 3D instead of a flat disc. Drawn as a shifted
    # darker disc clipped visually by the main disc on top of it.
    draw_list.add_circle_filled(
        moon_cx + moon_r * 0.18, moon_cy + moon_r * 0.12, moon_r * 0.96,
        imgui.get_color_u32_rgba(0.82, 0.74, 0.52, 0.22 * alpha), 48)
    # Large maria (dark surface patches)
    maria = [
        (-0.32,  0.08, 0.22, 0.18),
        ( 0.10, -0.18, 0.26, 0.15),
        (-0.05,  0.38, 0.20, 0.17),
        ( 0.32,  0.32, 0.14, 0.16),
    ]
    for (mx, my, mr, ma) in maria:
        draw_list.add_circle_filled(
            moon_cx + mx * moon_r, moon_cy + my * moon_r,
            mr * moon_r,
            imgui.get_color_u32_rgba(0.62, 0.54, 0.36, ma * alpha), 32)
    # Craters: shaded ring (dark shadow side + light rim highlight) on top
    # of a fill so they read as round pits. Shadow biased lower-right to
    # match the terminator direction.
    craters = [
        # (dx, dy, r, strength)
        (-0.20,  0.10, 0.075, 1.00),
        ( 0.25, -0.05, 0.060, 0.95),
        (-0.02, -0.28, 0.050, 0.90),
        ( 0.10,  0.42, 0.040, 0.85),
        (-0.40, -0.12, 0.035, 0.85),
        ( 0.42,  0.12, 0.055, 0.90),
        (-0.30,  0.48, 0.028, 0.80),
        ( 0.05,  0.08, 0.032, 0.80),
        ( 0.32, -0.35, 0.030, 0.80),
        (-0.55,  0.20, 0.024, 0.75),
    ]
    crater_fill = imgui.get_color_u32_rgba(0.78, 0.70, 0.48, 0.55 * alpha)
    crater_shadow = imgui.get_color_u32_rgba(0.45, 0.38, 0.22, 0.72 * alpha)
    crater_rim = imgui.get_color_u32_rgba(1.0, 0.98, 0.80, 0.55 * alpha)
    for (dx, dy, cr_rel, strength) in craters:
        cx = moon_cx + dx * moon_r
        cy = moon_cy + dy * moon_r
        cr = cr_rel * moon_r
        # Fill
        draw_list.add_circle_filled(cx, cy, cr, crater_fill, 18)
        # Shadow arc (lower-right crescent) — offset darker disc
        draw_list.add_circle_filled(
            cx + cr * 0.22, cy + cr * 0.22, cr * 0.82,
            crater_shadow, 18)
        # Upper-left rim highlight
        draw_list.add_circle(
            cx - cr * 0.08, cy - cr * 0.08, cr * 0.92,
            crater_rim, 20, max(1.2, cr * 0.08 * strength))

    # Elliott + FunGen-in-basket on a LIT bike (no silhouette). The logo
    # is painted later by the main splash loop via place_logo; render_bg
    # draws the bike, the rider, and the basket cradle that holds the logo.
    bx, by, scale = _et_pose(t, w, h)
    frame_col = imgui.get_color_u32_rgba(0.72, 0.18, 0.14, 0.98 * alpha)
    frame_hi = imgui.get_color_u32_rgba(0.95, 0.40, 0.28, 0.98 * alpha)
    tire_col = imgui.get_color_u32_rgba(0.12, 0.12, 0.14, 0.98 * alpha)
    rim_col = imgui.get_color_u32_rgba(0.80, 0.82, 0.88, 0.98 * alpha)
    spoke_col = imgui.get_color_u32_rgba(0.70, 0.72, 0.78, 0.95 * alpha)
    hub_col = imgui.get_color_u32_rgba(0.90, 0.92, 0.95, 0.98 * alpha)
    wicker = imgui.get_color_u32_rgba(0.72, 0.48, 0.20, 0.98 * alpha)
    wicker_dk = imgui.get_color_u32_rgba(0.48, 0.30, 0.12, 0.98 * alpha)
    hoodie = imgui.get_color_u32_rgba(0.88, 0.18, 0.18, 0.98 * alpha)
    hoodie_dk = imgui.get_color_u32_rgba(0.58, 0.10, 0.10, 0.98 * alpha)
    jeans = imgui.get_color_u32_rgba(0.18, 0.28, 0.55, 0.98 * alpha)
    skin = imgui.get_color_u32_rgba(0.98, 0.78, 0.58, 0.98 * alpha)
    hair = imgui.get_color_u32_rgba(0.35, 0.22, 0.10, 0.98 * alpha)

    wheel_r = scale * 0.38
    wl_cx = bx - scale * 0.55
    wr_cx = bx + scale * 0.55
    wheel_cy = by + scale * 0.45
    wheel_angle = t * 9.0
    for wcx in (wl_cx, wr_cx):
        # Tire (dark band), rim (light), hub, rotating spokes.
        draw_list.add_circle(wcx, wheel_cy, wheel_r, tire_col, 28, 5.0)
        draw_list.add_circle(wcx, wheel_cy, wheel_r * 0.88, rim_col, 28, 2.2)
        for si in range(10):
            a = wheel_angle + si * (math.pi / 5)
            sx1 = wcx + math.cos(a) * wheel_r * 0.18
            sy1 = wheel_cy + math.sin(a) * wheel_r * 0.18
            sx2 = wcx + math.cos(a) * wheel_r * 0.86
            sy2 = wheel_cy + math.sin(a) * wheel_r * 0.86
            draw_list.add_line(sx1, sy1, sx2, sy2, spoke_col, 1.4)
        draw_list.add_circle_filled(wcx, wheel_cy, wheel_r * 0.16,
                                     hub_col, 14)

    seat_x = bx - scale * 0.05
    seat_y = wheel_cy - scale * 0.50
    hdlbar_x = wr_cx + scale * 0.10
    hdlbar_y = wheel_cy - scale * 0.35
    # Red frame tubes
    draw_list.add_line(wl_cx, wheel_cy, seat_x, seat_y, frame_col, 5.0)
    draw_list.add_line(seat_x, seat_y, wr_cx, wheel_cy, frame_col, 5.0)
    draw_list.add_line(seat_x, seat_y, hdlbar_x, hdlbar_y, frame_col, 5.0)
    draw_list.add_line(wl_cx, wheel_cy, hdlbar_x, hdlbar_y, frame_col, 3.5)
    # Highlight on top tube
    draw_list.add_line(
        seat_x, seat_y - 2, hdlbar_x, hdlbar_y - 2, frame_hi, 1.6)

    # Crank + pedals
    crank_cx = bx - scale * 0.02
    crank_cy = wheel_cy - scale * 0.05
    crank_r = scale * 0.18
    pedal_angle = wheel_angle
    p1_x = crank_cx + math.cos(pedal_angle) * crank_r
    p1_y = crank_cy + math.sin(pedal_angle) * crank_r
    p2_x = crank_cx + math.cos(pedal_angle + math.pi) * crank_r
    p2_y = crank_cy + math.sin(pedal_angle + math.pi) * crank_r
    draw_list.add_line(crank_cx, crank_cy, p1_x, p1_y, rim_col, 2.5)
    draw_list.add_line(crank_cx, crank_cy, p2_x, p2_y, rim_col, 2.5)
    draw_list.add_circle_filled(p1_x, p1_y, scale * 0.05, tire_col, 10)
    draw_list.add_circle_filled(p2_x, p2_y, scale * 0.05, tire_col, 10)

    # Handlebar grips
    draw_list.add_circle_filled(
        hdlbar_x, hdlbar_y, scale * 0.05, tire_col, 12)

    # Wicker basket cradle — see-through: only the rim + back weave lines
    # are drawn in render_bg, with NO solid fill, so the sky shows through
    # the weave behind the logo. The front weave (slats crossing over the
    # logo) is painted in render_fg once the logo itself has been drawn.
    basket_cx, basket_cy = _basket_center(bx, by, scale)
    logo_size_est = scale * 0.60  # keep in sync with place_logo
    basket_w = logo_size_est * 1.10
    basket_h = logo_size_est * 0.50  # half the logo height — user spec
    bx_l = basket_cx - basket_w / 2
    bx_r = basket_cx + basket_w / 2
    by_t = basket_cy - basket_h / 2
    by_b = basket_cy + basket_h / 2
    # Stash geometry so render_fg can draw the front slats over the logo
    # without recomputing. Per-frame stash (not persistent across runs).
    splash._et_basket_geom = (bx_l, bx_r, by_t, by_b)

    # Rim — thick rounded rectangle outline (the basket silhouette)
    draw_list.add_rect(
        bx_l, by_t, bx_r, by_b, wicker, scale * 0.07, 0, 4.5)
    # Inner rim edge for depth
    draw_list.add_rect(
        bx_l + 3, by_t + 3, bx_r - 3, by_b - 3,
        wicker_dk, scale * 0.06, 0, 1.6)
    # Back weave lines (behind logo — mostly hidden once the logo draws
    # on top, but the parts sticking beyond the logo corners stay visible)
    for i in range(1, 4):
        sy = by_t + (by_b - by_t) * i / 4
        draw_list.add_line(bx_l + 3, sy, bx_r - 3, sy, wicker_dk, 1.6)
    # Rim highlight strip on the top
    draw_list.add_rect_filled(
        bx_l - 2, by_t - 3, bx_r + 2, by_t + 4, wicker_dk, 2)
    # Mount strap from basket to handlebar
    draw_list.add_line(
        basket_cx, by_b, hdlbar_x, hdlbar_y, wicker_dk, 3.0)

    # Elliott — red hoodie, blue jeans, peach face with messy hair
    rider_hip_x = seat_x
    rider_hip_y = seat_y - scale * 0.02
    rider_shoulder_x = seat_x + scale * 0.07
    rider_shoulder_y = seat_y - scale * 0.48
    rider_head_x = rider_shoulder_x + scale * 0.03
    rider_head_y = rider_shoulder_y - scale * 0.22
    # Torso (hoodie)
    draw_list.add_quad_filled(
        rider_hip_x - scale * 0.17, rider_hip_y,
        rider_hip_x + scale * 0.17, rider_hip_y,
        rider_shoulder_x + scale * 0.20, rider_shoulder_y,
        rider_shoulder_x - scale * 0.20, rider_shoulder_y,
        hoodie)
    # Hoodie cuff stripe
    draw_list.add_line(
        rider_hip_x - scale * 0.17, rider_hip_y - 2,
        rider_hip_x + scale * 0.17, rider_hip_y - 2,
        hoodie_dk, 3.0)
    # Head — skin + hair cap
    head_r = scale * 0.20
    draw_list.add_circle_filled(
        rider_head_x, rider_head_y, head_r, skin, 24)
    draw_list.add_circle_filled(
        rider_head_x, rider_head_y - head_r * 0.35,
        head_r * 0.95, hair, 24)
    # Eye dot
    draw_list.add_circle_filled(
        rider_head_x + head_r * 0.40, rider_head_y + head_r * 0.05,
        head_r * 0.08,
        imgui.get_color_u32_rgba(0.1, 0.1, 0.1, 0.95 * alpha), 10)
    # Arms reaching to handlebars
    draw_list.add_line(
        rider_shoulder_x + scale * 0.14, rider_shoulder_y + scale * 0.06,
        hdlbar_x, hdlbar_y, hoodie, 5.0)
    draw_list.add_line(
        rider_shoulder_x + scale * 0.02, rider_shoulder_y + scale * 0.06,
        hdlbar_x - scale * 0.06, hdlbar_y + scale * 0.02, hoodie, 4.5)
    # Jeans legs to pedals
    for (foot_x, foot_y) in [(p1_x, p1_y), (p2_x, p2_y)]:
        mid_x = (rider_hip_x + foot_x) * 0.5 + scale * 0.05
        mid_y = (rider_hip_y + foot_y) * 0.5 - scale * 0.08
        draw_list.add_line(
            rider_hip_x, rider_hip_y, mid_x, mid_y, jeans, 6.0)
        draw_list.add_line(
            mid_x, mid_y, foot_x, foot_y, jeans, 5.5)
    # Forest silhouette
    ground_y = h * 0.82
    for (tx_t, tw, th_) in state['trees']:
        draw_list.add_rect_filled(
            tx_t + tw * 0.40, ground_y - 8,
            tx_t + tw * 0.60, ground_y + 6,
            imgui.get_color_u32_rgba(0.02, 0.02, 0.02, alpha))
        draw_list.add_triangle_filled(
            tx_t, ground_y,
            tx_t + tw, ground_y,
            tx_t + tw * 0.5, ground_y - th_ * 0.55,
            imgui.get_color_u32_rgba(0.02, 0.02, 0.02, alpha))
        draw_list.add_triangle_filled(
            tx_t + tw * 0.1, ground_y - th_ * 0.3,
            tx_t + tw * 0.9, ground_y - th_ * 0.3,
            tx_t + tw * 0.5, ground_y - th_,
            imgui.get_color_u32_rgba(0.02, 0.02, 0.02, alpha))
    draw_list.add_rect_filled(
        0, ground_y, w, h,
        imgui.get_color_u32_rgba(0.02, 0.02, 0.02, alpha))

    # Glowing fingertip in the lower-left (E.T.'s healing glow)
    finger_pulse = 0.6 + 0.4 * math.sin(t * 2.8)
    fx = w * 0.10
    fy = h * 0.88
    for rad, aa in ((40, 0.10), (22, 0.25), (12, 0.60), (5, 1.0)):
        draw_list.add_circle_filled(
            fx, fy, rad * finger_pulse,
            imgui.get_color_u32_rgba(1.0, 0.55, 0.15,
                                      aa * finger_pulse * alpha))

    # Reese's Pieces floating upward
    for i in range(8):
        rp_rel = ((i / 8.0) + (t * 0.15) % 1.0) % 1.0
        rp_x = w * 0.08 + math.sin(rp_rel * math.pi * 3 + i) * 25
        rp_y = h - rp_rel * h * 0.6 - 40
        rp_col_idx = i % 3
        rp_col = [(1.0, 0.55, 0.10),
                  (1.0, 0.90, 0.25),
                  (0.85, 0.15, 0.18)][rp_col_idx]
        draw_list.add_circle_filled(
            rp_x, rp_y, 7.5,
            imgui.get_color_u32_rgba(*rp_col, 0.92 * alpha))
        draw_list.add_circle_filled(
            rp_x - 2, rp_y - 2, 2.5,
            imgui.get_color_u32_rgba(1.0, 1.0, 0.85, 0.8 * alpha))


def render_fg(splash, logo_x, logo_y, logo_size, laser_time):
    """Front of the wicker basket (drawn OVER the logo so the weave reads
    as see-through), plus the 'FunGen phone home' speech bubble."""
    draw_list = imgui.get_window_draw_list()

    # Front basket slats on top of the logo. Uses geometry stashed by
    # render_bg so the front weave aligns perfectly with the rim.
    geom = getattr(splash, '_et_basket_geom', None)
    if geom is not None:
        bx_l, bx_r, by_t, by_b = geom
        wicker = imgui.get_color_u32_rgba(0.72, 0.48, 0.20, 0.92)
        wicker_dk = imgui.get_color_u32_rgba(0.42, 0.26, 0.10, 0.95)
        # Vertical slats — 6 evenly spaced across the basket width
        for i in range(1, 7):
            sx_v = bx_l + (bx_r - bx_l) * i / 7
            draw_list.add_line(
                sx_v, by_t + 4, sx_v, by_b - 4, wicker, 2.2)
        # Two horizontal weave bands over the logo
        for frac in (0.30, 0.70):
            sy = by_t + (by_b - by_t) * frac
            draw_list.add_line(bx_l + 3, sy, bx_r - 3, sy, wicker_dk, 1.8)
        # Subtle darker shadow along the bottom rim (depth cue)
        draw_list.add_line(
            bx_l + 4, by_b - 2, bx_r - 4, by_b - 2, wicker_dk, 2.0)

    # Bubble sits above and slightly to the right of the logo
    bub_w = logo_size * 1.45
    bub_h = logo_size * 0.65
    bub_x = logo_x + logo_size * 0.35
    bub_y = logo_y - bub_h - logo_size * 0.18
    # Clamp so the bubble doesn't fly off the top of the window
    bub_y = max(12.0, bub_y)
    bub_r = bub_h * 0.28

    white = imgui.get_color_u32_rgba(0.98, 0.98, 0.98, 0.95)
    stroke = imgui.get_color_u32_rgba(0.05, 0.05, 0.10, 0.92)
    ink = imgui.get_color_u32_rgba(0.08, 0.08, 0.15, 0.98)

    # Bubble body (rounded rect)
    draw_list.add_rect_filled(
        bub_x, bub_y, bub_x + bub_w, bub_y + bub_h, white, bub_r)
    draw_list.add_rect(
        bub_x, bub_y, bub_x + bub_w, bub_y + bub_h, stroke,
        bub_r, 0, 2.5)

    # Speech-bubble tail: three shrinking circles from bubble down to logo.
    tail_start_x = bub_x + bub_w * 0.25
    tail_start_y = bub_y + bub_h
    tail_end_x = logo_x + logo_size * 0.55
    tail_end_y = logo_y + logo_size * 0.15
    for i, frac in enumerate((0.25, 0.55, 0.80)):
        tcx = tail_start_x + (tail_end_x - tail_start_x) * frac
        tcy = tail_start_y + (tail_end_y - tail_start_y) * frac
        tr = bub_h * (0.18 - i * 0.05)
        draw_list.add_circle_filled(tcx, tcy, tr, white, 14)
        draw_list.add_circle(tcx, tcy, tr, stroke, 14, 2.0)

    # Text "FunGen phone home", two lines, centered in the bubble
    line1 = "FunGen"
    line2 = "phone home"
    imgui.set_window_font_scale(1.6)
    s1 = imgui.calc_text_size(line1)
    s2 = imgui.calc_text_size(line2)
    total_h = s1[1] + s2[1] + 3
    cy_text = bub_y + (bub_h - total_h) / 2
    x1 = bub_x + (bub_w - s1[0]) / 2
    x2 = bub_x + (bub_w - s2[0]) / 2
    draw_list.add_text(x1, cy_text, ink, line1)
    draw_list.add_text(x2, cy_text + s1[1] + 3, ink, line2)
    imgui.set_window_font_scale(1.0)

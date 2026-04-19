"""Super Mario Bros. world 1-1 vibe: cyan sky, 8-bit clouds and hills, warp
pipes, ?-blocks with spinning coins, walking Goombas, jumping Mario."""

import math

import imgui


def _mario_pose(t, w, h):
    """Compute Mario's running + hopping animation state.

    Shared between ``place_logo`` (which runs before ``render_bg`` in the
    splash loop) and ``render_bg`` itself, so the logo-as-head can ride on
    the exact same position/jump-height the body is drawn at.
    """
    ground_y = h * 0.82
    mario_cycle = 9.0
    mp = (t % mario_cycle) / mario_cycle
    mx = -80 + mp * (w + 160)
    hop_period = 1.6
    hp = (t % hop_period) / hop_period
    is_airborne = hp > 0.55
    if is_airborne:
        hop_t = (hp - 0.55) / 0.45
        jump_h = 90 * math.sin(hop_t * math.pi)
    else:
        jump_h = 0
    my = ground_y - 80 - jump_h
    stride = (t * 6.0) % 1.0 if not is_airborne else 0.5
    mario_sz = 180
    return mx, my, mario_sz, is_airborne, stride


def place_logo(splash, width, height, current_time):
    """Logo = Mario's head. Tracks his run + jump; face stripped, cap drawn
    on top by ``render_fg``."""
    mx, my, mario_sz, _airborne, _stride = _mario_pose(
        current_time, width, height)
    # Head center in _draw_mario: cx=mx, cy=my - s*0.45. Logo is sized a bit
    # bigger than the original head disc (head_r = s*0.25, diameter = s*0.5)
    # so the FunGen mark reads clearly with a cap perched on top.
    logo_size = mario_sz * 0.72
    logo_x = mx - logo_size / 2
    logo_y = (my - mario_sz * 0.48) - logo_size / 2
    # Mario is already bouncing — don't layer the default float offset.
    return logo_x, logo_y, logo_size, 0.0


def render_fg(splash, logo_x, logo_y, logo_size, laser_time):
    """Red Mario cap perched on top of the logo-head. Drawn after the logo
    itself so the brim sits on top of the logo's upper edge. The cap dome
    lives entirely above the logo — only the brim touches the head — so the
    FunGen mark stays mostly uncovered. F-badge (not M) on the dome."""
    draw_list = imgui.get_window_draw_list()
    red = imgui.get_color_u32_rgba(0.92, 0.15, 0.12, 0.98)
    white = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.98)
    shadow = imgui.get_color_u32_rgba(0.0, 0.0, 0.0, 0.35)

    cap_cx = logo_x + logo_size / 2
    cap_r = logo_size * 0.42
    # Dome center is above the logo top. Only ~15% of the cap radius dips
    # into the logo — enough to meet the brim cleanly, not enough to cover
    # the FunGen glyph.
    cap_cy = logo_y - cap_r * 0.55
    draw_list.add_circle_filled(cap_cx, cap_cy, cap_r, red, 28)

    # Brim — horizontal red strip AT the logo's top edge, extending outward
    # past the logo on both sides.
    brim_y_top = logo_y - cap_r * 0.06
    brim_y_bot = brim_y_top + cap_r * 0.26
    brim_left = cap_cx - cap_r * 1.35
    brim_right = cap_cx + cap_r * 1.35
    draw_list.add_rect_filled(
        brim_left, brim_y_top, brim_right, brim_y_bot, red,
        cap_r * 0.10)
    draw_list.add_rect_filled(
        brim_left, brim_y_bot - 1, brim_right, brim_y_bot + 2, shadow, 0.0)

    # F badge on the cap
    fr_cx = cap_cx
    fr_cy = cap_cy - cap_r * 0.05
    fr_r = cap_r * 0.40
    draw_list.add_circle_filled(fr_cx, fr_cy, fr_r, white, 18)
    # F strokes — vertical stem + top crossbar + (shorter) middle bar.
    stroke = 3.5
    v_x = fr_cx - fr_r * 0.30
    v_top_y = fr_cy - fr_r * 0.50
    v_bot_y = fr_cy + fr_r * 0.52
    draw_list.add_line(v_x, v_top_y, v_x, v_bot_y, red, stroke)
    draw_list.add_line(
        v_x, v_top_y, v_x + fr_r * 0.65, v_top_y, red, stroke)
    draw_list.add_line(
        v_x, fr_cy - fr_r * 0.02,
        v_x + fr_r * 0.45, fr_cy - fr_r * 0.02, red, stroke)


def _draw_mario(draw_list, cx, cy, s, t, alpha, stride=0.0, airborne=False,
                skip_head=False):
    """Small Mario sprite in running / airborne poses.

    When ``skip_head`` is True, the head + cap + face + 'M' badge are omitted
    (the caller will paint the FunGen logo there instead).
    """
    s = float(s)
    red = imgui.get_color_u32_rgba(0.92, 0.15, 0.12, 0.98 * alpha)
    blue = imgui.get_color_u32_rgba(0.15, 0.35, 0.85, 0.98 * alpha)
    skin = imgui.get_color_u32_rgba(0.98, 0.78, 0.55, 0.98 * alpha)
    brown = imgui.get_color_u32_rgba(0.35, 0.18, 0.08, 0.98 * alpha)
    white = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.98 * alpha)
    yellow = imgui.get_color_u32_rgba(1.0, 0.90, 0.25, 0.98 * alpha)
    shoe_brown = imgui.get_color_u32_rgba(0.25, 0.12, 0.05, 0.98 * alpha)
    head_r = s * 0.25
    head_cx = cx
    head_cy = cy - s * 0.45
    if not skip_head:
        draw_list.add_circle_filled(head_cx, head_cy, head_r, skin, 20)
        draw_list.add_rect_filled(
            head_cx - head_r * 0.9, head_cy + head_r * 0.15,
            head_cx + head_r * 0.9, head_cy + head_r * 0.40,
            brown, head_r * 0.12)
        draw_list.add_circle_filled(
            head_cx, head_cy + head_r * 0.10,
            head_r * 0.30,
            imgui.get_color_u32_rgba(0.95, 0.65, 0.45,
                                      0.98 * alpha), 14)
        for sgn in (-1, +1):
            draw_list.add_circle_filled(
                head_cx + sgn * head_r * 0.30,
                head_cy - head_r * 0.20,
                head_r * 0.12, white, 10)
            draw_list.add_circle_filled(
                head_cx + sgn * head_r * 0.30,
                head_cy - head_r * 0.20,
                head_r * 0.06,
                imgui.get_color_u32_rgba(0.05, 0.05, 0.05,
                                          0.98 * alpha), 8)
        # Red cap
        draw_list.add_circle_filled(
            head_cx, head_cy - head_r * 0.08, head_r * 1.02, red, 24)
        draw_list.add_rect_filled(
            head_cx - head_r * 1.10, head_cy - head_r * 0.02,
            head_cx + head_r * 1.10, head_cy + head_r * 1.20,
            skin)
        draw_list.add_rect_filled(
            head_cx - head_r * 0.3, head_cy - head_r * 0.12,
            head_cx + head_r * 1.30, head_cy + head_r * 0.12,
            red, head_r * 0.12)
        mcircle_cx = head_cx + head_r * 0.15
        mcircle_cy = head_cy - head_r * 0.45
        mcircle_r = head_r * 0.28
        draw_list.add_circle_filled(
            mcircle_cx, mcircle_cy, mcircle_r, white, 16)
        m_col = imgui.get_color_u32_rgba(0.92, 0.15, 0.12, 0.98 * alpha)
        draw_list.add_line(
            mcircle_cx - mcircle_r * 0.45, mcircle_cy + mcircle_r * 0.40,
            mcircle_cx - mcircle_r * 0.25, mcircle_cy - mcircle_r * 0.50,
            m_col, 3.0)
        draw_list.add_line(
            mcircle_cx - mcircle_r * 0.25, mcircle_cy - mcircle_r * 0.50,
            mcircle_cx, mcircle_cy + mcircle_r * 0.10,
            m_col, 3.0)
        draw_list.add_line(
            mcircle_cx, mcircle_cy + mcircle_r * 0.10,
            mcircle_cx + mcircle_r * 0.25, mcircle_cy - mcircle_r * 0.50,
            m_col, 3.0)
        draw_list.add_line(
            mcircle_cx + mcircle_r * 0.25, mcircle_cy - mcircle_r * 0.50,
            mcircle_cx + mcircle_r * 0.45, mcircle_cy + mcircle_r * 0.40,
            m_col, 3.0)
    # Torso / shirt
    shirt_top_y = cy - s * 0.18
    shirt_bot_y = cy + s * 0.00
    draw_list.add_rect_filled(
        cx - s * 0.22, shirt_top_y,
        cx + s * 0.22, shirt_bot_y,
        red, s * 0.04)
    # Overalls
    overalls_top_y = cy - s * 0.06
    overalls_bot_y = cy + s * 0.30
    draw_list.add_rect_filled(
        cx - s * 0.26, overalls_top_y,
        cx + s * 0.26, overalls_bot_y,
        blue, s * 0.05)
    draw_list.add_rect_filled(
        cx - s * 0.14, shirt_top_y,
        cx - s * 0.08, overalls_top_y + s * 0.04, blue)
    draw_list.add_rect_filled(
        cx + s * 0.08, shirt_top_y,
        cx + s * 0.14, overalls_top_y + s * 0.04, blue)
    for sgn in (-1, +1):
        draw_list.add_circle_filled(
            cx + sgn * s * 0.10, overalls_top_y + s * 0.03,
            s * 0.035, yellow, 10)
    # Gloved hands
    if airborne:
        for sgn in (-1, +1):
            hx = cx + sgn * s * 0.30
            hy = shirt_top_y - s * 0.18
            draw_list.add_circle_filled(hx, hy, s * 0.11, white, 14)
            draw_list.add_line(
                cx + sgn * s * 0.22, shirt_top_y + s * 0.06,
                hx, hy, red, 8.0)
    else:
        for sgn in (-1, +1):
            arm_offset = (stride - 0.5) * 2 * sgn
            hx = cx + sgn * s * 0.30 + arm_offset * s * 0.10
            hy = shirt_top_y + s * 0.05 - abs(arm_offset) * s * 0.08
            draw_list.add_circle_filled(hx, hy, s * 0.11, white, 14)
            draw_list.add_line(
                cx + sgn * s * 0.22, shirt_top_y + s * 0.06,
                hx, hy, red, 8.0)
    # Feet / shoes
    if airborne:
        for sgn in (-1, +1):
            shx = cx + sgn * s * 0.10
            shy = overalls_bot_y - s * 0.02
            draw_list.add_rect_filled(
                shx - s * 0.16, shy,
                shx + s * 0.16 + sgn * s * 0.04,
                shy + s * 0.12,
                shoe_brown, s * 0.05)
    else:
        for leg_i, sgn in enumerate((-1, +1)):
            phase_off = 0.5 if sgn == +1 else 0.0
            ls = (stride + phase_off) % 1.0
            fwd = math.sin(ls * math.pi * 2) * s * 0.18
            lift = max(0.0, math.sin(ls * math.pi * 2)) * s * 0.14
            shx = cx + fwd
            shy = overalls_bot_y - lift
            draw_list.add_rect_filled(
                shx - s * 0.14, shy,
                shx + s * 0.14 + (1 if fwd > 0 else -1) * s * 0.04,
                shy + s * 0.11,
                shoe_brown, s * 0.05)


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height
    t = current_time
    # World scale — everything in the scene (pipes, blocks, goombas, hills,
    # clouds, shrubs, dirt tiles) is multiplied by this so the decor reads
    # as chunky NES-console artwork rather than small badges.
    S = 1.6
    draw_list.add_rect_filled(
        0, 0, w, h,
        imgui.get_color_u32_rgba(0.35, 0.60, 1.0, alpha))
    ground_y = h * 0.82
    grass_h = int(8 * S)
    draw_list.add_rect_filled(
        0, ground_y, w, ground_y + grass_h,
        imgui.get_color_u32_rgba(0.20, 0.70, 0.20, alpha))
    draw_list.add_rect_filled(
        0, ground_y + grass_h, w, h,
        imgui.get_color_u32_rgba(0.70, 0.35, 0.15, alpha))
    tile = int(18 * S)
    for row in range(int((h - ground_y - grass_h) / tile) + 1):
        ry = ground_y + grass_h + row * tile
        row_off = tile // 2 if row % 2 else 0
        for col in range(int(w / tile) + 2):
            gx = col * tile - row_off
            draw_list.add_rect(
                gx, ry, gx + tile, ry + tile,
                imgui.get_color_u32_rgba(0.45, 0.20, 0.08,
                                          0.85 * alpha),
                0, 0, 1.0)
    cloud_y_positions = [(0.12, 0.35, 2.0),
                         (0.18, 0.72, 1.4),
                         (0.10, 0.08, 1.0)]
    for (cy_rel, phase, speed) in cloud_y_positions:
        drift = ((t * 12 * speed) + phase * w) % (w + 300 * S) - 150 * S
        cx_c = drift
        cy_c = h * cy_rel
        white = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.97 * alpha)
        for (cox, cr) in [(-40 * S, 18 * S), (-15 * S, 25 * S),
                           (0, 32 * S), (20 * S, 26 * S),
                           (45 * S, 20 * S)]:
            draw_list.add_circle_filled(
                cx_c + cox, cy_c, cr, white, 20)
        outline = imgui.get_color_u32_rgba(0.0, 0.0, 0.0,
                                            0.18 * alpha)
        draw_list.add_line(
            cx_c - 55 * S, cy_c + 14 * S,
            cx_c + 60 * S, cy_c + 14 * S,
            outline, 2)

    # Rolling green hills
    hill_y = ground_y - 5 * S
    for (hx_rel, hr_base) in [(0.20, 85), (0.55, 110), (0.85, 70)]:
        hr = hr_base * S
        hcx = hx_rel * w
        draw_list.add_circle_filled(
            hcx, hill_y + hr * 0.8, hr,
            imgui.get_color_u32_rgba(0.10, 0.60, 0.20,
                                      0.95 * alpha), 28)
        draw_list.add_circle_filled(
            hcx + hr * 0.2, hill_y + hr * 0.9, hr * 0.85,
            imgui.get_color_u32_rgba(0.06, 0.45, 0.15,
                                      0.85 * alpha), 24)

    # Small shrubs
    for (bx_rel, bsz_base) in [(0.05, 18), (0.40, 22), (0.72, 16),
                                (0.93, 20)]:
        bsz = bsz_base * S
        bcx = bx_rel * w
        bcy = ground_y - bsz * 0.5
        for (box, br) in [(-bsz * 0.6, bsz * 0.7),
                           (0, bsz),
                           (bsz * 0.6, bsz * 0.7)]:
            draw_list.add_circle_filled(
                bcx + box, bcy, br,
                imgui.get_color_u32_rgba(0.05, 0.50, 0.12,
                                          0.95 * alpha), 18)

    # Two warp pipes
    for (px_rel, pipe_h_base) in [(0.12, 100), (0.88, 130)]:
        pipe_h = pipe_h_base * S
        pcx = px_rel * w
        p_top_y = ground_y - pipe_h
        shaft_half = 28 * S
        shaft_hi = 22 * S
        cap_half = 36 * S
        cap_hi = 28 * S
        cap_h = 18 * S
        collar_y = 14 * S
        draw_list.add_rect_filled(
            pcx - shaft_half, p_top_y + collar_y,
            pcx + shaft_half, ground_y,
            imgui.get_color_u32_rgba(0.10, 0.60, 0.18, alpha))
        draw_list.add_rect_filled(
            pcx - shaft_half, p_top_y + collar_y,
            pcx - shaft_hi, ground_y,
            imgui.get_color_u32_rgba(0.35, 0.85, 0.30, alpha))
        draw_list.add_rect_filled(
            pcx - cap_half, p_top_y, pcx + cap_half, p_top_y + cap_h,
            imgui.get_color_u32_rgba(0.10, 0.60, 0.18, alpha))
        draw_list.add_rect_filled(
            pcx - cap_half, p_top_y, pcx - cap_hi, p_top_y + cap_h,
            imgui.get_color_u32_rgba(0.35, 0.85, 0.30, alpha))
        draw_list.add_rect_filled(
            pcx - cap_half, p_top_y + collar_y,
            pcx + cap_half, p_top_y + cap_h,
            imgui.get_color_u32_rgba(0.05, 0.40, 0.10, alpha))
        draw_list.add_rect_filled(
            pcx - 30 * S, p_top_y + 2 * S,
            pcx + 30 * S, p_top_y + 6 * S,
            imgui.get_color_u32_rgba(0.02, 0.15, 0.04, alpha))

    # Floating ?-blocks
    block_positions = [(0.28, 0.40), (0.48, 0.35), (0.68, 0.42)]
    for i, (bx_rel, by_rel) in enumerate(block_positions):
        bcx = bx_rel * w
        bcy = by_rel * h
        bsize = 38 * S
        draw_list.add_rect_filled(
            bcx - bsize / 2, bcy - bsize / 2,
            bcx + bsize / 2, bcy + bsize / 2,
            imgui.get_color_u32_rgba(0.95, 0.70, 0.10, alpha), 3 * S)
        draw_list.add_rect(
            bcx - bsize / 2, bcy - bsize / 2,
            bcx + bsize / 2, bcy + bsize / 2,
            imgui.get_color_u32_rgba(0.25, 0.10, 0.02, alpha),
            3 * S, 0, 2.5 * S)
        for (rx, ry) in [(-0.40, -0.40), (0.40, -0.40),
                          (-0.40, 0.40), (0.40, 0.40)]:
            draw_list.add_circle_filled(
                bcx + rx * bsize, bcy + ry * bsize,
                2.5 * S,
                imgui.get_color_u32_rgba(0.20, 0.08, 0.02,
                                          alpha), 8)
        imgui.set_window_font_scale(2.0 * S)
        qmk = "?"
        qsz = imgui.calc_text_size(qmk)
        draw_list.add_text(
            bcx - qsz[0] / 2, bcy - qsz[1] / 2,
            imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.98 * alpha),
            qmk)
        imgui.set_window_font_scale(1.0)
        # Spinning gold coin
        coin_y_bob = math.sin(t * 3 + i) * 4 * S
        coin_cx = bcx
        coin_cy = bcy - bsize - 22 * S + coin_y_bob
        coin_w = abs(math.cos(t * 6 + i)) * 14 * S + 3 * S
        coin_h = 14 * S
        draw_list.add_rect_filled(
            coin_cx - coin_w, coin_cy - coin_h,
            coin_cx + coin_w, coin_cy + coin_h,
            imgui.get_color_u32_rgba(1.0, 0.85, 0.20, alpha),
            coin_w * 0.5)
        if coin_w > 8 * S:
            draw_list.add_rect_filled(
                coin_cx - coin_w * 0.4, coin_cy - 10 * S,
                coin_cx + coin_w * 0.4, coin_cy + 10 * S,
                imgui.get_color_u32_rgba(1.0, 0.95, 0.60,
                                          0.85 * alpha),
                coin_w * 0.3)

    # Goombas walking
    for (gx_base, gspeed_base, gphase) in [(0.35, 45, 0.0),
                                            (0.65, -35, 2.5)]:
        gspeed = gspeed_base * S
        if gspeed > 0:
            gcx = ((gx_base * w + t * gspeed) %
                   (w + 100 * S)) - 50 * S
        else:
            gcx = (w + 50 * S - ((t * abs(gspeed) + gx_base * w)
                                 % (w + 100 * S)))
        gcy = ground_y - 18 * S
        waddle = math.sin(t * 6 + gphase) * 3 * S
        draw_list.add_rect_filled(
            gcx - 26 * S, gcy - 18 * S, gcx + 26 * S, gcy + 6 * S,
            imgui.get_color_u32_rgba(0.55, 0.30, 0.10, alpha), 14 * S)
        draw_list.add_rect_filled(
            gcx - 26 * S, gcy + 0, gcx + 26 * S, gcy + 6 * S,
            imgui.get_color_u32_rgba(0.40, 0.22, 0.06, alpha))
        draw_list.add_rect_filled(
            gcx - 22 * S, gcy + 6 * S, gcx + 22 * S, gcy + 20 * S,
            imgui.get_color_u32_rgba(0.90, 0.70, 0.30, alpha))
        draw_list.add_triangle_filled(
            gcx - 16 * S, gcy + 4 * S, gcx - 4 * S, gcy + 4 * S,
            gcx - 10 * S, gcy + 10 * S,
            imgui.get_color_u32_rgba(0.05, 0.05, 0.05,
                                      0.98 * alpha))
        draw_list.add_triangle_filled(
            gcx + 4 * S, gcy + 4 * S, gcx + 16 * S, gcy + 4 * S,
            gcx + 10 * S, gcy + 10 * S,
            imgui.get_color_u32_rgba(0.05, 0.05, 0.05,
                                      0.98 * alpha))
        draw_list.add_rect_filled(
            gcx - 14 * S, gcy + 10 * S, gcx - 6 * S, gcy + 18 * S,
            imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.98 * alpha))
        draw_list.add_rect_filled(
            gcx + 6 * S, gcy + 10 * S, gcx + 14 * S, gcy + 18 * S,
            imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 0.98 * alpha))
        draw_list.add_rect_filled(
            gcx - 12 * S, gcy + 12 * S, gcx - 8 * S, gcy + 17 * S,
            imgui.get_color_u32_rgba(0.05, 0.05, 0.05,
                                      0.98 * alpha))
        draw_list.add_rect_filled(
            gcx + 8 * S, gcy + 12 * S, gcx + 12 * S, gcy + 17 * S,
            imgui.get_color_u32_rgba(0.05, 0.05, 0.05,
                                      0.98 * alpha))
        draw_list.add_rect_filled(
            gcx - 20 * S + waddle, ground_y - 6 * S,
            gcx - 4 * S + waddle, ground_y,
            imgui.get_color_u32_rgba(0.30, 0.15, 0.05, alpha))
        draw_list.add_rect_filled(
            gcx + 4 * S - waddle, ground_y - 6 * S,
            gcx + 20 * S - waddle, ground_y,
            imgui.get_color_u32_rgba(0.30, 0.15, 0.05, alpha))

    # Mario animation — body only; the FunGen logo is placed over the head
    # slot by ``place_logo`` (same pose function, so they stay in lockstep).
    mx, my, mario_sz, is_airborne, stride = _mario_pose(t, w, h)
    _draw_mario(draw_list, mx, my, mario_sz, t, alpha,
                stride=stride, airborne=is_airborne, skip_head=True)

"""NYC alley at night: brick walls, fire escape, sewer cover, four turtles'
colored bandannas floating across with their weapons, pizza slice spinning.
One of the four heads is the FunGen logo, wearing a cyan ninja bandanna."""

import math

import imgui

# Shared turtle-head layout. Must match the list in render_bg so place_logo
# can position the FunGen head on the same path as the other turtles.
_TURTLES = [
    ((0.95, 0.15, 0.15), "R", 35, 0.0, 0.40),  # idx 0 — FunGen hero
    ((0.15, 0.45, 0.95), "L", 28, 1.5, 0.32),
    ((1.00, 0.55, 0.10), "M", 42, 3.2, 0.48),
    ((0.65, 0.30, 0.85), "D", 31, 4.7, 0.28),
]

_TURTLE_CYCLE = 12.0
_HERO_IDX = 0  # which of the four turtles carries the FunGen logo


def _turtle_pose(i, t, w, h):
    """Position of turtle ``i`` at time ``t``. Must match render_bg exactly."""
    _col, _weapon, _speed, ph, y_rel = _TURTLES[i]
    ft = ((t + ph) % _TURTLE_CYCLE) / _TURTLE_CYCLE
    bx_c = -120 + ft * (w + 240)
    by_c = h * y_rel + math.sin(ft * math.pi * 3 + i) * 30
    return bx_c, by_c


def place_logo(splash, width, height, current_time):
    """Logo = the FunGen ninja turtle's head, riding along with the other
    three heads. The bandanna strap is drawn on top of the logo by
    render_fg; bandanna tails trail behind it from render_bg."""
    bx_c, by_c = _turtle_pose(_HERO_IDX, current_time, width, height)
    head_sz = max(68, int(min(width, height) * 0.11))
    # Logo stands in for the head — slightly bigger than the other turtles'
    # head circles (which are head_sz * 0.9 diameter) so the FunGen mark
    # reads clearly even at floating-head scale.
    logo_size = head_sz * 1.30
    logo_x = bx_c - logo_size / 2
    logo_y = by_c - logo_size / 2
    return logo_x, logo_y, logo_size, 0.0


def _draw_turtle_head(draw_list, cx, cy, s, bandanna_col, alpha):
    """Green turtle head with a colored bandanna and two eye-slits."""
    s = float(s)
    green = imgui.get_color_u32_rgba(0.25, 0.65, 0.20, 0.97 * alpha)
    green_dk = imgui.get_color_u32_rgba(0.15, 0.45, 0.12, 0.97 * alpha)
    draw_list.add_circle_filled(cx, cy, s * 0.45, green, 20)
    band_a = 0.97 * alpha
    band_col = imgui.get_color_u32_rgba(*bandanna_col, band_a)
    draw_list.add_rect_filled(
        cx - s * 0.55, cy - s * 0.15,
        cx + s * 0.55, cy + s * 0.05, band_col)
    # Bandanna tails flowing out the left side
    draw_list.add_triangle_filled(
        cx - s * 0.55, cy - s * 0.10,
        cx - s * 1.05, cy + s * 0.08,
        cx - s * 0.55, cy + s * 0.05, band_col)
    draw_list.add_triangle_filled(
        cx - s * 0.50, cy + s * 0.02,
        cx - s * 1.00, cy + s * 0.28,
        cx - s * 0.50, cy + s * 0.18, band_col)
    # Eye slits
    for sgn in (-1, +1):
        draw_list.add_rect_filled(
            cx + sgn * s * 0.20 - s * 0.08, cy - s * 0.08,
            cx + sgn * s * 0.20 + s * 0.08, cy - s * 0.01,
            imgui.get_color_u32_rgba(1, 1, 1, 0.95 * alpha))
        draw_list.add_rect_filled(
            cx + sgn * s * 0.20 - s * 0.03, cy - s * 0.08,
            cx + sgn * s * 0.20 + s * 0.03, cy - s * 0.01,
            imgui.get_color_u32_rgba(0, 0, 0, 0.98 * alpha))
    draw_list.add_line(
        cx - s * 0.10, cy + s * 0.22,
        cx + s * 0.10, cy + s * 0.22,
        green_dk, 2.0)


def _draw_turtle_weapon(draw_list, cx, cy, s, kind, t, alpha):
    """Weapon primitive: katana / sai / nunchaku / bo staff."""
    s = float(s)
    grey = imgui.get_color_u32_rgba(0.88, 0.90, 0.95, 0.98 * alpha)
    grey_dk = imgui.get_color_u32_rgba(0.55, 0.60, 0.70, 0.95 * alpha)
    brown = imgui.get_color_u32_rgba(0.50, 0.28, 0.10, 0.98 * alpha)
    brown_dk = imgui.get_color_u32_rgba(0.30, 0.15, 0.05, 0.98 * alpha)
    gold = imgui.get_color_u32_rgba(0.85, 0.70, 0.18, 0.98 * alpha)
    black = imgui.get_color_u32_rgba(0.05, 0.03, 0.05, 0.98 * alpha)
    bob = math.sin(t * 3) * 4
    wx = cx + s * 0.40
    wy = cy + bob
    if kind == "L":
        blade_top = wy - s * 0.95
        blade_bot = wy + s * 0.05
        draw_list.add_rect_filled(
            wx - s * 0.04, blade_top,
            wx + s * 0.04, blade_bot, grey, s * 0.02)
        draw_list.add_line(
            wx, blade_top + s * 0.03,
            wx, blade_bot - s * 0.02,
            grey_dk, 2.0)
        draw_list.add_circle_filled(
            wx, blade_bot + s * 0.02, s * 0.10, gold, 16)
        draw_list.add_circle(
            wx, blade_bot + s * 0.02, s * 0.10, black, 16, 1.5)
        handle_top = blade_bot + s * 0.07
        handle_bot = wy + s * 0.40
        draw_list.add_rect_filled(
            wx - s * 0.05, handle_top,
            wx + s * 0.05, handle_bot,
            imgui.get_color_u32_rgba(0.85, 0.15, 0.15,
                                      0.98 * alpha), s * 0.02)
        for bi in range(4):
            by = handle_top + (handle_bot - handle_top) * (bi + 0.5) / 4
            draw_list.add_line(
                wx - s * 0.05, by - 2,
                wx + s * 0.05, by + 2, black, 1.8)
        draw_list.add_circle_filled(
            wx, handle_bot + s * 0.02, s * 0.04,
            brown_dk, 12)
    elif kind == "R":
        shaft_top = wy - s * 0.70
        shaft_bot = wy + s * 0.05
        draw_list.add_rect_filled(
            wx - s * 0.035, shaft_top,
            wx + s * 0.035, shaft_bot, grey, s * 0.015)
        draw_list.add_triangle_filled(
            wx - s * 0.035, shaft_top,
            wx + s * 0.035, shaft_top,
            wx, shaft_top - s * 0.09, grey)
        for sgn in (-1, +1):
            prong_base_x = wx + sgn * s * 0.02
            prong_base_y = wy - s * 0.22
            prong_mid_x = wx + sgn * s * 0.22
            prong_mid_y = wy - s * 0.18
            prong_tip_x = wx + sgn * s * 0.20
            prong_tip_y = wy - s * 0.48
            draw_list.add_line(
                prong_base_x, prong_base_y,
                prong_mid_x, prong_mid_y, grey, 5.0)
            draw_list.add_line(
                prong_mid_x, prong_mid_y,
                prong_tip_x, prong_tip_y, grey, 5.0)
            draw_list.add_triangle_filled(
                prong_tip_x - sgn * s * 0.025,
                prong_tip_y + s * 0.02,
                prong_tip_x + sgn * s * 0.025,
                prong_tip_y + s * 0.02,
                prong_tip_x + sgn * s * 0.01,
                prong_tip_y - s * 0.05, grey)
        grip_top = shaft_bot + s * 0.01
        grip_bot = wy + s * 0.32
        draw_list.add_rect_filled(
            wx - s * 0.055, grip_top,
            wx + s * 0.055, grip_bot,
            brown, s * 0.02)
        for bi in range(3):
            by = grip_top + (grip_bot - grip_top) * (bi + 0.5) / 3
            draw_list.add_line(
                wx - s * 0.055, by,
                wx + s * 0.055, by, brown_dk, 1.5)
        draw_list.add_circle_filled(
            wx, grip_bot + s * 0.02, s * 0.04,
            grey, 12)
    elif kind == "M":
        baton1_top = (wx - s * 0.15, wy - s * 0.55)
        baton1_bot = (wx - s * 0.10, wy - s * 0.10)
        baton2_top = (wx + s * 0.15, wy - s * 0.10)
        baton2_bot = (wx + s * 0.20, wy + s * 0.35)
        for (b_top, b_bot) in [(baton1_top, baton1_bot),
                                 (baton2_top, baton2_bot)]:
            draw_list.add_line(
                b_top[0], b_top[1],
                b_bot[0], b_bot[1], brown, 9.0)
            draw_list.add_circle_filled(
                b_top[0], b_top[1], s * 0.035, brown_dk, 12)
            draw_list.add_circle_filled(
                b_bot[0], b_bot[1], s * 0.035, brown_dk, 12)
            mid_x = (b_top[0] + b_bot[0]) * 0.5
            mid_y = (b_top[1] + b_bot[1]) * 0.5
            draw_list.add_circle_filled(
                mid_x, mid_y, s * 0.04, black, 10)
        chain_pts = [
            (baton1_bot[0], baton1_bot[1]),
            ((baton1_bot[0] + baton2_top[0]) * 0.5 - s * 0.02,
             (baton1_bot[1] + baton2_top[1]) * 0.5 - s * 0.01),
            ((baton1_bot[0] + baton2_top[0]) * 0.5 + s * 0.02,
             (baton1_bot[1] + baton2_top[1]) * 0.5 + s * 0.01),
            (baton2_top[0], baton2_top[1]),
        ]
        for (lx, ly) in chain_pts:
            draw_list.add_circle(
                lx, ly, s * 0.035, grey, 10, 2.2)
    elif kind == "D":
        pole_top = (wx - s * 0.35, wy - s * 0.75)
        pole_bot = (wx + s * 0.35, wy + s * 0.55)
        draw_list.add_line(
            pole_top[0], pole_top[1],
            pole_bot[0], pole_bot[1],
            brown, s * 0.10)
        for f in (0.08, 0.92):
            gx = pole_top[0] + (pole_bot[0] - pole_top[0]) * f
            gy = pole_top[1] + (pole_bot[1] - pole_top[1]) * f
            dx_ = pole_bot[0] - pole_top[0]
            dy_ = pole_bot[1] - pole_top[1]
            dlen = math.hypot(dx_, dy_)
            perp_x = -dy_ / dlen
            perp_y = dx_ / dlen
            draw_list.add_line(
                gx - perp_x * s * 0.06, gy - perp_y * s * 0.06,
                gx + perp_x * s * 0.06, gy + perp_y * s * 0.06,
                brown_dk, 5.0)
        for (px, py) in (pole_top, pole_bot):
            draw_list.add_circle_filled(
                px, py, s * 0.045, brown_dk, 12)


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height
    t = current_time
    # Sky gradient (dark urban orange sodium glow)
    for i in range(30):
        f = i / 30
        y0 = f * h
        y1 = (f + 1 / 30) * h + 1
        r_ = 0.10 + 0.05 * f
        g_ = 0.05 + 0.02 * f
        b_ = 0.08 + 0.02 * f
        draw_list.add_rect_filled(
            0, y0, w, y1, imgui.get_color_u32_rgba(r_, g_, b_, alpha))
    floor_y = h * 0.72
    draw_list.add_rect_filled(
        0, floor_y, w, h,
        imgui.get_color_u32_rgba(0.05, 0.04, 0.04, alpha))
    # Brick walls: repeating brick pattern on left and right sides
    wall_w = w * 0.18
    for side in (0, 1):
        bx0 = 0 if side == 0 else w - wall_w
        draw_list.add_rect_filled(
            bx0, 0, bx0 + wall_w, floor_y,
            imgui.get_color_u32_rgba(0.35, 0.15, 0.10, alpha))
        brick_w = 36
        brick_h = 14
        for row in range(int(floor_y / brick_h) + 1):
            oy = row * brick_h
            row_off = (brick_w / 2) if (row % 2) else 0
            for col in range(int(wall_w / brick_w) + 2):
                ox = bx0 + col * brick_w - row_off
                draw_list.add_rect(
                    ox, oy, ox + brick_w, oy + brick_h,
                    imgui.get_color_u32_rgba(0.18, 0.08, 0.05,
                                              0.8 * alpha), 0, 0, 1.2)
                draw_list.add_line(
                    ox, oy, ox + brick_w, oy,
                    imgui.get_color_u32_rgba(0.50, 0.22, 0.14,
                                              0.6 * alpha), 1.0)
    # Fire escape zigzag on the left wall
    esc_x = wall_w * 0.85
    for i in range(5):
        yy = floor_y - i * 70 - 30
        draw_list.add_rect_filled(
            esc_x, yy, esc_x + 40, yy + 4,
            imgui.get_color_u32_rgba(0.15, 0.12, 0.10, alpha))
        draw_list.add_line(
            esc_x + (40 if i % 2 == 0 else 0), yy,
            esc_x + (0 if i % 2 == 0 else 40), yy - 60,
            imgui.get_color_u32_rgba(0.15, 0.12, 0.10, alpha), 2.5)
    # Sewer cover
    sewer_cx = w * 0.5
    sewer_cy = floor_y + (h - floor_y) * 0.55
    sewer_r = min(w, h) * 0.08
    draw_list.add_circle_filled(
        sewer_cx, sewer_cy, sewer_r,
        imgui.get_color_u32_rgba(0.18, 0.14, 0.10, alpha), 32)
    draw_list.add_circle(
        sewer_cx, sewer_cy, sewer_r,
        imgui.get_color_u32_rgba(0.08, 0.06, 0.04, alpha), 32, 2.5)
    for i in range(-2, 3):
        for j in range(-2, 3):
            hx = sewer_cx + i * sewer_r * 0.30
            hy = sewer_cy + j * sewer_r * 0.30
            if (hx - sewer_cx) ** 2 + (hy - sewer_cy) ** 2 <= (sewer_r * 0.85) ** 2:
                draw_list.add_circle(
                    hx, hy, sewer_r * 0.10,
                    imgui.get_color_u32_rgba(0.08, 0.06, 0.04,
                                              0.8 * alpha),
                    12, 1.2)

    # Four turtles floating across. Index _HERO_IDX is the FunGen ninja —
    # its head is drawn by the main loop (the logo); we skip the default
    # head circle here but still paint its bandanna tails + weapon so the
    # silhouette still reads as a turtle. Override its bandanna color to
    # cyan to distinguish it from the other three.
    head_sz = max(68, int(min(w, h) * 0.11))
    weapon_s = head_sz * 1.7
    hero_color = (0.30, 0.85, 1.00)  # cyan — FunGen brand
    for i in range(len(_TURTLES)):
        col, initial, _speed, _ph, _y_rel = _TURTLES[i]
        bx_c, by_c = _turtle_pose(i, t, w, h)
        if i == _HERO_IDX:
            # Bandanna tails streaming behind the logo-head. The strap +
            # knot are drawn on top of the logo in render_fg.
            band_col = imgui.get_color_u32_rgba(*hero_color, 0.97 * alpha)
            tail_root_x = bx_c - head_sz * 0.55
            draw_list.add_triangle_filled(
                tail_root_x, by_c - head_sz * 0.10,
                tail_root_x - head_sz * 0.90, by_c + head_sz * 0.08,
                tail_root_x, by_c + head_sz * 0.05, band_col)
            draw_list.add_triangle_filled(
                tail_root_x + 4, by_c + head_sz * 0.02,
                tail_root_x - head_sz * 0.80, by_c + head_sz * 0.28,
                tail_root_x + 4, by_c + head_sz * 0.18, band_col)
            # Stash the head position so render_fg can align the strap.
            splash._tmnt_hero_head = (bx_c, by_c, head_sz)
        else:
            _draw_turtle_head(draw_list, bx_c, by_c, head_sz, col, alpha)
        _draw_turtle_weapon(draw_list, bx_c, by_c, weapon_s, initial,
                            t, alpha)

    # Pizza slice flying across diagonally
    pz_cycle = 7.0
    pz_t = (t % pz_cycle) / pz_cycle
    if pz_t < 0.60:
        f = pz_t / 0.60
        px = -60 + f * (w + 120)
        py = h * 0.18 + f * (h * 0.45)
        pr = 70
        rot = t * 4
        c, s_ = math.cos(rot), math.sin(rot)
        def _rot(x, y_):
            return (px + x * c - y_ * s_, py + x * s_ + y_ * c)
        crust_pts = [(0, -pr * 1.08),
                      (pr * 1.02, pr * 0.58),
                      (-pr * 1.02, pr * 0.58)]
        rc = [_rot(x, y_) for (x, y_) in crust_pts]
        draw_list.add_triangle_filled(
            rc[0][0], rc[0][1], rc[1][0], rc[1][1],
            rc[2][0], rc[2][1],
            imgui.get_color_u32_rgba(0.70, 0.40, 0.10,
                                      0.98 * alpha))
        cheese_pts = [(0, -pr), (pr * 0.92, pr * 0.52),
                       (-pr * 0.92, pr * 0.52)]
        rch = [_rot(x, y_) for (x, y_) in cheese_pts]
        draw_list.add_triangle_filled(
            rch[0][0], rch[0][1], rch[1][0], rch[1][1],
            rch[2][0], rch[2][1],
            imgui.get_color_u32_rgba(0.98, 0.80, 0.25,
                                      0.98 * alpha))
        peps = [(0, -pr * 0.35),
                (pr * 0.30, pr * 0.08),
                (-pr * 0.30, pr * 0.08),
                (pr * 0.12, pr * 0.30),
                (-pr * 0.15, pr * 0.30)]
        for (pox, poy) in peps:
            rpx, rpy = _rot(pox, poy)
            draw_list.add_circle_filled(
                rpx, rpy, pr * 0.13,
                imgui.get_color_u32_rgba(0.85, 0.12, 0.12,
                                          0.98 * alpha), 14)
            draw_list.add_circle(
                rpx, rpy, pr * 0.13,
                imgui.get_color_u32_rgba(0.50, 0.05, 0.05,
                                          0.85 * alpha), 14, 1.5)
        for bi in range(6):
            fb = bi / 5
            bump_x = -pr * 0.95 + fb * pr * 1.9
            bump_y = pr * 0.55
            rbx, rby = _rot(bump_x, bump_y)
            draw_list.add_circle_filled(
                rbx, rby, pr * 0.08,
                imgui.get_color_u32_rgba(0.80, 0.48, 0.15,
                                          0.95 * alpha), 10)


def render_fg(splash, logo_x, logo_y, logo_size, laser_time):
    """Cyan ninja bandanna strap across the logo-head, drawn after the
    logo texture so it lands on top like a real headband."""
    draw_list = imgui.get_window_draw_list()
    cyan = imgui.get_color_u32_rgba(0.30, 0.85, 1.0, 0.95)
    cyan_dk = imgui.get_color_u32_rgba(0.10, 0.45, 0.70, 0.95)
    # Strap across the upper-middle of the logo (forehead bandanna),
    # extending past the sides to match the other turtles' bandanna width.
    strap_y_top = logo_y + logo_size * 0.24
    strap_y_bot = logo_y + logo_size * 0.38
    strap_x_l = logo_x - logo_size * 0.10
    strap_x_r = logo_x + logo_size * 1.10
    draw_list.add_rect_filled(
        strap_x_l, strap_y_top, strap_x_r, strap_y_bot, cyan, 2)
    draw_list.add_rect_filled(
        strap_x_l, strap_y_bot - 2, strap_x_r, strap_y_bot + 1,
        cyan_dk, 0)
    # Knot on the left where the tails emerge
    knot_x = strap_x_l + 2
    knot_y = (strap_y_top + strap_y_bot) / 2
    draw_list.add_circle_filled(
        knot_x, knot_y, logo_size * 0.055, cyan, 14)
    draw_list.add_circle(
        knot_x, knot_y, logo_size * 0.055, cyan_dk, 14, 1.4)

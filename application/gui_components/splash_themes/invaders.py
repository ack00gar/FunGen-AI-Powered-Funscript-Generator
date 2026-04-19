"""Space Invaders: starfield, marching alien fleet, UFO, alien return-fire.

The logo itself acts as the player ship — ``render_fg`` fires bullets from the
logo's position, resolves hits against the fleet, and spawns explosions.
``render_bg`` publishes ``splash._inv_fleet_state`` so ``render_fg`` can do
collision detection without reimplementing the march math.
"""

import math
import random

import glfw
import imgui


# Pixel-bitmap sprites ('#' = on). Two frames per enemy for march animation.
_SQUID = (
    [
        "...##...",
        "..####..",
        ".######.",
        "##.##.##",
        "########",
        ".#.##.#.",
        "#.#..#.#",
        ".#.##.#.",
    ],
    [
        "...##...",
        "..####..",
        ".######.",
        "##.##.##",
        "########",
        ".######.",
        "#.#..#.#",
        "#.#..#.#",
    ],
)

_CRAB = (
    [
        "..#.....#..",
        "...#...#...",
        "..#######..",
        ".##.###.##.",
        "###########",
        "#.#######.#",
        "#.#.....#.#",
        "...##.##...",
    ],
    [
        "..#.....#..",
        "#..#...#..#",
        "#.#######.#",
        "###.###.###",
        "###########",
        ".#########.",
        "..#.....#..",
        ".#.......#.",
    ],
)

_OCTO = (
    [
        "....####....",
        ".##########.",
        "############",
        "###..##..###",
        "############",
        "...##..##...",
        "..##.##.##..",
        "##........##",
    ],
    [
        "....####....",
        ".##########.",
        "############",
        "###..##..###",
        "############",
        "..###..###..",
        ".##..##..##.",
        "..##....##..",
    ],
)

_UFO = [
    ".....#######.....",
    "....#########....",
    "...###########...",
    "..##.##.##.##.##.",
    ".###############.",
    "###.##.###.##.###",
    "..###.......###..",
    "....#.......#....",
]


def _blit(draw_list, sprite, x, y, sz, r, g, b, aa):
    col = imgui.get_color_u32_rgba(r, g, b, aa)
    for pr, row_str in enumerate(sprite):
        py = y + pr * sz
        for pc, ch in enumerate(row_str):
            if ch == '#':
                sx2 = x + pc * sz
                draw_list.add_rect_filled(sx2, py, sx2 + sz, py + sz, col)


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height

    if not hasattr(splash, '_inv_stars'):
        splash._inv_stars = [(random.uniform(0, 1), random.uniform(0, 1),
                              random.uniform(0.3, 1.0), random.uniform(0.5, 2.0))
                             for _ in range(90)]
        splash._inv_alive = [[True] * 11 for _ in range(5)]
        splash._inv_respawn_time = current_time

    # Auto-respawn: if fleet wiped out, wait 1.2s then bring it back
    alive_count = sum(1 for row in splash._inv_alive for v in row if v)
    if alive_count == 0 and current_time - splash._inv_respawn_time > 1.2:
        splash._inv_alive = [[True] * 11 for _ in range(5)]
        splash._inv_respawn_time = current_time
    if alive_count > 0:
        splash._inv_respawn_time = current_time

    # 1. Starfield
    for sx, sy, brightness, twinkle in splash._inv_stars:
        a = alpha * brightness * (0.5 + 0.5 * math.sin(current_time * twinkle + sx * 12))
        size = 1.0 + brightness * 0.9
        draw_list.add_circle_filled(sx * w, sy * h, size,
                                    imgui.get_color_u32_rgba(1.0, 1.0, 1.0, a))

    # 2. Scoreboard (top strip) — "score" tracks kill count for fun
    imgui.set_window_font_scale(1.0)
    score_a = alpha * 0.85
    score_col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, score_a)
    lbl_y = 12
    kill_count = getattr(splash, '_inv_kill_count', 0)
    score_val = kill_count * 30
    hi_val = max(9990, score_val + 1000)
    labels = [("SCORE<1>", f"{score_val:04d}", 0.22),
              ("HI-SCORE", f"{hi_val:04d}", 0.50),
              ("CREDIT", "01", 0.78)]
    for lbl, val, fx in labels:
        lbl_size = imgui.calc_text_size(lbl)
        val_size = imgui.calc_text_size(val)
        lx = fx * w - lbl_size[0] / 2
        vx = fx * w - val_size[0] / 2
        draw_list.add_text(lx, lbl_y, score_col, lbl)
        draw_list.add_text(vx, lbl_y + lbl_size[1] + 4,
                           imgui.get_color_u32_rgba(1.0, 0.3, 0.3, score_a), val)

    # 3. Alien fleet — 5 rows × 11 cols, fast march
    px = max(3, int(min(w, h) / 280))
    cell_w = 14 * px
    cell_h = 11 * px
    cols = 11
    rows = 5
    fleet_w = cols * cell_w
    margin = 60
    range_x = max(40, w - 2 * margin - fleet_w)

    FLEET_PERIOD = 8.0
    frame = int(current_time * 1.8) % 2
    phase = (current_time % FLEET_PERIOD) / FLEET_PERIOD
    tri = 2 * abs(phase - 0.5)
    fleet_x = margin + tri * range_x
    fleet_x = int(fleet_x / px) * px
    bounces = int(current_time / (FLEET_PERIOD / 2))
    descent_step = 14
    max_descent = 180
    fleet_y_top = 70 + (bounces * descent_step) % (max_descent + 1)

    row_sprites = [_SQUID, _CRAB, _CRAB, _OCTO, _OCTO]
    row_colors = [(1.0, 0.4, 0.85), (0.4, 1.0, 1.0), (0.4, 1.0, 1.0),
                  (1.0, 1.0, 0.35), (0.4, 1.0, 0.45)]

    for ri in range(rows):
        sprite = row_sprites[ri][frame]
        sprite_w_px = len(sprite[0]) * px
        offset_x = (cell_w - sprite_w_px) // 2
        r, g, b = row_colors[ri]
        ay = fleet_y_top + ri * cell_h
        for ci in range(cols):
            if not splash._inv_alive[ri][ci]:
                continue
            ax = fleet_x + ci * cell_w + offset_x
            _blit(draw_list, sprite, ax, ay, px, r, g, b, alpha)

    # Publish fleet state for the fg method to do hit detection
    splash._inv_fleet_state = {
        'x': fleet_x, 'y_top': fleet_y_top,
        'cell_w': cell_w, 'cell_h': cell_h,
        'cols': cols, 'rows': rows, 'px': px,
    }

    # 4. UFO crosses the top every ~7s
    ufo_period = 7.0
    ufo_phase = current_time % ufo_period
    ufo_traverse = 2.4
    if ufo_phase < ufo_traverse:
        ufo_t = ufo_phase / ufo_traverse
        ufo_w_px = 17 * px
        ufo_x = -ufo_w_px + ufo_t * (w + ufo_w_px)
        ufo_y = 26
        _blit(draw_list, _UFO, ufo_x, ufo_y, px, 1.0, 0.25, 0.25, alpha)

    # 5. Alien return fire
    for bi, t_offset in enumerate((0.0, 1.3)):
        ap = (current_time + t_offset) % 2.4
        if ap > 1.1:
            continue
        at = ap / 1.1
        col_pick = (int(current_time / 2.4) * 7 + bi * 5) % cols
        src_x = fleet_x + col_pick * cell_w + cell_w / 2
        src_y = fleet_y_top + rows * cell_h
        end_y = h - 30
        ab_y = src_y + at * (end_y - src_y)
        zz = math.sin(at * 14) * px
        col = imgui.get_color_u32_rgba(1.0, 0.8, 0.3, alpha * 0.9)
        draw_list.add_rect_filled(src_x + zz - px / 2, ab_y,
                                  src_x + zz + px / 2, ab_y + 3 * px, col)

    imgui.set_window_font_scale(1.0)


def render_fg(splash, logo_x, logo_y, logo_size, laser_time):
    """Invaders foreground: fire bullets from the (moving) logo, resolve hits,
    spawn explosions. The logo IS the player ship."""
    draw_list = imgui.get_window_draw_list()
    window_width, window_height = glfw.get_window_size(splash.window)
    w, h = window_width, window_height
    current_time = laser_time + 0.3

    if not hasattr(splash, '_inv_bullets'):
        splash._inv_bullets = []
        splash._inv_explosions = []
        splash._inv_last_fire_time = current_time - 10.0
        splash._inv_kill_count = 0

    fleet = getattr(splash, '_inv_fleet_state', None)
    if fleet is None:
        return

    px = fleet['px']

    FIRE_INTERVAL = 0.75
    if current_time - splash._inv_last_fire_time > FIRE_INTERVAL:
        splash._inv_bullets.append({
            't0': current_time,
            'x': logo_x + logo_size / 2,
            'y0': logo_y + logo_size * 0.18,
        })
        splash._inv_last_fire_time = current_time

    BULLET_SPEED = h * 0.50
    fx = fleet['x']
    fy = fleet['y_top']
    cw = fleet['cell_w']
    ch = fleet['cell_h']
    cols = fleet['cols']
    rows = fleet['rows']

    kept = []
    for b in splash._inv_bullets:
        dt = current_time - b['t0']
        by = b['y0'] - dt * BULLET_SPEED
        if by < 40:
            continue

        hit = False
        rel_x = b['x'] - fx
        if 0 <= rel_x < cols * cw:
            ci = int(rel_x / cw)
            for ri in range(rows - 1, -1, -1):
                row_top = fy + ri * ch
                row_bot = row_top + ch
                if row_top <= by <= row_bot and splash._inv_alive[ri][ci]:
                    splash._inv_alive[ri][ci] = False
                    splash._inv_kill_count = getattr(splash, '_inv_kill_count', 0) + 1
                    splash._inv_explosions.append({
                        't0': current_time,
                        'x': fx + ci * cw + cw / 2,
                        'y': row_top + ch / 2,
                        'r0': cw * 0.5,
                    })
                    hit = True
                    break
        if hit:
            continue

        beam_col = imgui.get_color_u32_rgba(1.0, 1.0, 1.0, 1.0)
        glow_col = imgui.get_color_u32_rgba(0.6, 1.0, 0.6, 0.6)
        beam_len = 5 * px
        draw_list.add_rect_filled(b['x'] - px, by,
                                  b['x'] + px, by + beam_len, glow_col)
        draw_list.add_rect_filled(b['x'] - px / 2, by,
                                  b['x'] + px / 2, by + beam_len, beam_col)
        kept.append(b)
    splash._inv_bullets = kept

    # Draw + cull explosions
    EXPL_DUR = 0.45
    kept_expl = []
    for ex in splash._inv_explosions:
        elapsed = current_time - ex['t0']
        if elapsed > EXPL_DUR:
            continue
        t = elapsed / EXPL_DUR
        a = 1.0 - t
        r = ex['r0'] * (1.0 + t * 2.2)
        draw_list.add_circle_filled(ex['x'], ex['y'], r,
                                    imgui.get_color_u32_rgba(1.0, 0.4, 0.1, a * 0.28))
        draw_list.add_circle_filled(ex['x'], ex['y'], r * 0.6,
                                    imgui.get_color_u32_rgba(1.0, 0.8, 0.3, a * 0.55))
        draw_list.add_circle_filled(ex['x'], ex['y'], r * 0.25,
                                    imgui.get_color_u32_rgba(1.0, 1.0, 0.95, a * 0.95))
        n_sparks = 8
        for i in range(n_sparks):
            ang = i * (math.tau / n_sparks) + t * 2.0
            sp_len = ex['r0'] * (0.8 + t * 2.0)
            sx = ex['x'] + math.cos(ang) * sp_len
            sy = ex['y'] + math.sin(ang) * sp_len
            draw_list.add_line(ex['x'], ex['y'], sx, sy,
                               imgui.get_color_u32_rgba(1.0, 0.75, 0.2, a * 0.8),
                               max(1.5, px * 0.7))
        kept_expl.append(ex)
    splash._inv_explosions = kept_expl

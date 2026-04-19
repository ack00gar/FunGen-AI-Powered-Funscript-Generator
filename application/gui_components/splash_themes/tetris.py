"""Tetris sim: 12×20 playfield, falling piece auto-rotates every 0.5s,
collides/locks, full rows flash white and collapse, stack resets on top-out."""

import math
import random

import imgui


_TETRO_ROTS = {
    'I': [
        [(0, 1), (1, 1), (2, 1), (3, 1)],
        [(2, 0), (2, 1), (2, 2), (2, 3)],
        [(0, 2), (1, 2), (2, 2), (3, 2)],
        [(1, 0), (1, 1), (1, 2), (1, 3)],
    ],
    'O': [[(1, 0), (2, 0), (1, 1), (2, 1)]] * 4,
    'T': [
        [(0, 1), (1, 1), (2, 1), (1, 0)],
        [(1, 0), (1, 1), (1, 2), (2, 1)],
        [(0, 1), (1, 1), (2, 1), (1, 2)],
        [(1, 0), (1, 1), (1, 2), (0, 1)],
    ],
    'S': [
        [(1, 1), (2, 1), (0, 2), (1, 2)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
        [(1, 1), (2, 1), (0, 2), (1, 2)],
        [(1, 0), (1, 1), (2, 1), (2, 2)],
    ],
    'Z': [
        [(0, 1), (1, 1), (1, 2), (2, 2)],
        [(2, 0), (1, 1), (2, 1), (1, 2)],
        [(0, 1), (1, 1), (1, 2), (2, 2)],
        [(2, 0), (1, 1), (2, 1), (1, 2)],
    ],
    'L': [
        [(0, 1), (1, 1), (2, 1), (2, 0)],
        [(1, 0), (1, 1), (1, 2), (2, 2)],
        [(0, 1), (1, 1), (2, 1), (0, 2)],
        [(0, 0), (1, 0), (1, 1), (1, 2)],
    ],
    'J': [
        [(0, 0), (0, 1), (1, 1), (2, 1)],
        [(1, 0), (2, 0), (1, 1), (1, 2)],
        [(0, 1), (1, 1), (2, 1), (2, 2)],
        [(1, 0), (1, 1), (0, 2), (1, 2)],
    ],
    # The FunGen piece — same shape as O (2x2 square), but rendered with
    # the FunGen logo as its face instead of a flat color.
    'F': [[(1, 0), (2, 0), (1, 1), (2, 1)]] * 4,
}

_TETRO_COLORS = {
    'I': (0.25, 0.95, 1.00),
    'O': (1.00, 0.95, 0.25),
    'T': (0.80, 0.35, 1.00),
    'S': (0.30, 1.00, 0.35),
    'Z': (1.00, 0.35, 0.35),
    'L': (1.00, 0.65, 0.25),
    'J': (0.25, 0.50, 1.00),
    'F': (0.95, 0.95, 0.95),   # placeholder, never actually drawn
}

# Regular tetrominoes — excludes F, which is spawned on a fixed cadence.
_TETRO_TYPES = [t for t in _TETRO_ROTS.keys() if t != 'F']

# Sentinel stored in state['grid'] for cells that belong to a locked F-tile.
# Collision detection uses these exactly like normal cells; rendering skips
# them because the 2x2 logo texture is painted on top instead.
_LOGO_MARKER = (-1.0, -1.0, -1.0)

# Every Nth spawned piece is a FunGen logo piece.
_LOGO_EVERY = 3


def place_logo(splash, width, height, current_time):
    """Tetris draws its own logo tiles (falling + all locked 2x2 logo
    pieces) via draw_list.add_image, so the shared splash loop's single
    logo image would be redundant. Send it far off-screen to hide it."""
    return -9999.0, -9999.0, 1.0, 0.0


def render_bg(splash, draw_list, window_width, window_height, current_time, alpha):
    w, h = window_width, window_height
    t = current_time

    COLS, ROWS = 12, 20
    cell = min((h - 120) // ROWS, w // 24)
    pf_w = COLS * cell
    pf_h = ROWS * cell
    pf_x = (w - pf_w) // 2
    pf_y = (h - pf_h) // 2

    state = getattr(splash, '_tetris_state', None)
    if state is None or 'grid' not in state:
        state = {
            'rng': random.Random(17),
            'grid': {},
            'piece': None,
            'last_t': t,
            'fall_speed': 12.0,
            'rotate_timer': 0.0,
            'flash_end': 0.0,
            'flash_rows': [],
            'piece_count': 0,
            'logo_tiles': [],  # list of (gx, gy) top-left of locked F tiles
        }
        splash._tetris_state = state

    def spawn():
        state['piece_count'] += 1
        # Every Nth piece is an F (FunGen) logo piece; rest are random.
        if state['piece_count'] % _LOGO_EVERY == 0:
            typ = 'F'
        else:
            typ = state['rng'].choice(_TETRO_TYPES)
        return {
            'type': typ, 'rot': 0,
            'x': (COLS - 4) // 2,
            'y': -1.0,
            'color': _TETRO_COLORS[typ],
        }

    def cells_of(piece):
        return _TETRO_ROTS[piece['type']][piece['rot']]

    def collides(piece, px, py, rot):
        for (cc, cr) in _TETRO_ROTS[piece['type']][rot]:
            gx = px + cc
            gy = py + cr
            if gx < 0 or gx >= COLS or gy >= ROWS:
                return True
            if gy >= 0 and (gx, gy) in state['grid']:
                return True
        return False

    if state['piece'] is None:
        state['piece'] = spawn()

    dt = max(0.0, min(0.2, t - state['last_t']))
    state['last_t'] = t
    flashing = t < state['flash_end']

    if not flashing:
        piece = state['piece']
        state['rotate_timer'] += dt
        if state['rotate_timer'] > 0.5:
            state['rotate_timer'] = 0.0
            new_rot = (piece['rot'] + 1) % 4
            if not collides(piece, piece['x'], int(piece['y']), new_rot):
                piece['rot'] = new_rot
        new_y = piece['y'] + state['fall_speed'] * dt
        if collides(piece, piece['x'], int(new_y), piece['rot']):
            locked_y = max(0, int(piece['y']))
            is_logo = piece['type'] == 'F'
            for (cc, cr) in cells_of(piece):
                gx = piece['x'] + cc
                gy = locked_y + cr
                if 0 <= gx < COLS and 0 <= gy < ROWS:
                    state['grid'][(gx, gy)] = (
                        _LOGO_MARKER if is_logo else piece['color'])
            if is_logo:
                # Record top-left of the 2x2 logo tile for rendering.
                # O/F rotation has cells at (1,0)(2,0)(1,1)(2,1) → top-left
                # in world coords is (piece.x + 1, locked_y + 0).
                state['logo_tiles'].append((piece['x'] + 1, locked_y))
            full = [r for r in range(ROWS)
                    if all((c, r) in state['grid'] for c in range(COLS))]
            if full:
                state['flash_rows'] = full
                state['flash_end'] = t + 0.35
            state['piece'] = spawn()
            p2 = state['piece']
            if collides(p2, p2['x'], max(0, int(p2['y'])), p2['rot']):
                state['grid'] = {}
                state['logo_tiles'] = []
        else:
            piece['y'] = new_y

    if state['flash_rows'] and t >= state['flash_end']:
        for row_to_clear in sorted(state['flash_rows']):
            new_grid = {}
            for (c, r), col in state['grid'].items():
                if r == row_to_clear:
                    continue
                if r < row_to_clear:
                    new_grid[(c, r + 1)] = col
                else:
                    new_grid[(c, r)] = col
            state['grid'] = new_grid
            # Apply same shift/remove rules to tracked logo tiles. A tile
            # occupies rows (ly, ly+1); any clear touching either row
            # destroys it.
            new_tiles = []
            for (lx, ly) in state['logo_tiles']:
                if ly == row_to_clear or (ly + 1) == row_to_clear:
                    continue
                if (ly + 1) < row_to_clear:
                    new_tiles.append((lx, ly + 1))
                else:
                    new_tiles.append((lx, ly))
            state['logo_tiles'] = new_tiles
        state['flash_rows'] = []

    # --- Draw playfield ---
    # Dark-blue backdrop gradient
    for i in range(32):
        f = i / 32
        y0 = f * h
        y1 = (f + 1 / 32) * h + 1
        r_ = 0.02 + 0.03 * f
        g_ = 0.03 + 0.01 * (1 - f)
        b_ = 0.12 * (1 - f) + 0.18 * f
        draw_list.add_rect_filled(
            0, y0, w, y1, imgui.get_color_u32_rgba(r_, g_, b_, alpha))

    draw_list.add_rect_filled(
        pf_x, pf_y, pf_x + pf_w, pf_y + pf_h,
        imgui.get_color_u32_rgba(0.04, 0.03, 0.10, alpha))
    draw_list.add_rect(
        pf_x - 3, pf_y - 3, pf_x + pf_w + 3, pf_y + pf_h + 3,
        imgui.get_color_u32_rgba(0.55, 0.35, 0.85, 0.9 * alpha),
        0, thickness=3)
    for c in range(COLS + 1):
        draw_list.add_line(
            pf_x + c * cell, pf_y, pf_x + c * cell, pf_y + pf_h,
            imgui.get_color_u32_rgba(0.2, 0.22, 0.4, 0.25 * alpha), 1)
    for r in range(ROWS + 1):
        draw_list.add_line(
            pf_x, pf_y + r * cell, pf_x + pf_w, pf_y + r * cell,
            imgui.get_color_u32_rgba(0.2, 0.22, 0.4, 0.25 * alpha), 1)

    def draw_cell(gc, gr, color, a=1.0):
        x0 = pf_x + gc * cell
        y0 = pf_y + gr * cell
        draw_list.add_rect_filled(
            x0, y0, x0 + cell - 1, y0 + cell - 1,
            imgui.get_color_u32_rgba(*color, a * alpha), 3)
        draw_list.add_line(
            x0, y0, x0 + cell - 1, y0,
            imgui.get_color_u32_rgba(1, 1, 1, 0.45 * a * alpha), 1.5)
        draw_list.add_line(
            x0, y0, x0, y0 + cell - 1,
            imgui.get_color_u32_rgba(1, 1, 1, 0.45 * a * alpha), 1.5)
        draw_list.add_line(
            x0 + cell - 1, y0, x0 + cell - 1, y0 + cell - 1,
            imgui.get_color_u32_rgba(0, 0, 0, 0.4 * a * alpha), 1.5)
        draw_list.add_line(
            x0, y0 + cell - 1, x0 + cell - 1, y0 + cell - 1,
            imgui.get_color_u32_rgba(0, 0, 0, 0.4 * a * alpha), 1.5)

    flash_strobe = 0.5 + 0.5 * math.sin(t * 60) if flashing else 1.0
    for (gc, gr), col in state['grid'].items():
        if col == _LOGO_MARKER:
            # Part of a locked 2x2 logo tile — rendered via logo_tiles
            # below, with optional flash overlay if the row is clearing.
            continue
        if flashing and gr in state['flash_rows']:
            draw_cell(gc, gr, (1.0, 1.0, 1.0), flash_strobe)
        else:
            draw_cell(gc, gr, col)

    # Locked FunGen logo tiles — each 2x2 drawn as the logo texture.
    logo_tex = getattr(splash, 'logo_texture', None)
    if logo_tex is not None:
        for (lx, ly) in state['logo_tiles']:
            x0 = pf_x + lx * cell
            y0 = pf_y + ly * cell
            x1 = x0 + 2 * cell - 1
            y1 = y0 + 2 * cell - 1
            # White flash overlay if either of the tile's rows is clearing.
            if flashing and (ly in state['flash_rows']
                             or (ly + 1) in state['flash_rows']):
                draw_list.add_rect_filled(
                    x0, y0, x1, y1,
                    imgui.get_color_u32_rgba(1, 1, 1, flash_strobe * alpha))
            else:
                draw_list.add_image(
                    logo_tex, (x0, y0), (x1, y1),
                    col=imgui.get_color_u32_rgba(1, 1, 1, alpha))

    if not flashing:
        piece = state['piece']
        py_int = int(piece['y'])
        py_frac = piece['y'] - py_int
        if piece['type'] == 'F':
            # Falling FunGen piece — single 2x2 logo instead of 4 cells.
            gc = piece['x'] + 1
            if logo_tex is not None:
                x0 = pf_x + gc * cell
                y0 = pf_y + (py_int + py_frac) * cell
                # Clip rendering when piece is still sliding in from the top
                if y0 + 2 * cell > pf_y:
                    draw_list.add_image(
                        logo_tex, (x0, y0),
                        (x0 + 2 * cell - 1, y0 + 2 * cell - 1),
                        col=imgui.get_color_u32_rgba(1, 1, 1, alpha))
        else:
            for (cc, cr) in cells_of(piece):
                gc = piece['x'] + cc
                gr = py_int + cr
                if gr < 0:
                    continue
                x0 = pf_x + gc * cell
                y0 = pf_y + (py_int + cr + py_frac) * cell
                draw_list.add_rect_filled(
                    x0, y0, x0 + cell - 1, y0 + cell - 1,
                    imgui.get_color_u32_rgba(*piece['color'], alpha), 3)
                draw_list.add_line(
                    x0, y0, x0 + cell - 1, y0,
                    imgui.get_color_u32_rgba(1, 1, 1, 0.5 * alpha), 1.5)
                draw_list.add_line(
                    x0, y0, x0, y0 + cell - 1,
                    imgui.get_color_u32_rgba(1, 1, 1, 0.5 * alpha), 1.5)

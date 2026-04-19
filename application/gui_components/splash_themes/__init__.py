"""Splash-screen theme renderers.

Each theme exports a ``render_bg(splash, draw_list, width, height, now, alpha)``
callable that draws the background animation for a single splash frame. Some
themes also export ``render_fg(splash, logo_x, logo_y, logo_size, laser_time)``
for a foreground overlay layered on top of the logo (laser eyes, lightsabers).

``splash`` is the live ``StandaloneSplashWindow`` instance. Themes read cached
per-session state from it (``splash._matrix_columns`` etc.) so the splash's
scene persists across frames without each theme having to know about its own
lifecycle.

Themes not yet extracted are handled by the method-name dispatch that falls
back to ``getattr(splash, "_render_<theme>_scene", None)`` in splash_screen.py.
Currently the only class-resident renderers are the big foreground laser-eye
and lightsaber-duel sequences for terminator / starwars.
"""

from . import (
    blade,
    breaking,
    bsod,
    clippy,
    et,
    invaders,
    mario,
    mars,
    matrix,
    pacman,
    sonic,
    starwars,
    terminator,
    tetris,
    tmnt,
    tron,
    xfiles,
)

_BG = {
    'matrix': matrix.render_bg,
    'terminator': terminator.render_bg,
    'tron': tron.render_bg,
    'starwars': starwars.render_bg,
    'breaking': breaking.render_bg,
    'invaders': invaders.render_bg,
    'mars': mars.render_bg,
    'clippy': clippy.render_bg,
    'tetris': tetris.render_bg,
    'pacman': pacman.render_bg,
    'blade': blade.render_bg,
    'bsod': bsod.render_bg,
    'sonic': sonic.render_bg,
    'xfiles': xfiles.render_bg,
    'tmnt': tmnt.render_bg,
    'et': et.render_bg,
    'mario': mario.render_bg,
}

_FG = {
    'invaders': invaders.render_fg,
    'mario': mario.render_fg,
    'et': et.render_fg,
    'pacman': pacman.render_fg,
    'tmnt': tmnt.render_fg,
}

# Themes that want to drive the logo's placement themselves (so the logo reads
# as a character's head, a ship, a block, etc. rather than a centered floater).
# Each callable returns (x, y, size, float_offset) — top-left of the logo box.
_PLACE = {
    'mario': mario.place_logo,
    'sonic': sonic.place_logo,
    'et': et.place_logo,
    'pacman': pacman.place_logo,
    'tetris': tetris.place_logo,
    'tmnt': tmnt.place_logo,
    'xfiles': xfiles.place_logo,
    'mars': mars.place_logo,
}


def render_bg(theme_name, splash, draw_list, width, height, now, alpha):
    fn = _BG.get(theme_name)
    if fn is not None:
        fn(splash, draw_list, width, height, now, alpha)
        return True
    return False


def render_fg(theme_name, splash, logo_x, logo_y, logo_size, laser_time):
    fn = _FG.get(theme_name)
    if fn is not None:
        fn(splash, logo_x, logo_y, logo_size, laser_time)
        return True
    return False


def place_logo(theme_name, splash, width, height, current_time):
    """Return (x, y, size, float_offset) for the logo, or None for default."""
    fn = _PLACE.get(theme_name)
    if fn is None:
        return None
    return fn(splash, width, height, current_time)

"""Render Nerd Font icon glyphs to RGBA buffers for use as UI icons.

Acts as a drop-in replacement for PNG emoji icons: the IconTextureManager
consults PNG_TO_GLYPH, and if an entry exists, renders the glyph via PIL
using the bundled Symbols Nerd Font instead of loading the PNG file.
Result: existing image_button call sites get monochrome Font Awesome /
Material / Codicons icons with zero call-site changes.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import numpy as np


_FONT_PATH = os.path.normpath(os.path.join(
    os.path.dirname(__file__), "..", "..",
    "assets", "fonts", "icons.ttf",
))


# Map UI icon filenames (as referenced by call sites) to Nerd Font codepoints.
# Empty string = no mapping, fall back to PNG.
PNG_TO_GLYPH = {
    # Playback
    "play.png":          "\uf04b",
    "pause.png":         "\uf04c",
    "stop.png":          "\uf04d",
    "next-frame.png":    "\uf051",
    "prev-frame.png":    "\uf048",
    "jump-start.png":    "\uf049",
    "jump-end.png":      "\uf050",
    "speed-realtime.png": "\uf017",   # clock
    "speed-slowmo.png":   "\uf251",   # hourglass-start
    "speed-max.png":      "\uf135",   # rocket
    "speaker-high.png":   "\uf028",
    "speaker-muted.png":  "\uf026",   # volume-off (FA4)

    # File / project
    "folder-open.png":   "\uf07c",
    "document-new.png":  "\uf15b",
    "save.png":          "\uf0c7",
    "save-as.png":       "\uf0c7",
    "export.png":        "\uf08e",
    "download.png":      "\uf019",
    "page-facing-up.png": "\uf15c",

    # Edit
    "edit.png":          "\uf044",
    "undo.png":          "\uf0e2",
    "redo.png":          "\uf01e",
    "trash.png":         "\uf1f8",
    "reset.png":         "\uf0e2",
    "plus-circle.png":   "\uf055",
    "check.png":         "\uf00c",
    "checkmark.png":     "\uf00c",

    # View / nav
    "zoom-in.png":       "\uf00e",
    "zoom-out.png":      "\uf010",
    "fullscreen.png":    "\uf065",
    "fullscreen-exit.png": "\uf066",
    "video-show.png":    "\uf03d",
    "video-hide.png":    "\uf070",

    # Status
    "warning.png":       "\uf071",
    "error.png":         "\uf057",

    # Tools / content
    "settings.png":      "\uf013",
    "wrench.png":        "\uf0ad",
    "robot.png":         "\uf188",   # bug (generic AI/bot)
    "gamepad.png":       "\uf11b",
    "flashlight.png":    "\uf0eb",
    "magic-wand.png":    "\uf0d0",
    "books.png":         "\uf02d",
    "chart-increasing.png": "\uf201",
    "counterclockwise-arrows.png": "\uf021",
    "satellite.png":     "\uf09e",   # rss (stream/broadcast)
    "energy-leaf.png":   "\uf06c",
    "user.png":          "\uf007",

    # Sidebar (control panel tabs)
    "sidebar-run.png":       "\uf135",   # rocket
    "sidebar-postproc.png":  "\uf0d0",   # magic-wand
    "sidebar-advanced.png":  "\uf013",   # gear
    "sidebar-device.png":    "\uf11b",   # gamepad
    "sidebar-stream.png":    "\uf09e",   # rss
    "sidebar-batch.png":     "\uf0c0",   # users
    "sidebar-metadata.png":  "\uf02b",   # tag
    "sidebar-configure.png": "\uf0ad",   # wrench
    "sidebar-subtitle.png":  "\uf27a",   # comment
}


_font_cache = {}  # size -> ImageFont


def _get_font(size: int):
    from PIL import ImageFont
    if size not in _font_cache:
        _font_cache[size] = ImageFont.truetype(_FONT_PATH, size)
    return _font_cache[size]


def render_glyph_rgba(
    glyph: str,
    size_px: int = 64,
    color: Tuple[int, int, int] = (230, 230, 232),
) -> Optional[np.ndarray]:
    """Render a single glyph to an RGBA uint8 numpy array (HxWx4, top-left origin)."""
    try:
        from PIL import Image, ImageDraw
    except Exception:
        return None
    if not os.path.exists(_FONT_PATH):
        return None
    # Render at 80% of canvas so glyph has margin; center it.
    font_size = int(size_px * 0.80)
    font = _get_font(font_size)
    img = Image.new("RGBA", (size_px, size_px), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    try:
        bbox = draw.textbbox((0, 0), glyph, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        x = (size_px - tw) // 2 - bbox[0]
        y = (size_px - th) // 2 - bbox[1]
    except Exception:
        x, y = size_px // 8, size_px // 8
    draw.text((x, y), glyph, font=font, fill=color + (255,))
    return np.asarray(img, dtype=np.uint8)


def glyph_for(icon_name: str) -> Optional[str]:
    g = PNG_TO_GLYPH.get(icon_name, "")
    return g if g else None

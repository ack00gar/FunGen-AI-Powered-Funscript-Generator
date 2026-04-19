"""VR stereo-panel / eye selection helper.

Setting: ``app_settings['vr_panel_selection']`` = 'left' | 'right' | 'full'.
``_rl`` layouts are handled here so downstream code never has to flip.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

LAYOUT_MONO = 'mono'
LAYOUT_SBS = 'sbs'
LAYOUT_TB = 'tb'

EYE_LEFT = 'left'
EYE_RIGHT = 'right'
EYE_FULL = 'full'

_VALID_EYES = (EYE_LEFT, EYE_RIGHT, EYE_FULL)


@dataclass(frozen=True)
class PanelRegion:
    """Crop rectangle in [0, 1] ratios. (0, 0, 1, 1) = full frame."""

    x: float
    y: float
    w: float
    h: float

    def is_full(self) -> bool:
        return (self.x == 0.0 and self.y == 0.0
                and self.w == 1.0 and self.h == 1.0)

    def pixel_rect(self, src_w: int, src_h: int) -> Tuple[int, int, int, int]:
        """Return integer (x, y, w, h) at the given source resolution."""
        cw = int(round(src_w * self.w))
        ch = int(round(src_h * self.h))
        cx = int(round(src_w * self.x))
        cy = int(round(src_h * self.y))
        return cx, cy, cw, ch

    def ffmpeg_crop(self, src_w: int, src_h: int) -> Optional[str]:
        """Return ``crop=W:H:X:Y`` for this region, or None if full-frame."""
        if self.is_full() or src_w <= 0 or src_h <= 0:
            return None
        cx, cy, cw, ch = self.pixel_rect(src_w, src_h)
        return f"crop={cw}:{ch}:{cx}:{cy}"


_FULL_REGION = PanelRegion(0.0, 0.0, 1.0, 1.0)


def layout_of(vr_input_format: Optional[str]) -> str:
    """Classify a vr_input_format string into one of the layout constants."""
    fmt = (vr_input_format or '').lower()
    if '_sbs' in fmt or '_lr' in fmt or '_rl' in fmt:
        return LAYOUT_SBS
    if '_tb' in fmt:
        return LAYOUT_TB
    return LAYOUT_MONO


def _is_reversed(vr_input_format: Optional[str]) -> bool:
    """True iff the FIRST half of the frame is actually the RIGHT eye."""
    return '_rl' in (vr_input_format or '').lower()


def normalize_eye(eye: Optional[str], default: str = EYE_LEFT) -> str:
    """Coerce a user-supplied eye value into one of the three constants."""
    if isinstance(eye, str):
        low = eye.strip().lower()
        if low in _VALID_EYES:
            return low
    return default


def resolve_eye(vr_input_format: Optional[str],
                eye: str = EYE_LEFT) -> PanelRegion:
    """Panel rectangle for the requested eye, with ``_rl`` reversal applied."""
    eye = normalize_eye(eye)
    layout = layout_of(vr_input_format)
    if layout == LAYOUT_MONO or eye == EYE_FULL:
        return _FULL_REGION
    if layout == LAYOUT_SBS:
        want_right_half = (eye == EYE_RIGHT)
        if _is_reversed(vr_input_format):
            want_right_half = not want_right_half
        return PanelRegion(0.5 if want_right_half else 0.0, 0.0, 0.5, 1.0)
    # LAYOUT_TB
    want_bottom_half = (eye == EYE_RIGHT)
    return PanelRegion(0.0, 0.5 if want_bottom_half else 0.0, 1.0, 0.5)


def shader_params(vr_input_format: Optional[str],
                  eye: str = EYE_LEFT) -> Tuple[str, bool]:
    """(stereo_format, use_right_eye) for VRDewarpShader.render_pass."""
    eye = normalize_eye(eye)
    layout = layout_of(vr_input_format)
    if layout == LAYOUT_MONO or eye == EYE_FULL:
        return LAYOUT_MONO, False
    want_right = (eye == EYE_RIGHT)
    if layout == LAYOUT_SBS and _is_reversed(vr_input_format):
        want_right = not want_right
    return layout, want_right


def read_setting(app_settings, default: str = EYE_LEFT) -> str:
    """Read ``vr_panel_selection`` from settings, with legacy ``vr_crop_panel`` fallback."""
    if app_settings is None:
        return default
    try:
        raw = app_settings.config.vr_display.panel_selection
    except Exception:
        raw = None
    if raw is not None and raw != "":
        return normalize_eye(raw, default=default)
    try:
        legacy = app_settings.get('vr_crop_panel', None)
    except Exception:
        legacy = None
    if isinstance(legacy, str):
        low = legacy.strip().lower()
        if low == 'second':
            return EYE_RIGHT
        if low == 'first':
            return EYE_LEFT
    return default

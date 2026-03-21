"""Non-blocking toast notification system for FunGen.

Renders stacked notifications in the top-right corner of the application window.
Notifications auto-dismiss after a configurable duration and fade out smoothly.

Usage:
    # From any component with access to app:
    app.notify("Export complete", "success")
    app.notify("Plugin failed: invalid parameters", "error")
    app.notify("Auto post-processing enabled", "info")
"""

import time
import threading
import imgui
from dataclasses import dataclass, field
from typing import List
from enum import Enum


class NotificationType(Enum):
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    INFO = "info"


# Left accent bar colors (RGBA)
_TYPE_COLORS = {
    NotificationType.SUCCESS: (0.2, 0.8, 0.3, 1.0),
    NotificationType.ERROR: (0.9, 0.25, 0.2, 1.0),
    NotificationType.WARNING: (0.95, 0.7, 0.1, 1.0),
    NotificationType.INFO: (0.3, 0.6, 0.95, 1.0),
}

_BG_COLOR = (0.15, 0.15, 0.18, 0.92)
_TEXT_COLOR = (0.9, 0.9, 0.9, 1.0)
_FADE_DURATION = 0.4
_MAX_VISIBLE = 4
_DEFAULT_DURATION = 4.0
_TOAST_WIDTH = 320
_TOAST_PADDING = 10
_ACCENT_WIDTH = 4
_MARGIN_TOP = 4  # Small gap below the y_offset passed by caller
_MARGIN_RIGHT = 12
_GAP = 6


@dataclass
class _Toast:
    message: str
    type: NotificationType
    duration: float
    created: float = field(default_factory=time.time)

    @property
    def age(self):
        return time.time() - self.created

    @property
    def alpha(self):
        remaining = self.duration - self.age
        if remaining <= 0:
            return 0.0
        if remaining < _FADE_DURATION:
            return remaining / _FADE_DURATION
        # Fade in
        if self.age < 0.15:
            return self.age / 0.15
        return 1.0

    @property
    def expired(self):
        return self.age >= self.duration


class NotificationManager:
    """Manages and renders toast notifications."""

    def __init__(self, app=None):
        self._toasts: List[_Toast] = []
        self._lock = threading.Lock()
        self._app = app

    def add(self, message: str, type_str: str = "info", duration: float = _DEFAULT_DURATION):
        """Add a notification. type_str: 'success', 'error', 'warning', 'info'."""
        # Check if toasts are disabled in settings
        if self._app and hasattr(self._app, 'app_settings'):
            if not self._app.app_settings.get("show_toast_notifications", True):
                return
        try:
            ntype = NotificationType(type_str)
        except ValueError:
            ntype = NotificationType.INFO

        # Error toasts stay longer
        if ntype == NotificationType.ERROR and duration == _DEFAULT_DURATION:
            duration = 6.0

        with self._lock:
            self._toasts.append(_Toast(message=message, type=ntype, duration=duration))
            while len(self._toasts) > _MAX_VISIBLE * 2:
                self._toasts.pop(0)

    def render(self, top_y_offset: float = 0):
        """Render all active toasts. top_y_offset is the Y position below toolbar."""
        with self._lock:
            self._toasts = [t for t in self._toasts if not t.expired]

        if not self._toasts:
            return

        viewport = imgui.get_main_viewport()
        if not viewport:
            return

        vp_x = viewport.pos[0]
        vp_y = viewport.pos[1]
        vp_w = viewport.size[0]

        draw_list = imgui.get_foreground_draw_list()
        y_offset = vp_y + top_y_offset + _MARGIN_TOP

        # Render newest at top (reversed order, take last N)
        visible = self._toasts[-_MAX_VISIBLE:]

        for toast in visible:
            alpha = toast.alpha
            if alpha <= 0.01:
                continue

            # Measure text
            text_size = imgui.calc_text_size(toast.message, wrap_width=_TOAST_WIDTH - _ACCENT_WIDTH - _TOAST_PADDING * 2)
            toast_h = max(32, text_size[1] + _TOAST_PADDING * 2)

            x = vp_x + vp_w - _TOAST_WIDTH - _MARGIN_RIGHT
            y = y_offset

            # Background
            bg = _BG_COLOR
            draw_list.add_rect_filled(
                x, y, x + _TOAST_WIDTH, y + toast_h,
                imgui.get_color_u32_rgba(bg[0], bg[1], bg[2], bg[3] * alpha),
                rounding=6.0
            )

            # Subtle border
            draw_list.add_rect(
                x, y, x + _TOAST_WIDTH, y + toast_h,
                imgui.get_color_u32_rgba(0.3, 0.3, 0.35, 0.5 * alpha),
                rounding=6.0
            )

            # Left accent bar
            accent = _TYPE_COLORS.get(toast.type, _TYPE_COLORS[NotificationType.INFO])
            draw_list.add_rect_filled(
                x, y, x + _ACCENT_WIDTH, y + toast_h,
                imgui.get_color_u32_rgba(accent[0], accent[1], accent[2], accent[3] * alpha),
                rounding=6.0,
                flags=imgui.DRAW_ROUND_CORNERS_LEFT
            )

            # Text (truncate to fit width)
            text_x = x + _ACCENT_WIDTH + _TOAST_PADDING
            text_y = y + (toast_h - text_size[1]) * 0.5
            max_chars = int((_TOAST_WIDTH - _ACCENT_WIDTH - _TOAST_PADDING * 2) / 7)  # ~7px per char
            display_msg = toast.message if len(toast.message) <= max_chars else toast.message[:max_chars - 3] + "..."
            draw_list.add_text(
                text_x, text_y,
                imgui.get_color_u32_rgba(_TEXT_COLOR[0], _TEXT_COLOR[1], _TEXT_COLOR[2], _TEXT_COLOR[3] * alpha),
                display_msg
            )

            y_offset += toast_h + _GAP

"""UI utilities for FunGen."""

from .icon_texture import get_icon_texture_manager
from .logo_texture import get_logo_texture_manager
from .button_styles import primary_button_style, destructive_button_style, success_button_style
from .keyboard_layout_detector import detect_keyboard_layout
from .system_scaling import get_system_scaling
from .imgui_helpers import (
    tooltip_if_hovered,
    DisabledScope,
    readonly_input,
    ScopedWidth,
    ScopedID,
    ScopedStyleColor,
    ScopedStyleVar,
    centered_text,
    help_marker,
    confirm_button,
)

__all__ = [
    'get_icon_texture_manager',
    'get_logo_texture_manager',
    'primary_button_style',
    'destructive_button_style',
    'success_button_style',
    'detect_keyboard_layout',
    'get_system_scaling',
    # ImGui helpers
    'tooltip_if_hovered',
    'DisabledScope',
    'readonly_input',
    'ScopedWidth',
    'ScopedID',
    'ScopedStyleColor',
    'ScopedStyleVar',
    'centered_text',
    'help_marker',
    'confirm_button',
]

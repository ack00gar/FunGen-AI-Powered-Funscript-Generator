"""
ImGui UI Helper Functions and Context Managers

This module provides reusable helper functions and context managers
for ImGui UI development, extracted from various UI components.
"""

import imgui
from typing import Optional, Callable


def tooltip_if_hovered(text: str) -> None:
    """
    Show tooltip when the last ImGui item is hovered.

    Args:
        text: Tooltip text to display

    Example:
        >>> imgui.button("Save")
        >>> tooltip_if_hovered("Save your changes to disk")
    """
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


class DisabledScope:
    """
    Context manager for disabled UI elements.

    When active, all UI elements within the context will be disabled
    (greyed out and non-interactive) with reduced opacity.

    Example:
        >>> with DisabledScope(some_condition):
        ...     imgui.button("This button might be disabled")
    """

    __slots__ = ("active",)

    def __init__(self, active: bool):
        """
        Initialize the disabled scope.

        Args:
            active: If True, elements will be disabled. If False, no effect.
        """
        self.active = active
        if active:
            imgui.internal.push_item_flag(imgui.internal.ITEM_DISABLED, True)
            imgui.push_style_var(imgui.STYLE_ALPHA, imgui.get_style().alpha * 0.5)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.active:
            imgui.pop_style_var()
            imgui.internal.pop_item_flag()


def readonly_input(label_id: str, value: str, width: int = -1) -> None:
    """
    Display a read-only input field.

    Args:
        label_id: ImGui label/ID for the input field
        value: Value to display (will show "Not set" if None/empty)
        width: Width of the input field in pixels (-1 for auto)

    Example:
        >>> readonly_input("##filepath", "/path/to/file.mp4", width=300)
    """
    if width is not None and width >= 0:
        imgui.push_item_width(width)
    imgui.input_text(label_id, value or "Not set", 256, flags=imgui.INPUT_TEXT_READ_ONLY)
    if width is not None and width >= 0:
        imgui.pop_item_width()


class ScopedWidth:
    """
    Context manager for scoped item width.

    Example:
        >>> with ScopedWidth(200):
        ...     imgui.input_text("##name", name_buffer, 256)
    """

    __slots__ = ("width",)

    def __init__(self, width: float):
        """
        Args:
            width: Width in pixels (or -1 for auto)
        """
        self.width = width

    def __enter__(self):
        imgui.push_item_width(self.width)
        return self

    def __exit__(self, exc_type, exc, tb):
        imgui.pop_item_width()


class ScopedID:
    """
    Context manager for scoped ImGui ID.

    Example:
        >>> for i, item in enumerate(items):
        ...     with ScopedID(i):
        ...         imgui.button("Delete")  # Each has unique ID
    """

    __slots__ = ("id",)

    def __init__(self, id_value):
        """
        Args:
            id_value: ID value (int or str)
        """
        self.id = id_value

    def __enter__(self):
        imgui.push_id(str(self.id))
        return self

    def __exit__(self, exc_type, exc, tb):
        imgui.pop_id()


class ScopedStyleColor:
    """
    Context manager for temporary style color changes.

    Example:
        >>> with ScopedStyleColor(imgui.COLOR_BUTTON, (1.0, 0.0, 0.0, 1.0)):
        ...     imgui.button("Red Button")
    """

    __slots__ = ("count",)

    def __init__(self, *color_pairs):
        """
        Args:
            *color_pairs: Pairs of (color_id, (r, g, b, a))

        Example:
            >>> ScopedStyleColor(
            ...     (imgui.COLOR_BUTTON, (1, 0, 0, 1)),
            ...     (imgui.COLOR_BUTTON_HOVERED, (0.8, 0, 0, 1))
            ... )
        """
        self.count = 0
        for color_id, color in color_pairs:
            imgui.push_style_color(color_id, *color)
            self.count += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.count > 0:
            imgui.pop_style_color(self.count)


class ScopedStyleVar:
    """
    Context manager for temporary style variable changes.

    Example:
        >>> with ScopedStyleVar(imgui.STYLE_ALPHA, 0.5):
        ...     imgui.text("Semi-transparent text")
    """

    __slots__ = ("count",)

    def __init__(self, *var_pairs):
        """
        Args:
            *var_pairs: Pairs of (var_id, value)

        Example:
            >>> ScopedStyleVar(
            ...     (imgui.STYLE_ALPHA, 0.5),
            ...     (imgui.STYLE_WINDOW_ROUNDING, 0.0)
            ... )
        """
        self.count = 0
        for var_id, value in var_pairs:
            imgui.push_style_var(var_id, value)
            self.count += 1

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.count > 0:
            imgui.pop_style_var(self.count)


def centered_text(text: str, offset_x: float = 0.0) -> None:
    """
    Display centered text.

    Args:
        text: Text to display
        offset_x: Additional horizontal offset

    Example:
        >>> centered_text("Welcome to FunGen!")
    """
    text_width = imgui.calc_text_size(text).x
    window_width = imgui.get_window_width()
    cursor_x = (window_width - text_width) * 0.5 + offset_x
    imgui.set_cursor_pos_x(cursor_x)
    imgui.text(text)


def help_marker(description: str, marker: str = "(?)") -> None:
    """
    Display a help marker with tooltip.

    Args:
        description: Help text to show in tooltip
        marker: Marker text to display (default: "(?)")

    Example:
        >>> imgui.text("Some Setting")
        >>> imgui.same_line()
        >>> help_marker("This setting controls the behavior of X")
    """
    imgui.text_disabled(marker)
    if imgui.is_item_hovered():
        imgui.begin_tooltip()
        imgui.push_text_wrap_pos(imgui.get_font_size() * 35.0)
        imgui.text_unformatted(description)
        imgui.pop_text_wrap_pos()
        imgui.end_tooltip()


def confirm_button(label: str, confirm_text: str = "Click again to confirm",
                   timeout_seconds: float = 2.0) -> bool:
    """
    Button that requires double-click confirmation.

    Args:
        label: Button label
        confirm_text: Text shown during confirmation wait
        timeout_seconds: How long to wait for confirmation

    Returns:
        True if confirmed, False otherwise

    Example:
        >>> if confirm_button("Delete All"):
        ...     delete_all_items()
    """
    import time

    state_key = f"confirm_{label}"
    if not hasattr(confirm_button, 'state'):
        confirm_button.state = {}

    current_time = time.time()
    if state_key in confirm_button.state:
        last_click_time = confirm_button.state[state_key]
        elapsed = current_time - last_click_time

        if elapsed < timeout_seconds:
            # Show confirmation button
            if imgui.button(f"{confirm_text}###{label}_confirm"):
                del confirm_button.state[state_key]
                return True
            return False
        else:
            # Timeout expired, reset
            del confirm_button.state[state_key]

    # First click
    if imgui.button(label):
        confirm_button.state[state_key] = current_time

    return False


__all__ = [
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

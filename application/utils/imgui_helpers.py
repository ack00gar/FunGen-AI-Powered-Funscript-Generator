"""Shared imgui utility classes."""

import imgui


class DisabledScope:
    """Context manager to disable imgui widgets with reduced alpha."""
    __slots__ = ("active",)

    def __init__(self, active):
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

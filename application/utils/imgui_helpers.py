"""Shared imgui utility classes."""

import imgui


def tooltip_if_hovered(text):
    """Show a tooltip when the last imgui item is hovered."""
    if imgui.is_item_hovered():
        imgui.set_tooltip(text)


def center_next_window(width, height=0):
    """Position the next window centered on the main viewport.

    Args:
        width: Window width in pixels.
        height: Window height (0 = auto-resize vertical).
    """
    mv = imgui.get_main_viewport()
    pos_x = mv.pos[0] + (mv.size[0] - width) * 0.5
    pos_y = mv.pos[1] + (mv.size[1] - max(height, 300)) * 0.5
    imgui.set_next_window_position(pos_x, pos_y, condition=imgui.APPEARING)
    imgui.set_next_window_size(width, height, condition=imgui.APPEARING)


def begin_modal_centered(name, width, height=0):
    """Open and begin a centered modal popup with auto-resize.

    Returns True if the popup is open and content should be rendered.
    Caller must call ``imgui.end_popup()`` when done.
    """
    imgui.open_popup(name)
    center_next_window(width, height)
    opened, _ = imgui.begin_popup_modal(
        name, True, flags=imgui.WINDOW_ALWAYS_AUTO_RESIZE
    )
    return opened


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

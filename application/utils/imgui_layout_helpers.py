"""Shared two-column layout helpers for consistent settings/info rendering."""
import imgui

_LABEL_MIN_WIDTH = 160  # minimum label column px (before font scaling)


def begin_settings_columns(col_id="settings_cols"):
    """Start a two-column layout for label : widget rows."""
    imgui.columns(2, col_id, border=False)
    scale = imgui.get_io().font_global_scale
    lw = max(_LABEL_MIN_WIDTH * scale, imgui.get_content_region_available_width() * 0.45)
    imgui.set_column_width(0, lw)


def end_settings_columns():
    imgui.columns(1)


def row_label(text, tooltip=None):
    """Render a left-column label, advance to widget column."""
    imgui.text(text)
    if tooltip and imgui.is_item_hovered():
        imgui.set_tooltip(tooltip)
    imgui.next_column()


def row_end():
    """Advance back to label column after widget."""
    imgui.next_column()


def row_separator():
    """Visual break between groups within a section."""
    end_settings_columns()
    imgui.spacing()
    imgui.separator()
    imgui.spacing()

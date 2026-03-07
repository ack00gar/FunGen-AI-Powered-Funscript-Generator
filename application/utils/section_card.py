"""Reusable section card context manager for wrapping UI content in styled containers."""

import imgui
from contextlib import contextmanager
from config.element_group_colors import CardColors


# Tier presets
_TIER_PRESETS = {
    "primary": {
        "bg": CardColors.PRIMARY_BG,
        "rounding": 6.0,
        "accent_width": 4.0,
        "padding": 8.0,
    },
    "secondary": {
        "bg": CardColors.SECONDARY_BG,
        "rounding": 4.0,
        "accent_width": 0.0,
        "padding": 6.0,
    },
    "inline": {
        "bg": CardColors.INLINE_BG,
        "rounding": 3.0,
        "accent_width": 0.0,
        "padding": 4.0,
    },
}

# Nesting depth counter — prevents nested channels_split assertions when
# secondary/inline cards are used inside primary cards.
_channel_split_depth = 0


@contextmanager
def section_card(label, tier="primary", accent_color=None, open_by_default=True):
    """Context manager that wraps content in a visually distinct card container.

    All tiers use draw_list channel splitting to draw background behind content
    (preventing the dimming effect of semi-transparent overlays), unless already
    inside a channel split from an outer card.

    Args:
        label: Display text for the card header. Also used as imgui ID.
        tier: Visual tier - "primary", "secondary", or "inline".
        accent_color: Optional RGBA tuple for the left accent bar (primary tier only).
        open_by_default: Whether the card starts expanded (collapsible header).

    Yields:
        bool: True if content should be rendered (header is open).
    """
    global _channel_split_depth

    preset = _TIER_PRESETS.get(tier, _TIER_PRESETS["primary"])
    bg = preset["bg"]
    rounding = preset["rounding"]
    accent_width = preset["accent_width"]
    padding = preset["padding"]

    # Use channel splitting when not already inside another split.
    # Nested splits cause imgui assertions, so inner cards fall back to
    # drawing background after content (acceptable at low alpha).
    use_channels = _channel_split_depth == 0

    imgui.spacing()

    # Record start position for background drawing
    draw_list = imgui.get_window_draw_list()
    region_start = imgui.get_cursor_screen_pos()
    content_width = imgui.get_content_region_available_width()

    if use_channels:
        _channel_split_depth += 1
        # Split channels BEFORE rendering content so background ends up behind it.
        # Channel 0 = background (rendered first), Channel 1 = content (rendered on top).
        draw_list.channels_split(2)
        draw_list.channels_set_current(1)  # All content goes to foreground channel

    # Indent content for padding + accent bar
    total_left_pad = padding + accent_width
    imgui.indent(total_left_pad)

    # Add top padding via dummy
    imgui.dummy(0, padding * 0.5)

    # Render collapsible header within the card
    flags = imgui.TREE_NODE_DEFAULT_OPEN if open_by_default else 0
    is_open, _ = imgui.collapsing_header(label, flags=flags)

    imgui.begin_group()
    try:
        yield is_open
    finally:
        imgui.end_group()

        # Add bottom padding
        imgui.dummy(0, padding * 0.5)

        # Measure the content bounds
        region_end_y = imgui.get_cursor_screen_pos()[1]

        # Unindent
        imgui.unindent(total_left_pad)

        bg_color = imgui.get_color_u32_rgba(*bg)
        x1, y1 = region_start[0], region_start[1]
        x2 = x1 + content_width
        y2 = region_end_y

        if use_channels:
            # Draw card background on channel 0 (behind content on channel 1)
            draw_list.channels_set_current(0)  # Background channel
            draw_list.add_rect_filled(x1, y1, x2, y2, bg_color, rounding)

            # Draw accent bar if configured
            if accent_width > 0 and accent_color:
                accent_u32 = imgui.get_color_u32_rgba(*accent_color)
                draw_list.add_rect_filled(
                    x1, y1,
                    x1 + accent_width, y2,
                    accent_u32, rounding
                )

            # Merge: channel 0 (bg) rendered first, then channel 1 (content) on top
            draw_list.channels_merge()
            _channel_split_depth -= 1
        else:
            # Nested inside another card's channel split — draw background on
            # the parent's background channel so it renders behind content.
            draw_list.channels_set_current(0)
            draw_list.add_rect_filled(x1, y1, x2, y2, bg_color, rounding)
            draw_list.channels_set_current(1)

        imgui.spacing()

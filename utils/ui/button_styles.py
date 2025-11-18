"""
Button styling utilities for visual hierarchy.
Provides context managers for PRIMARY, DESTRUCTIVE, and SECONDARY button styles.
"""

import imgui
from contextlib import contextmanager
from config.element_group_colors import ButtonColors


@contextmanager
def primary_button_style():
    """
    Context manager for PRIMARY button style (positive/affirmative actions).

    Use for actions like: Start, Create, Save, Generate, Apply, Confirm, etc.

    Example:
        with primary_button_style():
            if imgui.button("Start Processing"):
                # Handle click
    """
    imgui.push_style_color(imgui.COLOR_BUTTON, *ButtonColors.PRIMARY)
    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *ButtonColors.PRIMARY_HOVERED)
    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *ButtonColors.PRIMARY_ACTIVE)
    try:
        yield
    finally:
        imgui.pop_style_color(3)


@contextmanager
def destructive_button_style():
    """
    Context manager for DESTRUCTIVE button style (dangerous/irreversible actions).

    Use for actions like: Delete, Clear, Abort, Remove, Unload, Reset, etc.

    Example:
        with destructive_button_style():
            if imgui.button("Delete Chapter"):
                # Handle click
    """
    imgui.push_style_color(imgui.COLOR_BUTTON, *ButtonColors.DESTRUCTIVE)
    imgui.push_style_color(imgui.COLOR_BUTTON_HOVERED, *ButtonColors.DESTRUCTIVE_HOVERED)
    imgui.push_style_color(imgui.COLOR_BUTTON_ACTIVE, *ButtonColors.DESTRUCTIVE_ACTIVE)
    try:
        yield
    finally:
        imgui.pop_style_color(3)


# SECONDARY buttons use ImGui's default styling (no context manager needed)
# Use for actions like: Browse, Edit, Cancel, Close, etc.
# Simply call imgui.button() without any style context manager.

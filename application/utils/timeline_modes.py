"""Timeline editing mode enums.

Defines formal state enums for the interactive timeline, replacing ad-hoc
boolean flags with a structured state machine.
"""
from enum import Enum


class TimelineMode(Enum):
    """Primary editing mode for the timeline."""
    SELECT = "select"            # Default: click to select/drag points, marquee selection
    ALTERNATING = "alternating"  # Click to place alternating top/bottom points
    RECORDING = "recording"      # Capture mouse position while video plays
    INJECTION = "injection"      # Click segments to inject intermediate points


class TimelineInteractionState(Enum):
    """Transient interaction state within any mode."""
    IDLE = "idle"
    DRAGGING_POINT = "dragging_point"
    MARQUEEING = "marqueeing"
    RANGE_SELECTING = "range_selecting"
    PANNING = "panning"

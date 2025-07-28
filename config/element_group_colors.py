"""
Color element groups for organizing UI colors by component.
This provides a structured way to manage colors for different UI elements.
"""

from config.constants_colors import CurrentTheme, RGBColors


class GaugeColors:
    """Color constants for gauge components."""
    BACKGROUND = CurrentTheme.GAUGE_BG
    BORDER = CurrentTheme.GAUGE_BORDER
    BAR_RED = CurrentTheme.GAUGE_BAR_RED
    BAR_GREEN = CurrentTheme.GAUGE_BAR_GREEN
    TEXT = CurrentTheme.GAUGE_TEXT
    VALUE_TEXT = CurrentTheme.GAUGE_VALUE_TEXT


class TimelineColors:
    """Color constants for interactive timeline components."""
    CANVAS_BACKGROUND = CurrentTheme.TIMELINE_CANVAS_BG
    CENTER_MARKER = CurrentTheme.TIMELINE_MARKER
    TIME_DISPLAY_TEXT = CurrentTheme.TIMELINE_TIME_TEXT
    GRID_LINES = CurrentTheme.TIMELINE_GRID
    GRID_MAJOR_LINES = CurrentTheme.TIMELINE_GRID_MAJOR
    GRID_LABELS = CurrentTheme.TIMELINE_GRID_LABELS
    AUDIO_WAVEFORM = CurrentTheme.TIMELINE_WAVEFORM
    SELECTED_POINT_BORDER = CurrentTheme.TIMELINE_SELECTED_BORDER
    PREVIEW_LINES = CurrentTheme.TIMELINE_PREVIEW_LINE
    PREVIEW_POINTS = CurrentTheme.TIMELINE_PREVIEW_POINT
    MARQUEE_SELECTION_FILL = CurrentTheme.TIMELINE_MARQUEE_FILL
    MARQUEE_SELECTION_BORDER = CurrentTheme.TIMELINE_MARQUEE_BORDER
    POINT_DEFAULT = CurrentTheme.TIMELINE_POINT_DEFAULT
    POINT_DRAGGING = CurrentTheme.TIMELINE_POINT_DRAGGING
    POINT_SELECTED = CurrentTheme.TIMELINE_POINT_SELECTED
    POINT_HOVER = CurrentTheme.TIMELINE_POINT_HOVER


class CompilerToolColors:
    """Color constants for TensorRT compiler components."""
    SUCCESS = CurrentTheme.COMPILER_SUCCESS
    ERROR = CurrentTheme.COMPILER_ERROR
    INFO = CurrentTheme.COMPILER_INFO
    WARNING = CurrentTheme.COMPILER_WARNING
    STARTING = CurrentTheme.COMPILER_STARTING
    STOPPING = CurrentTheme.COMPILER_STOPPING
    OUTPUT_TEXT = CurrentTheme.COMPILER_OUTPUT_TEXT


class LRDialColors:
    """Color constants for LR dial components."""
    BACKGROUND = CurrentTheme.LR_DIAL_BACKGROUND
    BORDER = CurrentTheme.LR_DIAL_BORDER
    LEFT_LABEL = CurrentTheme.LR_DIAL_LEFT_LABEL
    RIGHT_LABEL = CurrentTheme.LR_DIAL_RIGHT_LABEL
    INDICATOR_LINE = CurrentTheme.LR_DIAL_INDICATOR_LINE
    INDICATOR_TIP = CurrentTheme.LR_DIAL_INDICATOR_TIP
    VALUE_TEXT = CurrentTheme.LR_DIAL_VALUE_TEXT


class VideoDisplayColors:
    """Color constants for video display components."""
    # Pose skeleton colors
    DOMINANT_LIMB = CurrentTheme.VIDEO_DOMINANT_LIMB
    DOMINANT_KEYPOINT = CurrentTheme.VIDEO_DOMINANT_KEYPOINT
    MUTED_LIMB = CurrentTheme.VIDEO_MUTED_LIMB
    MUTED_KEYPOINT = CurrentTheme.VIDEO_MUTED_KEYPOINT
    
    # Motion mode colors
    MOTION_UNDETERMINED = CurrentTheme.VIDEO_MOTION_UNDETERMINED
    MOTION_THRUSTING = CurrentTheme.VIDEO_MOTION_THRUSTING
    MOTION_RIDING = CurrentTheme.VIDEO_MOTION_RIDING
    
    # ROI and tracking colors
    ROI_DRAWING = CurrentTheme.VIDEO_ROI_DRAWING
    ROI_BORDER = CurrentTheme.VIDEO_ROI_BORDER
    TRACKING_POINT = CurrentTheme.VIDEO_TRACKING_POINT
    FLOW_VECTOR = CurrentTheme.VIDEO_FLOW_VECTOR
    
    # Overlay colors
    BOX_LABEL = CurrentTheme.VIDEO_BOX_LABEL
    OCCLUSION_WARNING = CurrentTheme.VIDEO_OCCLUSION_WARNING
    
    # Additional video display colors
    PERSISTENT_REFINED_TRACK = CurrentTheme.VIDEO_PERSISTENT_REFINED_TRACK
    ACTIVE_INTERACTOR = CurrentTheme.VIDEO_ACTIVE_INTERACTOR
    LOCKED_PENIS = CurrentTheme.VIDEO_LOCKED_PENIS
    FILL_COLOR = CurrentTheme.VIDEO_FILL_COLOR
    ALIGNED_FALLBACK = CurrentTheme.VIDEO_ALIGNED_FALLBACK
    INFERRED_BOX = CurrentTheme.VIDEO_INFERRED_BOX


class ControlPanelColors:
    """Color constants for control panel components."""
    ACTIVE_PROGRESS = CurrentTheme.CONTROL_PANEL_ACTIVE_PROGRESS
    COMPLETED_PROGRESS = CurrentTheme.CONTROL_PANEL_COMPLETED_PROGRESS
    SUB_PROGRESS = CurrentTheme.CONTROL_PANEL_SUB_PROGRESS


class VideoNavigationColors:
    """Color constants for video navigation components."""
    BACKGROUND = CurrentTheme.NAV_BACKGROUND
    ICON = CurrentTheme.NAV_ICON
    SCRIPTING_BORDER = CurrentTheme.NAV_SCRIPTING_BORDER
    SELECTION_PRIMARY = CurrentTheme.NAV_SELECTION_PRIMARY
    SELECTION_SECONDARY = CurrentTheme.NAV_SELECTION_SECONDARY
    TEXT_BLACK = CurrentTheme.NAV_TEXT_BLACK
    TEXT_WHITE = CurrentTheme.NAV_TEXT_WHITE
    MARKER = CurrentTheme.NAV_MARKER


class AppGUIColors:
    """Color constants for app GUI components."""
    MARKER = CurrentTheme.APP_MARKER

    """Color constants for general UI sliders."""
    FPS_TARGET_MARKER = RGBColors.FPS_TARGET_MARKER
    FPS_TRACKER_MARKER = RGBColors.FPS_TRACKER_MARKER
    FPS_PROCESSOR_MARKER = RGBColors.FPS_PROCESSOR_MARKER
    
    # Additional RGB colors
    TEAL = RGBColors.TEAL

    TRANSPARENT = CurrentTheme.TRANSPARENT
    SEMI_TRANSPARENT = CurrentTheme.SEMI_TRANSPARENT


class SegmentColors:
    """Color constants for video segment components."""
    BJ = CurrentTheme.SEGMENT_BJ
    HJ = CurrentTheme.SEGMENT_HJ
    NR = CurrentTheme.SEGMENT_NR
    CG_MISS = CurrentTheme.SEGMENT_CG_MISS
    REV_CG_DOG = CurrentTheme.SEGMENT_REV_CG_DOG
    FOOTJ = CurrentTheme.SEGMENT_FOOTJ
    BOOBJ = CurrentTheme.SEGMENT_BOOBJ
    CLOSEUP = CurrentTheme.SEGMENT_CLOSEUP
    DEFAULT = CurrentTheme.SEGMENT_DEFAULT


class BoxStyleColors:
    """Color constants for box style components."""
    GENERAL = CurrentTheme.BOX_GENERAL
    PREF_PENIS = CurrentTheme.BOX_PREF_PENIS
    LOCKED_PENIS = CurrentTheme.BOX_LOCKED_PENIS
    PUSSY = CurrentTheme.BOX_PUSSY
    BUTT = CurrentTheme.BOX_BUTT
    TRACKED = CurrentTheme.BOX_TRACKED
    TRACKED_ALT = CurrentTheme.BOX_TRACKED_ALT
    GENERAL_DETECTION = CurrentTheme.BOX_GENERAL_DETECTION
    EXCLUDED = CurrentTheme.BOX_EXCLUDED


class FloatingWidgetColors:
    """Color constants for floating widget components."""
    BACKGROUND = CurrentTheme.FLOATING_WIDGET_BG
    BORDER = CurrentTheme.FLOATING_WIDGET_BORDER
    TEXT = CurrentTheme.FLOATING_WIDGET_TEXT


class GeneralColors:
    """General color constants for common UI elements."""
    WHITE = CurrentTheme.WHITE
    WHITE_DARK = CurrentTheme.WHITE_DARK
    BLACK = CurrentTheme.BLACK
    GRAY = CurrentTheme.GRAY
    GRAY_LIGHT = CurrentTheme.GRAY_LIGHT
    GRAY_DARK = CurrentTheme.GRAY_DARK
    GRAY_MEDIUM = CurrentTheme.GRAY_MEDIUM
    RED = CurrentTheme.RED
    RED_LIGHT = CurrentTheme.RED_LIGHT
    RED_DARK = CurrentTheme.RED_DARK
    GREEN = CurrentTheme.GREEN
    GREEN_LIGHT = CurrentTheme.GREEN_LIGHT
    GREEN_DARK = CurrentTheme.GREEN_DARK
    BLUE = CurrentTheme.BLUE
    BLUE_LIGHT = CurrentTheme.BLUE_LIGHT
    BLUE_DARK = CurrentTheme.BLUE_DARK
    YELLOW = CurrentTheme.YELLOW
    YELLOW_LIGHT = CurrentTheme.YELLOW_LIGHT
    YELLOW_DARK = CurrentTheme.YELLOW_DARK
    CYAN = CurrentTheme.CYAN
    MAGENTA = CurrentTheme.MAGENTA
    ORANGE = CurrentTheme.ORANGE
    ORANGE_LIGHT = CurrentTheme.ORANGE_LIGHT
    ORANGE_DARK = CurrentTheme.ORANGE_DARK
    PURPLE = CurrentTheme.PURPLE
    PINK = CurrentTheme.PINK
    BROWN = CurrentTheme.BROWN 
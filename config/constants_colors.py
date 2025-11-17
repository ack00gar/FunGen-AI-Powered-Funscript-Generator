"""
Color constants for GUI components with theme support.
Centralizes all color definitions to replace hardcoded RGBA values throughout the codebase.
All constants are named by their actual color, not their purpose.
Naming scheme: COLOR_SHADE (e.g., BLUE_LIGHT, GRAY_DARK)
"""

# TODO: Structure colors. Global base colors (full opaque), class themes with modified base colors.

# Dark Theme Colors (RGBA format: red, green, blue, alpha) - values from 0.0 to 1.0
class DarkTheme:
    # Basic Colors
    RED = (0.9, 0.2, 0.2, 1.0)
    RED_LIGHT = (0.9, 0.4, 0.4, 0.8)
    RED_DARK = (0.7, 0.2, 0.2, 1.0)

    GREEN = (0.2, 0.8, 0.2, 1.0)
    GREEN_LIGHT = (0.4, 0.9, 0.4, 0.8)
    GREEN_DARK = (0.2, 0.7, 0.2, 1.0)

    BLUE = (0.2, 0.2, 0.9, 1.0)
    BLUE_LIGHT = (0.7, 0.9, 1.0, 1.0)
    BLUE_DARK = (0.2, 0.35, 0.6, 0.6)

    YELLOW = (0.9, 0.9, 0.2, 1.0)
    YELLOW_LIGHT = (0.9, 0.9, 0.3, 0.8)
    YELLOW_DARK = (0.8, 0.8, 0.2, 1.0)

    CYAN = (0.1, 0.9, 0.9, 1.0)
    MAGENTA = (0.9, 0.2, 0.9, 1.0)

    WHITE = (1.0, 1.0, 1.0, 1.0)
    WHITE_DARK = (0.9, 0.9, 0.9, 1.0)
    BLACK = (0.0, 0.0, 0.0, 1.0)
    GRAY = (0.7, 0.7, 0.7, 1.0)
    GRAY_LIGHT = (0.8, 0.8, 0.8, 1.0)
    GRAY_DARK = (0.3, 0.3, 0.3, 1.0)
    GRAY_MEDIUM = (0.5, 0.5, 0.5, 1.0)

    ORANGE = (1.0, 0.6, 0.0, 1.0)
    ORANGE_LIGHT = (1.0, 0.8, 0.4, 0.8)
    ORANGE_DARK = (0.9, 0.6, 0.0, 1.0)
    
    BROWN = (0.9, 0.6, 0.2, 0.8)
    PURPLE = (0.7, 0.7, 0.9, 0.8)
    PINK = (0.9, 0.3, 0.6, 0.8)

    TRANSPARENT = (0, 0, 0, 0)
    SEMI_TRANSPARENT = (0, 0, 0, 0.4)
    
    # App GUI Colors
    APP_MARKER = (*RED[:3], 0.85)  # Red marker for app GUI
    FLOATING_WIDGET_BG = (0.1, 0.1, 0.1, 1.0) # dark gray
    FLOATING_WIDGET_BORDER = GRAY_MEDIUM
    FLOATING_WIDGET_TEXT = WHITE_DARK
    ENERGY_SAVER_INDICATOR = GREEN
    VERSION_CURRENT_HIGHLIGHT = GREEN
    VERSION_CHANGELOG_TEXT = GRAY
    VIDEO_STATUS_FUNGEN = GREEN
    VIDEO_STATUS_OTHER = YELLOW
    BACKGROUND_CLEAR = (0.06, 0.06, 0.06, 1.0)  # Dark gray background

    # Menu Colors
    FRAME_OFFSET = YELLOW

    # Gauge-specific colors
    GAUGE_BG = FLOATING_WIDGET_BG
    GAUGE_BORDER = FLOATING_WIDGET_BORDER
    GAUGE_BAR_RED = RED_DARK
    GAUGE_BAR_GREEN = GREEN_DARK
    GAUGE_TEXT = FLOATING_WIDGET_TEXT
    GAUGE_VALUE_TEXT = YELLOW

    # LR Dial Colors
    LR_DIAL_BACKGROUND = FLOATING_WIDGET_BG
    LR_DIAL_BORDER = FLOATING_WIDGET_BORDER
    LR_DIAL_LEFT_LABEL = RED
    LR_DIAL_RIGHT_LABEL = BLUE
    LR_DIAL_INDICATOR_LINE = CYAN
    LR_DIAL_INDICATOR_TIP = YELLOW
    LR_DIAL_VALUE_TEXT = YELLOW

    # Timeline Colors
    TIMELINE_CANVAS_BG = (0.08, 0.08, 0.1, 1.0) # dark gray
    TIMELINE_MARKER = (*RED[:3], 0.9)
    TIMELINE_TIME_TEXT = (*WHITE[:3], 0.7)
    TIMELINE_GRID = (*GRAY_DARK[:3], 0.8)
    TIMELINE_GRID_MAJOR = (*GRAY_DARK[:3], 0.9)
    TIMELINE_GRID_LABELS = GRAY
    TIMELINE_WAVEFORM = BLUE_DARK
    TIMELINE_SELECTED_BORDER = (0.6, 0.0, 0.0, 1.0) # dark red
    TIMELINE_PREVIEW_LINE = (*ORANGE[:3], 0.9)
    TIMELINE_PREVIEW_POINT = ORANGE
    TIMELINE_MARQUEE_FILL = (0.5, 0.5, 1.0, 0.3) # light blue
    TIMELINE_MARQUEE_BORDER = (0.8, 0.8, 1.0, 0.7) # light blue
    TIMELINE_POINT_DEFAULT = GREEN
    TIMELINE_POINT_DRAGGING = RED
    TIMELINE_POINT_SELECTED = (1.0, 0.0, 0.0, 1.0) # dark red
    TIMELINE_POINT_HOVER = (0.5, 1.0, 0.5, 1.0) # light green
    ULTIMATE_AUTOTUNE_PREVIEW = (0.2, 1.0, 0.5, 0.5)  # A semi-transparent green

    # Compiler Colors
    COMPILER_SUCCESS = GREEN
    COMPILER_ERROR = RED
    COMPILER_INFO = GRAY
    COMPILER_WARNING = YELLOW
    COMPILER_STARTING = ORANGE
    COMPILER_STOPPING = YELLOW
    COMPILER_OUTPUT_TEXT = BLUE_LIGHT

    # Video Display Colors
    VIDEO_DOMINANT_LIMB = (*GREEN[:3], 0.95)
    VIDEO_DOMINANT_KEYPOINT = (1.0, 0.5, 0.1, 1.0) # orange
    VIDEO_MUTED_LIMB = (0.2, 0.6, 1.0, 0.4) # muted cyan
    VIDEO_MUTED_KEYPOINT = (*RED[:3], 0.5)
    VIDEO_MOTION_UNDETERMINED = (*YELLOW[:3], 0.9)
    VIDEO_MOTION_THRUSTING = (*GREEN[:3], 0.95)
    VIDEO_MOTION_RIDING = (*MAGENTA[:3], 0.95)
    VIDEO_ROI_DRAWING = (*YELLOW[:3], 0.7)
    VIDEO_ROI_BORDER = (*CYAN[:3], 0.7)
    VIDEO_TRACKING_POINT = (*GREEN[:3], 0.9)
    VIDEO_FLOW_VECTOR = (*RED[:3], 0.9)
    VIDEO_BOX_LABEL = WHITE
    VIDEO_OCCLUSION_WARNING = (*ORANGE[:3], 0.95)
    VIDEO_PERSISTENT_REFINED_TRACK = CYAN
    VIDEO_ACTIVE_INTERACTOR = YELLOW
    VIDEO_LOCKED_PENIS = GREEN_DARK
    VIDEO_FILL_COLOR = (*GREEN[:3], 0.4)
    VIDEO_ALIGNED_FALLBACK = (*ORANGE[:3], 0.9)
    VIDEO_INFERRED_BOX = (*PURPLE[:3], 0.85)

    # Navigation Colors
    NAV_BACKGROUND = (0.1, 0.1, 0.12, 1.0) # dark gray
    NAV_ICON = (*YELLOW[:3], 0.9)
    NAV_SCRIPTING_BORDER = (*YELLOW[:3], 0.6)
    NAV_SELECTION_PRIMARY = (*GREEN[:3], 0.95)
    NAV_SELECTION_SECONDARY = (0.3, 0.5, 1.0, 0.95) # blue
    NAV_TEXT_BLACK = BLACK
    NAV_TEXT_WHITE = WHITE
    NAV_MARKER = (*WHITE[:3], 0.7)

    # Box Style Colors
    BOX_GENERAL = (*GRAY_LIGHT[:3], 0.7)
    BOX_PREF_PENIS = (*GREEN[:3], 0.9)
    BOX_LOCKED_PENIS = (*CYAN[:3], 0.8)
    BOX_PUSSY = (1.0, 0.5, 0.8, 0.8) # pink
    BOX_BUTT = (*BROWN[:3], 0.8)
    BOX_TRACKED = (*YELLOW[:3], 0.8)
    BOX_TRACKED_ALT = (*GRAY[:3], 0.7)
    BOX_GENERAL_DETECTION = (0.2, 0.5, 1.0, 0.6) # blue
    BOX_EXCLUDED = (0.5, 0.1, 0.1, 0.5) # dark red

    # Segment Colors
    SEGMENT_BJ = (*GREEN_LIGHT[:3], 0.8)
    SEGMENT_HJ = (*GREEN_LIGHT[:3], 0.8)
    SEGMENT_NR = (*GRAY[:3], 0.7)
    SEGMENT_CG_MISS = (0.4, 0.4, 0.9, 0.8) # blue
    SEGMENT_REV_CG_DOG = (*BROWN[:3], 0.8)
    SEGMENT_CG = (0.3, 0.5, 0.9, 0.8)  # lighter blue for Cowgirl
    SEGMENT_MISS = (0.5, 0.3, 0.9, 0.8)  # purple-blue for Missionary
    SEGMENT_REV_CG = (0.7, 0.5, 0.3, 0.8)  # lighter brown for Reverse Cowgirl
    SEGMENT_DOG = (0.5, 0.4, 0.2, 0.8)  # darker brown for Doggy
    SEGMENT_FOOTJ = (*YELLOW_LIGHT[:3], 0.8)
    SEGMENT_BOOBJ = (*PINK[:3], 0.8)
    SEGMENT_CLOSEUP = (*PURPLE[:3], 0.8)
    SEGMENT_INTRO = (0.5, 0.6, 0.7, 0.7)  # light gray-blue
    SEGMENT_OUTRO = (0.6, 0.5, 0.6, 0.7)  # light gray-purple
    SEGMENT_TRANSITION = (0.6, 0.6, 0.5, 0.7)  # light gray-yellow
    SEGMENT_DEFAULT = TRANSPARENT

    # Control Panel Colors
    CONTROL_PANEL_ACTIVE_PROGRESS = (0.2, 0.6, 1.0, 1.0) # blue
    CONTROL_PANEL_COMPLETED_PROGRESS = (0.2, 0.7, 0.2, 1.0) # green
    CONTROL_PANEL_SUB_PROGRESS = (0.3, 0.7, 1.0, 1.0) # cyan
    
    # Control Panel Status Indicator Colors
    CONTROL_PANEL_STATUS_READY = GREEN
    CONTROL_PANEL_STATUS_WARNING = ORANGE
    CONTROL_PANEL_STATUS_ERROR = RED
    CONTROL_PANEL_STATUS_INFO = BLUE_LIGHT
    
    # Control Panel Section Header Colors
    CONTROL_PANEL_SECTION_HEADER = WHITE_DARK

    # Update Settings Dialog Colors
    UPDATE_TOKEN_VALID = GREEN
    UPDATE_TOKEN_INVALID = RED
    UPDATE_TOKEN_WARNING = ORANGE
    UPDATE_TOKEN_SET = GREEN
    UPDATE_TOKEN_NOT_SET = ORANGE
    UPDATE_DIALOG_TEXT = WHITE
    UPDATE_DIALOG_GRAY_TEXT = GRAY

    # Button Palette Colors (for visual hierarchy)
    # PRIMARY buttons (positive/affirmative actions: Start, Create, Save, etc.)
    BUTTON_PRIMARY = (0.2, 0.5, 0.9, 1.0)  # Blue
    BUTTON_PRIMARY_HOVERED = (0.3, 0.6, 1.0, 1.0)  # Lighter blue
    BUTTON_PRIMARY_ACTIVE = (0.15, 0.4, 0.75, 1.0)  # Darker blue

    # DESTRUCTIVE buttons (dangerous/irreversible actions: Delete, Clear, Abort, etc.)
    BUTTON_DESTRUCTIVE = (0.8, 0.2, 0.2, 1.0)  # Red
    BUTTON_DESTRUCTIVE_HOVERED = (0.9, 0.3, 0.3, 1.0)  # Lighter red
    BUTTON_DESTRUCTIVE_ACTIVE = (0.7, 0.1, 0.1, 1.0)  # Darker red

    # SECONDARY buttons (default/neutral actions: Browse, Edit, Cancel, etc.)
    # These use ImGui's default button colors (no custom styling needed)


# Current theme (default to dark)
CurrentTheme = DarkTheme

class RGBColors:
    # RGB Colors (for legacy compatibility - 0-255 range)
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GREY = (128, 128, 128)
    GREY_LIGHT = (180, 180, 180)
    GREY_DARK = (64, 64, 64)
    RED = (255, 0, 0)
    GREEN = (0, 255, 0)
    GREEN_LIGHT = (128, 255, 128)
    GREEN_DARK = (0, 128, 0)
    BLUE = (0, 0, 255)
    BLUE_DARK = (0, 0, 128)
    YELLOW = (255, 255, 0)
    TEAL = (0, 220, 220)
    CYAN = (0, 255, 255)
    ORANGE = (255, 165, 0)
    ORANGE_LIGHT = (255, 180, 0)
    MAGENTA = (255, 0, 255)
    PURPLE = (128, 0, 128)


    TIMELINE_HEATMAP = [
        (30, 144, 255),   # Dodger Blue
        (34, 139, 34),    # Lime Green
        (255, 215, 0),    # Gold
        (220, 20, 60),    # Crimson
        (147, 112, 219),  # Medium Purple
        (37, 22, 122)     # Dark Blue (Cap color)
    ]
    TIMELINE_COLOR_SPEED_STEP = 250.0  # Speed (pixels/sec) used to step through the color map.
    TIMELINE_HEATMAP_BACKGROUND = (20, 20, 25, 255)
    TIMELINE_COLOR_ALPHA = 0.9 # alpha value for the timeline heatmap

    # General Slider Colors
    FPS_TARGET_MARKER = YELLOW
    FPS_TRACKER_MARKER = GREEN
    FPS_PROCESSOR_MARKER = RED

    # OBJECT DETECTION CLASS COLORS
    CLASS_COLORS = {
        "penis": RED,
        "glans": GREEN_DARK,
        "pussy": BLUE,
        "butt": ORANGE_LIGHT,
        "anus": PURPLE,
        "breast": ORANGE,
        "navel": CYAN,
        "hand": MAGENTA,
        "face": GREEN,
        "foot": (165, 42, 42), # Brown
        "hips center": BLACK,
        "locked_penis": CYAN,
    }

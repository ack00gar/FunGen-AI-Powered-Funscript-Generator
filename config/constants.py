import platform
import os
import enum
from typing import Dict, List, Tuple, Any
from enum import Enum, auto

# Attempt to import torch for device detection, but fail gracefully if it's not available.
try:
    import torch
except ImportError:
    torch = None

####################################################################################################
# META & VERSIONING
####################################################################################################
APP_NAME = "FunGen"
APP_VERSION = "0.5.4"
APP_WINDOW_TITLE = f"{APP_NAME} v{APP_VERSION} - AI Computer Vision"
FUNSCRIPT_AUTHOR = "FunGen"

# --- Component Versions ---
OBJECT_DETECTION_VERSION = "1.0.0"
TRACKING_VERSION = "0.1.1"
FUNSCRIPT_FORMAT_VERSION = "1.0"
FUNSCRIPT_METADATA_VERSION = "0.2.0"  # For chapters and other metadata
CONFIG_VERSION = 1


####################################################################################################
# FILE & PATHS
####################################################################################################
SETTINGS_FILE = "settings.json"
AUTOSAVE_FILE = "autosave.fgnstate"
DEFAULT_AUTOSAVE_INTERVAL_SECONDS = 300

# --- Logging Configuration ---
# Maximum size per log file before rotation (bytes) and number of backups to keep
LOG_MAX_BYTES = 5 * 1024 * 1024  # 5 MB
LOG_BACKUP_COUNT = 3
PROJECT_FILE_EXTENSION = ".fgnproj"
DEFAULT_OUTPUT_FOLDER = "output"

# --- Send2Trash Configuration ---
SEND2TRASH_MAX_ATTEMPTS = 3
SEND2TRASH_RETRY_DELAY = 2  # Delay in seconds between retry attempts

# --- TensorRT Compiler UI ---
TENSORRT_OUTPUT_DISPLAY_HEIGHT = 150  # Height in pixels for subprocess output display

# --- Internet Connection Test ---
INTERNET_TEST_HOSTS = [
    ("8.8.8.8", 53),      # Google DNS
    ("1.1.1.1", 53),      # Cloudflare DNS
    ("208.67.222.222", 53) # OpenDNS
]


####################################################################################################
# SYSTEM & PERFORMANCE
####################################################################################################
# Determines the compute device for ML models (e.g., 'cuda', 'mps', 'cpu').
# This is detected once and used by both Stage 1 and the live tracker.
DEVICE = 'cpu'
if torch:
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':
        DEVICE = 'mps'
    elif torch.cuda.is_available():
        DEVICE = 'cuda'

# The side length of the square input image for the YOLO model.
YOLO_INPUT_SIZE = 640
# Default target height for oscillation processing/downscaling to match model input characteristics
DEFAULT_OSCILLATION_PROCESSING_TARGET_HEIGHT = YOLO_INPUT_SIZE

# GPU Unwarp Configuration (VR video processing optimization)
ENABLE_GPU_UNWARP = True  # Use GPU shader unwrapping instead of CPU v360 filter
GPU_UNWARP_BACKEND = 'auto'  # 'metal', 'opengl', 'auto'
FALLBACK_TO_FFMPEG_V360 = True  # Use v360 if GPU unwarp unavailable

# Fallback for determining producer/consumer counts if os.cpu_count() fails.
DEFAULT_FALLBACK_CPU_CORES = 4

class ProcessingSpeedMode(Enum):
    REALTIME = "Real Time"
    SLOW_MOTION = "Slow-mo"
    MAX_SPEED = "Max Speed"


####################################################################################################
# UI COLORS (Toolbar button states)
####################################################################################################
# Button color scheme for different states
# Format: (R, G, B, A) normalized to 0.0-1.0 range

# Green: For active "running" states (Play when playing, Start Tracking when tracking)
TOOLBAR_BUTTON_GREEN_ACTIVE = (0.0, 0.7, 0.0, 0.7)
TOOLBAR_BUTTON_GREEN_HOVERED = (0.0, 0.85, 0.0, 0.85)
TOOLBAR_BUTTON_GREEN_PRESSED = (0.0, 0.6, 0.0, 0.9)

# Blue: For toggle features (Mode, Show Video, Speed modes, Timeline toggles, etc.)
TOOLBAR_BUTTON_BLUE_ACTIVE = (0.3, 0.5, 0.7, 0.8)
TOOLBAR_BUTTON_BLUE_HOVERED = (0.4, 0.6, 0.8, 0.9)
TOOLBAR_BUTTON_BLUE_PRESSED = (0.2, 0.4, 0.6, 1.0)

# Red: For "stop" or "inactive but important" states
TOOLBAR_BUTTON_RED_ACTIVE = (0.7, 0.0, 0.0, 0.7)
TOOLBAR_BUTTON_RED_HOVERED = (0.85, 0.0, 0.0, 0.85)
TOOLBAR_BUTTON_RED_PRESSED = (0.6, 0.0, 0.0, 0.9)

# Default: Normal button state (inactive)
TOOLBAR_BUTTON_DEFAULT = (0.2, 0.2, 0.2, 0.5)
TOOLBAR_BUTTON_DEFAULT_HOVERED = (0.3, 0.3, 0.3, 0.7)
TOOLBAR_BUTTON_DEFAULT_PRESSED = (0.15, 0.15, 0.15, 0.9)


# TrackerMode enum removed - now using dynamic tracker discovery system
# See config/tracker_discovery.py for the new dynamic approach

# Default tracker will be resolved dynamically from available trackers
DEFAULT_TRACKER_NAME = "axis_projection_working"  # Internal name, resolved at runtime

####################################################################################################
# AI & MODELS
####################################################################################################
AI_MODEL_EXTENSIONS_FILTER = "AI Models (.pt .onnx .engine .mlpackage),.pt;.onnx;.engine;.mlpackage|All Files,*.*"
AI_MODEL_TOOLTIP_EXTENSIONS = ".pt, .onnx, .engine, .mlpackage"


####################################################################################################
# KEYBOARD SHORTCUTS
####################################################################################################
MOD_KEY = "SUPER" if platform.system() == "Darwin" else "CTRL"

DEFAULT_SHORTCUTS = {
    # File Operations (Standard shortcuts across all platforms)
    "save_project": f"{MOD_KEY}+S",
    "open_project": f"{MOD_KEY}+O",

    # Video Navigation (Layout-independent - work on all keyboards)
    "seek_next_frame": "RIGHT_ARROW",
    "seek_prev_frame": "LEFT_ARROW",
    "pan_timeline_left": "ALT+LEFT_ARROW",
    "pan_timeline_right": "ALT+RIGHT_ARROW",
    "jump_to_start": "HOME",
    "jump_to_end": "END",

    # Point Navigation
    "jump_to_next_point": "UP_ARROW",     # Up arrow for next point
    "jump_to_prev_point": "DOWN_ARROW",   # Down arrow for previous point
    "jump_to_next_point_alt": ".",        # Alternative: period
    "jump_to_prev_point_alt": ",",        # Alternative: comma

    # Point Value Adjustment (nudge selected points)
    "nudge_selection_pos_up": "SHIFT+UP_ARROW",      # Shift+Up to raise selected point value
    "nudge_selection_pos_down": "SHIFT+DOWN_ARROW",  # Shift+Down to lower selected point value

    # Timeline View Controls
    "zoom_in_timeline": f"{MOD_KEY}+EQUAL",   # CTRL+= (same key as + on most keyboards)
    "zoom_out_timeline": f"{MOD_KEY}+MINUS",  # CTRL+-

    # Window Toggles
    "toggle_video_display": "V",
    "toggle_timeline2": "T",
    "toggle_gauge_window": "G",
    "toggle_3d_simulator": "S",
    "toggle_movement_bar": "M",
    "toggle_chapter_list": "L",
    "toggle_heatmap": "H",
    "toggle_funscript_preview": "P",
    "toggle_waveform": "W",
    "toggle_video_feed": "F",
    "reset_timeline_view": "R",

    # Editing (Layout-independent modifier+key combinations)
    "delete_selected_point": "DELETE",
    "delete_selected_point_alt": "BACKSPACE",
    "delete_selected_chapter": "DELETE",
    "delete_selected_chapter_alt": "BACKSPACE",
    "delete_points_in_chapter": "SHIFT+DELETE",
    "delete_points_in_chapter_alt": "SHIFT+BACKSPACE",
    "select_all_points": f"{MOD_KEY}+A",
    "deselect_all_points": f"{MOD_KEY}+D",
    "undo_timeline1": f"{MOD_KEY}+Z",
    "redo_timeline1": f"{MOD_KEY}+Y",
    "undo_timeline2": f"{MOD_KEY}+ALT+Z",
    "redo_timeline2": f"{MOD_KEY}+ALT+Y",
    "copy_selection": f"{MOD_KEY}+C",
    "paste_selection": f"{MOD_KEY}+V",

    # Playback (Universal)
    "toggle_playback": "SPACE",

    # Add Points (Number row - layout-independent)
    "add_point_0"   : "0",
    "add_point_10"  : "1",
    "add_point_20"  : "2",
    "add_point_30"  : "3",
    "add_point_40"  : "4",
    "add_point_50"  : "5",
    "add_point_60"  : "6",
    "add_point_70"  : "7",
    "add_point_80"  : "8",
    "add_point_90"  : "9",
    "add_point_100" : "=",           # Changed from "°" (not available on most keyboards)

    # Chapter Creation (Video Editing Style - universal letters)
    "set_chapter_start": "I",        # In-point (start of chapter)
    "set_chapter_end": "O",          # Out-point (end of chapter)
}


####################################################################################################
# UI & DISPLAY
####################################################################################################
# --- Window & Layout ---
DEFAULT_WINDOW_WIDTH = 1800
DEFAULT_WINDOW_HEIGHT = 1000
DEFAULT_UI_LAYOUT = "fixed"  # "fixed" or "floating"

# --- UI Behavior ---
MAX_HISTORY_DISPLAY = 10  # Max number of actions to show in the Undo/Redo history display.
UI_PREVIEW_UPDATE_INTERVAL_S = 1.0  # Interval for updating graphs during live tracking.
DEFAULT_CHAPTER_BAR_HEIGHT = 20  # Height in pixels of the chapter bar.

# --- Timeline & Heatmap Colors (now imported from constants_colors.py) ---
# Timeline colors are now managed through constants_colors.py


####################################################################################################
# INTERFACE PERFORMANCE SETTINGS
####################################################################################################
# --- Font Scale Options ---
FONT_SCALE_LABELS = ["70%", "80%", "90%", "100%", "110%", "125%", "150%", "175%", "200%"]
FONT_SCALE_VALUES = [0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
DEFAULT_FONT_SCALE = 1.0

# --- Timeline Pan Speed ---
TIMELINE_PAN_SPEED_MIN = 1
TIMELINE_PAN_SPEED_MAX = 50
DEFAULT_TIMELINE_PAN_SPEED = 5

# --- Energy Saver Settings ---
ENERGY_SAVER_NORMAL_FPS_MIN = 10
ENERGY_SAVER_THRESHOLD_MIN = 10
ENERGY_SAVER_IDLE_FPS_MIN = 1
DEFAULT_ENERGY_SAVER_NORMAL_FPS = 60
DEFAULT_ENERGY_SAVER_THRESHOLD_SECONDS = 30.0
DEFAULT_ENERGY_SAVER_IDLE_FPS = 10


####################################################################################################
# OBJECT DETECTION & CLASSES
####################################################################################################
CLASS_NAMES_TO_IDS = {
    'face': 0, 'hand': 1, 'penis': 2, 'glans': 3, 'pussy': 4, 'butt': 5,
    'anus': 6, 'breast': 7, 'navel': 8, 'foot': 9, 'hips center': 10
}
CLASS_IDS_TO_NAMES = {v: k for k, v in CLASS_NAMES_TO_IDS.items()}
CLASSES_TO_DISCARD_BY_DEFAULT = ["anus"]

INTERACTION_ZONES = {
    "Cowgirl / Missionary": [6, 7, 12, 13],          # Left/Right Shoulder, Left/Right Hip
    "Rev. Cowgirl / Doggy": [6, 7, 12, 13, 14, 15],   # Shoulders, Hips, Knees
    "Blowjob":              [1, 2, 3, 4, 5, 6, 7],   # Nose, Eyes, Ears, Shoulders (Head region)
    "Handjob":              [8, 9, 10, 11],           # Left/Right Elbow, Left/Right Wrist
    "Boobjob":              [6, 7, 12, 13],          # Shoulders and Hips (Torso area)
    "Footjob":              [14, 15, 16, 17]         # Left/Right Knee, Left/Right Ankle
}

POSE_STABILITY_THRESHOLD = 2.5

####################################################################################################
# FUNSCRIPT & CHAPTERS
####################################################################################################

DEFAULT_CHAPTER_FPS = 30.0
POSITION_INFO_MAPPING = {
    # Scripted positions (category: Position)
    # Combined types (auto-detected by AI)
    "CG/Miss.": {"long_name": "Cowgirl / Missionary", "short_name": "CG/Miss.", "category": "Position"},
    "R.CG/Dog.": {"long_name": "Rev. Cowgirl / Doggy", "short_name": "R.CG/Dog.", "category": "Position"},
    # Individual types (user can manually categorize)
    "CG": {"long_name": "Cowgirl", "short_name": "CG", "category": "Position"},
    "Miss.": {"long_name": "Missionary", "short_name": "Miss.", "category": "Position"},
    "R.CG": {"long_name": "Reverse Cowgirl", "short_name": "R.CG", "category": "Position"},
    "Dog.": {"long_name": "Doggy", "short_name": "Dog.", "category": "Position"},
    # Other positions
    "BJ": {"long_name": "Blowjob", "short_name": "BJ", "category": "Position"},
    "HJ": {"long_name": "Handjob", "short_name": "HJ", "category": "Position"},
    "FootJ": {"long_name": "Footjob", "short_name": "FootJ", "category": "Position"},
    "BoobJ": {"long_name": "Boobjob", "short_name": "BoobJ", "category": "Position"},
    # Non-scripted positions (category: Not Relevant)
    "NR": {"long_name": "Not Relevant", "short_name": "NR", "category": "Not Relevant"},
    "C-Up": {"long_name": "Close Up", "short_name": "C-Up", "category": "Not Relevant"},
    "Intro": {"long_name": "Intro", "short_name": "Intro", "category": "Not Relevant"},
    "Outro": {"long_name": "Outro", "short_name": "Outro", "category": "Not Relevant"},
    "Trans": {"long_name": "Transition", "short_name": "Trans", "category": "Not Relevant"},
}

# Chapter/Segment metadata enums
class ChapterSegmentType(Enum):
    """
    Defines the type/category of a chapter segment.
    Used to classify what kind of content the chapter contains.
    """
    POSITION = "Position"           # Sex position/act (most common)
    TRANSITION = "Transition"       # Scene transition/movement
    INTRO = "Intro"                 # Video intro/setup
    OUTRO = "Outro"                 # Video outro/ending
    NOT_RELEVANT = "Not Relevant"   # Non-relevant content
    UNKNOWN = "Unknown"             # Unclassified

    @classmethod
    def get_default(cls):
        """Get the default segment type."""
        return cls.POSITION

    @classmethod
    def get_all_values(cls):
        """Get list of all segment type values."""
        return [member.value for member in cls]

    @classmethod
    def get_all_names(cls):
        """Get list of all segment type names (for dropdowns)."""
        return [member.name for member in cls]

    @classmethod
    def get_user_category_options(cls):
        """Get simplified category options for user-facing UI (Chapter Type Manager)."""
        return [cls.POSITION.value, cls.NOT_RELEVANT.value]

    @classmethod
    def get_default_for_new_type(cls):
        """Get the default category for creating new custom types."""
        return cls.POSITION.value


class ChapterSource(Enum):
    """
    Tracks how a chapter was created/generated.
    Used for auditing and understanding chapter provenance.
    """
    MANUAL = "manual"                           # User created manually
    MANUAL_DRAG = "manual_drag"                 # User drag-created on timeline
    MANUAL_MERGE = "manual_merge"               # User merged chapters
    MANUAL_SPLIT = "manual_split"               # User split chapter
    STAGE2 = "stage2"                           # AI Stage 2 detection
    STAGE3 = "stage3"                           # AI Stage 3 mixed processing
    STAGE2_FUNSCRIPT = "stage2_funscript"       # From Stage 2 funscript metadata
    STAGE3_FUNSCRIPT = "stage3_funscript"       # From Stage 3 funscript metadata
    IMPORTED = "imported"                       # Imported from external file
    KEYBOARD_SHORTCUT = "keyboard_shortcut"     # Created via keyboard shortcut
    GAP_FILL = "gap_fill"                       # Auto-created to fill gap
    TRACK_AND_MERGE = "track_and_merge"         # Created from track gap & merge operation

    @classmethod
    def get_default(cls):
        """Get the default source."""
        return cls.MANUAL

    @classmethod
    def get_all_values(cls):
        """Get list of all source values."""
        return [member.value for member in cls]

    @classmethod
    def is_ai_generated(cls, source: str) -> bool:
        """Check if source indicates AI generation."""
        ai_sources = {cls.STAGE2.value, cls.STAGE3.value,
                     cls.STAGE2_FUNSCRIPT.value, cls.STAGE3_FUNSCRIPT.value}
        return source in ai_sources

    @classmethod
    def is_user_created(cls, source: str) -> bool:
        """Check if source indicates user creation."""
        user_sources = {cls.MANUAL.value, cls.MANUAL_DRAG.value,
                       cls.MANUAL_MERGE.value, cls.MANUAL_SPLIT.value,
                       cls.KEYBOARD_SHORTCUT.value}
        return source in user_sources


####################################################################################################
# TRACKING & OPTICAL FLOW DEFAULTS
####################################################################################################
DEFAULT_TRACKER_CONFIDENCE_THRESHOLD = 0.4
DEFAULT_TRACKER_ROI_PADDING = 20
DEFAULT_LIVE_TRACKER_SENSITIVITY = 70.0
DEFAULT_LIVE_TRACKER_Y_OFFSET = 0.0
DEFAULT_LIVE_TRACKER_X_OFFSET = 0.0
DEFAULT_LIVE_TRACKER_BASE_AMPLIFICATION = 1.4
DEFAULT_CLASS_AMP_MULTIPLIERS = {"face": 1.25, "hand": 1.5}
DEFAULT_ROI_PERSISTENCE_FRAMES = 450  # 15 seconds @ 30fps (was 180 = 6s, increased for better occlusion handling)
DEFAULT_ROI_SMOOTHING_FACTOR = 0.6
DEFAULT_ROI_UPDATE_INTERVAL = 100
DEFAULT_ROI_NARROW_FACTOR_HJBJ = 0.5
DEFAULT_MIN_ROI_DIM_HJBJ = 10
CLASS_STABILITY_WINDOW = 10
DEFAULT_DIS_FLOW_PRESET = "ULTRAFAST"
DEFAULT_DIS_FINEST_SCALE = 5
DEFAULT_FLOW_HISTORY_SMOOTHING_WINDOW = 3
INVERSION_DETECTION_SPLIT_RATIO = 4.0
MOTION_INVERSION_THRESHOLD = 1.2


####################################################################################################
# STAGE 1: VIDEO DECODING & DETECTION
####################################################################################################
STAGE1_FRAME_QUEUE_MAXSIZE = 99
DEFAULT_S1_NUM_PRODUCERS = 1
DEFAULT_S1_NUM_CONSUMERS = max(os.cpu_count() // 2, 1) if os.cpu_count() else 2


####################################################################################################
# STAGE 2: ANALYSIS & REFINEMENT
####################################################################################################
DEFAULT_S2_OF_WORKERS = min(4, max(1, os.cpu_count() // 4 if os.cpu_count() else 1))
PENIS_CLASS_NAME = "penis"
GLANS_CLASS_NAME = "glans"
CLASS_PRIORITY_ANALYSIS = {"pussy": 8, "butt": 7, "face": 6, "hand": 5, "breast": 4, "foot": 3}
LEAD_BODY_PARTS = ["pussy", "butt", "face", "hand"]
CLASS_INTERACTION_PRIORITY = ["pussy", "butt", "face", "hand", "breast", "foot"]
DEFAULT_S2_ATR_PASS_COUNT = 10

STATUS_DETECTED = "Detected"
STATUS_INTERPOLATED = "Interpolated"
STATUS_OPTICAL_FLOW = "OpticalFlow"
STATUS_SMOOTHED = "Smoothed"
STATUS_POSE_INFERRED = "Pose_Inferred"
STATUS_INFERRED_RELATIVE = "Inferred_Relative"
STATUS_OF_RECOVERED = "OF_Recovered"
STATUS_EXCLUDED_VR = "Excluded_VR_Filter_Peripheral"

S2_LOCKED_PENIS_DEACTIVATION_SECONDS = 3.0

S2_PENIS_INTERPOLATION_MAX_GAP_FRAMES = 30
S2_LOCKED_PENIS_EXTENDED_INTERPOLATION_MAX_FRAMES = 180
S2_CONTACT_EXTENDED_INTERPOLATION_MAX_FRAMES = 5
S2_CONTACT_OPTICAL_FLOW_MAX_GAP_FRAMES = 20
S2_PENIS_LENGTH_SMOOTHING_WINDOW = 15
S2_PENIS_ABSENCE_THRESHOLD_FOR_HEIGHT_RESET = 180
S2_RTS_WINDOW_PADDING = 20
S2_SMOOTH_MAX_FLICKER_DURATION = 60

S2_LEADER_INERTIA_FACTOR = 2  # 1.3  # Challenger must be 30% faster than the incumbent leader.
S2_LEADER_COOLDOWN_SECONDS = 3  # 1.5  # 0.8  # Cooldown period (in seconds) after a leader change.
S2_VELOCITY_SMOOTHING_WINDOW = 7  # 5 # Number of frames to average velocity over.
S2_LEADER_MIN_VELOCITY_THRESHOLD = 5  #0.8 # Pixels/frame. A challenger must move faster than this to be considered a new leader.


####################################################################################################
# STAGE 3: OPTICAL FLOW PROCESSING
####################################################################################################
DEFAULT_S3_WARMUP_FRAMES = 10


####################################################################################################
# AUTO POST-PROCESSING DEFAULTS
####################################################################################################
DEFAULT_AUTO_POST_AMP_CONFIG = {
    "Default": {
        "sg_window": 7, "sg_polyorder": 3, "rdp_epsilon": 15.0, "scale_factor": 1.0, "center_value": 50,
        "clamp_lower": 10, "clamp_upper": 90, "output_min": 0, "output_max": 100
    },
    "Cowgirl / Missionary": {
        "sg_window": 11, "sg_polyorder": 3, "rdp_epsilon": 15.0, "scale_factor": 1.1, "center_value": 50,
        "clamp_lower": 10, "clamp_upper": 90, "output_min": 0, "output_max": 100
    },
    "Rev. Cowgirl / Doggy": {
        "sg_window": 7, "sg_polyorder": 3, "rdp_epsilon": 15.0, "scale_factor": 1.1, "center_value": 50,
        "clamp_lower": 10, "clamp_upper": 90, "output_min": 0, "output_max": 100
    },
    "Blowjob": {
        "sg_window": 7, "sg_polyorder": 3, "rdp_epsilon": 15.0, "scale_factor": 1.3, "center_value": 60,
        "clamp_lower": 10, "clamp_upper": 90, "output_min": 30, "output_max": 100
    },
    "Handjob": {
        "sg_window": 7, "sg_polyorder": 3, "rdp_epsilon": 15.0, "scale_factor": 1.3, "center_value": 60,
        "clamp_lower": 10, "clamp_upper": 90, "output_min": 30, "output_max": 100
    },
    "Boobjob": {
        "sg_window": 7, "sg_polyorder": 3, "rdp_epsilon": 15.0, "scale_factor": 1.2, "center_value": 55,
        "clamp_lower": 10, "clamp_upper": 90, "output_min": 30, "output_max": 100
    },
    "Footjob": {
        "sg_window": 7, "sg_polyorder": 3, "rdp_epsilon": 15.0, "scale_factor": 1.2, "center_value": 50,
        "clamp_lower": 10, "clamp_upper": 90, "output_min": 30, "output_max": 100
    },
    "Close Up": {
        "sg_window": 7, "sg_polyorder": 3, "rdp_epsilon": 15.0, "scale_factor": 1.0, "center_value": 100,
        "clamp_lower": 100, "clamp_upper": 100, "output_min": 100, "output_max": 100
    },
    "Not Relevant": {
        "sg_window": 7, "sg_polyorder": 3, "rdp_epsilon": 15.0, "scale_factor": 1.0, "center_value": 100,
        "clamp_lower": 100, "clamp_upper": 100, "output_min": 100, "output_max": 100
    }
}

# These global fallbacks are now derived from the "Default" profile for consistency.
DEFAULT_AUTO_POST_SG_WINDOW = DEFAULT_AUTO_POST_AMP_CONFIG["Default"]["sg_window"]
DEFAULT_AUTO_POST_SG_POLYORDER = DEFAULT_AUTO_POST_AMP_CONFIG["Default"]["sg_polyorder"]
DEFAULT_AUTO_POST_RDP_EPSILON = DEFAULT_AUTO_POST_AMP_CONFIG["Default"]["rdp_epsilon"]

# The old global clamping constants are no longer the primary source of truth, but can be kept for other uses if needed.
# It's better to derive them from the new dictionary as well to maintain a single source of truth.
DEFAULT_AUTO_POST_CLAMP_LOW = DEFAULT_AUTO_POST_AMP_CONFIG["Default"]["clamp_lower"]
DEFAULT_AUTO_POST_CLAMP_HIGH = DEFAULT_AUTO_POST_AMP_CONFIG["Default"]["clamp_upper"]

####################################################################################################
# DEFAULT MODELS & DOWNLOADS
####################################################################################################
DEFAULT_MODELS_DIR = "models"
MODEL_DOWNLOAD_URLS = {
    "detection_pt": "https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/releases/download/models-v1.1.0/FunGen-12s-pov-1.1.0.pt",
    "detection_mlpackage_zip": "https://github.com/ack00gar/FunGen-AI-Powered-Funscript-Generator/releases/download/models-v1.1.0/FunGen-12s-pov-1.1.0.mlpackage.zip",
    "pose_pt": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt"
}

# Splash screen emoji URLs (optional decorative assets)
# Downloaded to assets/splash/ subdirectory
SPLASH_EMOJI_URLS = {
    'splash/peach.png': 'https://em-content.zobj.net/source/apple/391/peach_1f351.png',
    'splash/eggplant.png': 'https://em-content.zobj.net/source/apple/391/eggplant_1f346.png',
    'splash/cherries.png': 'https://em-content.zobj.net/source/apple/391/cherries_1f352.png',
    'splash/droplet.png': 'https://em-content.zobj.net/source/apple/391/sweat-droplets_1f4a6.png',
    'splash/lips.png': 'https://em-content.zobj.net/source/apple/391/mouth_1f444.png',
    'splash/tongue.png': 'https://em-content.zobj.net/source/apple/391/tongue_1f445.png',
    'splash/woman-face.png': 'https://em-content.zobj.net/source/apple/391/woman_1f469.png',
    'splash/hot-pepper.png': 'https://em-content.zobj.net/source/apple/391/hot-pepper_1f336-fe0f.png',
    'splash/fire.png': 'https://em-content.zobj.net/source/apple/391/fire_1f525.png',
    'splash/banana.png': 'https://em-content.zobj.net/source/apple/391/banana_1f34c.png',
    'splash/lollipop.png': 'https://em-content.zobj.net/source/apple/391/lollipop_1f36d.png'
}

# UI control icon URLs (Apple emoji style)
# These icons are used throughout the UI for buttons and controls
# Downloaded to assets/ui/icons/ subdirectory
UI_CONTROL_ICON_URLS = {
    # Playback controls (video display overlay)
    'ui/icons/jump-start.png': 'https://em-content.zobj.net/source/apple/391/last-track-button_23ee-fe0f.png',
    'ui/icons/prev-frame.png': 'https://em-content.zobj.net/source/apple/391/fast-reverse-button_23ea.png',
    'ui/icons/play.png': 'https://em-content.zobj.net/source/apple/391/play-button_25b6-fe0f.png',
    'ui/icons/pause.png': 'https://em-content.zobj.net/source/apple/391/pause-button_23f8-fe0f.png',
    'ui/icons/stop.png': 'https://em-content.zobj.net/source/apple/391/stop-button_23f9-fe0f.png',
    'ui/icons/next-frame.png': 'https://em-content.zobj.net/source/apple/391/fast-forward-button_23e9.png',
    'ui/icons/jump-end.png': 'https://em-content.zobj.net/source/apple/391/next-track-button_23ed-fe0f.png',

    # View/zoom controls - using magnifying glass (+ and - emojis blocked by CDN)
    'ui/icons/zoom-in.png': 'https://em-content.zobj.net/source/apple/391/magnifying-glass-tilted-right_1f50e.png',
    'ui/icons/zoom-out.png': 'https://em-content.zobj.net/source/apple/391/magnifying-glass-tilted-left_1f50d.png',
    'ui/icons/reset.png': 'https://em-content.zobj.net/source/apple/391/counterclockwise-arrows-button_1f504.png',

    # Fullscreen controls - cinema for enter, door for exit
    'ui/icons/fullscreen.png': 'https://em-content.zobj.net/source/apple/391/cinema_1f3a6.png',
    'ui/icons/fullscreen-exit.png': 'https://em-content.zobj.net/source/apple/391/door_1f6aa.png',

    # Video display control
    'ui/icons/video-camera.png': 'https://em-content.zobj.net/source/apple/391/movie-camera_1f3a5.png',
    'ui/icons/video-show.png': 'https://em-content.zobj.net/source/apple/391/movie-camera_1f3a5.png',  # 🎥 movie camera for show video
    'ui/icons/video-hide.png': 'https://em-content.zobj.net/source/apple/391/no-one-under-eighteen_1f51e.png',  # 🔞 no one under 18 for hide video

    # General UI controls
    'ui/icons/settings.png': 'https://em-content.zobj.net/source/apple/391/gear_2699-fe0f.png',
    'ui/icons/gamepad.png': 'https://em-content.zobj.net/source/apple/391/video-game_1f3ae.png',

    # File operations
    'ui/icons/folder.png': 'https://em-content.zobj.net/source/apple/391/file-folder_1f4c1.png',
    'ui/icons/folder-open.png': 'https://em-content.zobj.net/source/apple/391/open-file-folder_1f4c2.png',
    'ui/icons/save.png': 'https://em-content.zobj.net/source/apple/391/floppy-disk_1f4be.png',
    'ui/icons/save-as.png': 'https://em-content.zobj.net/source/apple/391/floppy-disk_1f4be.png',  # Using same as save
    'ui/icons/document-new.png': 'https://em-content.zobj.net/source/apple/391/page-facing-up_1f4c4.png',
    'ui/icons/import.png': 'https://em-content.zobj.net/source/apple/391/inbox-tray_1f4e5.png',
    'ui/icons/export.png': 'https://em-content.zobj.net/source/apple/391/outbox-tray_1f4e4.png',

    # Edit operations
    'ui/icons/undo.png': 'https://em-content.zobj.net/source/apple/391/right-arrow-curving-left_21a9-fe0f.png',
    'ui/icons/redo.png': 'https://em-content.zobj.net/source/apple/391/left-arrow-curving-right_21aa-fe0f.png',
    'ui/icons/edit.png': 'https://em-content.zobj.net/source/apple/391/pencil_270f-fe0f.png',
    'ui/icons/trash.png': 'https://em-content.zobj.net/source/apple/391/wastebasket_1f5d1-fe0f.png',

    # Chapter management
    'ui/icons/plus-circle.png': 'https://em-content.zobj.net/source/apple/391/plus_2795.png',  # Fixed: original URL was 403
    'ui/icons/merge.png': 'https://em-content.zobj.net/source/apple/391/link_1f517.png',
    'ui/icons/scissors.png': 'https://em-content.zobj.net/source/apple/391/scissors_2702-fe0f.png',

    # Status indicators
    'ui/icons/check.png': 'https://em-content.zobj.net/source/apple/391/check-mark-button_2705.png',
    'ui/icons/checkmark.png': 'https://em-content.zobj.net/source/apple/391/check-mark_2714-fe0f.png',  # ✓ simple checkmark
    'ui/icons/error.png': 'https://em-content.zobj.net/source/apple/391/cross-mark_274c.png',
    'ui/icons/warning.png': 'https://em-content.zobj.net/source/apple/391/warning_26a0-fe0f.png',
    'ui/icons/energy-leaf.png': 'https://em-content.zobj.net/source/apple/391/leaf-fluttering-in-wind_1f343.png',  # 🍃 leaf for energy saver
    'ui/icons/red-circle.png': 'https://em-content.zobj.net/source/apple/391/large-red-circle_1f534.png',  # 🔴 red circle for real-time (fixed: original was 403)
    'ui/icons/wrench.png': 'https://em-content.zobj.net/source/apple/391/wrench_1f527.png',  # 🔧 wrench for tools/devices

    # Chapter source indicators
    'ui/icons/robot.png': 'https://em-content.zobj.net/source/apple/391/robot_1f916.png',  # 🤖 AI generated
    'ui/icons/user.png': 'https://em-content.zobj.net/source/apple/391/person_1f9d1.png',  # 👤 User created
    'ui/icons/download.png': 'https://em-content.zobj.net/source/apple/391/down-arrow_2b07-fe0f.png',  # ⬇️ Imported

    # Timeline/number indicators (for toolbar) - using keycap digits
    'ui/icons/keycap-1.png': 'https://em-content.zobj.net/source/apple/391/keycap-digit-one_31-fe0f-20e3.png',  # 1️⃣ keycap digit one
    'ui/icons/keycap-2.png': 'https://em-content.zobj.net/source/apple/391/keycap-digit-two_32-fe0f-20e3.png',  # 2️⃣ keycap digit two

    # Chapter/content indicators
    'ui/icons/books.png': 'https://em-content.zobj.net/source/apple/391/books_1f4da.png',  # 📚 books for chapter list

    # Network/streaming indicators
    'ui/icons/satellite.png': 'https://em-content.zobj.net/source/apple/391/satellite-antenna_1f4e1.png',
    'ui/icons/flashlight.png': 'https://em-content.zobj.net/source/apple/391/flashlight_1f526.png',  # 🔦 flashlight for device control
    'ui/icons/page-facing-up.png': 'https://em-content.zobj.net/source/apple/391/page-facing-up_1f4c4.png',  # 📄 page for script loaded indicator
    'ui/icons/counterclockwise-arrows.png': 'https://em-content.zobj.net/source/apple/391/counterclockwise-arrows-button_1f504.png',  # 🔄 sync toggle for Handy pause/resume

    # Enhancement/optimization indicators
    'ui/icons/rocket.png': 'https://em-content.zobj.net/source/apple/391/rocket_1f680.png',  # 🚀 rocket
    'ui/icons/magic-wand.png': 'https://em-content.zobj.net/source/apple/391/magic-wand_1fa84.png',  # 🪄 magic wand for Ultimate Autotune

    # Tracking/automation indicators
    'ui/icons/robot.png': 'https://em-content.zobj.net/source/apple/391/robot_1f916.png',  # 🤖 robot for tracking
    'ui/icons/sparkles.png': 'https://em-content.zobj.net/source/apple/391/sparkles_2728.png',  # ✨ sparkles for auto post-processing
    'ui/icons/bot.png': 'https://em-content.zobj.net/source/apple/391/robot_1f916.png',  # 🤖 bot for live tracking button
    'ui/icons/eraser.png': 'https://em-content.zobj.net/source/apple/391/eraser_1f9fd.png',  # 🧽 eraser for clear chapters
    'ui/icons/nerd-face.png': 'https://em-content.zobj.net/source/apple/391/nerd-face_1f913.png',  # 🤓 nerd face for Expert/Simple mode toggle

    # Playback speed mode indicators
    'ui/icons/speed-realtime.png': 'https://em-content.zobj.net/source/apple/391/person-walking_1f6b6.png',  # 🚶 person walking for Real Time
    'ui/icons/speed-slowmo.png': 'https://em-content.zobj.net/source/apple/391/turtle_1f422.png',  # 🐢 turtle for Slow-mo
    'ui/icons/speed-max.png': 'https://em-content.zobj.net/source/apple/391/rabbit_1f407.png',  # 🐇 rabbit for Max Speed

    # Visualization/analysis indicators
    'ui/icons/chart-increasing.png': 'https://em-content.zobj.net/source/apple/391/chart-increasing_1f4c8.png',  # 📈 chart for 3D simulator
}

####################################################################################################
# UPDATER & GITHUB
####################################################################################################
DEFAULT_COMMIT_FETCH_COUNT = 15


STATUS_SYNTHESIZED_KALMAN = "Synthesized_Kalman"
STATUS_GENERATED_PROPAGATED = "Generated_Propagated"
STATUS_GENERATED_LINEAR = "Generated_Linear"
STATUS_GENERATED_RTS = "Generated_RTS"

# Thresholds from generate_in_between_boxes.py
SHORT_GAP_THRESHOLD = 2
LONG_GAP_THRESHOLD = 30
RTS_WINDOW_PADDING = 20 # Frames to include before start_frame and after end_frame for RTS window

# --- Constants from new helper scripts ---
# From smooth_tracked_classes.py
SIZE_SMOOTHING_FRAMES_CONST = 30

# From generate_tracked_classes.py
FILTER_BOXES_AREA_TO_LOCKED_CONST = {
    "pussy": 10, "butt": 40, "face": 15, "hand": 6, "breast": 25, "foot": 6,
}
FILTER_BOXES_AREA_TO_LOCKED_MIN_CONST = {"foot": 1}
CENTER_SCREEN_CONST = 320  # Assuming YOLO input size of 640 / 2. Should be dynamic.
CENTER_SCREEN_FOCUS_AREA_CONST = 320  # Example, make dynamic or pass based on yolo_input_size

import platform
import os
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
APP_VERSION = "0.5.0"
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
PROJECT_FILE_EXTENSION = ".fgnproj"
DEFAULT_OUTPUT_FOLDER = "output"


####################################################################################################
# SYSTEM & PERFORMANCE
####################################################################################################
# Determines the compute device for ML models (e.g., 'cuda', 'mps', 'cpu').
# This is detected once and used by both Stage 1 and the live tracker.
DEVICE = 'cpu'
if torch:
    if platform.processor() == 'arm' and platform.system() == 'Darwin':
        DEVICE = 'mps'
    elif torch.cuda.is_available():
        DEVICE = 'cuda'

# The side length of the square input image for the YOLO model.
YOLO_INPUT_SIZE = 640
# Fallback for determining producer/consumer counts if os.cpu_count() fails.
DEFAULT_FALLBACK_CPU_CORES = 4

class TrackerMode(Enum):
    LIVE_YOLO_ROI = "Live Optical Flow (YOLO ROI)"
    LIVE_USER_ROI = "Live Optical Flow (User ROI)"
    OFFLINE_2_STAGE = "YOLO AI (2 Stages, legacy)"
    OFFLINE_3_STAGE = "YOLO AI + Opt. Flow (3 Stages)"


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
    "seek_next_frame": "RIGHT_ARROW",
    "seek_prev_frame": "LEFT_ARROW",
    "jump_to_next_point": ".",
    "jump_to_prev_point": ",",
    "pan_timeline_left": "ALT+LEFT_ARROW",
    "pan_timeline_right": "ALT+RIGHT_ARROW",
    "delete_selected_point": "DELETE",
    "delete_selected_point_alt": "BACKSPACE",
    "select_all_points": f"{MOD_KEY}+A",
    "undo_timeline1": f"{MOD_KEY}+Z",
    "redo_timeline1": f"{MOD_KEY}+Y",
    "undo_timeline2": f"{MOD_KEY}+ALT+Z",
    "redo_timeline2": f"{MOD_KEY}+ALT+Y",
    "copy_selection": f"{MOD_KEY}+C",
    "paste_selection": f"{MOD_KEY}+V",
    "toggle_playback": "SPACE",
    "add_point_0" : "0",
    "add_point_10" : "1",
    "add_point_20" : "2",
    "add_point_30" : "3",
    "add_point_40" : "4",
    "add_point_50" : "5",
    "add_point_60" : "6",
    "add_point_70" : "7",
    "add_point_80" : "8",
    "add_point_90" : "9",
    "add_point_100" : "°",
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

# --- Timeline & Heatmap Colors ---
TIMELINE_HEATMAP_COLORS = [
    (30, 144, 255),   # Dodger Blue
    (34, 139, 34),    # Lime Green
    (255, 215, 0),    # Gold
    (220, 20, 60),    # Crimson
    (147, 112, 219),  # Medium Purple
    (37, 22, 122)     # Dark Blue (Cap color)
]
TIMELINE_COLOR_SPEED_STEP = 250.0  # Speed (pixels/sec) used to step through the color map.
TIMELINE_COLOR_ALPHA = 0.9

# --- Default Chapter Colors (used in video_segment.py) ---
DEFAULT_CHAPTER_COLORS = {
    "BJ": (0.9, 0.4, 0.4, 0.8),
    "HJ": (0.4, 0.9, 0.4, 0.8),
    "NR": (0.6, 0.6, 0.6, 0.7),
    "CG/Miss.": (0.4, 0.4, 0.9, 0.8),
    "R.CG/Dog.": (0.9, 0.6, 0.2, 0.8),
    "FootJ": (0.9, 0.9, 0.3, 0.8),
    "BoobJ": (0.9, 0.3, 0.6, 0.8),
    "C-Up": (0.7, 0.7, 0.9, 0.8),
    "default": (0.5, 0.5, 0.5, 0.7)
}


####################################################################################################
# OBJECT DETECTION & CLASSES
####################################################################################################
CLASS_NAMES_TO_IDS = {
    'face': 0, 'hand': 1, 'penis': 2, 'glans': 3, 'pussy': 4, 'butt': 5,
    'anus': 6, 'breast': 7, 'navel': 8, 'foot': 9, 'hips center': 10
}
CLASS_IDS_TO_NAMES = {v: k for k, v in CLASS_NAMES_TO_IDS.items()}
CLASS_COLORS = {
    "penis": (255, 0, 0), "glans": (0, 128, 0), "pussy": (0, 0, 255), "butt": (0, 180, 255),
    "anus": (128, 0, 128), "breast": (255, 165, 0), "navel": (0, 255, 255),
    "hand": (255, 0, 255), "face": (0, 255, 0), "foot": (165, 42, 42),
    "hips center": (0, 0, 0), "locked_penis": (0, 255, 255),
}
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
SCENE_DETECTION_DEFAULT_THRESHOLD = 27.0
DEFAULT_CHAPTER_FPS = 30.0
POSITION_INFO_MAPPING = {
    "NR": {"long_name": "Not Relevant", "short_name": "NR"},
    "C-Up": {"long_name": "Close Up", "short_name": "C-Up"},
    "CG/Miss.": {"long_name": "Cowgirl / Missionary", "short_name": "CG/Miss."},
    "R.CG/Dog.": {"long_name": "Rev. Cowgirl / Doggy", "short_name": "R.CG/Dog"},
    "BJ": {"long_name": "Blowjob", "short_name": "BJ"},
    "HJ": {"long_name": "Handjob", "short_name": "HJ"},
    "FootJ": {"long_name": "Footjob", "short_name": "FootJ"},
    "BoobJ": {"long_name": "Boobjob", "short_name": "BoobJ"},
}


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
DEFAULT_ROI_PERSISTENCE_FRAMES = 180
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

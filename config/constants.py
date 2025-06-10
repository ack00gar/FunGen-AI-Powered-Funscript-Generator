import os
import platform


VERSION = "0.5.0"
OBJECT_DETECTION_VERSION = "1.0.0"
TRACKING_VERSION = "0.1.1"
FUNSCRIPT_VERSION = "0.2.0"
CONFIG_VERSION = 1

SETTINGS_FILE = "settings.json"
AUTOSAVE_FILE = "autosave.fgnstate"
PROJECT_FILE_EXTENSION = ".fgnproj"

# For Chapters export in funscript
DEFAULT_CHAPTER_FPS = 30.0

# Undo-Redo Manager display
MAX_HISTORY_DISPLAY = 10

##################################################################################################
# Live Optical Flow Tracker
##################################################################################################

DEFAULT_SENSITIVITY = 50.0
DEFAULT_Y_OFFSET = 0.0
DEFAULT_X_OFFSET = 0.0
DEFAULT_BASE_AMPLIFICATION = 1.5
DEFAULT_ROI_PERSISTENCE_FRAMES = 120  # 2 seconds at 60fps
DEFAULT_ROI_SMOOTHING_FACTOR = 0.6
DEFAULT_ROI_UPDATE_INTERVAL = 100
DEFAULT_ROI_NARROW_FACTOR_HJBJ = 0.5
DEFAULT_MIN_ROI_DIM_HJBJ = 10

# Define default multipliers for specific classes
DEFAULT_CLASS_AMP_MULTIPLIERS = {
    "face": 2.5,  # Face interaction gets 50% more amplification
    "hand": 2.0,  # Hand interaction gets 30% more amplification
}

##################################################################################################
# STAGE 1
##################################################################################################

QUEUE_MAXSIZE = 99

##################################################################################################
# STAGE 2
##################################################################################################

# --- Constants from original stage_2_OF_filler_chapters_dev.py ---
PENIS_CLASS_NAME_CONST = "penis"
GLANS_CLASS_NAME_CONST = "glans"
CLASS_PRIORITY_CONST = {"pussy": 8, "butt": 7, "face": 6, "hand": 5, "breast": 4, "foot": 3}
LEAD_BODY_PARTS_CONST = ["pussy", "butt", "face", "hand"]
POSITION_INFO_MAPPING_CONST = {
    "NR": {"long_name": "Not Relevant", "short_name": "NR"},
    "C-Up": {"long_name": "Close Up", "short_name": "C-Up"},
    "CG/Miss.":   {"long_name": "Cowgirl / Missionary", "short_name": "CG/Miss."},
    "R.CG/Dog.":    {"long_name": "Rev. Cowgirl / Doggy", "short_name": "R.CG/Dog"},
    "BJ":      {"long_name": "Blowjob", "short_name": "BJ"},
    "HJ":      {"long_name": "Handjob", "short_name": "HJ"},
    "FootJ":    {"long_name": "Footjob", "short_name": "FootJ"},
    "BoobJ":  {"long_name": "Boobjob", "short_name": "BoobJ"},
    # Deprecating old combined keys by removing them or pointing them to a default
    # The 'face' and 'hand' keys are now ambiguous and should be removed to enforce the new specific logic.
}



CLASSES_TO_DISCARD_CONST = ["anus"]

PENIS_INTERPOLATION_MAX_GAP_CONST = 30
LOCKED_PENIS_EXTENDED_INTERPOLATION_MAX_FRAMES_CONST = 180
CONTACT_EXTENDED_INTERPOLATION_MAX_FRAMES_CONST = 5
CONTACT_OPTICAL_FLOW_MAX_GAP_FRAMES_CONST = 20
OPTICAL_FLOW_CALC_ROI_MARGIN_CONST = 20
PENIS_OF_ROI_MARGIN_CONST = 5
PENIS_BASE_Y2_MARGIN_CONST = 5
PENIS_LENGTH_SMOOTHING_WINDOW_CONST = 15
PENIS_ABSENCE_THRESHOLD_FOR_HEIGHT_RESET_CONST = 180
SMOOTH_MAX_FLICKER_DURATION_CONST = 60
SMOOTH_MAX_SHORT_DURATION_CONST = 180
SMOOTH_MIN_BRACKETING_DURATION_CONST = 10
SMOOTH_MAX_INTER_GAP_CONST = 1
SMOOTH_PRIORITY_ADVANTAGE_FOR_B_TO_RESIST_FLICKER_OVERRIDE_CONST = 2

STATUS_DETECTED = "Detected"
STATUS_INTERPOLATED = "Interpolated"
STATUS_OPTICAL_FLOW = "OpticalFlow"
STATUS_SYNTHESIZED_KALMAN = "Synthesized_Kalman"
STATUS_SMOOTHED = "Smoothed"
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

# From generate_cuts_and_positions.py
# FPS will be dynamic. These are multipliers or absolute frame counts.
# Example: FRAME_CHECK_INTERVAL_SECONDS = 2 -> fps * 2
MAX_NO_PENIS_FRAGMENTS_CONST = 8  # (e.g., 8 * 2-second fragments = 16 seconds)
MAX_PENIS_WIDTH_CHANGE_CONST = 3.0
MAX_PENIS_WIDTH_CHANGE_SUDDEN_CONST = 3.0
MAX_PENIS_WIDTH_DIST_DELTA_CONST = 2.5
MAX_PENIS_WIDTH_DIST_DELTA_SUDDEN_CONST = 1.5
# MIN_PENIS_DETECTIONS_PER_FRAGMENT_FACTOR = 1/4 (e.g., fps / 4)
# PENIS_DETECTIONS_OUTLIER_THRESHOLD_FACTOR = 1/6 (e.g., fps / 6)



##################################################################################################
# PERFORMANCE
##################################################################################################

RENDER_RESOLUTION = 640
#TEXTURE_RESOLUTION = RENDER_RESOLUTION * 1.3  # Texture size that is used to texture the opengl sphere
#YOLO_BATCH_SIZE = 1 if platform.system() == "Darwin" else 30  # Mac doesn't support batching. Note TensorRT (.engine) and .onnx is compiled for a batch size of 30
#YOLO_PERSIST = True  # Big impact on performance but also improves tracking



##################################################################################################
# OBJECT DETECTION
##################################################################################################

CLASS_TYPES = {
    'face': 0, 'hand': 1, 'penis': 2, 'glans': 3, 'pussy': 4, 'butt': 5,
    'anus': 6, 'breast': 7, 'navel': 8, 'foot': 9, 'hips center': 10
}
CLASS_REVERSE_MATCH = {
    0: 'penis', 1: 'glans', 2: 'pussy', 3: 'butt', 4: 'anus', 5: 'breast',
    6: 'navel', 7: 'hand', 8: 'face', 9: 'foot', 10: 'hips center'
}
CLASS_PRIORITY_ORDER = {
    "glans": 0, "penis": 1, "breast": 2, "navel": 3, "pussy": 4, "butt": 5, "face": 6
}
CLASS_NAMES = {
    'face': 0,
    'hand': 1,  # 'left hand': 1, 'right hand': 1,
    'penis': 2,
    'glans': 3,
    'pussy': 4,
    'butt': 5,
    'anus': 6,
    'breast': 7,
    'navel': 8,
    'foot': 9, 'left foot': 9, 'right foot': 9,
    'hips center': 10
}
CLASS_COLORS = {
    "locked_penis": (0, 255, 255), # yellow
    "penis": (255, 0, 0),  # red
    "glans": (0, 128, 0),  # green
    "pussy": (0, 0, 255),  # blue
    "butt": (0, 180, 255),  # deep yellow
    "anus": (128, 0, 128),  # purple
    "breast": (255, 165, 0),  # orange
    "navel": (0, 255, 255),  # cyan
    "hand": (255, 0, 255),  # magenta
    "left hand": (255, 0, 255),  # magenta
    "right hand": (255, 0, 255),  # magenta
    "face": (0, 255, 0),  # lime
    "foot": (165, 42, 42),  # brown
    "left foot": (165, 42, 42),  # brown
    "right foot": (165, 42, 42),  # brown
    "hips center": (0, 0, 0)
}
HEATMAP_COLORS = [
    [0, 0, 0],  # Black (no action)
    [30, 144, 255],  # Dodger blue
    [34, 139, 34],  # Lime green
    [255, 215, 0],  # Gold
    [220, 20, 60],  # Crimson
    [147, 112, 219],  # Medium purple
    [37, 22, 122]  # Dark blue
]

# --- Color definitions ---
HEATMAP_COLORS_TIMELINE = [
    # [0, 0, 0],  # Black (for speed <= 0)
    [30, 144, 255],  # Dodger blue
    [34, 139, 34],  # Lime green
    [255, 215, 0],  # Gold
    [220, 20, 60],  # Crimson
    [147, 112, 219],  # Medium purple
    [37, 22, 122]  # Dark blue (cap color for high speed)
]
STEP_SIZE_TIMELINE = 250.0
TIMELINE_COLOR_ALPHA = 0.9


##################################################################################################
# DIV
##################################################################################################

FUNSCRIPT_AUTHOR = "FunGen"
OLD_FUNSCRIPT_AUTHOR = "FunGen_k00gar_AI"
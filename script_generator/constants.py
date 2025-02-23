import os
import platform

import torch

VERSION = "0.3.0"
OBJECT_DETECTION_VERSION = "1.0.0"
TRACKING_VERSION = "0.1.0"
FUNSCRIPT_VERSION = "0.1.0"
CONFIG_VERSION = 1

##################################################################################################
# PERFORMANCE
##################################################################################################

RENDER_RESOLUTION = 640
TEXTURE_RESOLUTION = RENDER_RESOLUTION * 1.3  # Texture size that is used to texture the opengl sphere
YOLO_BATCH_SIZE = 1 if platform.system() == "Darwin" or not torch.cuda.is_available() else 30  # Mac doesn't support batching. Note TensorRT (.engine) is compiled for a batch size of 30
YOLO_PERSIST = True  # Big impact on performance but also improves tracking

##################################################################################################
# ADVANCED
##################################################################################################

YOLO_CONF = 0.3
VR_TO_2D_PITCH = -21  # The dataset is trained on -25
UPDATE_PROGRESS_INTERVAL = 0.2  # Updates progress in the console and in gui
STEP_SIZE = 120  # Define custom colormap based on Lucife's heatmapColors | Speed step size for color transitions
QUEUE_MAXSIZE = 100  # Bounded queue size to avoid memory blow-up as raw frames consume a lot of memory, does not increase performance

##################################################################################################
# DEV
##################################################################################################

# when enabled the queue will be processed one by one (use it on (QUEUE_MAXSIZE / frame rate) seconds longer videos or less)
# raw frames take a lot of memory (RAM) so don't set the queue to high
SEQUENTIAL_MODE = False
if SEQUENTIAL_MODE:
    QUEUE_MAXSIZE = 3000

##################################################################################################
# DEFAULT CONFIG
##################################################################################################

DEFAULT_CONFIG = {
    "config_version": CONFIG_VERSION,
    "ffmpeg_path": None,
    "ffprobe_path": None,
    "ffmpeg_hwaccel": None,
    "yolo_model_path": None,
    "copy_funscript_to_movie_dir": True,
    "funscript_output_dir": None,
    "make_funscript_backup": True,
    "log_level": "INFO",
    "tracking_logic_version": 1
}

##################################################################################################
# PROG
##################################################################################################

RUN_POSE_MODEL = False
YOLO_POSE_MODEL = None  # YOLO("models/yolo11n-pose.mlpackage", task="pose") #TODO pose model?
VALID_VIDEO_READERS = ["FFmpeg", "FFmpeg + OpenGL (Windows)"]
VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv"}

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
    'hand': 1, 'left hand': 1, 'right hand': 1,
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

##################################################################################################
# PATHS
##################################################################################################

PROJECT_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_PATH = os.path.join(PROJECT_PATH, "output")
MODELS_PATH = os.path.join(PROJECT_PATH, "models")
MODEL_FILENAMES = [
    "FunGen-12s-pov-1.0.0.engine",
    "FunGen-12s-pov-1.0.0.pt",
    "FunGen-12s-pov-1.0.0.mlpackage",
    "FunGen-12s-pov-1.0.0.onnx",
    "FunGen-12s-mix-1.0.0.engine",
    "FunGen-12s-mix-1.0.0.mlpackage",
    "FunGen-12s-mix-1.0.0.pt",
    "FunGen-12s-mix-1.0.0.onnx",
    "FunGen-11n-mix-1.0.0.engine",
    "FunGen-11n-mix-1.0.0.mlpackage",
    "FunGen-11n-mix-1.0.0.onnx",
    "FunGen-11n-mix-1.0.0.pt",
    "FunGen-11s-mix-1.0.0.engine",
    "FunGen-11s-mix-1.0.0.mlpackage",
    "FunGen-11s-mix-1.0.0.onnx",
    "FunGen-11s-mix-1.0.0.pt",
    "k00gar-11n-RGB-200ep-best.mlpackage",
    "k00gar-11n-RGB-200ep-best.pt",
    "k00gar-11n-RGB-200ep-best.onnx"
]
LOGO = os.path.join(PROJECT_PATH, "resources", "logo.png")
ICON = os.path.join(PROJECT_PATH, "resources", "icon.ico")
CONFIG_FILE_PATH = os.path.join(PROJECT_PATH, "config.json")

##################################################################################################
# DEBUG VIDEO
##################################################################################################

FUNSCRIPT_BUFFER_SIZE = 500

##################################################################################################
# KEY CODES
##################################################################################################

LEFT = 2424832
RIGHT = 2555904
SPACE = 32
Q = 113
COMMA = 44
PERIOD = 46
LEFT_BRACKET = 91
RIGHT_BRACKET = 93

##################################################################################################
# DIV
##################################################################################################

FUNSCRIPT_AUTHOR = "FunGen"
OLD_FUNSCRIPT_AUTHOR = "FunGen_k00gar_AI"
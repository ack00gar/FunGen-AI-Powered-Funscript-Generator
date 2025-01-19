import shutil
import platform
import os

VERSION = "0.0.1_25-01-16"

# ffmpeg and ffprobe paths - replace with your own if not in your system path
win_ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"
mac_ffmpeg_path = "/usr/local/bin/ffmpeg"
lin_ffmpeg_path = "/usr/bin/ffmpeg"

win_ffprobe_path = "C:/ffmpeg/bin/ffprobe.exe"
mac_ffprobe_path = "/usr/local/bin/ffprobe"
lin_ffprobe_path = "/usr/bin/ffprobe"

# Yolo detection settings
MAX_FRAME_HEIGHT = 1080

PROJECT_PATH = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(PROJECT_PATH, "output")

# Define the base models directory
MODELS_DIR = "models"

# List of model filenames
MODEL_FILENAMES = [
    "k00gar-11n-200ep-best.mlpackage",
    "k00gar-11n-200ep-best.pt",
    "k00gar-11n-200ep-best.onnx",
    "yolo11n-pose.mlpackage",
    "yolo11n-pose.pt",
    "yolo11n-pose.onnx"
]


def find_ffmpeg_path():
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        return ffmpeg_path
    else:
        print("ffmpeg not found in PATH, defaulting to platform values in params/config.py")

        if platform.system() == "Windows":
            return win_ffmpeg_path
        elif platform.system() == "Darwin":
            return mac_ffmpeg_path
        elif platform.system() == "Linux":
            return lin_ffmpeg_path
        else:
            raise FileNotFoundError("ffmpeg not found in PATH and no default path for this OS")


def find_ffprobe_path():
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        return ffprobe_path
    else:
        print("ffprobe not found in PATH, defaulting to platform values in params/config.py")

        if platform.system() == "Windows":
            return win_ffprobe_path
        elif platform.system() == "Darwin":
            return mac_ffprobe_path
        elif platform.system() == "Linux":
            return lin_ffprobe_path
        else:
            raise FileNotFoundError("ffprobe not found in PATH and no default path for this OS")


# Set the paths in your config
FFMPEG_PATH = find_ffmpeg_path()
FFPROBE_PATH = find_ffprobe_path()

print(f"ffmpeg_path: {FFMPEG_PATH}")
print(f"ffprobe_path: {FFPROBE_PATH}")

# Construct full paths using os.path.join
YOLO_MODELS = [os.path.join(MODELS_DIR, filename) for filename in MODEL_FILENAMES]

# Class types
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

# Define class names with their corresponding indices
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

# Define colors for each class
CLASS_COLORS = {
    "penis": (255, 0, 0),  # red
    "glans": (0, 128, 0),  # green
    "pussy": (0, 0, 255),  # blue
    "butt": (255, 255, 0),  # yellow
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

# Define custom colormap based on Lucife's heatmapColors
HEATMAP_COLORS = [
    [0, 0, 0],  # Black (no action)
    [30, 144, 255],  # Dodger blue
    [34, 139, 34],  # Lime green
    [255, 215, 0],  # Gold
    [220, 20, 60],  # Crimson
    [147, 112, 219],  # Medium purple
    [37, 22, 122]  # Dark blue
]
STEP_SIZE = 120  # Speed step size for color transitions

VW_FILTER_COEFF = 2.0
LOGO = os.path.join(PROJECT_PATH, "resources", "logo.png")
ICON = os.path.join(PROJECT_PATH, "resources", "icon.ico")
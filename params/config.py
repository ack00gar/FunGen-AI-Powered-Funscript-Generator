import shutil
import platform
import os

# ffmpeg and ffprobe paths - replace with your own if not in your system path
win_ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"
mac_ffmpeg_path = "/usr/local/bin/ffmpeg"
lin_ffmpeg_path = "/usr/bin/ffmpeg"

win_ffprobe_path = "C:/ffmpeg/bin/ffprobe.exe"
mac_ffprobe_path = "/usr/local/bin/ffprobe"
lin_ffprobe_path = "/usr/bin/ffprobe"

# First, we will try to find the ffmpeg and ffprobe paths in the system PATH

def find_ffmpeg_path():
    # Try to find ffmpeg in the system PATH
    ffmpeg_path = shutil.which("ffmpeg")
    if ffmpeg_path:
        print(f"Found ffmpeg in PATH: {ffmpeg_path}")
        return ffmpeg_path
    else:
        print("ffmpeg not found in PATH, defaulting to platform values in params/config.py")
        # Fallback paths for different operating systems
        if platform.system() == "Windows":
            return win_ffmpeg_path  # Example default path for Windows
        elif platform.system() == "Darwin":  # macOS
            return mac_ffmpeg_path  # Example default path for macOS
        elif platform.system() == "Linux":
            return lin_ffmpeg_path  # Example default path for Linux
        else:
            raise FileNotFoundError("ffmpeg not found in PATH and no default path for this OS")

def find_ffprobe_path():
    # Try to find ffprobe in the system PATH
    ffprobe_path = shutil.which("ffprobe")
    if ffprobe_path:
        print(f"Found ffprobe in PATH: {ffprobe_path}")
        return ffprobe_path
    else:
        print("ffprobe not found in PATH, defaulting to platform values in params/config.py")
        # Fallback paths for different operating systems
        if platform.system() == "Windows":
            return win_ffprobe_path  # Example default path for Windows
        elif platform.system() == "Darwin":  # macOS
            return mac_ffprobe_path  # Example default path for macOS
        elif platform.system() == "Linux":
            return lin_ffprobe_path  # Example default path for Linux
        else:
            raise FileNotFoundError("ffprobe not found in PATH and no default path for this OS")

# Set the paths in your config
ffmpeg_path = find_ffmpeg_path()
ffprobe_path = find_ffprobe_path()

print(f"ffmpeg_path: {ffmpeg_path}")
print(f"ffprobe_path: {ffprobe_path}")

# models path

# Define the base models directory
models_dir = "models"

# List of model filenames
model_filenames = [
    "k00gar-11n-200ep-best.mlpackage",
    "k00gar-11n-200ep-best.pt",
    "k00gar-11n-200ep-best.onnx",
    "yolo11n-pose.mlpackage",
    "yolo11n-pose.pt",
    "yolo11n-pose.onnx"
]

# Construct full paths using os.path.join
yolo_models = [os.path.join(models_dir, filename) for filename in model_filenames]

# Class types
class_types = {'face': 0, 'hand': 1, 'penis': 2, 'glans': 3, 'pussy': 4, 'butt': 5,
                       'anus': 6, 'breast': 7, 'navel': 8, 'foot': 9, 'hips center': 10}

class_reverse_match = {0: 'penis', 1: 'glans', 2: 'pussy', 3: 'butt', 4: 'anus', 5: 'breast',
                       6: 'navel', 7: 'hand', 8: 'face', 9: 'foot', 10: 'hips center'}

class_priority_order = {
            "glans": 0, "penis": 1, "breast": 2, "navel": 3, "pussy": 4, "butt": 5, "face": 6
        }

# Define class names with their corresponding indices
class_names = {
    'face': 0,
    'hand':1, 'left hand': 1, 'right hand': 1,
    'penis': 2,
    'glans': 3,
    'pussy': 4,
    'butt': 5,
    'anus': 6,
    'breast': 7,
    'navel': 8,
    'foot':9, 'left foot': 9, 'right foot': 9,
    'hips center': 10
}

# Define colors for each class
class_colors = {
    "penis": (255, 0, 0),               # red
    "glans": (0, 128, 0),               # green
    "pussy": (0, 0, 255),               # blue
    "butt": (255, 255, 0),              # yellow
    "anus": (128, 0, 128),              # purple
    "breast": (255, 165, 0),            # orange
    "navel": (0, 255, 255),             # cyan
    "hand": (255, 0, 255),              # magenta
    "left hand": (255, 0, 255),         # magenta
    "right hand": (255, 0, 255),        # magenta
    "face": (0, 255, 0),                # lime
    "foot": (165, 42, 42),              # brown
    "left foot": (165, 42, 42),         # brown
    "right foot": (165, 42, 42),        # brown
    "hips center": (0, 0, 0)
}

# Define custom colormap based on Lucife's heatmapColors
heatmap_colors = [
    [0, 0, 0],  # Black (no action)
    [30, 144, 255],  # Dodger blue
    [34, 139, 34],  # Lime green
    [255, 215, 0],  # Gold
    [220, 20, 60],  # Crimson
    [147, 112, 219],  # Medium purple
    [37, 22, 122]  # Dark blue
]
step_size = 120  # Speed step size for color transitions

vw_filter_coeff = 2.0

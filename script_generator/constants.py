import os

from script_generator.utils.config import get_yolo_model_path

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
MODELS_DIR = "models"
MODEL_FILENAMES = [
#    "k00gar-11n-200ep-best.mlpackage",
#    "k00gar-11n-200ep-best.pt",
#    "k00gar-11n-200ep-best.onnx",
    "k00gar-11n-RGB-200ep-best.mlpackage",
    "k00gar-11n-RGB-200ep-best.pt",
    "k00gar-11n-RGB-200ep-best.onnx",
    "yolo11n-pose.mlpackage",
    "yolo11n-pose.pt",
    "yolo11n-pose.onnx"
]
LOGO = os.path.join(PROJECT_PATH, "resources", "logo.png")
ICON = os.path.join(PROJECT_PATH, "resources", "icon.ico")
YOLO_MODELS = [os.path.join(MODELS_DIR, filename) for filename in MODEL_FILENAMES]
MODEL_PATH = str(os.path.join(PROJECT_PATH, get_yolo_model_path(YOLO_MODELS)))
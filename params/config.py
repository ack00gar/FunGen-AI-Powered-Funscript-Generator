#models path
yolo_models = ["models/k00gar-11n-200ep-best.mlpackage",
                "models/k00gar-11n-200ep-best.pt",
                "models/k00gar-11n-200ep-best.onnx",
                "models/yolo11n-pose.mlpackage",
                "models/yolo11n-pose.pt",
                "models/yolo11n-pose.onnx"]

# ffmpeg and ffprobe paths
ffmpeg_path = "/usr/local/bin/ffmpeg"
ffprobe_path = "/usr/local/bin/ffprobe"

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

vw_filter_coeff = 10.0

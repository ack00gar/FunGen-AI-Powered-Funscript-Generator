# ffmpeg and ffprobe paths
ffmpeg_path = "/usr/local/bin/ffmpeg"
ffprobe_path = "/usr/local/bin/ffprobe"

# Class types
class_types = {'face': 0, 'hand': 1, 'penis': 2, 'glans': 3, 'pussy': 4, 'butt': 5,
                       'anus': 6, 'breast': 7, 'navel': 8, 'foot': 9}

class_reverse_match = {0: 'penis', 1: 'glans', 2: 'pussy', 3: 'butt', 4: 'anus', 5: 'breast',
                       6: 'navel', 7: 'hand', 8: 'face', 9: 'foot'}

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
    'foot':9, 'left foot': 9, 'right foot': 9
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
    "right foot": (165, 42, 42)         # brown
}
import platform

from ultralytics import YOLO

from script_generator.constants import MODEL_PATH
from script_generator.utils.config import find_ffmpeg_path, find_ffprobe_path

VERSION = "0.0.1_25-01-16"

##################################################################################################
# FFMPEG
##################################################################################################

# ffmpeg and ffprobe paths - replace with your own if not in your system path
win_ffmpeg_path = "C:/ffmpeg/bin/ffmpeg.exe"
mac_ffmpeg_path = "/usr/local/bin/ffmpeg"
lin_ffmpeg_path = "/usr/bin/ffmpeg"
win_ffprobe_path = "C:/ffmpeg/bin/ffprobe.exe"
mac_ffprobe_path = "/usr/local/bin/ffprobe"
lin_ffprobe_path = "/usr/bin/ffprobe"

##################################################################################################
# PERFORMANCE
##################################################################################################

RENDER_RESOLUTION = 1080 # Above 1080x1080 doesn't seem to improve detections
TEXTURE_RESOLUTION = 1440 # Size that is used to texture the opengl sphere
YOLO_BATCH_SIZE = 1 if platform.system() == "Darwin" else 30 # Mac doesn't support batching due to onnx

##################################################################################################
# ADVANCED / DEVELOPMENT
##################################################################################################


PITCH=-25
YAW=0
YOLO_CONF = 0.3

RUN_POSE_MODEL = False
YOLO_MODEL = YOLO(MODEL_PATH, task="detect")
YOLO_POSE_MODEL = None # YOLO("models/yolo11n-pose.mlpackage", task="pose")

# Set the paths in your config


SUBTRACT_THREADS_FROM_FFMPEG = 0
UPDATE_PROGRESS_INTERVAL = 0.25 # Updates progress in the console and in gui
# when enabled the queue will be processed one by one (use it on (QUEUE_MAXSIZE / frame rate) seconds longer videos or less)
# raw frames take a lot of memory (RAM) so don't set the queue to high
SEQUENTIAL_MODE = False
QUEUE_MAXSIZE = 3000 if SEQUENTIAL_MODE else 500 # Bounded queue size to avoid memory blow-up
DEBUG_PATH = "C:/cvr/funscript-generator/tmp_output"
PROGRESS_BAR = True # disable when you want to print messages while debugging

# Define custom colormap based on Lucife's heatmapColors
STEP_SIZE = 120  # Speed step size for color transitions
VW_FILTER_COEFF = 2.0


FFMPEG_PATH = find_ffmpeg_path(win_ffmpeg_path, mac_ffmpeg_path, lin_ffmpeg_path)
FFPROBE_PATH = find_ffprobe_path(win_ffprobe_path, mac_ffprobe_path, lin_ffprobe_path)
print(f"ffmpeg_path: {FFMPEG_PATH}")
print(f"ffprobe_path: {FFPROBE_PATH}")



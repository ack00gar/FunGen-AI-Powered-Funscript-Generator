import platform

from ultralytics import YOLO

from script_generator.constants import MODEL_PATH
from script_generator.utils.config import find_ffmpeg_path, find_ffprobe_path
from script_generator.debug.logger import logger

VERSION = "0.1.0"

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

RENDER_RESOLUTION = 640
TEXTURE_RESOLUTION = RENDER_RESOLUTION * 1.3 # Texture size that is used to texture the opengl sphere
YOLO_BATCH_SIZE = 1 if platform.system() == "Darwin" else 30 # Mac doesn't support batching
YOLO_PERSIST = True # Big impact on performance but also improves tracking

##################################################################################################
# ADVANCED
##################################################################################################

YOLO_CONF = 0.3
VR_TO_2D_PITCH=-25 # The dataset is trained on -25
UPDATE_PROGRESS_INTERVAL = 0.2 # Updates progress in the console and in gui
STEP_SIZE = 120  # Define custom colormap based on Lucife's heatmapColors | Speed step size for color transitions
QUEUE_MAXSIZE = 100 # Bounded queue size to avoid memory blow-up as raw frames consume a lot of memory, does not increase performance

##################################################################################################
# DEV
##################################################################################################

# when enabled the queue will be processed one by one (use it on (QUEUE_MAXSIZE / frame rate) seconds longer videos or less)
# raw frames take a lot of memory (RAM) so don't set the queue to high
SEQUENTIAL_MODE = False
if SEQUENTIAL_MODE:
    QUEUE_MAXSIZE = 3000

##################################################################################################
# PROG
##################################################################################################

RUN_POSE_MODEL = False
YOLO_POSE_MODEL = None # YOLO("models/yolo11n-pose.mlpackage", task="pose") #TODO pose model?
VALID_VIDEO_READERS = ["FFmpeg", "FFmpeg + OpenGL (Windows)"]
FFMPEG_PATH = find_ffmpeg_path(win_ffmpeg_path, mac_ffmpeg_path, lin_ffmpeg_path)
FFPROBE_PATH = find_ffprobe_path(win_ffprobe_path, mac_ffprobe_path, lin_ffprobe_path)
logger.info(f"ffmpeg_path: {FFMPEG_PATH}")
logger.info(f"ffprobe_path: {FFPROBE_PATH}")



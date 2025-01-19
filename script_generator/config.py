import os
import platform

from ultralytics import YOLO

from config import PROJECT_PATH
from script_generator.object_detection.utils import get_yolo_model_path

##################################################################################################
# CONFIGURATION
##################################################################################################

# Paths (hardcoded for now)
MODEL_PATH = "C:/cvr/funscript-generator/VR-Funscript-AI-Generator/models/k00gar-11n-200ep-best.pt"


##################################################################################################
# DETECTION
##################################################################################################

# VIDEO PREPARATION
RENDER_RESOLUTION = 1080
TEXTURE_RESOLUTION = 1440
PITCH=-25
YAW=0

# YOLO
YOLO_CONF = 0.3
YOLO_BATCH_SIZE = 1 if platform.system() == "Darwin" else 30 # Mac doesn't support batching due to onnx
RUN_POSE_MODEL = False

MODEL_PATH = str(os.path.join(PROJECT_PATH, get_yolo_model_path()))
YOLO_MODEL = YOLO(MODEL_PATH, task="detect")
YOLO_POSE_MODEL = None # YOLO("models/yolo11n-pose.mlpackage", task="pose")


##################################################################################################
# ADVANCED / DEVELOPMENT
##################################################################################################

SUBTRACT_THREADS_FROM_FFMPEG = 0
UPDATE_PROGRESS_INTERVAL = 0.25 # Updates progress in the console and in gui
# when enabled the queue will be processed one by one (use it on (QUEUE_MAXSIZE / frame rate) seconds longer videos or less)
# raw frames take a lot of memory (RAM) so don't set the queue to high
SEQUENTIAL_MODE = False
QUEUE_MAXSIZE = 3000 if SEQUENTIAL_MODE else 500 # Bounded queue size to avoid memory blow-up
DEBUG_PATH = "C:/cvr/funscript-generator/tmp_output"
PROGRESS_BAR = True # disable when you want to print messages while debugging

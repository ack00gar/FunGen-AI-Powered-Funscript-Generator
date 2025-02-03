import os

import torch
from ultralytics import YOLO

from script_generator.constants import MODELS_PATH, MODEL_FILENAMES, OBJECT_DETECTION_VERSION
from script_generator.debug.logger import log, log_od
from script_generator.utils.file import get_output_file_path
from script_generator.utils.helpers import is_mac
from script_generator.utils.json_utils import get_data_file_info
from script_generator.utils.msgpack_utils import save_msgpack_json, load_msgpack_json
from script_generator.utils.version import version_is_less_than


def get_yolo_model_path():
    yolo_models = [os.path.join(MODELS_PATH, filename) for filename in MODEL_FILENAMES]
    # Check if the device is an Apple device
    if is_mac():
        log.info(f"Apple device detected, loading {yolo_models[0]} for MPS inference.")
        return yolo_models[0]

    # Check if CUDA is available (for GPU support)
    elif torch.cuda.is_available():
        log.info(f"CUDA is available, loading {yolo_models[1]} for GPU inference.")
        return yolo_models[1]

    # Fallback to ONNX model for other platforms without CUDA
    else:
        log.info("CUDA not available, if this is unexpected, please install CUDA and check your version of torch.")
        log.info("You might need to install a dependency with the following command (example):")
        log.info("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        log.info(f"Falling back to CPU inference, loading {yolo_models[2]}.")
        log.info("WARNING: CPU inference may be slow on some devices.")

        return yolo_models[2]

def load_yolo_model(yolo_model_path):
    if not yolo_model_path or not os.path.exists(str(yolo_model_path)):
        log.warn("The YOLO model is missing. Please download and place the appropriate YOLO model in the models directory.")
        return None

    return YOLO(yolo_model_path, task="detect")


def get_raw_yolo_file_info(state):
    result_msgpack = get_data_file_info(state.video_path, ".msgpack", "rawyolo")
    if result_msgpack[0]:
        return result_msgpack

    return False, None, None


def save_yolo_data(state, data):
    path, _ = get_output_file_path(state.video_path, ".msgpack", "rawyolo")
    json_data = {"version": OBJECT_DETECTION_VERSION, "data": data}
    save_msgpack_json(path, json_data)


def load_yolo_data(state):
    exists, path, filename = get_raw_yolo_file_info(state)
    if not exists:
        return False, None, path, filename

    json = load_msgpack_json(path)

    if not isinstance(json, dict) or not json.get("version") or version_is_less_than(json["version"], OBJECT_DETECTION_VERSION) or not json.get("data"):
        if version_is_less_than(json["version"], OBJECT_DETECTION_VERSION):
            log_od.warn(f"A raw yolo was found but was skipped due to an outdated version: {path}")
        return False, None, path, filename

    return True, json["data"], path, filename

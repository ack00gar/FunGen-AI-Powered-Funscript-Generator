import os

import torch
from ultralytics import YOLO

from script_generator.constants import MODELS_PATH, MODEL_FILENAMES, OBJECT_DETECTION_VERSION
from script_generator.debug.logger import log
from script_generator.utils.file import get_output_file_path
from script_generator.utils.helpers import is_mac
from script_generator.utils.json_utils import get_data_file_info
from script_generator.utils.msgpack_utils import save_msgpack_json, load_msgpack_json


def find_model(extension):
    """Finds and returns the first model that matches the given file extension."""
    for filename in MODEL_FILENAMES:
        if filename.endswith(extension):
            model_path = os.path.join(MODELS_PATH, filename)
            if os.path.exists(model_path):
                return model_path
    return None


def is_compute_compatible(required_major=8, required_minor=9):
    """Checks if the current GPU has the required compute capability."""
    if torch.cuda.is_available():
        gpu_major, gpu_minor = torch.cuda.get_device_capability()
        if gpu_major > required_major or (gpu_major == required_major and gpu_minor >= required_minor):
            return True
        else:
            log.info(f"❌ Incompatible NVIDIA Cuda Compute Capability: Expected {required_major}.{required_minor}, but found {gpu_major}.{gpu_minor}. Falling back to .pt model for compatibility.")
        return False
    return False


def get_yolo_model_path():
    """Selects the appropriate YOLO model based on platform and hardware capabilities."""

    model_checks = [
        (".mlpackage", is_mac(), "Apple device detected, using MPS inference."),
        (".engine", torch.cuda.is_available() and is_compute_compatible(), "CUDA available and compatible, using GPU inference with TensorRT."),
        (".pt", torch.cuda.is_available(), "CUDA available, using GPU inference (TensorRT model not found or not compatible with Compute capability)."),
        (".onnx", True, "CUDA not available, using ONNX model for CPU inference.")
    ]

    for ext, condition, message in model_checks:
        if condition and (model_path := find_model(ext)):
            log.info(f"{message} Loading {model_path}.")
            if ext == ".onnx":
                log.info("WARNING: CPU inference may be slow on some devices.")
            return model_path

    log.error("No suitable model found. Please make sure to download one of our models and place it in the models directory.")
    return None

def load_yolo_model(yolo_model_path):
    if not yolo_model_path or not os.path.exists(yolo_model_path):
        if yolo_model_path:
            log.warn(f"No file found at specified YOLO model path: {yolo_model_path}")
        else:
            log.warn("The YOLO model is missing. Please download and place the appropriate YOLO model in the models directory.")
        return None

    log.info(f"Loading YOLO model: {yolo_model_path}")
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

    # TODO re-enable
    # if not isinstance(json, dict) or not json.get("version") or version_is_less_than(json["version"], OBJECT_DETECTION_VERSION) or not json.get("data"):
    #     if version_is_less_than(json["version"], OBJECT_DETECTION_VERSION):
    #         # TODO add message box and reprocess if out of date
    #         log_od.warn(f"A raw yolo was found but was skipped due to an outdated version: {path}")
    #     return False, None, path, filename

    return True, json["data"], path, filename

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


def find_model_by_extension(extension):
    """Finds the first model that matches the given file extension."""
    for filename in MODEL_FILENAMES:
        if filename.endswith(extension):
            return os.path.join(MODELS_PATH, filename)
    return None

def get_yolo_model_path():
    """Selects the appropriate YOLO model based on platform and hardware capabilities."""
    if is_mac():
        model_path = find_model_by_extension(".mlpackage")
        if model_path:
            log.info(f"Apple device detected, loading {model_path} for MPS inference.")
            return model_path

    if torch.cuda.is_available():
        model_path = find_model_by_extension(".engine")
        if model_path:
            log.info(f"CUDA is available, loading {model_path} for GPU inference.")
            return model_path

        model_path = find_model_by_extension(".pt")
        if model_path:
            log.info(f"CUDA is available, loading {model_path} for GPU inference.")
            return model_path

    # Default to ONNX for CPU-based inference
    model_path = find_model_by_extension(".onnx")
    if model_path:
        log.info("CUDA not available, falling back to ONNX model for CPU inference.")
        log.info("WARNING: CPU inference may be slow on some devices.")
        return model_path

    # Fallback in case no model is found
    log.error("No suitable model found. Check your MODEL_FILENAMES list.")
    return None

def load_yolo_model(yolo_model_path):
    if not yolo_model_path or not os.path.exists(str(yolo_model_path)):
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

import os
import platform
from tkinter import messagebox

import torch

from config import YOLO_MODELS, CLASS_REVERSE_MATCH, OUTPUT_PATH
from script_generator.gui.utils.widgets import Widgets
from script_generator.object_detection.box_record import BoxRecord
from script_generator.object_detection.object_detection_result import ObjectDetectionResult
from script_generator.utils.file import get_output_file_path


def check_skip_object_detection(state):
    raw_yolo_path, raw_yolo_filename = get_output_file_path(state.video_path, "_rawyolo.json")
    if os.path.exists(raw_yolo_path):
        skip_detection = Widgets.messagebox(
            "Detection File Conflict",
            f"The file already exists. What would you like to do?\n{raw_yolo_filename}",
            "Use Existing",
            "Generate New"
        )
        if skip_detection:
            print(f"File {raw_yolo_path} already exists. Skipping detections and loading file content...")
            return True
        else:
            os.remove(raw_yolo_path)

    return False

def get_yolo_model_path():
    # Check if the device is an Apple device
    if platform.system() == 'Darwin':
        print(f"Apple device detected, loading {YOLO_MODELS[0]} for MPS inference.")
        return YOLO_MODELS[0]

    # Check if CUDA is available (for GPU support)
    elif torch.cuda.is_available():
        print(f"CUDA is available, loading {YOLO_MODELS[1]} for GPU inference.")
        return YOLO_MODELS[1]

    # Fallback to ONNX model for other platforms without CUDA
    else:
        print("CUDA not available, if this is unexpected, please install CUDA and check your version of torch.")
        print("You might need to install a dependency with the following command (example):")
        print("pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118")
        print(f"Falling back to CPU inference, loading {YOLO_MODELS[2]}.")
        print("WARNING: CPU inference may be slow on some devices.")

        return YOLO_MODELS[2]


def make_data_boxes(records):
    """
    Convert YOLO records into BoxRecord objects.
    :param records: List of YOLO detection records.
    :return: A Result object containing BoxRecord instances.
    """
    result = ObjectDetectionResult()  # Create a Result instance
    for record in records:
        frame_idx, cls, conf, x1, y1, x2, y2, track_id = record
        box = [x1, y1, x2, y2]
        class_name = CLASS_REVERSE_MATCH.get(cls, 'unknown')
        box_record = BoxRecord(box, conf, cls, class_name, track_id)
        result.add_record(frame_idx, box_record)
    return result

def parse_yolo_data_looking_for_penis(data, start_frame):
    """
    Parse YOLO data to find the first instance of a penis.
    :param data: The YOLO detection data.
    :param start_frame: The starting frame for the search.
    :return: The frame ID where the penis is first detected.
    """
    consecutive_frames = 0
    frame_detected = 0
    penis_frame = 0

    penis_cls = 0
    glans_cls = 1

    for line in data: #
        frame_idx, cls, conf, x1, y1, x2, y2, track_id = line
        class_name = CLASS_REVERSE_MATCH.get(cls, 'unknown')
        if frame_idx >= start_frame and cls == penis_cls and conf >= 0.5:
            penis_frame = frame_idx

            # TODO re-enable logic
            return penis_frame

        if frame_idx == penis_frame and cls == glans_cls and conf >= 0.5:
            if frame_detected == 0:
                frame_detected = frame_idx
                consecutive_frames += 1
            elif frame_idx == frame_detected + 1:
                consecutive_frames += 1
                frame_detected = frame_idx
            else:
                consecutive_frames = 0
                frame_detected = 0

            if consecutive_frames >= 2:
                print(f"First instance of Glans/Penis found in frame {frame_idx - 4}")
                return frame_idx - 4
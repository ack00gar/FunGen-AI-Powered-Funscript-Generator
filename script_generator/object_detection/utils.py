import os
import platform

import torch

from script_generator.constants import CLASS_REVERSE_MATCH, YOLO_MODELS
from script_generator.gui.utils.widgets import Widgets
from script_generator.object_detection.box_record import BoxRecord
from script_generator.object_detection.object_detection_result import ObjectDetectionResult
from script_generator.utils.file import get_output_file_path
from script_generator.utils.logger import logger


def check_skip_object_detection(state, root):
    raw_yolo_path, raw_yolo_filename = get_output_file_path(state.video_path, "_rawyolo.json")
    if os.path.exists(raw_yolo_path):
        skip_detection = Widgets.messagebox(
            "Detection File Conflict",
            f"The file already exists. What would you like to do?\n{raw_yolo_filename}",
            "Use Existing",
            "Generate New",
            root
        )
        if skip_detection:
            logger.info(f"File {raw_yolo_path} already exists. Skipping detections and loading file content...")
            return True
        else:
            os.remove(raw_yolo_path)

    return False

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
                logger.info(f"First instance of Glans/Penis found in frame {frame_idx - 4}")
                return frame_idx - 4
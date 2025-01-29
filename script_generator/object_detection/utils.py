import os

from script_generator.constants import CLASS_REVERSE_MATCH
from script_generator.debug.logger import logger
from script_generator.gui.utils.widgets import Widgets
from script_generator.object_detection.box_record import BoxRecord
from script_generator.object_detection.object_detection_result import ObjectDetectionResult
from script_generator.utils.json_utils import get_data_file_info


def check_skip_object_detection(state, root):
    exists, path, filename = get_raw_yolo_file_info(state)
    if exists:
        choice = Widgets.messagebox(
            "Detection File Conflict",
            f"The file already exists. What would you like to do?\n{filename}",
            "Generate New",
            "Use Existing",
            root,
            height=150
        )
        if choice == "no":
            logger.info(f"File {path} already exists. Skipping detections and loading file content...")
            return "use_existing"
        elif choice == "yes":
            os.remove(path)
            return "generate"

        return "cancel"

    return "generate"


def get_raw_yolo_file_info(state):
    result_msgpack = get_data_file_info(state.video_path, "_rawyolo.msgpack")
    if result_msgpack[0]:
        return result_msgpack

    result_json = get_data_file_info(state.video_path, "_rawyolo.json")
    if result_json[0]:
        return result_json

    return False, None, None

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

    penis_cls = 0
    prev_frame = 0
    cons_frames = 0
    threshold = 5

    for line in data:
        frame_idx, cls, conf, x1, y1, x2, y2, track_id = line
        if frame_idx >= start_frame and cls == penis_cls and conf >= 0.5:
            penis_frame = frame_idx
            if prev_frame == frame_idx - 1:
                cons_frames += 1
            else:
                cons_frames = 0
            prev_frame = frame_idx

            if cons_frames > threshold:
                logger.info(f"First instance of Glans/Penis found in frame {frame_idx - threshold}")
                return penis_frame - threshold

import json
import os
import time

from script_generator.debug.logger import logger
from script_generator.utils.file import get_output_file_path


def write_json_to_file(file_path, data):
    """
    Write data to a JSON file, always overwriting if it exists.
    :param file_path: The path to the output file.
    :param data: The data to write.
    """
    logger.info(f"Exporting data...")
    export_start = time.time()

    # Ensure the directory exists
    directory = os.path.dirname(file_path)
    if not os.path.exists(directory):
        os.makedirs(directory)

        # Write the data to the file (overwrites if it exists)
    with open(file_path, 'w') as f:
        json.dump(data, f)
    export_end = time.time()
    logger.info(f"Done in {export_end - export_start} seconds.")


def load_json_from_file(file_path):
    """
    Load YOLO data from a JSON file.
    :param file_path: Path to the JSON file.
    :return: The loaded data.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Could not load json from file as it does not exist: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)
        logger.info(f"Loaded data from {file_path}, length: {len(data)}")
    return data


def get_data_file_info(video_path, suffix):
    raw_yolo_path, raw_yolo_filename = get_output_file_path(video_path, suffix)
    if os.path.exists(raw_yolo_path):
        file_size = os.path.getsize(raw_yolo_path)
        if file_size <= 5: # prevent jsons with []
            logger.warn(f"Raw YOLO data file is too small or empty: {raw_yolo_path} (size: {file_size} bytes)")
            try:
                os.remove(raw_yolo_path)
                logger.info(f"Deleted small raw YOLO data file: {raw_yolo_path}")
            except OSError as e:
                logger.error(f"Error deleting raw YOLO file {raw_yolo_path}: {e}")
            return False, None, None
        else:
            return True, raw_yolo_path, raw_yolo_filename
    return False, None, None

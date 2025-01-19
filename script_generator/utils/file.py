import json
import os
import time

from config import OUTPUT_PATH


def write_dataset(file_path, data):
    """
    Write data to a JSON file, always overwriting if it exists.
    :param file_path: The path to the output file.
    :param data: The data to write.
    """
    print(f"Exporting data...")
    export_start = time.time()
    # Write the data to the file (overwrites if it exists)
    with open(file_path, 'w') as f:
        json.dump(data, f)
    export_end = time.time()
    print(f"Done in {export_end - export_start} seconds.")

def load_json_from_file(file_path):
    """
    Load YOLO data from a JSON file.
    :param file_path: Path to the JSON file.
    :return: The loaded data.
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
        print(f"Loaded data from {file_path}, length: {len(data)}")
    return data

def get_output_file_path(video_path, suffix):
    """"
    Get the OUTPUT_PATH filename for a specific suffix (e.g. _raw_yolo.json)
    """
    filename = f"{os.path.basename(video_path)[:-4]}{suffix}"
    path = os.path.join(OUTPUT_PATH, filename)
    return path, filename
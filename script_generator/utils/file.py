import os

from script_generator.constants import OUTPUT_PATH

def get_output_file_path(video_path, suffix, add_spoiler_prefix=False):
    """"
    Get the OUTPUT_PATH filename for a specific suffix (e.g. _raw_yolo.json)
    """
    filename_base = os.path.basename(video_path)[:-4]
    filename = f"{'SPOILER_' if add_spoiler_prefix else ''}{filename_base}{suffix}"
    path = os.path.join(OUTPUT_PATH, filename_base, filename)
    return path, filename

def check_create_output_folder(video_path):
    output_dir, _ = get_output_file_path(video_path, "")
    directory = os.path.dirname(output_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
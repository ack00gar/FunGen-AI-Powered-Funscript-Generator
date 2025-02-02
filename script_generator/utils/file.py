import os

from script_generator.constants import OUTPUT_PATH, VIDEO_EXTENSIONS


def get_output_file_path(video_path, suffix, filename=None, add_spoiler_prefix=False):
    """"
    Get the OUTPUT_PATH filename for a specific suffix (e.g. _raw_yolo.json)
    """
    filename_base = os.path.basename(video_path)[:-4]
    filename_output = f"{'SPOILER_' if add_spoiler_prefix else ''}{filename if filename else filename_base}{suffix}"
    path = os.path.join(OUTPUT_PATH, filename_base, filename_output)
    return path, filename_output

def check_create_output_folder(video_path):
    output_dir, _ = get_output_file_path(video_path, "")
    directory = os.path.dirname(output_dir)
    if not os.path.exists(directory):
        os.makedirs(directory)

def ensure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)


def get_video_files(folder):
    """Recursively find all video files in the given folder."""
    video_files = []

    for root, _, files in os.walk(folder):
        for file in files:
            if any(file.lower().endswith(ext) for ext in VIDEO_EXTENSIONS):
                video_files.append(os.path.join(root, file))

    return video_files
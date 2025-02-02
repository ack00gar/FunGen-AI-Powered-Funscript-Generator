import json
import os

from script_generator.constants import FUNSCRIPT_AUTHOR, FUNSCRIPT_VERSION
from script_generator.debug.logger import log


def load_funscript_json(funscript_path):
    try:
        with open(funscript_path, "r") as f:
            funscript_data = json.load(f)
        actions = funscript_data.get("actions", [])
        funscript_times = [action["at"] for action in actions]
        funscript_positions = [action["pos"] for action in actions]
        return funscript_times, funscript_positions
    except FileNotFoundError:
        log.error(f"Funscript file not found at {funscript_path}")
        raise

def load_funscript(funscript_path):
    if not os.path.exists(funscript_path):
        log.error(f"Funscript not found at {funscript_path}")
        return None, None, None, None

    with open(funscript_path, 'r') as f:
        try:
            log.info(f"Loading funscript from {funscript_path}")
            data = json.load(f)
        except json.JSONDecodeError as e:
            log.error(f"JSONDecodeError: {e}")
            log.error(f"Error occurred at line {e.lineno}, column {e.colno}")
            log.error("Dumping the problematic JSON data:")
            f.seek(0)  # Reset file pointer to the beginning
            log.error(f.read())
            return None, None, None, None

    times = []
    positions = []

    for action in data['actions']:
        times.append(action['at'])
        positions.append(action['pos'])
    log.info(f"Loaded funscript with {len(times)} actions")

    # Access the chapters
    chapters = data.get("metadata", {}).get("chapters", [])

    relevant_chapters_export = []
    irrelevant_chapters_export = []
    # Print the chapters
    for chapter in chapters:
        if len(chapter['startTime']) > 8:
            chapter['startTime'] = chapter['startTime'][:8]
        if len(chapter['endTime']) > 8:
            chapter['endTime'] = chapter['endTime'][:8]
        log.info(f"Chapter: {chapter['name']}, Start: {chapter['startTime']}, End: {chapter['endTime']}")
        # convert 00:00:00 to milliseconds
        start_time_ms = int(chapter['startTime'].split(':')[0]) * 60 * 60 * 1000 + int(
            chapter['startTime'].split(':')[1]) * 60 * 1000 + int(chapter['startTime'].split(':')[2]) * 1000
        end_time_ms = int(chapter['endTime'].split(':')[0]) * 60 * 60 * 1000 + int(
            chapter['endTime'].split(':')[1]) * 60 * 1000 + int(chapter['endTime'].split(':')[2]) * 1000
        if chapter['name'] in ['POV Kissing', 'Close Up', 'Asslicking', 'Creampie']:
            irrelevant_chapters_export.append([chapter['name'], start_time_ms, end_time_ms])
        else:  # if chapter['name'] == 'Blow Job':
            relevant_chapters_export.append([chapter['name'], start_time_ms, end_time_ms])

    return times, positions, relevant_chapters_export, irrelevant_chapters_export

# TODO replace with proper JSON serialization
def write_funscript(distances, output_path, fps):
    with open(output_path, 'w') as f:
        f.write(f'{{"version":"{FUNSCRIPT_VERSION}","inverted":false,"range":100,"author":"{FUNSCRIPT_AUTHOR}","actions":[{{"at":0,"pos":100}},')
        i = 0
        for frame, position in distances:
            time_ms = int(frame * 1000 / fps)
            if i > 0:
                f.write(",")
            f.write(f'{{"at":{time_ms},"pos":{int(position)}}}')
            i += 1
        f.write("]}")
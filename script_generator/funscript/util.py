import json

from script_generator.debug.logger import logger


def load_funscript_json(funscript_path):
    try:
        with open(funscript_path, "r") as f:
            funscript_data = json.load(f)
        actions = funscript_data.get("actions", [])
        funscript_times = [action["at"] for action in actions]
        funscript_positions = [action["pos"] for action in actions]
        return funscript_times, funscript_positions
    except FileNotFoundError:
        logger.error(f"Funscript file not found at {funscript_path}")
        raise
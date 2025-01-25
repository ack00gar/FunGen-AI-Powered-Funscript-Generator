import json

import numpy as np

from script_generator.utils.file import get_output_file_path, load_json_from_file
from script_generator.utils.logger import logger


class DebugData:
    def __init__(self, state):
        self.app_state = state
        self.logs = {}

    def add_frame(self, frame_id, variables=None, bounding_boxes=None):
        if frame_id not in self.logs:
            self.logs[frame_id] = {"variables": {}, "bounding_boxes": []}

        if variables:
            self.logs[frame_id]["variables"].update(variables)

        if bounding_boxes:
            self.logs[frame_id]["bounding_boxes"].extend(bounding_boxes)

    def save_debug_file(self):
        debug_log_path, _ = get_output_file_path(self.app_state.video_path, "_debug_logs.json")
        try:
            with open(debug_log_path, "w") as f:
                json.dump(self.logs, f, indent=4, default=self._default_serializer)
            logger.info(f"Debug data saved to {debug_log_path}")
        except Exception as e:
            logger.error(f"Failed to save logs: {e}")

    @staticmethod
    def _default_serializer(obj):
        """Convert NumPy types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def load_logs(state):
    log_path, _ = get_output_file_path(state.video_path, "_debug_logs.json")
    return load_json_from_file(log_path)
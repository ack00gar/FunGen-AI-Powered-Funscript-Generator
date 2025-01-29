from script_generator.utils.file import get_output_file_path
from script_generator.utils.msgpack_utils import save_msgpack_json, load_msgpack_json


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
        path, _ = get_output_file_path(self.app_state.video_path, "_debug_logs.msgpack")
        save_msgpack_json(path, self.logs)


def load_debug_metrics(state):
    base_path, _ = get_output_file_path(state.video_path, "_debug_logs.msgpack")
    return load_msgpack_json(base_path)
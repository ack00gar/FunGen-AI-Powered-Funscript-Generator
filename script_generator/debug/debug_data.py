from script_generator.constants import TRACKING_VERSION
from script_generator.debug.logger import log_tr
from script_generator.utils.file import get_output_file_path
from script_generator.utils.json_utils import get_data_file_info
from script_generator.utils.msgpack_utils import save_msgpack_json, load_msgpack_json
from script_generator.utils.version import version_is_less_than


class DebugData:
    def __init__(self, state):
        self.app_state = state
        self.metrics = {}

    def add_frame(self, frame_id, variables=None, bounding_boxes=None):
        if frame_id not in self.metrics:
            self.metrics[frame_id] = {"variables": {}, "bounding_boxes": []}

        if variables:
            self.metrics[frame_id]["variables"].update(variables)

        if bounding_boxes:
            self.metrics[frame_id]["bounding_boxes"].extend(bounding_boxes)

    def save_debug_file(self):
        save_debug_metrics(self.app_state, self.metrics)

def save_debug_metrics(state, data):
    path, _ = get_output_file_path(state.video_path, ".msgpack", "metrics")
    json_data = {"version": TRACKING_VERSION, "data": data}
    save_msgpack_json(path, json_data)

def load_debug_metrics(state):
    exists, path, filename = get_metrics_file_info(state)
    if not exists:
        return False, None, path, filename

    json = load_msgpack_json(path)

    if not isinstance(json, dict) or not json.get("version") or version_is_less_than(json["version"], TRACKING_VERSION) or not json.get("data"):
        log_tr.warn(f"Debug metrics file was found but was skipped due to an outdated version: {path}")
        return False, None, path, filename

    return True, json["data"], path, filename

def get_metrics_file_info(state):
    result_msgpack = get_data_file_info(state.video_path, ".msgpack", "metrics")
    if result_msgpack[0]:
        return result_msgpack

    return False, None, None
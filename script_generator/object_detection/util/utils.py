from script_generator.utils.json_utils import get_data_file_info


def get_metrics_file_info(state):
    result_msgpack = get_data_file_info(state.video_path, "_debug_logs.msgpack")
    if result_msgpack[0]:
        return result_msgpack

    result_json = get_data_file_info(state.video_path, "_debug_logs.json")
    if result_json[0]:
        return result_json

    return False, None, None
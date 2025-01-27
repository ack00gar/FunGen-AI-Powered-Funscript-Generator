from script_generator.utils.json import get_data_file_info


def get_metrics_file_info(state):
    """
    Checks if the data_logs metrics file exists
    """
    return get_data_file_info(state, "_debug_logs.json")
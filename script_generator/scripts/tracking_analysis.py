from script_generator.analysis.workers.analyze_tracking_results_v1 import analyze_tracking_results_v1
from script_generator.analysis.workers.analyze_tracking_results_v2 import analyze_tracking_results_v2
from script_generator.debug.logger import log_tr
from script_generator.funscript.create_funscript import create_funscript
from script_generator.object_detection.util.data import get_raw_yolo_file_info
from script_generator.utils.data_classes.meta_data import MetaData


def tracking_analysis(state):
    exists, raw_yolo_path, _ = get_raw_yolo_file_info(state)
    if not exists:
        log_tr.info(f"Raw yolo json not found in {raw_yolo_path}")
        return

    # Get meta file
    meta = MetaData.get_create_meta(state)

    # Performing the tracking part and generation of the raw funscript data
    state.funscript_data = analyze_tracking_results_v1(state) if state.tracking_logic_version == 1 else analyze_tracking_results_v2(state)


    # Save debug file
    state.debug_data.save_debug_file()

    if not state.funscript_data:
        return

    # Generate funscript
    create_funscript(state)

    # Update meta file
    meta.finish_tracking_analysis(state)

    log_tr.info(f"Finished processing video: {state.video_path}")

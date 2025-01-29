import os

from script_generator.object_detection.analyze_tracking_results import analyze_tracking_results
from script_generator.object_detection.utils import make_data_boxes, parse_yolo_data_looking_for_penis, get_raw_yolo_file_info
from script_generator.utils.file import get_output_file_path
from script_generator.utils.json_utils import load_json_from_file
from script_generator.debug.logger import logger
from script_generator.utils.msgpack_utils import load_msgpack_json
from script_generator.video.info.video_info import get_cropped_dimensions
from utils.lib_FunscriptHandler import FunscriptGenerator


def tracking_analysis(state):
    # Load YOLO detection results from file
    exists, raw_yolo_path, _ = get_raw_yolo_file_info(state)
    if not exists:
        logger.info(f"Raw yolo json not found in {raw_yolo_path}")
        return

    yolo_data = load_msgpack_json(raw_yolo_path)

    state.set_video_info()
    video_info = state.video_info

    results = make_data_boxes(yolo_data)

    # Looking for the first instance of penis within the YOLO results
    first_penis_frame = parse_yolo_data_looking_for_penis(yolo_data, 0)

    if first_penis_frame is None:
        logger.error(f"No penis instance found in video. Further tracking is not possible.")
        return

    # Deciding whether we start from there or from a user-specified later frame
    # state.frame_start = max(max(first_penis_frame - int(video_info.fps), state.frame_start - int(video_info.fps)), 0)
    state.frame_start_track = max(max(first_penis_frame - int(video_info.fps), state.frame_start - int(video_info.fps)), 0)

    state.frame_end = state.video_info.total_frames if not state.frame_end else state.frame_end

    # logger.info(f"Frame Start adjusted to: {state.frame_start}")
    logger.info(f"Frame Start adjusted to: {state.frame_start_track}")

    # Performing the tracking part and generation of the raw funscript data
    state.funscript_data = analyze_tracking_results(state, results)

    state.debug_data.save_debug_file()

    funscript_handler = FunscriptGenerator()

    # Simplifying the funscript data and generating the file
    funscript_handler.generate(state)

    # Optionally, compare generated funscript with reference funscript if specified, or a simple generic report
    funscript_handler.create_report_funscripts(state)

    logger.info(f"Finished processing video: {state.video_path}")

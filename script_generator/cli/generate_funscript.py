import argparse
import os

from config import VALID_VIDEO_READERS
from script_generator.scripts.generate_funscript import generate_funscript
from script_generator.debug.logger import logger
from script_generator.state.app_state import AppState
from script_generator.utils.helpers import is_mac

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    parser = argparse.ArgumentParser(
        description="Generate a funscript file from a video."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file."
    )
    parser.add_argument(
        "--reuse-yolo",
        type=bool,
        default=False,
        help="Use an existing raw YOLO output file instead of generating a new one. Default is False."
    )
    parser.add_argument(
        "--copy-funscript",
        type=bool,
        default=True,
        help="Copies final funscript to the movie directory. Default is True."
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        default=0,
        help="The starting frame number for processing. Default is 0 (start from the beginning)."
    )
    parser.add_argument(
        "--frame-end",
        type=int or None,
        default=None,
        help="The ending frame number for processing. Default is None (process till the end)."
    )
    parser.add_argument(
        "--video-reader",
        type=str,
        default=None,
        help=f"Video reader to use. Valid options: {', '.join(VALID_VIDEO_READERS)}. Default is platform-specific."
    )
    parser.add_argument(
        "--save-debug-file",
        type=bool,
        default=True,
        help="Saves a debug file to disk with all collected metrics. Also allows you to re-use tracking data. Default is True."
    )
    parser.add_argument(
        "--boost-enabled",
        type=bool,
        default=True,
        help="Funscript tweaking setting: Enable boosting to adjust the motion range dynamically. Default is True."
    )
    parser.add_argument(
        "--boost-up-percent",
        type=int,
        default=10,
        help="Funscript tweaking setting: Increase the peaks by a specified percentage to enhance upper motion limits. Default is 10%."
    )
    parser.add_argument(
        "--boost-down-percent",
        type=int,
        default=15,
        help="Funscript tweaking setting: Reduce the lower peaks by a specified percentage to limit downward motion. Default is 15%."
    )
    parser.add_argument(
        "--threshold-enabled",
        type=bool,
        default=True,
        help="Funscript tweaking setting: Enable thresholding to control motion mapping within specified bounds. Default is True."
    )
    parser.add_argument(
        "--threshold-low",
        type=int,
        default=10,
        help="Funscript tweaking setting: Values below this threshold are mapped to 0, limiting lower boundary motion. Default is 10."
    )
    parser.add_argument(
        "--threshold-high",
        type=int,
        default=90,
        help="Funscript tweaking setting: Values above this threshold are mapped to 100, limiting upper boundary motion. Default is 90."
    )
    parser.add_argument(
        "--vw-simplification-enabled",
        type=bool,
        default=True,
        help="Funscript tweaking setting: Simplify the generated script to reduce the number of points, making it user-friendly. Default is True."
    )
    parser.add_argument(
        "--vw-factor",
        type=float,
        default=8.0,
        help="Funscript tweaking setting: Determines the degree of simplification. Higher values lead to fewer points. Default is 8.0."
    )
    parser.add_argument(
        "--rounding",
        type=int,
        default=5,
        help="Funscript tweaking setting: Set the rounding factor for script values to adjust precision. Default is 5."
    )
    args = parser.parse_args()

    video_reader = args.video_reader
    default_video_reader = "FFmpeg" if is_mac() else "FFmpeg + OpenGL (Windows)"
    if video_reader and video_reader not in VALID_VIDEO_READERS:
        logger.warning(
            f"Invalid video reader specified: {video_reader}. Using default: {default_video_reader}."
        )
        args.video_reader = default_video_reader

    try:
        logger.info(f"Processing video: {args.video_path}")
        state = AppState(is_cli=True)
        state.video_path = args.video_path
        state.use_existing_raw_yolo = args.reuse_yolo
        state.copy_funscript_to_movie_dir = args.copy_funscript
        state.frame_start = args.frame_start
        state.frame_end = args.frame_end
        state.video_reader = args.video_reader
        state.save_debug_file = args.save_debug_file
        state.boost_enabled = args.boost_enabled
        state.boost_up_percent = args.boost_up_percent
        state.boost_down_percent = args.boost_down_percent
        state.threshold_enabled = args.threshold_enabled
        state.threshold_low = args.threshold_low
        state.threshold_high = args.threshold_high
        state.vw_simplification_enabled = args.vw_simplification_enabled
        state.vw_factor = args.vw_factor
        state.rounding = args.rounding

        generate_funscript(state)
        logger.info("Funscript generation complete.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

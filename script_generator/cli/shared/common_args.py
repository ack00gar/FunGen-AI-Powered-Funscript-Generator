import argparse
import os

from script_generator.constants import VALID_VIDEO_READERS
from script_generator.debug.logger import log
from script_generator.state.app_state import AppState
from ultralytics import settings

# TODO this is a workaround and needs to be fixed properly
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
settings.update({"sync": False}) # Disable analytics and crash reporting

def add_shared_generate_funscript_args(parser: argparse.ArgumentParser) -> None:
    """
    Add all the arguments that multiple commands might share when generating a funscript.
    """
    parser.add_argument(
        "--reuse-yolo",
        action="store_true",
        help="Re-use an existing raw YOLO output file if available."
    )
    parser.add_argument(
        "--copy-funscript",
        action="store_true",
        help="Copies final funscript to the movie directory."
    )
    parser.add_argument(
        "--frame-start",
        type=int,
        help="The starting frame number for processing."
    )
    parser.add_argument(
        "--frame-end",
        type=int,
        help="The ending frame number for processing."
    )
    parser.add_argument(
        "--video-reader",
        type=str,
        help=f"Video reader to use. Valid options: {', '.join(VALID_VIDEO_READERS)}."
    )
    parser.add_argument(
        "--save-debug-file",
        action="store_true",
        help="Saves a debug file to disk with all collected metrics."
    )
    parser.add_argument(
        "--boost-enabled",
        action="store_true",
        help="Funscript tweaking: Enable boosting to adjust motion range dynamically."
    )
    parser.add_argument(
        "--boost-up-percent",
        type=int,
        help="Funscript tweaking: Increase the peaks by a specified percentage."
    )
    parser.add_argument(
        "--boost-down-percent",
        type=int,
        help="Funscript tweaking: Reduce lower peaks by a specified percentage."
    )
    parser.add_argument(
        "--threshold-enabled",
        action="store_true",
        help="Funscript tweaking: Enable thresholding to control motion mapping."
    )
    parser.add_argument(
        "--threshold-low",
        type=int,
        help="Funscript tweaking: Values below this threshold become 0."
    )
    parser.add_argument(
        "--threshold-high",
        type=int,
        help="Funscript tweaking: Values above this threshold become 100."
    )
    parser.add_argument(
        "--vw-simplification-enabled",
        action="store_true",
        help="Funscript tweaking: Simplify the script to reduce points."
    )
    parser.add_argument(
        "--vw-factor",
        type=float,
        help="Funscript tweaking: Factor controlling the degree of simplification."
    )
    parser.add_argument(
        "--rounding",
        type=int,
        help="Funscript tweaking: Rounding factor for script values."
    )

def validate_and_adjust_args(args: argparse.Namespace) -> None:
    """
    If user provided a video_reader that isn't valid, override it with a sensible default.
    Also do any other cross-argument validations or adjustments here.
    """
    default_video_reader = "FFmpeg" # if is_mac() else "FFmpeg + OpenGL (Windows)"
    if args.video_reader and args.video_reader not in VALID_VIDEO_READERS:
        log.warning(
            f"Invalid video reader specified: {args.video_reader}. "
            f"Using default: {default_video_reader}."
        )
        args.video_reader = default_video_reader

def parse_args() -> tuple[argparse.Namespace, set]:
    """
    Parse command-line arguments and return them along with a set of explicitly provided argument names.
    """
    parser = argparse.ArgumentParser()

    # Add all arguments
    add_shared_generate_funscript_args(parser)

    # Use parse_known_args to track explicitly provided arguments
    args, unknown = parser.parse_known_args()

    # Extract explicitly provided argument names (strip "--")
    provided_args = {arg.lstrip("-").replace("-", "_") for arg in unknown}

    return args, provided_args

def build_app_state_from_args(args: argparse.Namespace, provided_args: set) -> AppState:
    """
    Create and populate an AppState from the parsed arguments,
    setting only attributes that were explicitly provided via the CLI.
    """
    state = AppState()

    if "video_path" in provided_args:
        state.video_path = args.video_path
    if "reuse_yolo" in provided_args:
        state.use_existing_raw_yolo = args.reuse_yolo
    if "copy_funscript" in provided_args:
        state.copy_funscript_to_movie_dir = args.copy_funscript
    if "frame_start" in provided_args:
        state.frame_start = args.frame_start
    if "frame_end" in provided_args:
        state.frame_end = args.frame_end
    if "video_reader" in provided_args:
        state.video_reader = args.video_reader
    if "save_debug_file" in provided_args:
        state.save_debug_file = args.save_debug_file

    # Boosting
    if "boost_enabled" in provided_args:
        state.boost_enabled = args.boost_enabled
    if "boost_up_percent" in provided_args:
        state.boost_up_percent = args.boost_up_percent
    if "boost_down_percent" in provided_args:
        state.boost_down_percent = args.boost_down_percent

    # Threshold
    if "threshold_enabled" in provided_args:
        state.threshold_enabled = args.threshold_enabled
    if "threshold_low" in provided_args:
        state.threshold_low = args.threshold_low
    if "threshold_high" in provided_args:
        state.threshold_high = args.threshold_high

    # Simplification + rounding
    if "vw_simplification_enabled" in provided_args:
        state.vw_simplification_enabled = args.vw_simplification_enabled
    if "vw_factor" in provided_args:
        state.vw_factor = args.vw_factor
    if "rounding" in provided_args:
        state.rounding = args.rounding

    state.set_is_cli(True)

    return state

import argparse
import sys

from script_generator.cli.shared.common_args import (
    add_shared_generate_funscript_args,
    validate_and_adjust_args,
    build_app_state_from_args,
)
from script_generator.cli.shared.generate_funscript import generate_funscript
from script_generator.debug.logger import log


def main():
    parser = argparse.ArgumentParser(
        description="Generate a funscript file from a single video."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file."
    )
    add_shared_generate_funscript_args(parser)

    args = parser.parse_args()
    validate_and_adjust_args(args)

    provided_args = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            flag_name = token.lstrip("-").replace("-", "_")
            provided_args.add(flag_name)
        else:
            provided_args.add("video_path")

    try:
        log.info(f"Processing video: {args.video_path}")

        state = build_app_state_from_args(args, provided_args)
        state.set_video_info()
        generate_funscript(state)
        log.info("Funscript generation complete.")

    except Exception as e:
        log.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

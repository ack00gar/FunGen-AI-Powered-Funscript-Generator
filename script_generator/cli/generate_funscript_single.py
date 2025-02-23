import argparse
import sys

from script_generator.cli.shared.common_args import (
    add_shared_generate_funscript_args,
    validate_and_adjust_args,
    build_app_state_from_args,
)
from script_generator.cli.shared.generate_funscript import generate_funscript_cli
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

    provided_args = {
        arg.lstrip("-").replace("-", "_")
        for arg in sys.argv[1:]
        if arg.startswith("--")
    }
    provided_args.add("video_path")

    explicit_args = {arg: getattr(args, arg) for arg in vars(args) if arg in provided_args}
    log.info(f"Provided command-line arguments: {explicit_args}")

    try:
        log.info(f"Processing video: {args.video_path}")

        state = build_app_state_from_args(args, provided_args)
        state.set_video_info()
        generate_funscript_cli(state)
        log.info("Funscript generation complete.")

    except Exception as e:
        log.error(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

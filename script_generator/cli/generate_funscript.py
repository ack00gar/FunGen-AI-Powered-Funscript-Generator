import argparse
from script_generator.scripts.generate_funscript import generate_funscript
from script_generator.utils.logger import logger


def main():
    parser = argparse.ArgumentParser(
        description="Generate a funscript file from a video."
    )
    parser.add_argument(
        "video_path",
        type=str,
        help="Path to the input video file."
    )
    args = parser.parse_args()

    try:
        logger.info(f"Processing video: {args.video_path}")
        generate_funscript(args.video_path)
        logger.info("Funscript generation complete.")
    except Exception as e:
        logger.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
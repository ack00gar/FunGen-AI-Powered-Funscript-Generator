import argparse
import os

from script_generator.constants import VALID_VIDEO_READERS
from script_generator.cli.shared.generate_funscript import generate_funscript
from script_generator.debug.logger import log
from script_generator.gui.app import start_app
from script_generator.state.app_state import AppState
from script_generator.utils.data_classes.meta_data import MetaData
from script_generator.utils.helpers import is_mac

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

def main():
    parser = argparse.ArgumentParser(
        description="Open gui based on a metadata.json. These files are automatically created when processing a video and stored to folder output/{video_name}/metadata.json"
    )
    parser.add_argument(
        "metadata_path",
        type=str,
        help="metadata.json file"
    )
    args = parser.parse_args()

    try:
        log.info(f"Opening gui for metadata.json file: {args.metadata_path}")
        meta = MetaData.load_meta(args.metadata_path)
        state = AppState()
        state.video_path = meta.video_info.path
        state.video_info = meta.video_info
        start_app(state)
    except Exception as e:
        log.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()

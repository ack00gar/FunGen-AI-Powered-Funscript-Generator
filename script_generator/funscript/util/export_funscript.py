import os
import shutil
from script_generator.debug.logger import log
from script_generator.funscript.util.check_existing_funscript import check_existing_funscript
from script_generator.state.app_state import AppState
from script_generator.utils.file import get_output_file_path


def export_funscript(state: AppState):
    destinations = {}
    video_folder = os.path.dirname(state.video_path)
    filename_base = os.path.splitext(os.path.basename(state.video_path))[0]
    output_path, _ = get_output_file_path(state.video_path, ".funscript")

    if state.copy_funscript_to_movie_dir:
        movie_dest = os.path.join(video_folder, f"{filename_base}.funscript")
        destinations["movie"] = movie_dest

    if state.funscript_output_dir:
        output_dest = os.path.join(state.funscript_output_dir, f"{filename_base}.funscript")
        destinations["output"] = output_dest

    for _, dest_path in destinations.items():
        file_exists, is_ours, backup_path, out_of_date = check_existing_funscript(dest_path, filename_base, state.make_funscript_backup)

        if not file_exists:
            shutil.copy(output_path, dest_path)
            log.info(f"Copied funscript to {dest_path}.")
        else:
            if is_ours:
                if backup_path:
                    log.info(
                        f"Funscript made by this app already exists at {dest_path}. "
                        f"Backing it up to {backup_path}."
                    )
                    shutil.move(dest_path, backup_path)
                shutil.copy(output_path, dest_path)
                log.info(f"Copied funscript to {dest_path}.")
            else:
                log.warning(
                    f"Skipping copy to {dest_path} because it's an existing funscript "
                    f"not created by this app."
                )

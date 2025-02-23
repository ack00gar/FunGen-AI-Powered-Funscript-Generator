import argparse
import concurrent.futures
import os
import platform
import sys
import time
from collections import deque
from datetime import timedelta

from script_generator.cli.shared.common_args import (
    add_shared_generate_funscript_args,
    validate_and_adjust_args,
    build_app_state_from_args,
)
from script_generator.constants import OUTPUT_PATH
from script_generator.debug.logger import log
from script_generator.funscript.util.check_existing_funscript import check_existing_funscript
from script_generator.utils.file import get_video_files
from script_generator.utils.terminal import open_new_terminal


def main():
    parser = argparse.ArgumentParser(
        description="Generate funscripts in parallel for all videos in a folder."
    )
    parser.add_argument("folder", type=str, help="Folder containing video files")
    parser.add_argument(
        "--replace-outdated",
        action="store_true",
        default=True,
        help="Will regenerate outdated funscripts."
    )
    parser.add_argument(
        "--replace-up-to-date",
        action="store_true",
        default=False,
        help="Will regenerate funscripts that are up to date and made by this app too."
    )
    parser.add_argument(
        "--process-external-script",
        action="store_true",
        default=False,
        help="Will generate a new script even when the movie already has a script not made by this app. Existing funscript will not be replaced."
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=2,
        help="Number of subprocesses to run in parallel. If you have beefy hardware 4 seems to be the sweet spot but technically your VRAM is the limit."
    )
    add_shared_generate_funscript_args(parser)

    args = parser.parse_args()
    validate_and_adjust_args(args)

    # Determine which CLI flags were provided
    provided_args = set()
    for token in sys.argv[1:]:
        if token.startswith("--"):
            provided_args.add(token.lstrip("-").replace("-", "_"))
        else:
            provided_args.add("folder")

    try:
        log.info(
            f"Processing folder: {args.folder}, "
            f"{'' if args.replace_outdated else 'not '}replacing outdated funscripts, "
            f"{'also ' if args.replace_up_to_date else 'not '}replacing funscripts with current version, "
            f"using up to {args.num_workers} worker(s)."
        )
        paths = get_video_files(args.folder)
        log.info(f"Found {len(paths)} file(s) in folder")

        # Build temporary app state (to access config variables and defaults)
        state = build_app_state_from_args(args, provided_args)

        files_none = []
        files_by_us = []
        files_by_us_out_of_date = []
        files_not_by_us = []
        files_not_by_us_regenerate = []

        for video_file in paths:
            filename_base = os.path.splitext(os.path.basename(video_file))[0]
            video_folder = os.path.dirname(video_file)
            funscript_path = os.path.join(video_folder, f"{filename_base}.funscript")
            has_raw_yolo = False

            file_exists, is_ours, _, out_of_date = check_existing_funscript(funscript_path, filename_base, False)

            # If there's a separate output directory, check there too
            if state.funscript_output_dir and os.path.exists(state.funscript_output_dir):
                folder_destination = os.path.join(state.funscript_output_dir, f"{filename_base}.funscript")
                file_exists_d, is_ours_d, _, out_of_date_d = check_existing_funscript(
                    folder_destination, filename_base, False
                )
                if not file_exists and file_exists_d:
                    file_exists = True
                if not is_ours and is_ours_d:
                    is_ours = True
                if not out_of_date and out_of_date_d:
                    out_of_date = out_of_date_d

            if os.path.exists(OUTPUT_PATH) and os.path.getsize(OUTPUT_PATH) > 100:
                raw_yolo_dest = os.path.join(OUTPUT_PATH, filename_base, "rawyolo.msgpack")
                has_raw_yolo = os.path.exists(raw_yolo_dest)

            if not file_exists:
                files_none.append(video_file)
            elif is_ours:
                if out_of_date:
                    files_by_us_out_of_date.append(video_file)
                else:
                    files_by_us.append(video_file)
            elif file_exists:
                if has_raw_yolo or not args.process_external_script:
                    files_not_by_us.append(video_file)
                else:
                    files_not_by_us_regenerate.append(video_file)

        def log_files_by_category(category, file_list):
            log.info(f"{'-' * 100}")
            log.info(f" {category} ({len(file_list)}):")
            log.info(f"{'-' * 100}")
            for f in file_list:
                log.info(f" - {f}")

        log_files_by_category("Files with no funscript", files_none)
        if len(files_not_by_us_regenerate) > 0:
            log_files_by_category("Files with funscript not by this app to generate", files_not_by_us_regenerate)
        if len(files_by_us) > 0:
            log_files_by_category(
                f"Files with existing funscript by this app (>= current version, will "
                f"{'' if args.replace_up_to_date else 'NOT '}be processed)",
                files_by_us
            )
        if len(files_by_us_out_of_date) > 0:
            log_files_by_category(
                f"Files with existing funscript by this app (< current version, will "
                f"{'' if args.replace_outdated else 'NOT '}be re-generated)",
                files_by_us_out_of_date
            )
        if len(files_not_by_us) > 0:
            log_files_by_category("Files with existing funscript not by this app (ignored)", files_not_by_us)
        log.info(f"{'-' * 100}")

        # Decide which files to process
        to_process = files_none.copy()
        if args.replace_outdated:
            to_process.extend(files_by_us_out_of_date)
        if args.replace_up_to_date:
            to_process.extend(files_by_us)
        if files_not_by_us_regenerate:
            to_process.extend(files_not_by_us_regenerate)

        if not to_process:
            log.info("No files need new funscript generation.")
            return

        tasks = deque(to_process)

        log.info(f"Starting batch generation with up to {args.num_workers} parallel subprocesses.")

        # Dictionary to keep track of the submitted tasks
        future_to_video = {}

        global_start_time = time.time()  # overall start time for the batch

        # Use a ThreadPoolExecutor to limit concurrency.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            running = []
            # Submit all tasks.
            while tasks:
                video_path = tasks.popleft()
                log.debug(f"[Submit] {video_path}")
                running.append(video_path)
                future = executor.submit(run_task_timed, video_path, args)
                future_to_video[future] = video_path
                time.sleep(0.1)  # slight delay to stagger submissions

            total_count = len(future_to_video)
            completed_count = 0
            processing_times = []  # times for completed tasks

            # Process futures as they complete.
            for future in concurrent.futures.as_completed(future_to_video):
                video_path = future_to_video[future]
                try:
                    ret, elapsed = future.result()
                except Exception as e:
                    log.error(f"Error processing {video_path}: {e}", exc_info=True)
                    ret, elapsed = -1, 0
                completed_count += 1
                processing_times.append(elapsed)

                # Gather names of movies still in progress.
                currently_processing = [
                    v for fut, v in future_to_video.items() if not fut.done()
                ]

                # Calculate total runtime and average completion time
                total_run_time = time.time() - global_start_time
                average_time = total_run_time / completed_count if completed_count > 0 else 0

                # Display first 5 movies, add '...' if there are more
                queue_count = len(currently_processing)
                if queue_count > 5:
                    queued_paths = "\n".join(f" - {movie}" for movie in currently_processing[:5]) + "\n..."
                else:
                    queued_paths = "\n".join(f" - {movie}" for movie in currently_processing) if currently_processing else "None"

                log.info(
                    f"Progress: {completed_count}/{total_count} completed | Queue: {queue_count} remaining | "
                    f"Total run time: {str(timedelta(seconds=int(total_run_time)))} | "
                    f"Average completion time: {str(timedelta(seconds=int(average_time)))}s "
                    "(calculated from completed movies, not ongoing processes)."
                    f"\nQueued Movies:\n{queued_paths}"
                )

                if ret == 0:
                    log.info(f"[Done] {video_path}")
                else:
                    log.warning(f"[Failed] {video_path} (exit code {ret})")

        log.info("All funscript generation tasks completed.")

    except Exception as e:
        log.error(f"An error occurred: {e}", exc_info=True)


def run_task(video_path, args):
    """
    Runs the funscript generation for one video in a new terminal window.
    Passes only relevant arguments to the process_file command.
    Returns the exit code (0 if successful).
    """
    # Escape ampersand only for Windows
    if platform.system() == "Windows":
        video_path = video_path.replace("&", "^&")

    # Define the arguments relevant to process_file
    process_file_args = {
        "reuse_yolo", "copy_funscript", "frame_start", "frame_end",
        "video_reader", "save_debug_file", "boost_enabled",
        "boost_up_percent", "boost_down_percent", "threshold_enabled",
        "threshold_low", "threshold_high", "vw_simplification_enabled",
        "vw_factor", "rounding"
    }

    # Build the command dynamically
    cmd = f'python -m script_generator.cli.generate_funscript_single "{video_path}"'

    for arg, value in vars(args).items():
        if arg in process_file_args and value is not None:
            cli_arg = f"--{arg.replace('_', '-')}"

            if isinstance(value, bool):
                if value:
                    cmd += f" {cli_arg}"
            else:
                cmd += f" {cli_arg}={value}"

    # Launch the process in a new terminal
    proc = open_new_terminal(cmd, relative_path_up=2)

    if proc is None:
        print(f"Failed to launch terminal for: {video_path}")
        return -1

    # Wait for the process to finish
    proc.wait()
    return proc.returncode


def run_task_timed(video_path, args):
    """
    Wraps run_task to record how long it takes.
    Returns a tuple: (exit code, elapsed_time_in_seconds).
    """
    start = time.time()
    ret = run_task(video_path, args)
    elapsed = time.time() - start
    return ret, elapsed


if __name__ == "__main__":
    main()

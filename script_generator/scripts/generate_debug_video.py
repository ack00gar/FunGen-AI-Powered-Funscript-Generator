import os
import subprocess

from config import FFMPEG_PATH
from script_generator.debug.video_player.play import play_debug_video
from script_generator.state.app_state import AppState
from script_generator.utils.file import get_output_file_path
from script_generator.utils.logger import logger
from script_generator.video.ffmpeg.hwaccel import get_hwaccel_read_args


def generate_debug_video(state: AppState, frame_start: int, frame_end: int):
    logger.info(f"Generating debug video from: {frame_start} to {frame_end}")
    temp_video_path = play_debug_video(state, frame_start, frame_end, save_video_mode=True)

    debug_video_path, _ = get_output_file_path(state.video_path, "_debug.mp4")
    ffmpeg_command = [
        FFMPEG_PATH, "-y",
        *get_hwaccel_read_args(state.ffmpeg_hwaccel),
        "-i", temp_video_path,
        "-c:v", "hevc_nvenc" if state.ffmpeg_hwaccel == "cuda" else "libx265",
        "-preset", "p4" if state.ffmpeg_hwaccel == "cuda" else "fast",
        *(["-b:v", "5000k"] if state.ffmpeg_hwaccel == "cuda" else []),
        *(["-crf", "26"] if state.ffmpeg_hwaccel != "cuda" else []),
        "-movflags", "+faststart",
        debug_video_path
    ]

    logger.info(f"Generating debug video with command: {' '.join(map(str, ffmpeg_command))}")

    subprocess.run(ffmpeg_command)

    os.remove(temp_video_path)
    logger.info(f"Debug video generated and stored in: {debug_video_path}")
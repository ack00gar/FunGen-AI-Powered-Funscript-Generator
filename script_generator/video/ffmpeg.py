import subprocess

from script_generator.video.video_info import VideoInfo
from config import FFPROBE_PATH, FFMPEG_PATH


def get_video_info(video_path):
    try:
        cmd = [
            FFPROBE_PATH,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,width,height,codec_name,nb_frames,duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            video_path,
        ]

        output = subprocess.check_output(cmd).decode("utf-8").splitlines()

        # Ensure the output has the correct number of fields
        if len(output) < 6:
            raise ValueError("FFProbe output is missing required fields.")

        # Parse metadata
        codec_name = output[0]
        width = int(output[1])
        height = int(output[2])
        r_frame_rate = output[3]
        duration = float(output[4])
        total_frames = int(output[5])

        # Calculate FPS
        num, den = map(int, r_frame_rate.split('/'))  # Split numerator and denominator
        fps = num / den  # Calculate FPS

        # If the width is 2x the height we are dealing with a VR video
        is_vr = height == width // 2

        return VideoInfo(video_path, codec_name, width, height, duration, total_frames, fps, is_vr)
    except Exception as e:
        print(f"Error initializing video info: {e}")
        raise

def is_cuda_supported():
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-hwaccels"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        return "cuda" in result.stdout.lower()
    except Exception as e:
        print(f"Error checking CUDA support: {e}")
        return False
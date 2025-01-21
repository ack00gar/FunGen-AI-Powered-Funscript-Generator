import subprocess

from script_generator.utils.logger import logger
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

        logger.info(f"Video Info: {codec_name}, {width}x{height}, {fps:.2f} fps, {total_frames} frames, {duration} seconds")

        # If the width is 2x the height we are dealing with a VR video
        is_vr = height == width // 2

        # TODO: make it possible to override this auto-detection

        if is_vr:
            logger.info("Video format: VR - Based on its 2:1 ratio, processing this video as a VR SBS video")
        else:
            logger.info("Video format: 2D - Based on its ratio, processing this video as a 2D video")

        return VideoInfo(video_path, codec_name, width, height, duration, total_frames, fps, is_vr)
    except Exception as e:
        logger.error(f"Error initializing video info: {e}")
        raise


def is_hwaccel_supported():
    """
    Check which hardware acceleration backends are supported by FFmpeg.
    Returns a dictionary with supported backends.
    """
    try:
        result = subprocess.run(
            [FFMPEG_PATH, "-hwaccels"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True
        )
        hwaccels = result.stdout.lower().replace("hardware acceleration methods:", "")
        hwaccel_lines = [line.strip() for line in str(hwaccels).splitlines() if line.strip()]
        logger.info(f"hardware acceleration methods: {', '.join(hwaccel_lines)}")

        # Check for supported hardware acceleration backends
        return {
            "cuda": "cuda" in hwaccels,
            "vaapi": "vaapi" in hwaccels,
            "amf": "amf" in hwaccels,
            "videotoolbox": "videotoolbox" in hwaccels,
            "qsv": "qsv" in hwaccels,  # Intel Quick Sync Video
            "d3d11va": "d3d11va" in hwaccels,  # Direct3D 11 (Windows)
            "opencl": "opencl" in hwaccels,  # OpenCL (cross-platform)
        }
    except Exception as e:
        logger.error(f"Error checking hardware acceleration support: {e}")
        return {
            "cuda": False,
            "vaapi": False,
            "amf": False,
            "videotoolbox": False,
            "qsv": False,
            "d3d11va": False,
            "opencl": False,
        }

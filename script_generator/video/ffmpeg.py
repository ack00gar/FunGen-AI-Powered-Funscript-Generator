import subprocess

from debug.errors import FFProbeError
from script_generator.utils.logger import logger
from script_generator.video.video_info import VideoInfo
from config import FFPROBE_PATH, FFMPEG_PATH


import json

def get_video_info(video_path):
    try:
        cmd = [
            FFPROBE_PATH,
            "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate,width,height,codec_name,nb_frames",
            "-show_entries", "format=duration",
            "-of", "json",
            video_path,
        ]

        output = subprocess.check_output(cmd, stderr=subprocess.STDOUT).decode("utf-8")
        info = json.loads(output)

        # Parse the first video stream
        stream = info.get("streams", [{}])[0]
        if not stream:
            raise FFProbeError("No video stream found in the file.")

        # Extract stream metadata
        codec_name = stream.get("codec_name", "unknown")
        width = int(stream.get("width", 0))
        height = int(stream.get("height", 0))
        r_frame_rate = stream.get("r_frame_rate", "0/1")
        nb_frames = stream.get("nb_frames", None)

        # Extract format-level metadata
        duration = float(info.get("format", {}).get("duration", 0))

        # Calculate FPS
        num, den = map(int, r_frame_rate.split('/'))
        fps = num / den if den > 0 else 0

        # Estimate frames if not available
        if nb_frames is None and duration > 0 and fps > 0:
            nb_frames = int(duration * fps)

        # Check if the video is VR (2:1 aspect ratio)
        is_vr = height == width // 2

        logger.info(f"Video Info: {codec_name}, {width}x{height}, {fps:.2f} fps, {nb_frames} frames, {duration:.2f} seconds, is vr: {is_vr}")

        if is_vr:
            logger.info("Video Format: VR SBS - Based on its 2:1 ratio")
        else:
            logger.info("Video Format: 2D - Based on its ratio")

        return VideoInfo(video_path, codec_name, width, height, duration, nb_frames, fps, is_vr)

    except subprocess.CalledProcessError as e:
        logger.error(f"FFProbe command failed: {e.output.decode('utf-8')}")
        raise FFProbeError("FFProbe command execution failed.")
    except (ValueError, KeyError, IndexError) as e:
        logger.error(f"Error parsing FFProbe output: {e}")
        raise FFProbeError("Failed to parse FFProbe output.")


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

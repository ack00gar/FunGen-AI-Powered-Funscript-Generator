import subprocess

from config import FFMPEG_PATH
from script_generator.debug.logger import logger


def get_hwaccel_support():
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

        return hwaccel_lines
    except Exception as e:
        logger.error(f"Error checking hardware acceleration support: {e}")
        return []

def get_preferred_hwaccel():
    hwaccel_support = get_hwaccel_support()
    hwaccel_priority = ["cuda", "vaapi", "amf", "videotoolbox", "qsv", "d3d11va"]
    for hw in hwaccel_priority:
        if hw in hwaccel_support:
            return hw
    return None

def get_hwaccel_read_args(hwaccel):
    if hwaccel == "cuda":
        return ["-hwaccel", "cuda", "-hwaccel_output_format", "cuda"]
    if hwaccel == "vaapi":
        return ["-hwaccel", "vaapi", "-hwaccel_device", "/dev/dri/renderD128"]
    if hwaccel == "amf":
        return ["-hwaccel", "amf"]
    if hwaccel == "videotoolbox":
        return ["-hwaccel", "videotoolbox"]
    if hwaccel == "qsv":
        return ["-hwaccel", "qsv"]
    if hwaccel == "d3d11va":
        return ["-hwaccel", "d3d11va"]
    return []
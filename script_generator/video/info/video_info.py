import json
import re
import subprocess
from dataclasses import dataclass, field

from config import RENDER_RESOLUTION, FFPROBE_PATH
from script_generator.debug.errors import FFProbeError
from script_generator.debug.logger import logger

@dataclass
class VideoInfo:
    path: str
    codec_name: str = None
    width: int = 0
    height: int = 0
    duration: float = 0.0
    total_frames: int = 0
    fps: float = 0.0
    is_vr: bool = False
    projection: str = field(init=False)
    fov: int = field(init=False)
    is_fisheye = False

    def __post_init__(self):
        info = get_projection_and_fov_from_filename(self.path)
        self.projection = info["projection"]
        self.fov = info["fov"]
        self.is_fisheye = info["is_fisheye"]


def get_projection_and_fov_from_filename(filename):
    filename = filename.replace("_FB360", "")
    projection = "LR_180"
    is_fisheye = False
    fov = 180

    patterns = [
        {"regex": r"180_sbs", "projection": "LR_180", "fov": 180, "is_fisheye": False},
        {"regex": r"_LR_180", "projection": "LR_180", "fov": 180, "is_fisheye": False},
        {"regex": r"_MONO_360", "projection": "MONO_360", "fov": 360, "is_fisheye": False},
        {"regex": r"_TB_360", "projection": "TB_360", "fov": 360, "is_fisheye": False},
        {"regex": r"_MKX200", "projection": "MKX200", "fov": 200, "is_fisheye": True},
        {"regex": r"_MKX220", "projection": "MKX220", "fov": 220, "is_fisheye": True},
        {"regex": r"_RF52", "projection": "RF52", "fov": 190, "is_fisheye": True},
        {"regex": r"_FISHEYE190", "projection": "FISHEYE190", "fov": 190, "is_fisheye": True},
        {"regex": r"_VRCA220", "projection": "VRCA220", "fov": 220, "is_fisheye": True},
        {"regex": r"_MKX200_alpha", "projection": "MKX200", "fov": 200, "is_fisheye": True},
        {"regex": r"_MKX220_alpha", "projection": "MKX220", "fov": 220, "is_fisheye": True},
        {"regex": r"_RF52_alpha", "projection": "RF52", "fov": 190, "is_fisheye": True},
        {"regex": r"_FISHEYE190_alpha", "projection": "FISHEYE190", "fov": 190, "is_fisheye": True},
        {"regex": r"_VRCA220_alpha", "projection": "VRCA220", "fov": 220, "is_fisheye": True},
        {"regex": r"180x180_3dh", "projection": "LR_180", "fov": 180, "is_fisheye": False},
        {"regex": r"VR180", "projection": "LR_180", "fov": 180, "is_fisheye": False},
        {"regex": r"oculusrift_", "projection": "LR_180", "fov": 180, "is_fisheye": False}
    ]

    for pattern in patterns:
        if re.search(pattern["regex"], filename):
            projection = pattern["projection"]
            fov = pattern["fov"]
            is_fisheye = pattern["is_fisheye"]
            break
    logger.info(f"Video Format: Projection={projection}, FOV={fov}, is_fisheye={is_fisheye}")

    return {"projection": projection, "fov": fov, "is_fisheye": is_fisheye}

def get_cropped_dimensions(video: VideoInfo):
    # We crop 2d squarely in the center so both vr and 2d have the same dimensions
    return RENDER_RESOLUTION, RENDER_RESOLUTION

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

        return VideoInfo(video_path, codec_name, width, height, duration, int(nb_frames), fps, is_vr)

    except subprocess.CalledProcessError as e:
        logger.error(f"FFProbe command failed: {e.output.decode('utf-8')}")
        raise FFProbeError("FFProbe command execution failed.")
    except (ValueError, KeyError, IndexError) as e:
        logger.error(f"Error parsing FFProbe output: {e}")
        raise FFProbeError("Failed to parse FFProbe output.")
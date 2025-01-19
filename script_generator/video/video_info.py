import re
from dataclasses import dataclass, field

from config import RENDER_RESOLUTION


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
            break

    return {"projection": projection, "fov": fov, "is_fisheye": is_fisheye}

def get_cropped_dimensions(video: VideoInfo):
    if video.is_vr:
        return RENDER_RESOLUTION, RENDER_RESOLUTION
    else:
        scaling_factor = min(1, RENDER_RESOLUTION / video.height)
        return int(video.width * scaling_factor), int(video.height * scaling_factor)
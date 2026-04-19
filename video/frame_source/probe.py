"""Lightweight video metadata probe backed by ffprobe.

Single place for "how many frames, what fps, what resolution, what duration"
across the app. Returns a dataclass.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

from video.ffmpeg_helpers import find_ffprobe, run as ffmpeg_run


@dataclass
class VideoProbe:
    path: str
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int


def _parse_rational(expr: str, fallback: float = 30.0) -> float:
    """Parse ffprobe fps expressions like '60000/1001' into a float."""
    if not expr:
        return fallback
    if "/" in expr:
        num, _, den = expr.partition("/")
        try:
            n = float(num)
            d = float(den)
            if d > 0:
                return n / d
        except ValueError:
            return fallback
    try:
        return float(expr)
    except ValueError:
        return fallback


def probe(video_path: str, timeout_s: float = 5.0) -> Optional[VideoProbe]:
    """Run ffprobe on ``video_path``, return video-stream metadata or None.

    Does not decode any frames. Typical cost: 20-80 ms per call.
    """
    cmd = [
        find_ffprobe(),
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        result = ffmpeg_run(cmd, timeout=timeout_s)
        if result.returncode != 0:
            return None
        data = json.loads(result.stdout.decode("utf-8", errors="replace"))
    except (OSError, ValueError, json.JSONDecodeError):
        return None

    streams = data.get("streams") or []
    if not streams:
        return None
    s = streams[0]

    width = int(s.get("width") or 0)
    height = int(s.get("height") or 0)

    # Prefer r_frame_rate (container-declared) over avg_frame_rate (computed
    # from frame count, skewed by VFR). Falls back cleanly for weird inputs.
    fps = _parse_rational(s.get("r_frame_rate", ""), fallback=0.0)
    if fps <= 0:
        fps = _parse_rational(s.get("avg_frame_rate", ""), fallback=30.0)
    if fps <= 0:
        fps = 30.0

    duration_sec = 0.0
    dur_raw = s.get("duration") or (data.get("format") or {}).get("duration")
    if dur_raw:
        try:
            duration_sec = float(dur_raw)
        except ValueError:
            duration_sec = 0.0

    total_frames = 0
    nb_raw = s.get("nb_frames")
    if nb_raw and nb_raw != "N/A":
        try:
            total_frames = int(nb_raw)
        except ValueError:
            total_frames = 0
    if total_frames <= 0 and duration_sec > 0:
        total_frames = int(duration_sec * fps)

    return VideoProbe(
        path=video_path,
        fps=fps,
        total_frames=total_frames,
        duration_sec=duration_sec,
        width=width,
        height=height,
    )

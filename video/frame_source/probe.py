"""Lightweight PyAV-backed video metadata probe.

Replaces ad-hoc ``cv2.VideoCapture(path); cap.get(CAP_PROP_FPS)`` calls
scattered around the app. Single place for "how many frames, what fps,
what resolution" — uses the same libav that drives the main playback
pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import av


@dataclass
class VideoProbe:
    path: str
    fps: float
    total_frames: int
    duration_sec: float
    width: int
    height: int


def probe(video_path: str) -> Optional[VideoProbe]:
    """Open the container, read video-stream metadata, close. Returns
    ``None`` on failure. Fast (no decode)."""
    container = None
    try:
        container = av.open(video_path)
        stream = container.streams.video[0]
        fps = float(stream.average_rate or stream.guessed_rate or 30.0)
        total_frames = int(stream.frames or 0)
        duration_sec = (
            float(container.duration / 1_000_000) if container.duration else 0.0
        )
        if total_frames <= 0 and duration_sec > 0:
            total_frames = int(duration_sec * fps)
        return VideoProbe(
            path=video_path,
            fps=fps,
            total_frames=total_frames,
            duration_sec=duration_sec,
            width=stream.width,
            height=stream.height,
        )
    except Exception:
        return None
    finally:
        if container is not None:
            try: container.close()
            except Exception: pass

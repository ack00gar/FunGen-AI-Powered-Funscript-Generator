"""Shared types for the frame-source subpackage.

SourceConfig describes how to open a video (path, filter chain, output
dims). The sentinel objects mark EOS and seek-flush events on the
internal queue. Callback type aliases let any future frame-source
backend expose the same contract to VideoProcessor.

Kept dependency-free (no libav / no ffmpeg imports) so swappable
backends can import it without dragging implementation deps in.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


SeekCallback = Callable[[int], None]
PlaybackStateCallback = Callable[[bool, float], None]
PositionCallback = Callable[[int], None]


@dataclass
class SourceConfig:
    """Everything a frame source needs to open a video at production fidelity."""
    video_path: str
    # Filter chain string in libavfilter syntax, applied after demux/decode.
    # Built by the integration layer (FFmpegBuildersMixin) so we match the
    # existing coord space byte-for-byte. e.g.
    #   "crop=4096:4096:0:0,v360=he:in_stereo=0:output=sg:...,format=bgr24"
    # The trailing ``format=bgr24`` is added by the backend if missing.
    filter_chain: str
    output_w: int
    output_h: int
    # Decode thread count hint. ``0`` means "AUTO" (let libav decide).
    decoder_threads: int = 0


# Sentinels placed on the internal frame queue to signal end-of-stream or
# a seek-induced drain. Using plain ``object()`` singletons is simpler than
# class-based sentinels and avoids pickling issues if the queue ever crosses
# a process boundary.
_EOS = object()
_FLUSH = object()

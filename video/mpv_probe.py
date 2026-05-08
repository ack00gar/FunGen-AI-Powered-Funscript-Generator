"""Headless libmpv metadata probe.

Drop-in alternative to ffprobe for the runtime video-info fetch. Spawns
a transient mpv player with audio + video disabled, plays the file long
enough to populate metadata, queries properties, then terminates. Keeps
a single instance available for sequential probes (Probe pool of one).

Returns the same dict shape as VideoProcessor._get_video_info so call
sites are interchangeable. Unmapped fields (bit_depth heuristics, audio
sub-fields not exposed by libmpv) fall back to filesystem stats and
reasonable defaults; callers that need full granularity should drop to
ffprobe via the legacy path.
"""
from __future__ import annotations

import logging
import os
import time
from typing import Any, Dict, Optional

from video.mpv_loader import mpv, mpv_available


def _safe_get(player, name: str, default=None):
    try:
        return player[name]
    except Exception:
        return default


def _track_audio_codec(player) -> tuple[str, int, int, int]:
    """Pull audio codec info from track-list. Returns
    (codec_name, bitrate, sample_rate, channels) with zeros on miss."""
    try:
        tracks = player.track_list or []
    except Exception:
        return "", 0, 0, 0
    for t in tracks:
        try:
            if t.get("type") == "audio":
                return (
                    str(t.get("codec") or ""),
                    int(t.get("demux-bitrate") or 0),
                    int(t.get("demux-samplerate") or 0),
                    int(t.get("demux-channel-count") or 0),
                )
        except Exception:
            continue
    return "", 0, 0, 0


def probe(filename: str, logger: Optional[logging.Logger] = None,
          timeout_s: float = 5.0) -> Optional[Dict[str, Any]]:
    """Read metadata via a transient libmpv instance. Returns None on any
    failure (caller falls back to ffprobe)."""
    log = logger or logging.getLogger("mpv_probe")
    if not mpv_available:
        return None
    if not filename or not os.path.isfile(filename):
        return None

    p = None
    try:
        p = mpv.MPV(audio="no", vo="null", ao="null", profile="fast",
                    keep_open="yes", pause="yes", loglevel="error")
    except Exception as e:
        log.debug(f"mpv probe init failed: {e}")
        return None

    try:
        try:
            p.play(filename)
        except Exception as e:
            log.debug(f"mpv probe play({filename}) failed: {e}")
            return None

        deadline = time.monotonic() + max(0.5, float(timeout_s))
        duration = 0.0
        while time.monotonic() < deadline:
            d = _safe_get(p, "duration")
            if d is not None and float(d) > 0:
                duration = float(d)
                break
            time.sleep(0.02)
        if duration <= 0:
            return None

        fps = 0.0
        for prop in ("container-fps", "estimated-vf-fps",
                     "fps", "video-fps"):
            v = _safe_get(p, prop)
            try:
                if v is not None and float(v) > 0:
                    fps = float(v)
                    break
            except (TypeError, ValueError):
                continue
        if fps <= 0:
            fps = 30.0

        width = int(_safe_get(p, "width") or 0)
        height = int(_safe_get(p, "height") or 0)
        # Some mpv builds expose dims only via video-params.
        if width <= 0 or height <= 0:
            vp = _safe_get(p, "video-params") or {}
            try:
                width = int(vp.get("w") or width)
                height = int(vp.get("h") or height)
            except Exception:
                pass

        codec_name = str(_safe_get(p, "video-codec-name") or
                         _safe_get(p, "video-codec") or "unknown")

        # Bit depth: libmpv exposes 'video-params/component-bits' on recent
        # builds; fall back to 8 if unknown.
        bit_depth = 8
        try:
            vp = _safe_get(p, "video-params") or {}
            bd = vp.get("component-bits") if isinstance(vp, dict) else None
            if bd:
                bit_depth = int(bd)
            else:
                pix = (vp.get("pixelformat") if isinstance(vp, dict) else "") or ""
                pix = str(pix).lower()
                if "10" in pix:
                    bit_depth = 10
                elif "12" in pix:
                    bit_depth = 12
        except Exception:
            pass

        total_frames = int(_safe_get(p, "estimated-frame-count") or 0)
        if total_frames <= 0 and fps > 0:
            total_frames = int(duration * fps)

        # Audio sub-fields.
        a_codec, a_bitrate, a_rate, a_chans = _track_audio_codec(p)
        has_audio = bool(a_codec)

        try:
            file_size = os.path.getsize(filename)
        except OSError:
            file_size = 0

        bitrate = 0
        try:
            br = _safe_get(p, "file-format-bitrate") or _safe_get(p, "video-bitrate")
            if br:
                bitrate = int(br)
        except Exception:
            pass
        if bitrate == 0 and duration > 0 and file_size > 0:
            bitrate = int(file_size * 8 / duration)

        return {
            "duration": duration,
            "total_frames": total_frames,
            "fps": fps,
            "width": width,
            "height": height,
            "has_audio": has_audio,
            "bit_depth": bit_depth,
            "file_size": file_size,
            "bitrate": bitrate,
            "is_vfr": False,  # libmpv doesn't differentiate r vs avg fps
            "filename": os.path.basename(filename),
            "codec_name": codec_name,
            "codec_long_name": codec_name,
            "audio_codec_name": a_codec,
            "audio_codec_long_name": a_codec,
            "audio_bitrate": a_bitrate,
            "audio_sample_rate": a_rate,
            "audio_channels": a_chans,
        }
    except Exception as e:
        log.debug(f"mpv probe error for {filename}: {e}")
        return None
    finally:
        if p is not None:
            try:
                p.terminate()
            except Exception:
                pass

"""Export a chapter as a stream-copied video clip plus a sliced funscript.

Uses PyAV to remux (no re-encode) the requested time window into a new mp4,
then writes a sibling .funscript with timestamps rebased to 0. Boundary
actions are injected at +0.001s from each edge by interpolating the source
funscript so the clipped script doesn't begin or end mid-stroke at an
arbitrary value.
"""
from __future__ import annotations

import json
import os
from typing import List, Optional, Tuple

import av

from common.frame_utils import frame_to_ms


_BOUNDARY_OFFSET_MS = 1  # 0.001s, avoids partial-stroke artifacts at boundaries


def _interpolate_pos(actions: List[dict], time_ms: float) -> int:
    """Linear interpolation of pos at time_ms. Returns nearest endpoint outside range."""
    if not actions:
        return 50
    if time_ms <= actions[0]['at']:
        return int(actions[0]['pos'])
    if time_ms >= actions[-1]['at']:
        return int(actions[-1]['pos'])
    # Binary search
    lo, hi = 0, len(actions) - 1
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if actions[mid]['at'] <= time_ms:
            lo = mid
        else:
            hi = mid
    a, b = actions[lo], actions[hi]
    span = b['at'] - a['at']
    if span <= 0:
        return int(a['pos'])
    t = (time_ms - a['at']) / span
    return int(round(a['pos'] + (b['pos'] - a['pos']) * t))


def slice_funscript_actions(actions: List[dict], start_ms: int, end_ms: int) -> List[dict]:
    """Return clipped funscript with timestamps rebased to 0 and boundary
    actions injected at +1ms / -1ms from the edges (interpolated)."""
    if not actions or end_ms <= start_ms:
        return []
    duration_ms = end_ms - start_ms
    keep = [a for a in actions if start_ms <= a['at'] <= end_ms]
    out = []
    # Leading boundary at 1ms with interpolated position at start_ms
    out.append({'at': _BOUNDARY_OFFSET_MS, 'pos': _interpolate_pos(actions, start_ms)})
    for a in keep:
        rebased = a['at'] - start_ms
        if rebased <= _BOUNDARY_OFFSET_MS or rebased >= (duration_ms - _BOUNDARY_OFFSET_MS):
            continue  # skip points that would collide with boundary markers
        out.append({'at': rebased, 'pos': int(a['pos'])})
    # Trailing boundary at duration-1ms
    out.append({'at': duration_ms - _BOUNDARY_OFFSET_MS,
                'pos': _interpolate_pos(actions, end_ms)})
    return out


def remux_video_segment(src_path: str, dst_path: str, start_s: float, end_s: float) -> bool:
    """Stream-copy [start_s, end_s] from src to dst via PyAV (no re-encode)."""
    if end_s <= start_s:
        return False
    in_ = av.open(src_path)
    out = av.open(dst_path, mode='w')
    stream_map = {}
    for s in in_.streams:
        if s.type in ('video', 'audio'):
            stream_map[s.index] = out.add_stream_from_template(s)
    try:
        in_.seek(int(start_s * av.time_base), any_frame=False, backward=True)
        for packet in in_.demux():
            if packet.dts is None or packet.stream.index not in stream_map:
                continue
            t = float(packet.pts * packet.stream.time_base) if packet.pts is not None else 0.0
            if t > end_s:
                break
            if t < start_s:
                continue
            packet.pts = int((t - start_s) / packet.stream.time_base)
            packet.dts = packet.pts
            packet.stream = stream_map[packet.stream.index]
            out.mux(packet)
    finally:
        out.close()
        in_.close()
    return True


def export_chapter_clip(src_video: str, dst_dir: str, chapter,
                        funscript_actions: List[dict], fps: float,
                        clip_basename: Optional[str] = None) -> Tuple[str, str]:
    """Export the chapter as a stream-copied clip + sliced funscript.

    Returns (video_path, funscript_path).
    """
    os.makedirs(dst_dir, exist_ok=True)
    if clip_basename is None:
        label = (chapter.position_short_name or chapter.position_long_name or "chapter")
        label = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(label))
        src_stem = os.path.splitext(os.path.basename(src_video))[0]
        clip_basename = f"{src_stem}__{label}_{int(chapter.start_frame_id)}-{int(chapter.end_frame_id)}"
    video_path = os.path.join(dst_dir, f"{clip_basename}.mp4")
    funscript_path = os.path.join(dst_dir, f"{clip_basename}.funscript")

    start_ms = int(frame_to_ms(int(chapter.start_frame_id), fps))
    end_ms = int(frame_to_ms(int(chapter.end_frame_id), fps))
    start_s = start_ms / 1000.0
    end_s = end_ms / 1000.0

    remux_video_segment(src_video, video_path, start_s, end_s)

    sliced = slice_funscript_actions(funscript_actions, start_ms, end_ms)
    payload = {"version": "1.0", "actions": sliced}
    with open(funscript_path, 'w') as f:
        json.dump(payload, f)

    return video_path, funscript_path

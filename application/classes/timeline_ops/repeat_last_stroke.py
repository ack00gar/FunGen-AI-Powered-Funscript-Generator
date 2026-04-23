"""Append a copy of the most recent stroke (last two actions before playhead)."""
from bisect import bisect_left

from .add_point import add_point


def repeat_last_stroke(tl) -> None:
    actions = tl._get_actions()
    if len(actions) < 2:
        tl.app.logger.info("Repeat stroke needs at least two prior points",
                           extra={'status_message': True})
        return
    ph = tl._playhead_ms()
    if ph is None:
        return

    timestamps = [a['at'] for a in actions]
    idx = bisect_left(timestamps, ph)
    last_idx = min(idx, len(actions)) - 1
    if last_idx < 1:
        tl.app.logger.info("Repeat stroke needs two points before the playhead",
                           extra={'status_message': True})
        return
    a1 = actions[last_idx - 1]
    a2 = actions[last_idx]
    dt = a2['at'] - a1['at']
    if dt <= 0:
        return

    new1_at = a2['at'] + dt
    new2_at = new1_at + dt
    fps = tl.app.processor.fps if tl.app.processor else 30.0
    tol = max(1, int(500 / max(1.0, fps)))

    def _has_neighbor(ts):
        j = bisect_left(timestamps, ts)
        for c in (j - 1, j):
            if 0 <= c < len(actions) and abs(actions[c]['at'] - ts) <= tol:
                return True
        return False

    if _has_neighbor(new1_at) or _has_neighbor(new2_at):
        tl.app.logger.info("Repeat stroke would overlap existing points",
                           extra={'status_message': True})
        return

    add_point(tl, new1_at, a1['pos'])
    add_point(tl, new2_at, a2['pos'])
    tl.app.logger.info(
        f"Repeated stroke at {new1_at}ms / {new2_at}ms (Timeline {tl.timeline_num})",
        extra={'status_message': True})

"""Scripted scrub pattern through FFmpegFrameSource with cache on vs off.

Simulates hover-over-timeline and point-jump navigation: repeated
get_frame() calls at nearby-but-not-monotonic frame indices. Each miss
normally costs an ffmpeg respawn; cache hits short-circuit the seek.
Needs a real video: pass `--video <path>`.
"""
from __future__ import annotations

import os
import time
from pathlib import Path

from ..harness import Report, Sample
from ..registry import register


@register(
    "scrub_cache",
    "Scripted scrub pattern via FFmpegFrameSource: cache off vs on (needs --video).",
)
def run(device: str, video: str, iters: int, warmup: int, **_) -> Report:
    if not video:
        # Probe a sensible default so a plain `bench run scrub_cache` works.
        candidates = [
            "/Users/k00gar/Downloads/test_koogar_extra_short_A.mp4",
        ]
        video = next((p for p in candidates if os.path.exists(p)), None)
        if not video:
            raise FileNotFoundError(
                "scrub_cache needs a --video path (defaults probed: "
                + ", ".join(candidates) + ")"
            )

    from video.frame_source.ffmpeg_source import FFmpegFrameSource
    from video.frame_source._types import SourceConfig

    def _open_source():
        cfg = SourceConfig(
            video_path=video,
            output_w=640, output_h=640,
            filter_chain="scale=640:640:force_original_aspect_ratio=decrease,pad=640:640:(ow-iw)/2:(oh-ih)/2:color=black",
        )
        src = FFmpegFrameSource(cfg)
        if not src.open():
            raise RuntimeError(f"failed to probe {video}")
        src.start(0)
        # Wait until first frame arrives so initial spawn cost isn't charged
        # to the first sample in the script.
        deadline = time.time() + 10.0
        while src._current_frame_index < 0 and time.time() < deadline:
            time.sleep(0.01)
        return src

    # Scripted scrub pattern -- representative of hover-over-timeline + arrow-nav.
    # Mix of new-target misses and within-window revisits so we measure both
    # sides of the cache. The "tight hover" variant (every 4th frame is a
    # revisit of the last miss) gives the upper-bound scrub-hover speedup.
    total = int(iters)
    near = 150
    script = []
    base = 100
    unique_targets = []
    for i in range(total):
        if i % 4 == 0:
            # New advance target: miss, then put in cache.
            idx = base + (i // 4) * near
            unique_targets.append(idx)
        elif i % 4 == 1:
            # Immediate revisit -> guaranteed hit.
            idx = unique_targets[-1]
        elif i % 4 == 2:
            # Small back-step off the last target -> miss on first, hit thereafter.
            idx = unique_targets[-1] - 10
        else:
            # Revisit 2 targets back if available -> LRU hit if still resident.
            idx = unique_targets[-2] if len(unique_targets) >= 2 else unique_targets[-1]
        script.append(idx)

    r = Report(
        name="scrub_cache",
        description=f"{total} scripted scrubs through {os.path.basename(video)} with cache OFF vs ON.",
        device=device,
    )

    # ---- CACHE OFF ----
    src = _open_source()
    try:
        if hasattr(src, "set_scrub_cache_size"):
            src.set_scrub_cache_size(0)
        # Warmup (hits the decode path so initial fragment costs are not charged).
        for i in script[:warmup]:
            src.get_frame(i, timeout=5.0, accurate=True)
        samples_off = []
        for idx in script[warmup:]:
            t0 = time.perf_counter()
            src.get_frame(idx, timeout=5.0, accurate=True)
            samples_off.append(time.perf_counter() - t0)
    finally:
        src.stop()

    r.add(Sample(label="cache OFF", samples_s=samples_off, device=device,
                 meta={"n": len(samples_off)}))

    # ---- CACHE ON (default size = 16) ----
    has_cache = hasattr(FFmpegFrameSource, "set_scrub_cache_size")
    if has_cache:
        src = _open_source()
        try:
            for i in script[:warmup]:
                src.get_frame(i, timeout=5.0, accurate=True)
            samples_on = []
            for idx in script[warmup:]:
                t0 = time.perf_counter()
                src.get_frame(idx, timeout=5.0, accurate=True)
                samples_on.append(time.perf_counter() - t0)
            stats = src.scrub_cache_stats
        finally:
            src.stop()
        r.add(Sample(label="cache ON (16)", samples_s=samples_on, device=device,
                     meta={"n": len(samples_on), "cache_stats": stats}))
    else:
        r.extra["cache_on_skipped"] = "FFmpegFrameSource has no scrub cache on this branch"

    return r

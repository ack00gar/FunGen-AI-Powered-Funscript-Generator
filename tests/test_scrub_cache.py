"""Tests for the scrub cache added to FFmpegFrameSource.

These tests exercise only the cache mechanics (put/get/eviction/stats/clear)
on a source object built without opening a real video or spawning ffmpeg.
End-to-end cache-hit timing needs a real video and belongs in a bench.
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pytest

from video.frame_source._types import SourceConfig
from video.frame_source.ffmpeg_source import FFmpegFrameSource


@pytest.fixture
def src() -> FFmpegFrameSource:
    """Build an FFmpegFrameSource without calling open() -- we only touch cache methods."""
    cfg = SourceConfig(
        video_path="/does/not/exist.mp4",
        output_w=64, output_h=64,
        filter_chain="",
    )
    return FFmpegFrameSource(cfg)


def _make_frame(i: int, h: int = 64, w: int = 64) -> np.ndarray:
    return np.full((h, w, 3), i % 255, dtype=np.uint8)


def test_cache_starts_empty(src):
    stats = src.scrub_cache_stats
    assert stats["size"] == 0
    assert stats["hits"] == 0
    assert stats["misses"] == 0


def test_cache_miss_returns_none_and_increments_stat(src):
    assert src._cache_get(42) is None
    assert src.scrub_cache_stats["misses"] == 1


def test_cache_put_then_hit(src):
    f = _make_frame(7)
    src._cache_put(7, f)
    got = src._cache_get(7)
    assert got is f
    stats = src.scrub_cache_stats
    assert stats["size"] == 1
    assert stats["hits"] == 1
    assert stats["misses"] == 0


def test_cache_lru_eviction_respects_max(src):
    src.set_scrub_cache_size(4)
    for i in range(6):
        src._cache_put(i, _make_frame(i))
    stats = src.scrub_cache_stats
    assert stats["size"] == 4
    # Oldest two should be evicted (0 and 1).
    assert src._cache_get(0) is None
    assert src._cache_get(1) is None
    # The rest should still be present.
    for i in range(2, 6):
        assert src._cache_get(i) is not None


def test_cache_hit_promotes_to_most_recent(src):
    src.set_scrub_cache_size(3)
    for i in range(3):
        src._cache_put(i, _make_frame(i))
    # Touch 0 to promote it.
    assert src._cache_get(0) is not None
    # Now insert a 4th -- eviction should drop 1, not 0.
    src._cache_put(99, _make_frame(99))
    assert src._cache_get(0) is not None
    assert src._cache_get(1) is None
    assert src._cache_get(2) is not None
    assert src._cache_get(99) is not None


def test_cache_clear_drops_everything(src):
    for i in range(5):
        src._cache_put(i, _make_frame(i))
    assert src.scrub_cache_stats["size"] == 5
    src._cache_clear()
    assert src.scrub_cache_stats["size"] == 0
    assert src._cache_get(0) is None


def test_set_scrub_cache_size_zero_disables(src):
    src.set_scrub_cache_size(0)
    src._cache_put(1, _make_frame(1))
    # Under a zero-max cache, put is effectively a no-op.
    assert src.scrub_cache_stats["size"] == 0
    assert src._cache_get(1) is None


def test_reapply_settings_clears_cache_even_without_new_config(src):
    src._cache_put(1, _make_frame(1))
    src._cache_put(2, _make_frame(2))
    assert src.scrub_cache_stats["size"] == 2
    # Calling reapply with no new_config and without a running decode loop
    # should still flush the cache (filter chain semantics).
    src.reapply_settings(None)
    assert src.scrub_cache_stats["size"] == 0


def test_cache_put_ignores_negative_index(src):
    src._cache_put(-1, _make_frame(0))
    assert src.scrub_cache_stats["size"] == 0


def test_cache_put_idempotent_for_same_index(src):
    f1 = _make_frame(5)
    src._cache_put(5, f1)
    # Inserting the same index again must not double-count or replace with newer array.
    src._cache_put(5, _make_frame(99))
    assert src.scrub_cache_stats["size"] == 1
    # The semantics we picked are "first write wins + LRU-promote on re-put".
    got = src._cache_get(5)
    assert got is f1

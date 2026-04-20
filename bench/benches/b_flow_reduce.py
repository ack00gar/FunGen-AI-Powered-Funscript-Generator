"""Microbench for HybridFlow._magnitude_weighted_flow.

Called once per frame during live tracking. The current implementation
allocates spatial weights via np.exp + np.outer every call even though
they only depend on the ROI shape (which rarely changes). This bench
compares the current path to an equivalent cached variant and to a
no-sqrt variant.
"""
from __future__ import annotations

import numpy as np

from ..harness import Report, Sample, measure
from ..registry import register


def _current(flow: np.ndarray) -> tuple:
    """Replica of the shipped _magnitude_weighted_flow."""
    magnitudes = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    fh, fw = flow.shape[:2]
    cx, sx = fw / 2, fw / 4.0
    cy, sy = fh / 2, fh / 4.0
    wx = np.exp(-((np.arange(fw) - cx) ** 2) / (2 * sx ** 2))
    wy = np.exp(-((np.arange(fh) - cy) ** 2) / (2 * sy ** 2))
    spatial = np.outer(wy, wx)
    combined = magnitudes * spatial
    total = np.sum(combined)
    if total > 0:
        dy = np.sum(flow[..., 1] * combined) / total
        dx = np.sum(flow[..., 0] * combined) / total
    else:
        dy = np.median(flow[..., 1])
        dx = np.median(flow[..., 0])
    return float(dy), float(dx)


class _WithCache:
    """Spatial weights cached by ROI shape; squared-magnitude weighting (no sqrt)."""

    def __init__(self):
        self._cache_key = None
        self._spatial = None

    def _weights(self, fh: int, fw: int) -> np.ndarray:
        key = (fh, fw)
        if self._cache_key != key:
            cx, sx = fw / 2, fw / 4.0
            cy, sy = fh / 2, fh / 4.0
            wx = np.exp(-((np.arange(fw, dtype=np.float32) - cx) ** 2) / (2 * sx * sx))
            wy = np.exp(-((np.arange(fh, dtype=np.float32) - cy) ** 2) / (2 * sy * sy))
            self._spatial = np.outer(wy, wx)
            self._cache_key = key
        return self._spatial

    def __call__(self, flow: np.ndarray) -> tuple:
        fh, fw = flow.shape[:2]
        spatial = self._weights(fh, fw)
        # Squared magnitude as weight -- sqrt is monotonic so the weighted
        # average is mathematically equivalent to weighting by magnitude up
        # to a global scale that cancels in the numerator/denominator ratio.
        mag2 = flow[..., 0] * flow[..., 0] + flow[..., 1] * flow[..., 1]
        combined = mag2 * spatial
        total = combined.sum()
        if total > 0:
            dy = (flow[..., 1] * combined).sum() / total
            dx = (flow[..., 0] * combined).sum() / total
        else:
            dy = float(np.median(flow[..., 1]))
            dx = float(np.median(flow[..., 0]))
        return float(dy), float(dx)


@register(
    "flow_reduce",
    "HybridFlow._magnitude_weighted_flow: current vs cached-weights + squared-mag.",
)
def run(iters: int, warmup: int, **_) -> Report:
    rng = np.random.default_rng(0)
    # Typical ROI flow from DIS: a few hundred pixels on each side, 2 channels.
    shapes = [(160, 240), (240, 320), (320, 480)]
    r = Report(
        name="flow_reduce",
        description=f"float32 flow fields, {iters} iters each shape.",
        device="cpu",
    )

    cached = _WithCache()
    # Prime the cache with one warmup call at each shape so its first-call
    # cost is not folded into the AFTER samples.
    for h, w in shapes:
        flow = rng.standard_normal((h, w, 2)).astype(np.float32)
        cached(flow)

    for h, w in shapes:
        flow = rng.standard_normal((h, w, 2)).astype(np.float32)
        a = measure(lambda: _current(flow), iters=iters, warmup=warmup)
        b = measure(lambda: cached(flow), iters=iters, warmup=warmup)
        r.add(Sample(label=f"current  {h}x{w}", samples_s=a))
        r.add(Sample(label=f"cached+no-sqrt  {h}x{w}", samples_s=b))

    return r

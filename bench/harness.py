"""Core measurement primitives: device sync, Sample/Report dataclasses, measure()."""
from __future__ import annotations

import gc
import statistics as _stats
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional


def _sync(device: str) -> None:
    if not device or device == "cpu":
        return
    try:
        import torch
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
            return
        if device == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            torch.mps.synchronize()
    except Exception:
        pass


def detect_device(preferred: str = "auto") -> str:
    if preferred and preferred != "auto":
        return preferred
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


@dataclass
class Sample:
    label: str
    samples_s: list
    device: str = "cpu"
    meta: dict = field(default_factory=dict)

    def _pct(self, p: float) -> float:
        if not self.samples_s:
            return 0.0
        s = sorted(self.samples_s)
        k = max(0, min(len(s) - 1, int(round((len(s) - 1) * p))))
        return s[k]

    @property
    def n(self) -> int: return len(self.samples_s)
    @property
    def mean(self) -> float: return _stats.fmean(self.samples_s) if self.samples_s else 0.0
    @property
    def stdev(self) -> float: return _stats.pstdev(self.samples_s) if len(self.samples_s) > 1 else 0.0
    @property
    def p50(self) -> float: return self._pct(0.50)
    @property
    def p95(self) -> float: return self._pct(0.95)
    @property
    def p99(self) -> float: return self._pct(0.99)
    @property
    def minv(self) -> float: return min(self.samples_s) if self.samples_s else 0.0


@dataclass
class Report:
    name: str
    description: str
    device: str
    samples: list = field(default_factory=list)
    extra: dict = field(default_factory=dict)

    def add(self, s: Sample) -> None:
        self.samples.append(s)

    def pretty(self) -> str:
        lines = ["", f"=== {self.name} ==="]
        if self.description:
            lines.append(self.description)
        lines.append(f"device: {self.device}  n per sample: {[s.n for s in self.samples]}")
        lines.append("")
        header = f"  {'label':<28} {'p50 ms':>9} {'p95 ms':>9} {'mean ms':>9} {'stdev ms':>10} {'n':>4}    vs baseline"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))
        baseline = self.samples[0] if self.samples else None
        for s in self.samples:
            ms = 1000.0
            speedup_str = ""
            if baseline and s is not baseline and baseline.mean > 0 and s.mean > 0:
                speedup_str = f"   {baseline.mean / s.mean:.2f}x"
            lines.append(
                f"  {s.label:<28} {s.p50*ms:>9.2f} {s.p95*ms:>9.2f} "
                f"{s.mean*ms:>9.2f} {s.stdev*ms:>10.2f} {s.n:>4}{speedup_str}"
            )
        if self.extra:
            lines.append("")
            for k, v in self.extra.items():
                lines.append(f"  [{k}] {v}")
        return "\n".join(lines)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "description": self.description,
            "device": self.device,
            "samples": [
                {
                    "label": s.label,
                    "n": s.n,
                    "mean_s": s.mean, "p50_s": s.p50, "p95_s": s.p95, "p99_s": s.p99,
                    "min_s": s.minv, "stdev_s": s.stdev,
                    "device": s.device, "meta": s.meta,
                }
                for s in self.samples
            ],
            "extra": self.extra,
        }


def measure(fn: Callable[..., Any], iters: int, warmup: int = 3, device: str = "cpu",
            args: tuple = (), kwargs: Optional[dict] = None) -> list:
    """Run fn warmup + iters times. Returns per-iter seconds."""
    kwargs = kwargs or {}
    for _ in range(warmup):
        fn(*args, **kwargs)
        _sync(device)
    gc.collect()
    out = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn(*args, **kwargs)
        _sync(device)
        out.append(time.perf_counter() - t0)
    return out

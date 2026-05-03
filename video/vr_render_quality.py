"""Adaptive quality ladder for the VR shader pipeline.

Four levels from L0 (best) to L3 (cheapest). An EMA-smoothed observer of the
shader pass time drives auto-step-up/down with hysteresis so the level does
not flap per frame.

Knobs per level:
  - supersample_factor: output FBO multiplier over target (1 or 2)
  - use_bicubic:       4-tap bicubic input sampling in the fragment shader
  - aniso_level:       GL_TEXTURE_MAX_ANISOTROPY on the input texture
  - mpv_scale:         name of mpv's `scale` property (ewa_lanczos / spline36
                        / bilinear)

Set mode='auto' to let the monitor pick; 'high' / 'medium' / 'low' pin a level.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Deque, Literal
from collections import deque


@dataclass(frozen=True)
class QualitySpec:
    name: str
    supersample_factor: float   # float: >1 supersample, <1 render smaller and upscale
    use_bicubic: bool
    aniso_level: float          # 1.0 = off, 4/8/16 = enabled
    mpv_scale: str              # mpv scale property value


LEVELS = [
    QualitySpec(name="L0",
                supersample_factor=2.0,
                use_bicubic=True,
                aniso_level=16.0,
                mpv_scale="ewa_lanczos"),
    QualitySpec(name="L1",
                supersample_factor=2.0,
                use_bicubic=False,
                aniso_level=16.0,
                mpv_scale="spline36"),
    QualitySpec(name="L2",
                supersample_factor=1.0,
                use_bicubic=False,
                aniso_level=4.0,
                mpv_scale="bilinear"),
    QualitySpec(name="L3",
                supersample_factor=1.0,
                use_bicubic=False,
                aniso_level=1.0,
                mpv_scale="bilinear"),
    QualitySpec(name="L4",
                supersample_factor=0.75,
                use_bicubic=False,
                aniso_level=1.0,
                mpv_scale="bilinear"),
    QualitySpec(name="L5",
                supersample_factor=0.5,
                use_bicubic=False,
                aniso_level=1.0,
                mpv_scale="bilinear"),
]

Mode = Literal["auto", "high", "medium", "low"]

_PIN_MAP = {"high": 0, "medium": 2, "low": 4}


class VRRenderQualityMonitor:
    """EMA-smoothed observer of shader pass ms + hysteresis-gated level changes.

    Usage:
      mon.record_pass_ms(shader._last_render_ms)   # each frame
      spec = mon.current_spec(mode='auto')         # apply spec.* before next pass
    """

    def __init__(self,
                 target_fps: float = 60.0,
                 step_down_ratio: float = 0.80,
                 step_up_ratio: float = 0.50,
                 min_frames_between_changes: int = 60,
                 ema_alpha: float = 0.10):
        self.target_fps = target_fps
        # Step down when EMA pass ms exceeds this fraction of the frame budget
        self._budget_ms = 1000.0 / target_fps
        self.step_down_ms = self._budget_ms * step_down_ratio
        # Step up when EMA pass ms is well under budget for a while
        self.step_up_ms = self._budget_ms * step_up_ratio
        self.min_frames_between_changes = min_frames_between_changes
        self.ema_alpha = ema_alpha
        self._ema_ms: float = 0.0
        self._samples: int = 0
        # Start at L2 (1.0x ss, no bicubic, 4x aniso). Avoids the cold-start
        # black screen on 8K VR where L0's 2x supersample overwhelms the GPU
        # before the EMA observes a single frame. Step up to L1/L0 only after
        # confirmed under-budget headroom.
        self._current_idx: int = 2
        self._frames_since_change: int = 0
        self._frames_over_budget: int = 0
        self._frames_under_budget: int = 0

    def record_pass_ms(self, ms: Optional[float]) -> None:
        """Feed one frame's shader pass time. None / zero samples are ignored."""
        if ms is None or ms <= 0:
            return
        if self._samples == 0:
            self._ema_ms = ms
        else:
            self._ema_ms = self.ema_alpha * ms + (1 - self.ema_alpha) * self._ema_ms
        self._samples += 1
        self._frames_since_change += 1
        if self._ema_ms > self.step_down_ms:
            self._frames_over_budget += 1
            self._frames_under_budget = 0
        elif self._ema_ms < self.step_up_ms:
            self._frames_under_budget += 1
            self._frames_over_budget = 0
        else:
            self._frames_over_budget = max(0, self._frames_over_budget - 1)
            self._frames_under_budget = max(0, self._frames_under_budget - 1)

    def _maybe_step_level(self) -> None:
        if self._frames_since_change < self.min_frames_between_changes:
            return
        # Need sustained over-budget to step DOWN (reduce quality).
        # 30 frames = 0.5s at 60fps -- responsive enough to avoid stutter.
        if self._frames_over_budget >= 30 and self._current_idx < len(LEVELS) - 1:
            self._current_idx += 1
            self._frames_since_change = 0
            self._frames_over_budget = 0
            return
        # Need sustained under-budget to step UP (increase quality).
        # 120 frames = 2s at 60fps -- quickly hunt for best quality when
        # GPU has headroom, previously 240 which took 4s.
        if self._frames_under_budget >= 120 and self._current_idx > 0:
            self._current_idx -= 1
            self._frames_since_change = 0
            self._frames_under_budget = 0

    def current_spec(self, mode: Mode = "auto") -> QualitySpec:
        """Return the spec to apply for the next render.

        mode 'auto': runs the hysteresis stepper and returns the current auto
        level. mode 'high'/'medium'/'low': pinned level, bypasses the stepper.
        """
        if mode in _PIN_MAP:
            return LEVELS[_PIN_MAP[mode]]
        self._maybe_step_level()
        return LEVELS[self._current_idx]

    @property
    def current_level_name(self) -> str:
        return LEVELS[self._current_idx].name

    @property
    def ema_ms(self) -> float:
        return self._ema_ms

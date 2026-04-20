"""HD -> imgsz resize end-to-end, with cv2 baseline vs torch GPU variant.

Replicates the resize + zero + paste sequence of _make_processing_frame.
The torch path matches the code landed on perf/gpu-resize and lets us
decide if GPU resize beats cv2.INTER_AREA at 8K-per-eye VR resolutions.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..harness import Report, Sample, _sync, measure
from ..registry import register


def _layout(imgsz: int, hd_w: int, hd_h: int):
    size = imgsz
    scale = size / max(hd_w, hd_h)
    new_w = int(hd_w * scale) & ~1
    new_h = int(hd_h * scale) & ~1
    new_w = min(new_w, size)
    new_h = min(new_h, size)
    pad_top = (size - new_h) // 2
    pad_left = (size - new_w) // 2
    return new_w, new_h, pad_top, pad_left


@register(
    "hd_resize",
    "HD -> imgsz resize + paste: cv2 baseline vs torch (device) end-to-end.",
)
def run(device: str, iters: int, warmup: int, imgsz: int, **_) -> Report:
    hd_w, hd_h = 2160, 2160
    new_w, new_h, pad_top, pad_left = _layout(imgsz, hd_w, hd_h)
    pad_bot = imgsz - new_h - pad_top
    pad_right = imgsz - new_w - pad_left

    rng = np.random.default_rng(0)
    hd = rng.integers(0, 255, (hd_h, hd_w, 3), dtype=np.uint8)
    buf = np.zeros((imgsz, imgsz, 3), dtype=np.uint8)

    # ---- Variant 1: cv2 full buf zero (current shipped baseline) ----
    def cv2_full_zero():
        resized = cv2.resize(hd, (new_w, new_h), interpolation=cv2.INTER_AREA)
        buf[:] = 0
        buf[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    # ---- Variant 2: cv2 pad-only zero ----
    def cv2_pad_only():
        resized = cv2.resize(hd, (new_w, new_h), interpolation=cv2.INTER_AREA)
        if pad_top:
            buf[:pad_top, :, :] = 0
        if pad_bot:
            buf[pad_top + new_h:, :, :] = 0
        if pad_left:
            buf[pad_top:pad_top + new_h, :pad_left, :] = 0
        if pad_right:
            buf[pad_top:pad_top + new_h, pad_left + new_w:, :] = 0
        buf[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

    r = Report(
        name="hd_resize",
        description=f"HD {hd_w}x{hd_h} -> {imgsz}x{imgsz} resize + paste (INTER_AREA vs torch {device}).",
        device=device,
    )

    a = measure(cv2_full_zero, iters=iters, warmup=warmup)
    b = measure(cv2_pad_only, iters=iters, warmup=warmup)
    r.add(Sample(label="cv2 + buf[:]=0", samples_s=a, device="cpu"))
    r.add(Sample(label="cv2 + pad-only zero", samples_s=b, device="cpu"))

    # ---- Variant 3: torch on the requested device (if available) ----
    torch_ok = False
    try:
        import torch
        import torch.nn.functional as F
        if device == "cuda" and not torch.cuda.is_available():
            r.extra["torch_skip"] = "cuda requested but not available"
        elif device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
            r.extra["torch_skip"] = "mps requested but not available"
        elif device == "cpu":
            # torch on cpu is pointless vs cv2 but kept for completeness
            torch_ok = True
        else:
            torch_ok = True
    except Exception as e:
        r.extra["torch_skip"] = f"torch import failed: {e}"

    if torch_ok:
        import torch
        import torch.nn.functional as F

        def torch_resize():
            # Upload HWC uint8 -> NCHW float32 on device, resize, download.
            t = torch.from_numpy(hd).to(device).permute(2, 0, 1).unsqueeze(0).float()
            out = F.interpolate(t, size=(new_h, new_w), mode="bilinear", antialias=True, align_corners=False)
            out = out.clamp_(0, 255).to(torch.uint8).squeeze(0).permute(1, 2, 0).contiguous()
            resized = out.cpu().numpy()
            if pad_top:
                buf[:pad_top, :, :] = 0
            if pad_bot:
                buf[pad_top + new_h:, :, :] = 0
            if pad_left:
                buf[pad_top:pad_top + new_h, :pad_left, :] = 0
            if pad_right:
                buf[pad_top:pad_top + new_h, pad_left + new_w:, :] = 0
            buf[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized

        try:
            c = measure(torch_resize, iters=iters, warmup=warmup, device=device)
            r.add(Sample(label=f"torch resize on {device}", samples_s=c, device=device))
        except Exception as e:
            r.extra["torch_run_error"] = str(e)

    return r

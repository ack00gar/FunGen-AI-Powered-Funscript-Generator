"""Reusable async DIS optical-flow worker for offline trackers.

Moves `cv2.DISOpticalFlow.calc()` (the dominant per-frame cost after YOLO)
off the main thread so the consumer loop can keep cvtColor + bookkeeping
tight while flow computation overlaps with decode + YOLO.

Main loop submits ``(frame_idx, prev_gray, curr_gray, roi, time_ms)`` jobs
(or None for a reset sentinel) and the worker calls back on_result with
``(frame_idx, time_ms, dy, dx, flow_meta)``. Order is preserved by the
FIFO queue.
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Optional, Tuple

import cv2
import numpy as np


_MIN_PATCH = 5


class AsyncDisFlowWorker:
    def __init__(self,
                 on_result: Callable[[int, int, float, float, dict], None],
                 preset: int = cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST,
                 logger: Optional[logging.Logger] = None,
                 input_queue_size: int = 16):
        self._on_result = on_result
        self._preset = preset
        self._logger = logger or logging.getLogger(__name__)
        self._in: "queue.Queue" = queue.Queue(maxsize=input_queue_size)
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._dis = None
        self._flow_time_s = 0.0
        self._submitted = 0
        self._completed = 0
        self._dropped = 0

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._dis = cv2.DISOpticalFlow.create(self._preset)
        self._thread = threading.Thread(
            target=self._loop, name="AsyncDisFlowWorker", daemon=True)
        self._thread.start()

    def stop(self, drain_timeout_s: float = 5.0) -> None:
        deadline = time.time() + drain_timeout_s
        while time.time() < deadline and not self._in.empty():
            time.sleep(0.01)
        self._stop.set()
        try:
            self._in.put_nowait(None)
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    def submit(self, frame_idx: int, prev_gray: np.ndarray,
               curr_gray: np.ndarray, roi: Tuple[int, int, int, int],
               time_ms: int, timeout_s: float = 0.05) -> bool:
        try:
            self._in.put((frame_idx, prev_gray, curr_gray, roi, time_ms),
                         timeout=timeout_s)
            self._submitted += 1
            return True
        except queue.Full:
            self._dropped += 1
            return False

    @property
    def total_flow_ms(self) -> float:
        return self._flow_time_s * 1000.0

    @property
    def submitted(self) -> int:
        return self._submitted

    @property
    def completed(self) -> int:
        return self._completed

    @property
    def dropped(self) -> int:
        return self._dropped

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                job = self._in.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:
                return
            f_idx, prev_gray, curr_gray, roi, time_ms = job
            if prev_gray is None or curr_gray is None:
                continue
            rx1, ry1, rx2, ry2 = roi
            prev_patch = prev_gray[ry1:ry2, rx1:rx2]
            curr_patch = curr_gray[ry1:ry2, rx1:rx2]
            if (prev_patch.shape != curr_patch.shape
                    or prev_patch.shape[0] <= _MIN_PATCH
                    or prev_patch.shape[1] <= _MIN_PATCH):
                continue
            t0 = time.perf_counter()
            try:
                flow = self._dis.calc(
                    np.ascontiguousarray(prev_patch),
                    np.ascontiguousarray(curr_patch),
                    None,
                )
            except cv2.error:
                flow = None
            self._flow_time_s += time.perf_counter() - t0
            if flow is None:
                continue
            # Magnitude-weighted averaging per axis
            try:
                magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
                weight_sum = float(magnitude.sum())
                if weight_sum > 1e-6:
                    dy = float((flow[..., 1] * magnitude).sum() / weight_sum)
                    dx = float((flow[..., 0] * magnitude).sum() / weight_sum)
                else:
                    dy = float(flow[..., 1].mean())
                    dx = float(flow[..., 0].mean())
            except Exception as e:
                self._logger.debug(f"flow aggregation error: {e}")
                continue
            meta = {'patch_h': flow.shape[0], 'patch_w': flow.shape[1]}
            try:
                self._on_result(f_idx, time_ms, dy, dx, dict(meta, flow=flow))
            except Exception as e:
                self._logger.debug(f"flow on_result error: {e}")
            self._completed += 1

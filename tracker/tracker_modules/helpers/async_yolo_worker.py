"""Reusable async YOLO worker for offline trackers.

Spins up one background thread that pulls ``(frame_idx, frame, payload)``
jobs off a bounded input queue, runs YOLO inference, and pushes
``(frame_idx, frame_h, frame_w, detections, payload)`` onto an output queue.
The tracker's main loop submits jobs, drains results non-blockingly after
each frame, and only pays for inference in parallel with its own DIS /
cvtColor / bookkeeping work.

Usage:

    from tracker.tracker_modules.helpers.async_yolo_worker import AsyncYoloWorker

    def apply_result(frame_idx, h, w, det_objs, payload):
        # update ROI / positions / overlays based on completed YOLO
        ...

    yolo = AsyncYoloWorker(model, conf=0.4, imgsz=640, device="mps",
                           on_result=apply_result, logger=logger)
    yolo.start()
    try:
        for frame_idx, frame, _ in stream_frames(...):
            if should_run_yolo:
                yolo.submit(frame_idx, frame, payload=("chap" if is_chap else ""))
            # DIS / cvtColor / bookkeeping on main thread
            ...
            yolo.drain()  # apply completed results
    finally:
        yolo.stop()            # drain in-flight + join worker
        elapsed_ms = yolo.total_inference_ms
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Any, Callable, Optional


class AsyncYoloWorker:
    """One-thread async YOLO inference with bounded input queue.

    The main thread stays free to run per-frame work (DIS, cvtColor, etc.)
    in parallel with inference. Results are applied via an `on_result`
    callback when the caller calls ``drain()`` (non-blocking) or on
    ``stop()`` (drains remaining).
    """

    def __init__(self,
                 model,
                 conf: float,
                 imgsz: int,
                 device: str,
                 on_result: Callable[[int, int, int, list, Any], None],
                 logger: Optional[logging.Logger] = None,
                 input_queue_size: int = 8,
                 batch_size: int = 1,
                 batch_timeout_s: float = 0.02):
        self._model = model
        self._conf = conf
        self._imgsz = imgsz
        self._device = device
        self._on_result = on_result
        self._logger = logger or logging.getLogger(__name__)

        self._in: "queue.Queue" = queue.Queue(maxsize=input_queue_size)
        self._out: "queue.Queue" = queue.Queue()
        self._stop = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._inference_time_s = 0.0
        self._submitted = 0
        self._completed = 0
        self._dropped = 0
        self._batch_size = max(1, int(batch_size))
        self._batch_timeout_s = batch_timeout_s
        # Flipped to False after first batch call raises on this device; we
        # then stay on single-frame GPU inference (no CPU fallback).
        self._batch_ok = self._batch_size > 1

    # ----- lifecycle -----

    def _probe_batch(self) -> None:
        """Detect at init whether the model accepts batch>1. Avoids the
        per-run WARNING when the engine is fixed-batch=1 (TRT default)."""
        if self._batch_size <= 1:
            return
        try:
            import numpy as _np
            dummy = [_np.zeros((self._imgsz, self._imgsz, 3), dtype=_np.uint8)
                     for _ in range(self._batch_size)]
            self._model(dummy, device=self._device, verbose=False,
                        conf=self._conf, imgsz=self._imgsz)
        except Exception as e:
            self._logger.info(
                f"YOLO model rejected batch={self._batch_size} "
                f"({type(e).__name__}); using single-frame inference.")
            self._batch_ok = False
            self._batch_size = 1

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._probe_batch()
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, name="AsyncYoloWorker", daemon=True)
        self._thread.start()

    def stop(self, drain_timeout_s: float = 10.0) -> None:
        """Wait for in-flight jobs to finish, apply their results, join worker."""
        deadline = time.time() + drain_timeout_s
        while time.time() < deadline:
            pending = self._in.qsize() + self._out.qsize()
            self.drain()
            if pending == 0 and self._in.empty() and self._out.empty():
                break
            time.sleep(0.01)
        self._stop.set()
        try:
            self._in.put_nowait(None)  # sentinel
        except queue.Full:
            pass
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

    # ----- public API -----

    def submit(self, frame_idx: int, frame, payload: Any = None,
               submit_timeout_s: float = 0.05) -> bool:
        """Queue a YOLO job. Returns False if the input queue was full and
        the submission was dropped (tracker should treat that frame as
        lacking a YOLO result for this cycle)."""
        try:
            self._in.put((frame_idx, frame, payload), timeout=submit_timeout_s)
            self._submitted += 1
            return True
        except queue.Full:
            self._dropped += 1
            return False

    def drain(self) -> int:
        """Apply any completed results via on_result. Non-blocking. Returns
        the count of results applied this call."""
        applied = 0
        while True:
            try:
                item = self._out.get_nowait()
            except queue.Empty:
                return applied
            f_idx, fh, fw, det_objs, payload = item
            try:
                self._on_result(f_idx, fh, fw, det_objs, payload)
            except Exception as e:
                self._logger.debug(f"AsyncYoloWorker on_result error: {e}")
            applied += 1
            self._completed += 1

    # ----- stats (read after stop) -----

    @property
    def total_inference_ms(self) -> float:
        return self._inference_time_s * 1000.0

    @property
    def submitted(self) -> int:
        return self._submitted

    @property
    def completed(self) -> int:
        return self._completed

    @property
    def dropped(self) -> int:
        """Frames whose YOLO job was dropped because the input queue was full."""
        return self._dropped

    # ----- worker loop -----

    def _loop(self) -> None:
        from tracker.tracker_modules.helpers.yolo_detection_helper import (
            run_detection as _yolo_run_detection,
            _parse_detections,
        )
        while not self._stop.is_set():
            try:
                job = self._in.get(timeout=0.2)
            except queue.Empty:
                continue
            if job is None:  # shutdown sentinel
                return

            # Collect a batch (up to batch_size or until batch_timeout_s
            # passes with no new job). We always have at least one job.
            jobs = [job]
            if self._batch_ok and self._batch_size > 1:
                deadline = time.perf_counter() + self._batch_timeout_s
                while len(jobs) < self._batch_size:
                    remaining = deadline - time.perf_counter()
                    if remaining <= 0:
                        break
                    try:
                        nxt = self._in.get(timeout=remaining)
                    except queue.Empty:
                        break
                    if nxt is None:
                        # Shutdown arrived mid-batch: process what we have
                        # and then terminate on next outer-loop iteration.
                        self._stop.set()
                        break
                    jobs.append(nxt)

            frames = [j[1] for j in jobs]
            t0 = time.perf_counter()
            dets_per_frame: list
            if len(frames) == 1 or not self._batch_ok:
                try:
                    dets = _yolo_run_detection(
                        self._model, frames[0],
                        conf=self._conf, imgsz=self._imgsz, device=self._device)
                    dets_per_frame = [dets]
                except Exception as e:
                    self._logger.debug(f"YOLO error frame {jobs[0][0]}: {e}")
                    dets_per_frame = [[]]
            else:
                try:
                    results = self._model(frames, device=self._device,
                                          verbose=False, conf=self._conf,
                                          imgsz=self._imgsz)
                    names = self._model.names
                    dets_per_frame = [_parse_detections([r], names) for r in results]
                except Exception as e:
                    # Batch failed on this device/model — stay on GPU but
                    # drop to single-frame mode permanently.
                    self._logger.warning(
                        f"YOLO batch inference failed ({e}); falling back "
                        f"to single-frame GPU inference.")
                    self._batch_ok = False
                    dets_per_frame = []
                    for fr, jb in zip(frames, jobs):
                        try:
                            dets_per_frame.append(_yolo_run_detection(
                                self._model, fr,
                                conf=self._conf, imgsz=self._imgsz,
                                device=self._device))
                        except Exception as e2:
                            self._logger.debug(f"YOLO error frame {jb[0]}: {e2}")
                            dets_per_frame.append([])
            self._inference_time_s += time.perf_counter() - t0

            for (f_idx, frame, payload), dets in zip(jobs, dets_per_frame):
                self._out.put((f_idx, frame.shape[0], frame.shape[1], dets, payload))

#!/usr/bin/env python3
"""
Live Hybrid Flow Tracker -- YOLO ROI detection + DIS optical flow.

Same core logic as the offline VR Hybrid Chapter-Aware tracker, but runs
frame-by-frame in real-time. Supports ranges, chapters, and all live
tracker features (device control, real-time preview, etc.).

Pipeline per frame:
  1. Run YOLO detection (every Nth frame) to find penis + nearest contact
  2. Compute padded ROI from detection boxes
  3. Run DIS ULTRAFAST optical flow in the ROI
  4. Extract magnitude-weighted vertical displacement (dy)
  5. Integrate dy into cumulative position with drift removal
  6. Normalize and emit funscript action
"""

import logging
import os
import time
import numpy as np
import cv2
from typing import Dict, Any, Optional, List, Tuple
from collections import deque
from itertools import islice

try:
    from ..core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult, StageDefinition
except ImportError:
    from tracker.tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult, StageDefinition

from config import constants as config_constants

# Contact classes that indicate interaction
CONTACT_CLASSES = {'pussy', 'butt', 'anus', 'face', 'hand', 'breast', 'foot'}

# ROI padding factor (fraction of ROI dimensions)
ROI_PADDING = 0.3

# Default YOLO detection interval (run every N frames)
DEFAULT_YOLO_INTERVAL = 5

# Flow history for normalization
FLOW_HISTORY_SIZE = 300  # ~10s at 30fps

# EMA smoothing alpha for position output
POSITION_SMOOTHING_ALPHA = 0.4

# Ignore very small flow values to avoid integrating optical-flow noise
FLOW_DEADZONE = 0.03

# Drift removal / normalization windows (in frames)
DRIFT_REMOVAL_WINDOW = 90
NORMALIZATION_WINDOW = 180

# If ROI shifts by more than this many pixels, reinitialize flow reference
ROI_SHIFT_RESET_THRESHOLD = 6

# Output normalization target: map recent p10-p90 spread to this amplitude span around 50
TARGET_HALF_RANGE = 41.0

# Slow baseline tracking for high-pass filtering cumulative displacement
BASELINE_ALPHA = 0.003

# Soft limiter to avoid hard clipping flat peaks at 0/100
SOFT_LIMIT_HALF_RANGE = 50.0
SOFT_LIMIT_SHARPNESS = 0.9

# Minimum ROI size in pixels
MIN_ROI_SIZE = 16


class HybridFlowTracker(BaseTracker):
    mutates_input_frame = False

    """
    Live tracker combining sparse YOLO detection with per-frame DIS optical flow.

    YOLO runs every N frames to locate the ROI (penis + nearest contact).
    DIS optical flow runs every frame in the ROI to track vertical motion.
    The cumulative vertical displacement is converted to a funscript position.
    """

    def __init__(self):
        super().__init__()
        self.app = None
        self.tracking_active = False
        self._initialized = False

        # YOLO
        self._yolo_model = None
        self._yolo_interval = DEFAULT_YOLO_INTERVAL
        self._yolo_confidence = 0.25

        # DIS optical flow
        self._dis = None
        self._prev_gray_roi = None
        self._prev_roi_for_flow = None

        # ROI state
        self._current_roi = None  # (x1, y1, x2, y2) in pixel coords
        self._last_penis_box = None
        self._last_contact_box = None
        self._frames_since_yolo = 0
        self._yolo_miss_count = 0

        # Signal processing
        self._flow_history = deque(maxlen=FLOW_HISTORY_SIZE)
        self._flow_raw_history = deque(maxlen=FLOW_HISTORY_SIZE)
        self._cumulative_pos = 0.0
        self._drift_buffer = deque(maxlen=FLOW_HISTORY_SIZE)
        self._smoothed_primary = 50.0
        self._smoothed_secondary = 50.0
        self._frame_count = 0
        self._roi_refreshed_this_frame = False

        # Normalization
        self._pos_min = 0.0
        self._pos_max = 0.0
        self._pos_range_ema = 1.0
        self._pos_center_ema = 0.0
        self._pos_baseline = 0.0

        # Horizontal flow for secondary axis
        self._dx_history = deque(maxlen=FLOW_HISTORY_SIZE)
        self._dx_raw_history = deque(maxlen=FLOW_HISTORY_SIZE)
        self._cumulative_dx = 0.0
        self._drift_buffer_dx = deque(maxlen=FLOW_HISTORY_SIZE)
        self._dx_min = 0.0
        self._dx_max = 0.0
        self._dx_range_ema = 1.0
        self._dx_center_ema = 0.0
        self._dx_baseline = 0.0

        # FPS tracking
        self.current_fps = 30.0
        self._fps_last_time = 0.0

        # Async YOLO worker state.
        self._yolo_thread = None
        self._yolo_shutdown = None
        self._yolo_wake = None
        self._yolo_pending_lock = None
        self._yolo_pending_frame = None
        self._yolo_pending_scale = (1.0, 1.0)
        self._yolo_result_lock = None
        self._yolo_result = None
        self._yolo_inflight = False
        self._yolo_calls = 0
        self._yolo_last_ms = 0.0

        # Funscript reference
        self.funscript = None

        # Last known positions for status
        self._last_primary_pos = 50
        self._last_secondary_pos = 50

    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="LIVE_HYBRID_FLOW",
            display_name="2D POV and VR Hybrid Flow",
            description="YOLO ROI detection with DIS optical flow for real-time tracking",
            category="live",
            version="1.0.0",
            author="FunGen",
            tags=["live", "hybrid", "yolo", "optical-flow", "dis", "vr", "2d", "pov"],
            requires_roi=False,
            supports_dual_axis=True,
            primary_axis="stroke",
            secondary_axis="roll",
        )

    def initialize(self, app_instance, **kwargs) -> bool:
        try:
            self.app = app_instance

            # Load YOLO model
            yolo_model_path = getattr(self.app, 'yolo_det_model_path', None)
            if not yolo_model_path:
                self.logger.error("YOLO detection model path not configured")
                return False

            if not os.path.exists(yolo_model_path):
                self.logger.error(f"YOLO model not found: {yolo_model_path}")
                return False

            from tracker.tracker_modules.helpers.yolo_detection_helper import (
                load_model as _yolo_load_model, run_detection as _yolo_run_detection,
            )
            self._yolo_run_detection = _yolo_run_detection
            self._yolo_model = _yolo_load_model(
                yolo_model_path,
                warmup_device=config_constants.DEVICE,
                warmup_imgsz=getattr(self.app, 'yolo_input_size', 640),
            )
            self.logger.debug(f"YOLO model loaded: {yolo_model_path}")

            # Initialize DIS optical flow
            self._dis = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)

            # Initialize funscript connection (same pattern as oscillation tracker)
            if hasattr(self, 'funscript') and self.funscript:
                pass  # Already have funscript from bridge
            elif hasattr(self.app, 'funscript') and self.app.funscript:
                self.funscript = self.app.funscript
            else:
                from funscript.multi_axis_funscript import MultiAxisFunscript
                self.funscript = MultiAxisFunscript(logger=self.logger)

            # Load settings
            self._load_settings()

            # Spatial-weight cache for _magnitude_weighted_flow. Key is the
            # (fh, fw) ROI shape; value is the pre-computed Gaussian weight
            # plane so np.exp / np.outer run once per ROI resize, not once
            # per frame.
            self._mwf_spatial_key: Optional[Tuple[int, int]] = None
            self._mwf_spatial: Optional[np.ndarray] = None

            self._initialized = True
            self.logger.debug("Hybrid Flow Tracker initialized")
            return True

        except Exception as e:
            self.logger.error(f"Initialization failed: {e}", exc_info=True)
            return False

    def _load_settings(self):
        if self.app and hasattr(self.app, 'app_settings'):
            settings = self.app.app_settings
            self._yolo_interval = settings.get("hybrid_flow_yolo_interval", DEFAULT_YOLO_INTERVAL)
            self._yolo_confidence = settings.get("hybrid_flow_confidence", 0.25)

    def start_tracking(self) -> bool:
        self.tracking_active = True
        self._reset_state()
        self._start_yolo_worker()
        self.logger.info("Hybrid Flow tracking started")
        return True

    def stop_tracking(self) -> bool:
        self.tracking_active = False
        self._stop_yolo_worker()
        self.logger.info("Hybrid Flow tracking stopped")
        return True

    def _start_yolo_worker(self) -> None:
        import threading as _threading
        if self._yolo_thread is not None and self._yolo_thread.is_alive():
            return
        self._yolo_shutdown = _threading.Event()
        self._yolo_wake = _threading.Event()
        self._yolo_pending_lock = _threading.Lock()
        self._yolo_result_lock = _threading.Lock()
        self._yolo_pending_frame = None
        self._yolo_result = None
        self._yolo_inflight = False
        self._yolo_thread = _threading.Thread(
            target=self._yolo_worker, name="HybridFlowYOLO", daemon=True)
        self._yolo_thread.start()

    def _stop_yolo_worker(self) -> None:
        if self._yolo_shutdown is not None:
            self._yolo_shutdown.set()
        if self._yolo_wake is not None:
            self._yolo_wake.set()
        t = self._yolo_thread
        if t is not None and t.is_alive():
            t.join(timeout=0.5)
        self._yolo_thread = None

    def _yolo_worker(self) -> None:
        while not self._yolo_shutdown.is_set():
            if not self._yolo_wake.wait(timeout=0.2):
                continue
            self._yolo_wake.clear()
            if self._yolo_shutdown.is_set():
                return
            with self._yolo_pending_lock:
                frame = self._yolo_pending_frame
                scale = self._yolo_pending_scale
                self._yolo_pending_frame = None
            if frame is None:
                continue
            self._yolo_inflight = True
            try:
                h, w = frame.shape[:2]
                # Original (source) frame dims for downstream consumers.
                src_h = int(h * scale[1])
                src_w = int(w * scale[0])
                t0 = time.perf_counter()
                result = self._run_yolo(frame, src_h, src_w, scale)
                self._yolo_last_ms = (time.perf_counter() - t0) * 1000.0
                self._yolo_calls += 1
                with self._yolo_result_lock:
                    self._yolo_result = result
            except Exception as e:
                self.logger.warning(f"YOLO worker error: {e}")
            finally:
                self._yolo_inflight = False

    def reset(self, reason: Optional[str] = None, **kwargs):
        self.tracking_active = False
        self._reset_state()

    def _reset_state(self):
        """Reset all per-session state."""
        self._prev_gray_roi = None
        self._prev_roi_for_flow = None
        self._current_roi = None
        self._last_penis_box = None
        self._last_contact_box = None
        self._frames_since_yolo = self._yolo_interval  # Force YOLO on first frame
        self._yolo_miss_count = 0
        self._flow_history.clear()
        self._flow_raw_history.clear()
        self._cumulative_pos = 0.0
        self._drift_buffer.clear()
        self._smoothed_primary = 50.0
        self._smoothed_secondary = 50.0
        self._frame_count = 0
        self._roi_refreshed_this_frame = False
        self._pos_min = 0.0
        self._pos_max = 0.0
        self._pos_range_ema = 1.0
        self._pos_center_ema = 0.0
        self._pos_baseline = 0.0
        self._dx_history.clear()
        self._dx_raw_history.clear()
        self._cumulative_dx = 0.0
        self._drift_buffer_dx.clear()
        self._dx_min = 0.0
        self._dx_max = 0.0
        self._dx_range_ema = 1.0
        self._dx_center_ema = 0.0
        self._dx_baseline = 0.0
        self._last_primary_pos = 50
        self._last_secondary_pos = 50

    # -------------------------------------------------------------------------
    # Frame processing
    # -------------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_time_ms: int,
                     frame_index: Optional[int] = None) -> TrackerResult:
        if frame is None or frame.size == 0:
            return TrackerResult(frame, None)

        self.live_overlay = {}

        if not self.tracking_active:
            return TrackerResult(frame, None, {})

        self._update_fps()
        self._frame_count += 1
        self._frames_since_yolo += 1

        h, w = frame.shape[:2]

        # Cold-start: run YOLO synchronously until we either get a detection
        # or hit a small frame cap. Without this, async YOLO takes ~50-100 ms
        # wall clock; at MAX_SPEED that window covers many video frames where
        # the tracker can't emit anything (no ROI -> no action). One blocking
        # call per startup frame trades a tiny pause for output that begins
        # as soon as the video has content YOLO can detect. The 30-frame cap
        # (0.5s at 60fps) bounds the worst-case startup wait for content with
        # a long intro / fade-in / black-frame lead.
        if self._frame_count <= 30 and self._current_roi is None:
            yolo_imgsz = int(getattr(self.app, 'yolo_input_size', 640))
            if w > yolo_imgsz:
                tw = yolo_imgsz
                th = max(1, int(h * yolo_imgsz / w))
                sync_frame = cv2.resize(frame, (tw, th))
                sync_scale = (w / tw, h / th)
            else:
                sync_frame = frame
                sync_scale = (1.0, 1.0)
            try:
                roi, penis_box, contact_box = self._run_yolo(sync_frame, h, w, sync_scale)
            except Exception as _e:
                roi, penis_box, contact_box = None, None, None
                self.logger.debug(f"sync YOLO on frame {self._frame_count} failed: {_e}")
            if roi is not None:
                self._current_roi = roi
                self._last_penis_box = penis_box
                self._last_contact_box = contact_box
                self._roi_refreshed_this_frame = True
                # We just ran YOLO; don't immediately dispatch another one.
                self._frames_since_yolo = 0

        # --- Step 1: YOLO dispatch + consume latest result (async) ---
        self._roi_refreshed_this_frame = False

        if self._frames_since_yolo >= self._yolo_interval:
            if self._yolo_thread is not None and self._yolo_thread.is_alive():
                # Resize once instead of frame.copy(): on 8K VR the copy is
                # ~30 ms bandwidth-bound; cv2.resize to the YOLO input size
                # is ~3-5 ms and ultralytics would resize internally anyway.
                # Bboxes come back in resized coords; we apply the inverse
                # scale in _run_yolo so callers still see source coords.
                yolo_imgsz = int(getattr(self.app, 'yolo_input_size', 640))
                src_h, src_w = frame.shape[:2]
                if src_w > yolo_imgsz:
                    tw = yolo_imgsz
                    th = max(1, int(src_h * yolo_imgsz / src_w))
                    submit_frame = cv2.resize(frame, (tw, th))
                    scale = (src_w / tw, src_h / th)
                else:
                    submit_frame = frame.copy()
                    scale = (1.0, 1.0)
                with self._yolo_pending_lock:
                    self._yolo_pending_frame = submit_frame
                    self._yolo_pending_scale = scale
                self._yolo_wake.set()
                self._frames_since_yolo = 0

        result = None
        if self._yolo_result_lock is not None:
            with self._yolo_result_lock:
                if self._yolo_result is not None:
                    result = self._yolo_result
                    self._yolo_result = None
        if result is not None:
            roi, penis_box, contact_box = result
            if roi is not None:
                old_roi = self._current_roi
                self._current_roi = roi
                self._last_penis_box = penis_box
                self._last_contact_box = contact_box
                self._yolo_miss_count = 0
                if old_roi != roi:
                    self._roi_refreshed_this_frame = True
            else:
                self._yolo_miss_count += 1
                if self._yolo_miss_count > 10 and self._current_roi is None:
                    margin_x = int(w * 0.2)
                    margin_y = int(h * 0.2)
                    self._current_roi = (margin_x, margin_y, w - margin_x, h - margin_y)
                    self._roi_refreshed_this_frame = True

        # --- Step 2: Optical flow in ROI ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dy, dx = 0.0, 0.0

        if self._current_roi is not None:
            rx1, ry1, rx2, ry2 = self._current_roi
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)
            current_roi = (rx1, ry1, rx2, ry2)

            if self._roi_shifted_for_flow(current_roi):
                self._prev_gray_roi = None

            roi_w = rx2 - rx1
            roi_h = ry2 - ry1

            if roi_w > MIN_ROI_SIZE and roi_h > MIN_ROI_SIZE:
                # DIS needs contiguous; this slice is also reused as prev next frame.
                curr_contig = np.ascontiguousarray(gray[ry1:ry2, rx1:rx2])

                if self._roi_refreshed_this_frame:
                    self._prev_gray_roi = None
                if self._prev_gray_roi is not None and self._prev_gray_roi.shape == curr_contig.shape:
                    try:
                        flow = self._dis.calc(self._prev_gray_roi, curr_contig, None)
                        if flow is not None:
                            dy, dx = self._magnitude_weighted_flow(flow)
                    except cv2.error:
                        pass

                self._prev_gray_roi = curr_contig
                self._prev_roi_for_flow = current_roi
            else:
                self._prev_gray_roi = None
                self._prev_roi_for_flow = None
        else:
            self._prev_gray_roi = None
            self._prev_roi_for_flow = None

        # --- Step 3: Convert flow to positions ---
        primary_pos = self._flow_to_position(dy, 'primary')
        secondary_pos = self._flow_to_position(dx, 'secondary')

        final_primary = int(round(np.clip(primary_pos, 0, 100)))
        final_secondary = int(round(np.clip(secondary_pos, 0, 100)))
        self._last_primary_pos = final_primary
        self._last_secondary_pos = final_secondary

        # --- Step 4: Axis routing (same pattern as oscillation tracker) ---
        axis_mode = getattr(self.app, 'tracking_axis_mode', 'both') if self.app else 'both'
        single_target = getattr(self.app, 'single_axis_output_target', 'primary') if self.app else 'primary'

        primary_to_write, secondary_to_write = None, None
        if axis_mode == 'both':
            primary_to_write = final_primary
            secondary_to_write = final_secondary
        elif axis_mode == 'vertical':
            if single_target == 'primary':
                primary_to_write = final_primary
            else:
                secondary_to_write = final_primary
        elif axis_mode == 'horizontal':
            if single_target == 'primary':
                primary_to_write = final_secondary
            else:
                secondary_to_write = final_secondary

        # No ROI yet means optical flow never ran -- dy/dx are 0 and the
        # "position" is just the center default. Skip the write so the
        # funscript doesn't fill with bogus pos=50 actions while async
        # YOLO is still warming up (visible at MAX_SPEED).
        emit_action = self._current_roi is not None
        if self.funscript and emit_action:
            self.funscript.add_action(
                timestamp_ms=frame_time_ms,
                primary_pos=primary_to_write,
                secondary_pos=secondary_to_write)

        action_log = []
        if emit_action:
            action_log.append({'at': frame_time_ms, 'pos': primary_to_write, 'secondary_pos': secondary_to_write})

        display_frame = frame
        self._draw_overlay()

        debug_info = {
            'position': final_primary,
            'secondary_position': final_secondary,
            'dy': dy,
            'dx': dx,
            'current_roi': self._current_roi,
            'yolo_miss_count': self._yolo_miss_count,
            'fps': self.current_fps,
        }

        return TrackerResult(
            processed_frame=display_frame,
            action_log=action_log,
            debug_info=debug_info,
            status_message=f"P:{final_primary} S:{final_secondary} | {self.current_fps:.0f}fps"
        )

    # -------------------------------------------------------------------------
    # YOLO detection
    # -------------------------------------------------------------------------

    def _run_yolo(self, frame: np.ndarray, h: int, w: int,
                  scale: Tuple[float, float] = (1.0, 1.0)
                  ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple], Optional[Tuple]]:
        """Run YOLO, return (roi_box, penis_box, contact_box) in source coords."""
        try:
            det_objs = self._yolo_run_detection(
                self._yolo_model, frame,
                conf=self._yolo_confidence,
                imgsz=getattr(self.app, 'yolo_input_size', 640),
                device=config_constants.DEVICE)
        except Exception:
            return None, None, None

        if not det_objs:
            return None, None, None

        sx, sy = scale
        scale_box = (lambda b: (b[0] * sx, b[1] * sy, b[2] * sx, b[3] * sy)) if (sx != 1.0 or sy != 1.0) else (lambda b: b)

        penis_box = None
        best_conf = 0.0
        contact_boxes = []

        for d in det_objs:
            if d.class_name == 'penis' and d.confidence > best_conf:
                penis_box = scale_box(d.bbox)
                best_conf = d.confidence
            elif d.class_name in CONTACT_CLASSES:
                contact_boxes.append(scale_box(d.bbox))

        if penis_box is None:
            return None, None, None

        # Find nearest contact
        px1, py1, px2, py2 = penis_box
        pcx, pcy = (px1 + px2) / 2, (py1 + py2) / 2
        best_contact = None
        best_dist = float('inf')
        for cb in contact_boxes:
            cx, cy = (cb[0] + cb[2]) / 2, (cb[1] + cb[3]) / 2
            dist = (pcx - cx) ** 2 + (pcy - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_contact = cb

        # Compute padded ROI
        roi_x1, roi_y1, roi_x2, roi_y2 = px1, py1, px2, py2
        if best_contact:
            roi_x1 = min(roi_x1, best_contact[0])
            roi_y1 = min(roi_y1, best_contact[1])
            roi_x2 = max(roi_x2, best_contact[2])
            roi_y2 = max(roi_y2, best_contact[3])

        roi_w = roi_x2 - roi_x1
        roi_h = roi_y2 - roi_y1
        pad_x = roi_w * ROI_PADDING
        pad_y = roi_h * ROI_PADDING

        roi = (
            max(0, int(roi_x1 - pad_x)),
            max(0, int(roi_y1 - pad_y)),
            min(w, int(roi_x2 + pad_x)),
            min(h, int(roi_y2 + pad_y)),
        )

        return roi, penis_box, best_contact

    # -------------------------------------------------------------------------
    # Flow processing
    # -------------------------------------------------------------------------

    def _magnitude_weighted_flow(self, flow: np.ndarray) -> Tuple[float, float]:
        """Magnitude-weighted dy/dx. Gaussian weights cached per ROI shape."""
        fh, fw = flow.shape[:2]
        key = (fh, fw)
        if getattr(self, "_mwf_spatial_key", None) != key:
            cx, sx = fw * 0.5, fw * 0.25
            cy, sy = fh * 0.5, fh * 0.25
            wx = np.exp(-((np.arange(fw, dtype=np.float32) - cx) ** 2) / (2 * sx * sx))
            wy = np.exp(-((np.arange(fh, dtype=np.float32) - cy) ** 2) / (2 * sy * sy))
            self._mwf_spatial = np.outer(wy, wx)
            self._mwf_spatial_key = key
        spatial = self._mwf_spatial

        fx = flow[..., 0]
        fy = flow[..., 1]
        mag2 = fx * fx + fy * fy
        combined = mag2 * spatial
        total = combined.sum()

        if total > 0:
            dy = (fy * combined).sum() / total
            dx = (fx * combined).sum() / total
        else:
            dy = float(np.median(fy))
            dx = float(np.median(fx))

        return float(dy), float(dx)

    def _flow_to_position(self, flow_val: float, axis: str) -> float:
        """Convert a flow component to 0-100 position with drift removal and normalization."""
        if abs(flow_val) < FLOW_DEADZONE:
            flow_val = 0.0

        if axis == 'primary':
            raw_history = self._flow_raw_history
            history = self._flow_history
            sign = -1.0  # down motion = insertion
        else:
            raw_history = self._dx_raw_history
            history = self._dx_history
            sign = 1.0

        # Velocity-like flow with median denoise + slow baseline to avoid drift.
        signed_flow = sign * flow_val
        raw_history.append(signed_flow)
        n_raw = len(raw_history)
        k_raw = 7 if n_raw >= 7 else n_raw
        recent_raw_list = list(islice(raw_history, n_raw - k_raw, n_raw))
        recent_raw_list.sort()
        flow_med = float(recent_raw_list[k_raw // 2])

        if axis == 'primary':
            self._pos_baseline = (1.0 - BASELINE_ALPHA) * self._pos_baseline + BASELINE_ALPHA * flow_med
            signal = flow_med - self._pos_baseline
        else:
            self._dx_baseline = (1.0 - BASELINE_ALPHA) * self._dx_baseline + BASELINE_ALPHA * flow_med
            signal = flow_med - self._dx_baseline

        history.append(signal)

        # Adaptive normalization
        if len(history) > 20:
            n_hist = len(history)
            k_hist = NORMALIZATION_WINDOW if n_hist >= NORMALIZATION_WINDOW else n_hist
            recent = np.fromiter(
                islice(history, n_hist - k_hist, n_hist),
                dtype=np.float32, count=k_hist)
            p10, p50, p90 = np.percentile(recent, (10, 50, 90))
            current_range = max(p90 - p10, 0.1)

            alpha = 0.015
            if axis == 'primary':
                self._pos_range_ema = (1 - alpha) * self._pos_range_ema + alpha * current_range
                self._pos_min = (1 - alpha) * self._pos_min + alpha * p10
                self._pos_max = (1 - alpha) * self._pos_max + alpha * p90
                self._pos_center_ema = (1 - alpha) * self._pos_center_ema + alpha * p50
                center = self._pos_center_ema
                half_range = max(self._pos_range_ema * 0.5, 0.05)
            else:
                self._dx_range_ema = (1 - alpha) * self._dx_range_ema + alpha * current_range
                self._dx_min = (1 - alpha) * self._dx_min + alpha * p10
                self._dx_max = (1 - alpha) * self._dx_max + alpha * p90
                self._dx_center_ema = (1 - alpha) * self._dx_center_ema + alpha * p50
                center = self._dx_center_ema
                half_range = max(self._dx_range_ema * 0.5, 0.05)

            normalized = 50.0 + ((signal - center) / half_range) * TARGET_HALF_RANGE
        else:
            normalized = 50.0 + signal * 200

        # Soft-limit extremes so peaks remain rounded instead of hard-clipped flat.
        normalized = 50.0 + SOFT_LIMIT_HALF_RANGE * np.tanh(
            ((normalized - 50.0) / SOFT_LIMIT_HALF_RANGE) * SOFT_LIMIT_SHARPNESS
        )

        # EMA smoothing
        if axis == 'primary':
            self._smoothed_primary = (1 - POSITION_SMOOTHING_ALPHA) * self._smoothed_primary + POSITION_SMOOTHING_ALPHA * normalized
            return self._smoothed_primary
        else:
            self._smoothed_secondary = (1 - POSITION_SMOOTHING_ALPHA) * self._smoothed_secondary + POSITION_SMOOTHING_ALPHA * normalized
            return self._smoothed_secondary

    def _roi_shifted_for_flow(self, current_roi: Tuple[int, int, int, int]) -> bool:
        """Return True when ROI moved enough to invalidate prev ROI patch for optical flow."""
        prev = self._prev_roi_for_flow
        if prev is None:
            return False

        return (
            abs(current_roi[0] - prev[0]) > ROI_SHIFT_RESET_THRESHOLD or
            abs(current_roi[1] - prev[1]) > ROI_SHIFT_RESET_THRESHOLD or
            abs(current_roi[2] - prev[2]) > ROI_SHIFT_RESET_THRESHOLD or
            abs(current_roi[3] - prev[3]) > ROI_SHIFT_RESET_THRESHOLD
        )

    # -------------------------------------------------------------------------
    # Helpers
    # -------------------------------------------------------------------------

    def _update_fps(self):
        """Update FPS calculation."""
        now = time.time()
        if self._fps_last_time > 0:
            dt = now - self._fps_last_time
            if dt > 0.001:
                self.current_fps = 1.0 / dt
        self._fps_last_time = now

    def _draw_overlay(self):
        """Populate live_overlay with ROI box and detection boxes."""
        if self._current_roi is not None:
            rx1, ry1, rx2, ry2 = self._current_roi
            self.live_overlay.setdefault('rects', []).append({
                'x1': rx1, 'y1': ry1, 'x2': rx2, 'y2': ry2,
                'color': (0, 1.0, 1.0, 1.0), 'thickness': 1.0, 'label': None
            })

        if self._last_penis_box is not None:
            px1, py1, px2, py2 = [int(v) for v in self._last_penis_box]
            self.live_overlay.setdefault('rects', []).append({
                'x1': px1, 'y1': py1, 'x2': px2, 'y2': py2,
                'color': (0, 1.0, 0, 1.0), 'thickness': 2.0, 'label': None
            })

        if self._last_contact_box is not None:
            cx1, cy1, cx2, cy2 = [int(v) for v in self._last_contact_box]
            self.live_overlay.setdefault('rects', []).append({
                'x1': cx1, 'y1': cy1, 'x2': cx2, 'y2': cy2,
                'color': (0, 1.0, 1.0, 1.0), 'thickness': 2.0, 'label': None
            })

    def get_status_info(self) -> Dict[str, Any]:
        """Get detailed status information."""
        return {
            'tracker': self.metadata.display_name,
            'active': self.tracking_active,
            'initialized': self._initialized,
            'last_position': self._last_primary_pos,
            'last_secondary': self._last_secondary_pos,
            'yolo_miss_count': self._yolo_miss_count,
            'fps': self.current_fps,
        }

    # -------------------------------------------------------------------------
    # Cleanup
    # -------------------------------------------------------------------------

    def cleanup(self):
        try:
            self._yolo_model = None
            self._dis = None
            self._prev_gray_roi = None
            if hasattr(self, '_flow_history'):
                self._flow_history.clear()
            if hasattr(self, '_drift_buffer'):
                self._drift_buffer.clear()
            if hasattr(self, '_dx_history'):
                self._dx_history.clear()
            if hasattr(self, '_drift_buffer_dx'):
                self._drift_buffer_dx.clear()
        except Exception:
            pass

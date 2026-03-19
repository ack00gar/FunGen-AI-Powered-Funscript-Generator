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

# Minimum ROI size in pixels
MIN_ROI_SIZE = 16


class HybridFlowTracker(BaseTracker):
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

        # ROI state
        self._current_roi = None  # (x1, y1, x2, y2) in pixel coords
        self._last_penis_box = None
        self._last_contact_box = None
        self._frames_since_yolo = 0
        self._yolo_miss_count = 0

        # Signal processing
        self._flow_history = deque(maxlen=FLOW_HISTORY_SIZE)
        self._cumulative_pos = 0.0
        self._drift_buffer = deque(maxlen=FLOW_HISTORY_SIZE)
        self._smoothed_primary = 50.0
        self._smoothed_secondary = 50.0
        self._frame_count = 0

        # Normalization
        self._pos_min = 0.0
        self._pos_max = 0.0
        self._pos_range_ema = 1.0

        # Horizontal flow for secondary axis
        self._dx_history = deque(maxlen=FLOW_HISTORY_SIZE)
        self._cumulative_dx = 0.0
        self._drift_buffer_dx = deque(maxlen=FLOW_HISTORY_SIZE)
        self._dx_min = 0.0
        self._dx_max = 0.0
        self._dx_range_ema = 1.0

        # FPS tracking
        self.current_fps = 30.0
        self._fps_last_time = 0.0

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

            from ultralytics import YOLO
            self._yolo_model = YOLO(yolo_model_path, task='detect')
            self.logger.info(f"YOLO model loaded: {yolo_model_path}")

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

            self._initialized = True
            self.logger.info("Hybrid Flow Tracker initialized")
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
        self.logger.info("Hybrid Flow tracking started")
        return True

    def stop_tracking(self) -> bool:
        self.tracking_active = False
        self.logger.info("Hybrid Flow tracking stopped")
        return True

    def reset(self, reason: Optional[str] = None, **kwargs):
        self.tracking_active = False
        self._reset_state()

    def _reset_state(self):
        """Reset all per-session state."""
        self._prev_gray_roi = None
        self._current_roi = None
        self._last_penis_box = None
        self._last_contact_box = None
        self._frames_since_yolo = self._yolo_interval  # Force YOLO on first frame
        self._yolo_miss_count = 0
        self._flow_history.clear()
        self._cumulative_pos = 0.0
        self._drift_buffer.clear()
        self._smoothed_primary = 50.0
        self._smoothed_secondary = 50.0
        self._frame_count = 0
        self._pos_min = 0.0
        self._pos_max = 0.0
        self._pos_range_ema = 1.0
        self._dx_history.clear()
        self._cumulative_dx = 0.0
        self._drift_buffer_dx.clear()
        self._dx_min = 0.0
        self._dx_max = 0.0
        self._dx_range_ema = 1.0
        self._last_primary_pos = 50
        self._last_secondary_pos = 50

    # -------------------------------------------------------------------------
    # Frame processing
    # -------------------------------------------------------------------------

    def process_frame(self, frame: np.ndarray, frame_time_ms: int,
                     frame_index: Optional[int] = None) -> TrackerResult:
        if frame is None or frame.size == 0:
            return TrackerResult(frame, None)

        if not self.tracking_active:
            return TrackerResult(frame, None, {})

        self._update_fps()
        self._frame_count += 1
        self._frames_since_yolo += 1

        h, w = frame.shape[:2]

        # --- Step 1: YOLO detection (every N frames) ---
        if self._frames_since_yolo >= self._yolo_interval:
            roi, penis_box, contact_box = self._run_yolo(frame, h, w)
            self._frames_since_yolo = 0

            if roi is not None:
                self._current_roi = roi
                self._last_penis_box = penis_box
                self._last_contact_box = contact_box
                self._yolo_miss_count = 0
            else:
                self._yolo_miss_count += 1
                if self._yolo_miss_count > 10 and self._current_roi is None:
                    margin_x = int(w * 0.2)
                    margin_y = int(h * 0.2)
                    self._current_roi = (margin_x, margin_y, w - margin_x, h - margin_y)

        # --- Step 2: Optical flow in ROI ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        dy, dx = 0.0, 0.0

        if self._current_roi is not None:
            rx1, ry1, rx2, ry2 = self._current_roi
            rx1, ry1 = max(0, rx1), max(0, ry1)
            rx2, ry2 = min(w, rx2), min(h, ry2)

            roi_w = rx2 - rx1
            roi_h = ry2 - ry1

            if roi_w > MIN_ROI_SIZE and roi_h > MIN_ROI_SIZE:
                curr_patch = gray[ry1:ry2, rx1:rx2]

                if self._prev_gray_roi is not None and self._prev_gray_roi.shape == curr_patch.shape:
                    try:
                        flow = self._dis.calc(
                            np.ascontiguousarray(self._prev_gray_roi),
                            np.ascontiguousarray(curr_patch),
                            None
                        )
                        if flow is not None:
                            dy, dx = self._magnitude_weighted_flow(flow)
                    except cv2.error:
                        pass

                self._prev_gray_roi = curr_patch.copy()
            else:
                self._prev_gray_roi = None

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

        # Write to funscript
        if self.funscript:
            self.funscript.add_action(
                timestamp_ms=frame_time_ms,
                primary_pos=primary_to_write,
                secondary_pos=secondary_to_write)

        action_log = [{'at': frame_time_ms, 'pos': primary_to_write, 'secondary_pos': secondary_to_write}]

        # --- Debug overlay ---
        display_frame = frame.copy()
        self._draw_overlay(display_frame)

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

    def _run_yolo(self, frame: np.ndarray, h: int, w: int
                  ) -> Tuple[Optional[Tuple[int, int, int, int]], Optional[Tuple], Optional[Tuple]]:
        """Run YOLO, return (roi_box, penis_box, contact_box) or (None, None, None)."""
        try:
            results = self._yolo_model(frame, device=config_constants.DEVICE, verbose=False,
                                       conf=self._yolo_confidence,
                                       imgsz=getattr(self.app, 'yolo_input_size', 640))
        except Exception:
            return None, None, None

        if not results or len(results) == 0:
            return None, None, None

        penis_box = None
        best_conf = 0.0
        contact_boxes = []

        for box in results[0].boxes:
            cls_id = int(box.cls[0])
            cls_name = self._yolo_model.names.get(cls_id, '')
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()

            if cls_name == 'penis' and conf > best_conf:
                penis_box = (x1, y1, x2, y2)
                best_conf = conf
            elif cls_name in CONTACT_CLASSES:
                contact_boxes.append((x1, y1, x2, y2))

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
        """Extract magnitude-weighted dy/dx from flow field."""
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

    def _flow_to_position(self, flow_val: float, axis: str) -> float:
        """Convert a flow component to 0-100 position with drift removal and normalization."""
        if axis == 'primary':
            self._cumulative_pos -= flow_val  # negate: down motion = insertion
            self._drift_buffer.append(self._cumulative_pos)
            cum = self._cumulative_pos
            drift_buf = self._drift_buffer
            history = self._flow_history
        else:
            self._cumulative_dx += flow_val
            self._drift_buffer_dx.append(self._cumulative_dx)
            cum = self._cumulative_dx
            drift_buf = self._drift_buffer_dx
            history = self._dx_history

        # Drift removal
        if len(drift_buf) > 10:
            drift = np.mean(drift_buf)
            detrended = cum - drift
        else:
            detrended = cum

        history.append(detrended)

        # Adaptive normalization
        if len(history) > 20:
            recent = np.array(history)
            p5 = np.percentile(recent, 5)
            p95 = np.percentile(recent, 95)
            current_range = max(p95 - p5, 0.1)

            alpha = 0.02
            if axis == 'primary':
                self._pos_range_ema = (1 - alpha) * self._pos_range_ema + alpha * current_range
                self._pos_min = (1 - alpha) * self._pos_min + alpha * p5
                self._pos_max = (1 - alpha) * self._pos_max + alpha * p95
                center = (self._pos_min + self._pos_max) / 2
                effective_range = max(self._pos_range_ema, 0.1)
            else:
                self._dx_range_ema = (1 - alpha) * self._dx_range_ema + alpha * current_range
                self._dx_min = (1 - alpha) * self._dx_min + alpha * p5
                self._dx_max = (1 - alpha) * self._dx_max + alpha * p95
                center = (self._dx_min + self._dx_max) / 2
                effective_range = max(self._dx_range_ema, 0.1)

            normalized = (detrended - center) / effective_range * 80 + 50
        else:
            normalized = 50.0 + detrended * 10

        # EMA smoothing
        if axis == 'primary':
            self._smoothed_primary = (1 - POSITION_SMOOTHING_ALPHA) * self._smoothed_primary + POSITION_SMOOTHING_ALPHA * normalized
            return self._smoothed_primary
        else:
            self._smoothed_secondary = (1 - POSITION_SMOOTHING_ALPHA) * self._smoothed_secondary + POSITION_SMOOTHING_ALPHA * normalized
            return self._smoothed_secondary

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

    def _draw_overlay(self, frame: np.ndarray):
        """Draw ROI box and detection boxes on frame."""
        if self._current_roi is not None:
            rx1, ry1, rx2, ry2 = self._current_roi
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (255, 255, 0), 1)

        if self._last_penis_box is not None:
            px1, py1, px2, py2 = [int(v) for v in self._last_penis_box]
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)

        if self._last_contact_box is not None:
            cx1, cy1, cx2, cy2 = [int(v) for v in self._last_contact_box]
            cv2.rectangle(frame, (cx1, cy1), (cx2, cy2), (0, 255, 255), 2)

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

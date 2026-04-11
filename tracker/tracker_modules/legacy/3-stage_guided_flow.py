"""
Guided Flow Tracker - YOLO-guided dense optical flow for offline funscript generation.

Uses Stage 2 detection data (chapters, locked penis, interactor boxes) to create
targeted ROIs for dense optical flow analysis. Different chapter types get different
measurement strategies.

Key differences from Stage 3 OF/Mixed:
- All flow is measured in targeted ROIs, not full-frame oscillation
- YOLO boxes define WHERE to measure, flow measures WHAT
- Chapter-aware strategies (penetration, oral, manual, riding, etc.)
- Optical flow tracks ROI position during YOLO occlusions
- Handles male thrusting (VR POV belly motion), grinding, hand+mouth combos
- Minimal internal smoothing — delegates to plugin pipeline (ultimate autotune etc.)
- Per-chapter normalization to 0-100
"""

import logging
import os
import time
import cv2
import numpy as np
from collections import deque
from typing import Dict, Any, Optional, List, Tuple, Callable
from multiprocessing import Event

from common.frame_utils import frame_to_ms
from scipy.signal import savgol_filter

try:
    from ..core.base_offline_tracker import BaseOfflineTracker, OfflineProcessingResult, OfflineProcessingStage
    from ..core.base_tracker import TrackerMetadata, StageDefinition
except ImportError:
    from tracker.tracker_modules.core.base_offline_tracker import BaseOfflineTracker, OfflineProcessingResult, OfflineProcessingStage
    from tracker.tracker_modules.core.base_tracker import TrackerMetadata, StageDefinition

try:
    from detection.cd.data_structures import Segment, FrameObject
    DETECTION_AVAILABLE = True
except ImportError:
    DETECTION_AVAILABLE = False

from funscript.multi_axis_funscript import MultiAxisFunscript
from config import constants


# ---------------------------------------------------------------------------
# Chapter strategy definitions
# ---------------------------------------------------------------------------

class ChapterStrategy:
    PENETRATION = "penetration"   # CG/Miss, Doggy — flow in contact zone
    ORAL = "oral"                 # BJ — face/head movement + hand stroking
    MANUAL = "manual"             # HJ — hand movement on penis
    RIDING = "riding"             # Cowgirl body movement (when penis invisible)
    BREAST = "breast"             # Boobjob
    FOOT = "foot"                 # Footjob
    UNKNOWN = "unknown"           # Fallback — full-ROI flow

# Map Stage 2 position names to strategies
POSITION_TO_STRATEGY = {
    "Cowgirl / Missionary": ChapterStrategy.PENETRATION,
    "Rev. Cowgirl / Doggy": ChapterStrategy.PENETRATION,
    "Blowjob": ChapterStrategy.ORAL,
    "Handjob": ChapterStrategy.MANUAL,
    "Boobjob": ChapterStrategy.BREAST,
    "Footjob": ChapterStrategy.FOOT,
    "Not Relevant": ChapterStrategy.UNKNOWN,
}

# ROI padding in pixels (YOLO-space, typically 640x640)
ROI_PAD = 30
# How many frames to hold a lost ROI before decaying
OCCLUSION_PATIENCE_FRAMES = 15
# Minimum ROI size
MIN_ROI_SIZE = 40
# Flow dampening when tracking occluded ROI
OCCLUSION_FLOW_DAMPING = 0.85
# Male thrust detection: minimum flow ratio (belly_flow / contact_flow)
MALE_THRUST_RATIO = 1.5
# Grinding detection: minimum dx/dy ratio
GRIND_DX_DY_RATIO = 1.3
# High-pass cutoff for drift removal (Hz). Strokes above this pass through.
# 0.25 Hz preserves even very slow strokes while removing camera/ROI drift.
HIGHPASS_CUTOFF_HZ = 0.25
# Maximum duration (seconds) for a "Not Relevant" segment to be auto-merged with neighbors
NR_MERGE_MAX_DURATION_S = 3.0


# ---------------------------------------------------------------------------
# DIS optical flow factory (per-process singleton)
# ---------------------------------------------------------------------------

_DIS_FLOW_INSTANCE = None

def _get_dis_flow():
    global _DIS_FLOW_INSTANCE
    if _DIS_FLOW_INSTANCE is None:
        _DIS_FLOW_INSTANCE = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
    return _DIS_FLOW_INSTANCE


# ---------------------------------------------------------------------------
# ROI helpers
# ---------------------------------------------------------------------------

def _clamp_roi(roi, img_h, img_w):
    """Clamp ROI (x1,y1,x2,y2) to image bounds and enforce minimum size."""
    x1, y1, x2, y2 = roi
    x1 = max(0, int(x1))
    y1 = max(0, int(y1))
    x2 = min(img_w, int(x2))
    y2 = min(img_h, int(y2))
    # Enforce minimum size
    if x2 - x1 < MIN_ROI_SIZE:
        cx = (x1 + x2) // 2
        x1 = max(0, cx - MIN_ROI_SIZE // 2)
        x2 = min(img_w, x1 + MIN_ROI_SIZE)
    if y2 - y1 < MIN_ROI_SIZE:
        cy = (y1 + y2) // 2
        y1 = max(0, cy - MIN_ROI_SIZE // 2)
        y2 = min(img_h, y1 + MIN_ROI_SIZE)
    return (x1, y1, x2, y2)


def _pad_roi(roi, pad, img_h, img_w):
    """Expand ROI by pad pixels and clamp."""
    x1, y1, x2, y2 = roi
    return _clamp_roi((x1 - pad, y1 - pad, x2 + pad, y2 + pad), img_h, img_w)


def _union_rois(*rois):
    """Union of multiple (x1,y1,x2,y2) ROIs."""
    rois = [r for r in rois if r is not None]
    if not rois:
        return None
    x1 = min(r[0] for r in rois)
    y1 = min(r[1] for r in rois)
    x2 = max(r[2] for r in rois)
    y2 = max(r[3] for r in rois)
    return (x1, y1, x2, y2)


def _box_to_roi(box_rec):
    """Convert a BoxRecord to (x1,y1,x2,y2) ROI tuple."""
    if box_rec is None:
        return None
    return tuple(box_rec.bbox)


def _roi_area(roi):
    if roi is None:
        return 0
    return max(0, roi[2] - roi[0]) * max(0, roi[3] - roi[1])


# ---------------------------------------------------------------------------
# Per-chapter signal processor
# ---------------------------------------------------------------------------

class ChapterProcessor:
    """Processes a single chapter segment with guided optical flow."""

    def __init__(self, segment, strategy, fps, is_vr, logger):
        self.segment = segment
        self.strategy = strategy
        self.fps = fps
        self.is_vr = is_vr
        self.logger = logger

        # DIS flow engine
        self.dis = _get_dis_flow()
        self.prev_gray = None

        # ROI tracking state
        self.current_roi = None          # Active ROI (x1,y1,x2,y2)
        self.roi_source = "none"         # "yolo" or "flow" or "none"
        self.occlusion_counter = 0       # Frames since last YOLO ROI
        self.belly_roi = None            # For male thrust detection

        # Raw signal buffers
        self.positions = []              # (frame_id, raw_position_0_100)
        self.secondary_positions = []    # (frame_id, secondary_0_100) for roll/grinding

        # Flow integration — drift removed offline in get_actions()
        self.flow_history_dy = deque(maxlen=5)  # Short median window for noise rejection
        self.flow_history_dx = deque(maxlen=5)
        self.integrated_dy = 0.0         # Raw cumulative vertical displacement
        self.integrated_dx = 0.0         # Raw cumulative horizontal displacement

    # ------------------------------------------------------------------
    # ROI extraction per strategy
    # ------------------------------------------------------------------

    def _extract_roi_penetration(self, frame_obj):
        """ROI for penetration chapters: contact zone between penis and interactor."""
        lp = frame_obj.locked_penis_state
        if not lp.active or not lp.box:
            return None, None

        penis_roi = lp.box  # Conceptual full penis box

        # Find the primary interactor (pussy/butt) in contact
        interactor_roi = None
        for contact in frame_obj.detected_contact_boxes:
            box = contact.get('box_rec') if isinstance(contact, dict) else contact
            if box is None:
                continue
            cname = box.class_name if hasattr(box, 'class_name') else ''
            if cname in ('pussy', 'butt', 'anus'):
                interactor_roi = _box_to_roi(box)
                break

        if interactor_roi is None:
            # Fallback: look in raw boxes
            for box in frame_obj.boxes:
                if box.class_name in ('pussy', 'butt', 'anus') and not box.is_excluded:
                    interactor_roi = _box_to_roi(box)
                    break

        # Contact zone = intersection of penis and interactor
        if interactor_roi:
            contact_roi = (
                max(penis_roi[0], interactor_roi[0]),
                max(penis_roi[1], interactor_roi[1]),
                min(penis_roi[2], interactor_roi[2]),
                min(penis_roi[3], interactor_roi[3]),
            )
            if _roi_area(contact_roi) > 0:
                return contact_roi, penis_roi
            # No intersection — use union (they're near each other)
            return _union_rois(penis_roi, interactor_roi), penis_roi

        # No interactor found — use penis box alone
        return penis_roi, penis_roi

    def _extract_roi_oral(self, frame_obj):
        """ROI for BJ: face + hand areas near penis."""
        lp = frame_obj.locked_penis_state
        penis_roi = lp.box if (lp.active and lp.box) else None

        face_roi = None
        hand_roi = None
        for box in frame_obj.boxes:
            if box.is_excluded:
                continue
            if box.class_name == 'face' and face_roi is None:
                face_roi = _box_to_roi(box)
            elif box.class_name == 'hand' and hand_roi is None:
                hand_roi = _box_to_roi(box)

        # Primary: face near penis. Secondary: hand.
        primary = face_roi or hand_roi
        if primary and penis_roi:
            return _union_rois(primary, penis_roi), penis_roi
        return primary or penis_roi, penis_roi

    def _extract_roi_manual(self, frame_obj):
        """ROI for HJ: hand area near penis."""
        lp = frame_obj.locked_penis_state
        penis_roi = lp.box if (lp.active and lp.box) else None

        hand_roi = None
        for box in frame_obj.boxes:
            if box.class_name == 'hand' and not box.is_excluded:
                hand_roi = _box_to_roi(box)
                break

        if hand_roi and penis_roi:
            return _union_rois(hand_roi, penis_roi), penis_roi
        return hand_roi or penis_roi, penis_roi

    def _extract_roi_breast(self, frame_obj):
        """ROI for boobjob: breast area near penis."""
        lp = frame_obj.locked_penis_state
        penis_roi = lp.box if (lp.active and lp.box) else None

        breast_roi = None
        for box in frame_obj.boxes:
            if box.class_name == 'breast' and not box.is_excluded:
                breast_roi = _box_to_roi(box)
                break

        if breast_roi and penis_roi:
            return _union_rois(breast_roi, penis_roi), penis_roi
        return breast_roi or penis_roi, penis_roi

    def _extract_belly_roi(self, frame_obj):
        """Extract belly region above the locked penis for male thrust detection (VR POV)."""
        lp = frame_obj.locked_penis_state
        if not lp.active or not lp.box:
            return None
        px1, py1, px2, py2 = lp.box
        # Belly = same x-range as penis, extending upward from penis top
        belly_height = (py2 - py1) * 1.5  # 1.5x penis height above
        return (px1, max(0, py1 - belly_height), px2, py1)

    def extract_roi(self, frame_obj):
        """Extract ROI based on chapter strategy. Returns (primary_roi, penis_roi)."""
        if self.strategy == ChapterStrategy.PENETRATION:
            return self._extract_roi_penetration(frame_obj)
        elif self.strategy == ChapterStrategy.ORAL:
            return self._extract_roi_oral(frame_obj)
        elif self.strategy == ChapterStrategy.MANUAL:
            return self._extract_roi_manual(frame_obj)
        elif self.strategy == ChapterStrategy.BREAST:
            return self._extract_roi_breast(frame_obj)
        else:
            # Fallback: use penis box or full frame
            lp = frame_obj.locked_penis_state
            if lp.active and lp.box:
                return lp.box, lp.box
            return None, None

    # ------------------------------------------------------------------
    # Signal extraction from optical flow
    # ------------------------------------------------------------------

    def _integrate_flow(self, smooth_dy, smooth_dx=0.0):
        """Accumulate flow displacement. Drift is removed later in get_actions()."""
        self.integrated_dy += smooth_dy
        self.integrated_dx += smooth_dx
        return self.integrated_dy, self.integrated_dx

    def _compute_flow_in_roi(self, gray_frame, roi, img_h, img_w):
        """Compute DIS flow in the given ROI. Returns (median_dy, median_dx, magnitude)."""
        roi = _clamp_roi(roi, img_h, img_w)
        x1, y1, x2, y2 = roi
        if x2 - x1 < 8 or y2 - y1 < 8:
            return 0.0, 0.0, 0.0

        current_patch = gray_frame[y1:y2, x1:x2]

        if self.prev_gray is None:
            return 0.0, 0.0, 0.0

        prev_patch = self.prev_gray[y1:y2, x1:x2]
        if prev_patch.shape != current_patch.shape:
            return 0.0, 0.0, 0.0

        flow = self.dis.calc(
            np.ascontiguousarray(prev_patch),
            np.ascontiguousarray(current_patch),
            None
        )
        if flow is None:
            return 0.0, 0.0, 0.0

        dy = np.median(flow[..., 1])
        dx = np.median(flow[..., 0])
        mag = np.sqrt(dy**2 + dx**2)
        return float(dy), float(dx), float(mag)

    def _extract_signal_penetration(self, frame_obj, gray, roi, penis_roi, img_h, img_w):
        """
        Penetration signal: use Stage 2 visible_part + flow validation.

        Stage 2's visible_part (0-100%) directly maps to funscript position:
        - 100% visible = fully extracted = position 100
        - 0% visible = fully inserted = position 0

        Flow is used to:
        1. Fill in during occlusions (integrated displacement)
        2. Detect male thrusting (VR POV)
        3. Detect grinding (horizontal dominance)
        """
        lp = frame_obj.locked_penis_state

        # Geometric signal from Stage 2 (already computed, position-based)
        geo_pos = lp.visible_part if lp.active else None
        if geo_pos is not None:
            geo_pos = max(0.0, min(100.0, geo_pos))

        # Flow in the contact zone
        dy, dx, mag = self._compute_flow_in_roi(gray, roi, img_h, img_w)
        self.flow_history_dy.append(dy)
        self.flow_history_dx.append(dx)

        smooth_dy = float(np.median(self.flow_history_dy)) if self.flow_history_dy else 0.0
        smooth_dx = float(np.median(self.flow_history_dx)) if self.flow_history_dx else 0.0
        pos_dy, pos_dx = self._integrate_flow(smooth_dy, smooth_dx)

        # Male thrust detection (VR POV): check belly region
        if self.is_vr and self.belly_roi:
            belly_dy, _, belly_mag = self._compute_flow_in_roi(gray, self.belly_roi, img_h, img_w)
            if belly_mag > 0.5 and mag < 0.3:
                # Body is moving but contact zone is still → male thrusting
                pos_dy, _ = self._integrate_flow(belly_dy)
                return 50.0 - pos_dy, dx

        # Grinding detection
        abs_dy = abs(smooth_dy)
        abs_dx = abs(smooth_dx)

        if abs_dx > 0.5 and abs_dy > 0.01 and abs_dx / max(abs_dy, 0.01) > GRIND_DX_DY_RATIO:
            return 50.0 - pos_dx, dx

        # Use geometric position if available, else drift-corrected flow
        if geo_pos is not None:
            return geo_pos, dx
        else:
            return 50.0 - pos_dy, dx

    def _extract_signal_oral(self, frame_obj, gray, roi, penis_roi, img_h, img_w):
        """
        Oral signal: face/head vertical movement relative to penis.

        Integrates flow with high-pass drift removal so the signal tracks
        HEAD POSITION oscillation. Downward head displacement = deeper = lower value.
        """
        dy, dx, mag = self._compute_flow_in_roi(gray, roi, img_h, img_w)
        self.flow_history_dy.append(dy)
        self.flow_history_dx.append(dx)

        smooth_dy = float(np.median(self.flow_history_dy)) if self.flow_history_dy else 0.0
        pos_dy, _ = self._integrate_flow(smooth_dy)

        return 50.0 - pos_dy, dx

    def _extract_signal_manual(self, frame_obj, gray, roi, penis_roi, img_h, img_w):
        """
        Manual (HJ) signal: hand vertical movement with drift removal.
        """
        dy, dx, mag = self._compute_flow_in_roi(gray, roi, img_h, img_w)
        self.flow_history_dy.append(dy)
        self.flow_history_dx.append(dx)

        smooth_dy = float(np.median(self.flow_history_dy)) if self.flow_history_dy else 0.0
        pos_dy, _ = self._integrate_flow(smooth_dy)

        return 50.0 - pos_dy, dx

    def _extract_signal_fallback(self, frame_obj, gray, roi, penis_roi, img_h, img_w):
        """Fallback: use Stage 2 geometric signal if available, else flow."""
        if hasattr(frame_obj, 'funscript_distance') and frame_obj.funscript_distance is not None:
            return float(frame_obj.funscript_distance), 0.0

        dy, dx, mag = self._compute_flow_in_roi(gray, roi, img_h, img_w)
        self.flow_history_dy.append(dy)
        smooth_dy = float(np.median(self.flow_history_dy)) if self.flow_history_dy else 0.0
        pos_dy, _ = self._integrate_flow(smooth_dy)
        return 50.0 - pos_dy, dx

    def extract_signal(self, frame_obj, gray, roi, penis_roi, img_h, img_w):
        """Extract primary and secondary signals based on strategy."""
        if self.strategy == ChapterStrategy.PENETRATION:
            return self._extract_signal_penetration(frame_obj, gray, roi, penis_roi, img_h, img_w)
        elif self.strategy == ChapterStrategy.ORAL:
            return self._extract_signal_oral(frame_obj, gray, roi, penis_roi, img_h, img_w)
        elif self.strategy == ChapterStrategy.MANUAL:
            return self._extract_signal_manual(frame_obj, gray, roi, penis_roi, img_h, img_w)
        elif self.strategy == ChapterStrategy.BREAST:
            return self._extract_signal_manual(frame_obj, gray, roi, penis_roi, img_h, img_w)
        else:
            return self._extract_signal_fallback(frame_obj, gray, roi, penis_roi, img_h, img_w)

    # ------------------------------------------------------------------
    # Main per-frame processing
    # ------------------------------------------------------------------

    def process_frame(self, frame_obj, gray_frame):
        """Process a single frame. Accumulates signal in self.positions."""
        img_h, img_w = gray_frame.shape[:2]
        frame_id = frame_obj.frame_id

        # 1. Try to extract ROI from YOLO data
        yolo_roi, penis_roi = self.extract_roi(frame_obj)

        if yolo_roi is not None:
            roi = _pad_roi(yolo_roi, ROI_PAD, img_h, img_w)
            self.current_roi = roi
            self.roi_source = "yolo"
            self.occlusion_counter = 0
            # Update belly ROI for male thrust detection
            if self.is_vr and self.strategy == ChapterStrategy.PENETRATION:
                belly = self._extract_belly_roi(frame_obj)
                if belly:
                    self.belly_roi = _clamp_roi(belly, img_h, img_w)
        elif self.current_roi is not None:
            # 2. Occluded: drift ROI using optical flow
            self.occlusion_counter += 1
            if self.occlusion_counter <= OCCLUSION_PATIENCE_FRAMES and self.prev_gray is not None:
                # Use flow to drift the ROI position
                x1, y1, x2, y2 = self.current_roi
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                w, h = x2 - x1, y2 - y1
                # Compute flow at the ROI center (small patch)
                patch_size = min(32, w // 2, h // 2)
                if patch_size >= 4:
                    pcx, pcy = int(cx), int(cy)
                    ps = patch_size
                    py1c = max(0, pcy - ps)
                    py2c = min(img_h, pcy + ps)
                    px1c = max(0, pcx - ps)
                    px2c = min(img_w, pcx + ps)
                    if py2c - py1c >= 4 and px2c - px1c >= 4:
                        prev_p = self.prev_gray[py1c:py2c, px1c:px2c]
                        curr_p = gray_frame[py1c:py2c, px1c:px2c]
                        if prev_p.shape == curr_p.shape:
                            drift_flow = self.dis.calc(
                                np.ascontiguousarray(prev_p),
                                np.ascontiguousarray(curr_p), None
                            )
                            if drift_flow is not None:
                                ddx = float(np.median(drift_flow[..., 0])) * OCCLUSION_FLOW_DAMPING
                                ddy = float(np.median(drift_flow[..., 1])) * OCCLUSION_FLOW_DAMPING
                                self.current_roi = _clamp_roi(
                                    (x1 + ddx, y1 + ddy, x2 + ddx, y2 + ddy), img_h, img_w
                                )
                self.roi_source = "flow"
                roi = self.current_roi
            else:
                # Patience expired — no ROI
                roi = None
                self.roi_source = "none"
        else:
            roi = None
            self.roi_source = "none"

        # 3. Extract signal
        if roi is not None:
            primary_pos, secondary_dx = self.extract_signal(
                frame_obj, gray_frame, roi, penis_roi, img_h, img_w
            )
        else:
            # No ROI — use Stage 2 fallback or hold at 50
            if hasattr(frame_obj, 'funscript_distance') and frame_obj.funscript_distance is not None:
                primary_pos = float(frame_obj.funscript_distance)
            else:
                primary_pos = 50.0
            secondary_dx = 0.0

        # 4. Record
        self.positions.append((frame_id, primary_pos))
        # Secondary axis: convert dx flow to 0-100 position
        secondary_pos = 50.0 + secondary_dx * 8.0
        self.secondary_positions.append((frame_id, max(0.0, min(100.0, secondary_pos))))

        # 5. Update prev frame
        self.prev_gray = gray_frame.copy()

    # ------------------------------------------------------------------
    # Post-processing: normalize per chapter
    # ------------------------------------------------------------------

    def get_actions(self, normalize=True):
        """
        Return funscript actions for this chapter.

        If normalize=True, scales the chapter's position range to use
        as much of the 0-100 range as the content warrants. This is a
        gentle per-chapter normalization, NOT aggressive percentile stretching.

        Returns: (primary_actions, secondary_actions) as lists of {at, pos}.
        """
        if not self.positions:
            return [], []

        frame_ids = np.array([p[0] for p in self.positions])
        raw_positions = np.array([p[1] for p in self.positions])

        # --- Step 1: Remove drift with zero-phase high-pass filter ---
        # This is the key advantage of offline processing: we can look at
        # the entire signal and cleanly separate oscillation from drift.
        if len(raw_positions) >= 30:
            try:
                from scipy.signal import butter, sosfiltfilt
                nyq = self.fps / 2.0
                cutoff = min(HIGHPASS_CUTOFF_HZ, nyq * 0.8)  # safety margin
                sos = butter(2, cutoff / nyq, btype='high', output='sos')
                detrended = sosfiltfilt(sos, raw_positions)
                # Re-center around 50
                raw_positions = detrended + 50.0
            except Exception:
                # Fallback: simple detrend (subtract linear fit)
                raw_positions = raw_positions - np.linspace(
                    raw_positions[0], raw_positions[-1], len(raw_positions)
                ) + 50.0

        # --- Step 2: Light SG smoothing to remove single-frame noise ---
        if len(raw_positions) >= 7:
            win = min(7, len(raw_positions))
            if win % 2 == 0:
                win -= 1
            if win >= 5:
                smoothed = savgol_filter(raw_positions, win, 2)
            else:
                smoothed = raw_positions.copy()
        else:
            smoothed = raw_positions.copy()

        # --- Step 3: Sliding-window amplitude normalization ---
        # Global normalization makes loud sections dominate and quiet ones
        # shallow. Instead, normalize locally so each ~3s window fills the range.
        if normalize and len(smoothed) > 10:
            from scipy.ndimage import maximum_filter1d, minimum_filter1d
            win_frames = max(15, int(self.fps * 3))  # 3-second window
            local_max = maximum_filter1d(smoothed, win_frames, mode='nearest')
            local_min = minimum_filter1d(smoothed, win_frames, mode='nearest')
            local_range = local_max - local_min

            # Normalize each sample by its local range
            valid = local_range > 0.3
            result = np.full_like(smoothed, 50.0)
            result[valid] = (smoothed[valid] - local_min[valid]) / local_range[valid] * 90.0 + 5.0
            smoothed = result

        # Final clamp to valid funscript range
        smoothed = np.clip(smoothed, 0, 100)

        # Convert to funscript actions
        primary_actions = []
        for i, fid in enumerate(frame_ids):
            ts_ms = frame_to_ms(fid, self.fps)
            primary_actions.append({'at': ts_ms, 'pos': int(round(smoothed[i]))})

        # Secondary axis
        secondary_actions = []
        if self.secondary_positions:
            sec_positions = np.array([p[1] for p in self.secondary_positions])
            for i, fid in enumerate(frame_ids):
                ts_ms = frame_to_ms(fid, self.fps)
                secondary_actions.append({'at': ts_ms, 'pos': int(round(sec_positions[i]))})

        return primary_actions, secondary_actions


# ---------------------------------------------------------------------------
# Main tracker class
# ---------------------------------------------------------------------------

class GuidedFlowTracker(BaseOfflineTracker):
    """
    YOLO-guided dense optical flow tracker for offline funscript generation.

    Stage 3 replacement that uses Stage 2 detection data to create targeted
    ROIs for precise optical flow measurement per chapter type.
    """

    def __init__(self):
        super().__init__()
        self.num_workers = 1  # Single-threaded for now; multiprocessing per chapter later

    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="OFFLINE_GUIDED_FLOW",
            display_name="Guided Flow (3-Stage)",
            description="YOLO-guided dense optical flow with chapter-aware analysis strategies",
            category="offline",
            version="0.1.0",
            author="FunGen",
            tags=["offline", "guided-flow", "stage3", "yolo", "optical-flow", "chapter-aware"],
            requires_roi=False,
            supports_dual_axis=True,
            primary_axis="stroke",
            secondary_axis="roll",
            stages=[
                StageDefinition(1, "Detection", "YOLO object detection",
                                produces_funscript=False, requires_previous=False, output_type="analysis"),
                StageDefinition(2, "Contact Analysis", "Scene segmentation and contact tracking",
                                produces_funscript=False, requires_previous=True, output_type="segmentation"),
                StageDefinition(3, "Guided Flow", "YOLO-guided optical flow funscript generation",
                                produces_funscript=True, requires_previous=True, output_type="funscript"),
            ],
            properties={
                "produces_funscript_in_stage3": True,
                "supports_batch": True,
                "requires_stage2_data": True,
                "is_stage3_tracker": True,
                "num_stages": 3,
            }
        )

    @property
    def processing_stages(self) -> List[OfflineProcessingStage]:
        return [OfflineProcessingStage.STAGE_3]

    @property
    def stage_dependencies(self) -> Dict[OfflineProcessingStage, List[OfflineProcessingStage]]:
        return {
            OfflineProcessingStage.STAGE_3: [
                OfflineProcessingStage.STAGE_1,
                OfflineProcessingStage.STAGE_2,
            ]
        }

    def initialize(self, app_instance, **kwargs) -> bool:
        try:
            self.app = app_instance
            if not DETECTION_AVAILABLE:
                self.logger.error("Detection module not available")
                return False
            self._initialized = True
            self.logger.info("Guided Flow tracker initialized")
            return True
        except Exception as e:
            self.logger.error(f"Init failed: {e}", exc_info=True)
            return False

    def can_resume_from_checkpoint(self, checkpoint_data):
        return False  # Not yet implemented

    def estimate_processing_time(self, stage, video_path, **kwargs):
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0.0
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            # ~40 FPS for targeted ROI flow (faster than full-frame oscillation)
            return frame_count / 40.0 + 30.0
        except Exception:
            return 300.0

    # ------------------------------------------------------------------
    # Stage 3 processing
    # ------------------------------------------------------------------

    def process_stage(self, stage, video_path, input_data=None, input_files=None,
                      output_directory=None, progress_callback=None, frame_range=None,
                      resume_data=None, **kwargs) -> OfflineProcessingResult:
        if stage != OfflineProcessingStage.STAGE_3:
            return OfflineProcessingResult(success=False, error_message=f"Only Stage 3 supported, got {stage}")
        if not self._initialized:
            return OfflineProcessingResult(success=False, error_message="Not initialized")

        input_files = input_files or {}
        input_data = input_data or {}
        start_time = time.time()
        self.processing_active = True

        try:
            # Load Stage 2 data (msgpack primary, SQLite fallback)
            stage2_path = input_files.get('stage2') or input_files.get('stage2_output')
            stage2_sqlite_path = input_files.get('stage2_sqlite')
            stage2_data = None

            if stage2_path and os.path.exists(stage2_path):
                stage2_data = self._load_stage2(stage2_path)

            if not stage2_data and stage2_sqlite_path and os.path.exists(stage2_sqlite_path):
                stage2_data = self._load_stage2_sqlite(stage2_sqlite_path)

            if not stage2_data:
                return OfflineProcessingResult(success=False, error_message="Stage 2 output not found")

            segments = stage2_data['segments']
            frame_objects_map = stage2_data['frame_objects']

            if not segments:
                return OfflineProcessingResult(success=False, error_message="No segments in Stage 2 data")

            # Determine video properties
            preprocessed_path = input_files.get('preprocessed_video')
            video_to_open = preprocessed_path if (preprocessed_path and os.path.exists(preprocessed_path)) else video_path

            cap = cv2.VideoCapture(video_to_open)
            if not cap.isOpened():
                return OfflineProcessingResult(success=False, error_message=f"Cannot open video: {video_to_open}")

            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            is_vr = self._detect_vr(cap)

            self.logger.info(f"Guided Flow: {len(segments)} chapters, {total_frames} frames, {fps:.1f} fps, VR={is_vr}")

            # Process each chapter
            all_primary = []
            all_secondary = []
            stop_event = self.stop_event or Event()

            # Merge short false-NR segments before filtering
            segments = self._merge_false_nr_segments(segments, fps, frame_objects_map)

            # Filter to relevant segments only
            relevant_segments = [
                s for s in segments
                if self._get_position(s) not in ("Not Relevant", "NR", "C-Up", "Close Up")
            ]

            total_relevant_frames = sum(
                s.end_frame_id - s.start_frame_id + 1 for s in relevant_segments
            )
            frames_done = 0

            for seg_idx, seg in enumerate(relevant_segments):
                if stop_event.is_set():
                    break

                position = self._get_position(seg)
                strategy = POSITION_TO_STRATEGY.get(position, ChapterStrategy.UNKNOWN)
                short_name = getattr(seg, 'position_short_name', position)

                self.logger.info(f"  Chapter {seg_idx+1}/{len(relevant_segments)}: "
                                 f"'{short_name}' ({strategy}) frames {seg.start_frame_id}-{seg.end_frame_id}")

                processor = ChapterProcessor(seg, strategy, fps, is_vr, self.logger)

                # Seek to chapter start
                cap.set(cv2.CAP_PROP_POS_FRAMES, seg.start_frame_id)

                for fid in range(seg.start_frame_id, seg.end_frame_id + 1):
                    if stop_event.is_set():
                        break

                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break

                    # Get Stage 2 frame object
                    frame_obj = frame_objects_map.get(fid)
                    if frame_obj is None:
                        continue

                    # Convert to grayscale
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                    processor.process_frame(frame_obj, gray)

                    frames_done += 1
                    if progress_callback and frames_done % 100 == 0:
                        progress_callback(
                            current_chapter_idx=seg_idx + 1,
                            total_chapters=len(relevant_segments),
                            chapter_name=short_name,
                            current_chunk_idx=seg_idx + 1,
                            total_chunks=len(relevant_segments),
                            total_frames_processed_overall=frames_done,
                            total_frames_to_process_overall=total_relevant_frames,
                            processing_fps=frames_done / max(0.01, time.time() - start_time),
                            time_elapsed=time.time() - start_time,
                            eta_seconds=(total_relevant_frames - frames_done) /
                                        max(1, frames_done / max(0.01, time.time() - start_time))
                        )

                # Collect chapter actions
                primary, secondary = processor.get_actions(normalize=True)
                all_primary.extend(primary)
                all_secondary.extend(secondary)

            cap.release()

            # Sort by timestamp
            all_primary.sort(key=lambda a: a['at'])
            all_secondary.sort(key=lambda a: a['at'])

            # Build funscript object — use set_axis_actions to bypass live-tracker
            # simplification filters (min_interval, collinear removal) that would
            # aggressively reduce the dense per-frame optical flow output
            funscript = MultiAxisFunscript(logger=self.logger)
            funscript.set_axis_actions('primary', all_primary)
            funscript.set_axis_actions('secondary', all_secondary)

            # Set chapters from segments
            funscript.set_chapters_from_segments(segments, fps)

            elapsed = time.time() - start_time
            effective_fps = frames_done / max(0.01, elapsed)
            self.logger.info(f"Guided Flow complete: {len(all_primary)} actions, "
                             f"{frames_done} frames in {elapsed:.1f}s ({effective_fps:.1f} fps)")

            self.processing_active = False
            return OfflineProcessingResult(
                success=True,
                output_data={
                    "success": True,
                    "funscript": funscript,
                    "total_frames_processed": frames_done,
                    "processing_method": "guided_flow",
                    "video_segments": [s.to_dict() for s in segments],
                },
                performance_metrics={
                    "processing_time_seconds": elapsed,
                    "total_actions": len(all_primary),
                    "effective_fps": effective_fps,
                }
            )

        except Exception as e:
            self.processing_active = False
            self.logger.error(f"Guided Flow error: {e}", exc_info=True)
            return OfflineProcessingResult(success=False, error_message=str(e))

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_stage2(self, path):
        """Load Stage 2 msgpack output into segments + frame_objects_map."""
        try:
            import msgpack
            with open(path, 'rb') as f:
                data = msgpack.load(f, raw=False)

            # Reconstruct Segment objects
            raw_segments = data.get('segments', [])
            segments = []
            for sd in raw_segments:
                if isinstance(sd, dict):
                    seg = Segment(
                        sd['start_frame_id'],
                        sd['end_frame_id'],
                        sd.get('class_name', sd.get('position_long_name', 'Not Relevant'))
                    )
                    segments.append(seg)
                elif isinstance(sd, Segment):
                    segments.append(sd)

            # Reconstruct FrameObject map
            # Overlay msgpack uses 'frames' key; legacy/direct uses 'frame_objects'
            raw_frames = data.get('frame_objects') or data.get('frames', {})
            frame_objects_map = {}
            yolo_input_size = data.get('yolo_input_size', 640)
            # Detect overlay format: list with 'yolo_boxes' key instead of 'detections'
            is_overlay_format = (isinstance(raw_frames, list) and raw_frames
                                 and isinstance(raw_frames[0], dict)
                                 and 'yolo_boxes' in raw_frames[0])

            if is_overlay_format:
                frame_objects_map = self._reconstruct_from_overlay(raw_frames, yolo_input_size)
            elif isinstance(raw_frames, dict):
                for fid_str, fo_data in raw_frames.items():
                    fid = int(fid_str) if isinstance(fid_str, str) else fid_str
                    if isinstance(fo_data, FrameObject):
                        frame_objects_map[fid] = fo_data
                    elif isinstance(fo_data, dict):
                        fo = FrameObject(fid, yolo_input_size, fo_data)
                        frame_objects_map[fid] = fo
            elif isinstance(raw_frames, list):
                for fo_data in raw_frames:
                    if isinstance(fo_data, FrameObject):
                        frame_objects_map[fo_data.frame_id] = fo_data
                    elif isinstance(fo_data, dict):
                        fid = fo_data.get('frame_id', fo_data.get('frame_pos', 0))
                        fo = FrameObject(fid, yolo_input_size, fo_data)
                        frame_objects_map[fid] = fo

            self.logger.info(f"Loaded {len(segments)} segments, {len(frame_objects_map)} frames from Stage 2")
            return {'segments': segments, 'frame_objects': frame_objects_map}

        except Exception as e:
            self.logger.error(f"Failed to load Stage 2 data: {e}", exc_info=True)
            return None

    @staticmethod
    def _reconstruct_from_overlay(frames_list, yolo_input_size):
        """Reconstruct FrameObjects from overlay msgpack format.

        The overlay format stores boxes under 'yolo_boxes' (BoxRecord.to_dict())
        and locked_penis data as a separate dict, while FrameObject's
        parse_raw_frame_data expects 'detections' with 'class' key.
        """
        from detection.cd.data_structures.box_records import BoxRecord, PoseRecord
        result = {}
        for fd in frames_list:
            fid = fd.get('frame_id', fd.get('frame_pos', 0))
            # Create FrameObject without raw data — we'll populate manually
            fo = FrameObject(fid, yolo_input_size, None)
            fo.assigned_position = fd.get('assigned_position', 'Not Relevant')
            fo.funscript_distance = fd.get('funscript_distance', 50)
            fo.is_occluded = fd.get('is_occluded', False)
            fo.motion_mode = fd.get('motion_mode', None)
            fo.dominant_pose_id = fd.get('dominant_pose_id', None)

            # Reconstruct boxes from overlay yolo_boxes dicts
            for bd in fd.get('yolo_boxes', []):
                box = BoxRecord(
                    fid,
                    bd.get('bbox'),
                    bd.get('confidence', 0.0),
                    bd.get('class_id', -1),
                    bd.get('class_name', ''),
                    yolo_input_size=yolo_input_size
                )
                box.is_excluded = bd.get('is_excluded', False)
                if bd.get('track_id') is not None:
                    box.track_id = bd['track_id']
                fo.boxes.append(box)

            # Reconstruct poses
            for pd in fd.get('poses', []):
                pr = PoseRecord(fid, pd.get('bbox'), pd.get('keypoints'))
                fo.poses.append(pr)

            # Reconstruct locked_penis_state from overlay dict
            lp_data = fd.get('locked_penis')
            if lp_data and isinstance(lp_data, dict):
                fo.locked_penis_state.active = True
                fo.locked_penis_state.box = tuple(lp_data['bbox']) if 'bbox' in lp_data else None

            result[fid] = fo
        return result

    def _load_stage2_sqlite(self, db_path):
        """Load Stage 2 data from SQLite database as fallback."""
        try:
            from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
            storage = Stage2SQLiteStorage(db_path)
            segments = storage.get_segments()
            min_frame, max_frame = storage.get_frame_range()
            frame_objects_map = storage.get_frame_objects_range(min_frame, max_frame) if max_frame > min_frame else {}
            self.logger.info(f"Loaded {len(segments)} segments, {len(frame_objects_map)} frames from Stage 2 SQLite")
            return {'segments': segments, 'frame_objects': frame_objects_map}
        except Exception as e:
            self.logger.error(f"Failed to load Stage 2 SQLite data: {e}", exc_info=True)
            return None

    def _merge_false_nr_segments(self, segments, fps, frame_objects_map):
        """Merge short NR/Close-Up segments with neighbors when flanked by same chapter type.

        Rule-based: NR segments shorter than NR_MERGE_MAX_DURATION_S are merged
        with their neighbor(s) if the same chapter type appears on both sides.
        If different types on each side, merge with the longer neighbor.

        Learning-enhanced: when a trained NR recovery model exists, also merge
        NR segments the classifier predicts as false.
        """
        NR_POSITIONS = {"Not Relevant", "NR", "C-Up", "Close Up"}
        if len(segments) < 3:
            return segments

        max_frames = int(NR_MERGE_MAX_DURATION_S * fps)
        merged = list(segments)
        changed = True

        while changed:
            changed = False
            new_list = []
            i = 0
            while i < len(merged):
                seg = merged[i]
                pos = self._get_position(seg)
                seg_len = seg.end_frame_id - seg.start_frame_id

                if pos in NR_POSITIONS and seg_len <= max_frames:
                    left = new_list[-1] if new_list else None
                    right = merged[i + 1] if i + 1 < len(merged) else None

                    left_pos = self._get_position(left) if left else None
                    right_pos = self._get_position(right) if right else None

                    should_merge = False
                    merge_target_pos = None

                    if left_pos and right_pos and left_pos not in NR_POSITIONS and right_pos not in NR_POSITIONS:
                        if left_pos == right_pos:
                            # Same type on both sides — merge all three
                            should_merge = True
                            merge_target_pos = left_pos
                        else:
                            # Different types — merge with longer neighbor
                            left_len = left.end_frame_id - left.start_frame_id
                            right_len = right.end_frame_id - right.start_frame_id
                            should_merge = True
                            merge_target_pos = left_pos if left_len >= right_len else right_pos

                    # Learning-enhanced merge check
                    if not should_merge:
                        try:
                            from funscript.learning.correction_model import CorrectionModel
                            model = CorrectionModel.load()
                            if model.nr_model is not None:
                                from funscript.learning.feature_extractor import extract_nr_segment_features
                                neighbor_ct = (left_pos or right_pos or "unknown")
                                neighbor_match = (left_pos == right_pos) if (left_pos and right_pos) else False
                                nr_features = extract_nr_segment_features(
                                    seg.start_frame_id, seg.end_frame_id, fps,
                                    neighbor_ct, neighbor_match, frame_objects_map,
                                ).reshape(1, -1)
                                prob = model.predict_nr_recovery(nr_features)
                                if prob[0] > 0.5:
                                    should_merge = True
                                    merge_target_pos = left_pos or right_pos or "unknown"
                        except Exception:
                            pass  # Learning not available yet — skip

                    if should_merge and merge_target_pos:
                        if left and merge_target_pos == self._get_position(left):
                            # Extend left segment to cover NR
                            left.end_frame_id = seg.end_frame_id
                            # If right has same type, merge right into left too
                            if right and self._get_position(right) == merge_target_pos:
                                left.end_frame_id = right.end_frame_id
                                i += 2  # Skip NR + right
                                changed = True
                                continue
                            i += 1
                            changed = True
                            continue
                        elif right and merge_target_pos == self._get_position(right):
                            # Extend right segment to cover NR
                            right.start_frame_id = seg.start_frame_id
                            i += 1  # Skip NR, right will be added normally
                            changed = True
                            continue

                new_list.append(seg)
                i += 1
            merged = new_list

        nr_removed = len(segments) - len(merged)
        if nr_removed > 0:
            self.logger.info(f"Merged {nr_removed} false NR segment(s) with neighbors")
        return merged

    def _detect_vr(self, cap):
        """Simple VR detection from aspect ratio."""
        w = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        h = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if h > 0:
            return (w / h) > 2.0
        return False

    def _get_position(self, segment):
        """Get position name from a segment (handles both Segment objects and dicts)."""
        if hasattr(segment, 'major_position'):
            return segment.major_position
        if hasattr(segment, 'position_long_name'):
            return segment.position_long_name
        if isinstance(segment, dict):
            return segment.get('class_name', segment.get('position_long_name', 'Not Relevant'))
        return 'Not Relevant'

    def cleanup(self):
        self.stop_processing()
        super().cleanup()

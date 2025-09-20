import numpy as np
import msgpack
import threading
import math
import os
from typing import List, Dict, Any, Optional, Tuple, Union
import logging
import cv2
from scipy.signal import savgol_filter, find_peaks
from simplification.cutil import simplify_coords_vw
from multiprocessing import Pool
import multiprocessing
import psutil
import time
import tempfile

from video import VideoProcessor
from config import constants
from funscript.dual_axis_funscript import DualAxisFunscript
from application.utils.rts_smoother import RTSSmoother
from application.utils.stage2_signal_enhancer import Stage2SignalEnhancer

# Import data structures from new modular files
from .data_structures import (
    FrameObject, LockedPenisState,
    BoxRecord, PoseRecord,
    BaseSegment, Segment
)

# AppStateContainer no longer used - replaced with direct app object usage

CONTACT_ONLY_FALLBACKS = {'hand', 'pussy', 'butt'}

# Logger will be passed as parameter to each function instead of using global logger


def _progress_update(callback, task_name, current, total, force_update=False):
    if callback:
        if force_update or current == 0 or current == total or (total > 0 and (current % (max(1, total // 20))) == 0):
            callback(task_name, current, total)


# Global ultrafast DIS cache (per process)
_GLOBAL_DIS_FLOW_ULTRAFAST = None

def _get_dis_flow_ultrafast():
    global _GLOBAL_DIS_FLOW_ULTRAFAST
    if _GLOBAL_DIS_FLOW_ULTRAFAST is None:
        flow = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
        try:
            if hasattr(flow, 'setFinestScale'):
                flow.setFinestScale(5)
        except Exception:
            pass
        _GLOBAL_DIS_FLOW_ULTRAFAST = flow
    return _GLOBAL_DIS_FLOW_ULTRAFAST


# --- Helper Functions ---
def _calculate_iou(box1: Tuple[float, ...], box2: Tuple[float, ...]) -> float:
    x1_1, y1_1, x2_1, y2_1 = box1
    x1_2, y1_2, x2_2, y2_2 = box2
    x1_inter = max(x1_1, x1_2)
    y1_inter = max(y1_1, y1_2)
    x2_inter = min(x2_1, x2_2)
    y2_inter = min(y2_1, y2_2)
    inter_area = max(0, x2_inter - x1_inter) * max(0, y2_inter - y1_inter)
    box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
    box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
    union_area = box1_area + box2_area - inter_area
    return inter_area / union_area if union_area > 0 else 0.0

def _get_aligned_fallback_boxes(
        potential_fallback_boxes: List[BoxRecord],
        dominant_pose: Optional[PoseRecord],
        alignment_ref_cx: float,
        max_alignment_offset: float
) -> List[BoxRecord]:
    """
    Selects fallback boxes associated with a dominant pose and in vertical alignment.
    """
    if not dominant_pose:
        return []

    # 1. Filter to boxes spatially associated with the dominant pose
    person_box = dominant_pose.bbox
    associated_boxes = [
        box for box in potential_fallback_boxes
        if (person_box[0] < box.cx < person_box[2]) and (person_box[1] < box.cy < person_box[3])
    ]

    # 2. Filter by vertical alignment
    aligned_boxes = [
        box for box in associated_boxes
        if abs(box.cx - alignment_ref_cx) < max_alignment_offset
    ]

    return aligned_boxes

def _assign_frame_position(contacts: List[Dict], logger: Optional[logging.Logger] = None) -> str:
    if not contacts:
        return "Not Relevant"

    # Use a set for faster 'in' checks
    detected_class_names = {contact["class_name"] for contact in contacts}
    
    # DEBUG: Log contact data for first few calls
    if hasattr(_assign_frame_position, '_call_count'):
        _assign_frame_position._call_count += 1
    else:
        _assign_frame_position._call_count = 1
    
    if _assign_frame_position._call_count <= 10:
        if logger:
            logger.debug(f"_assign_frame_position call #{_assign_frame_position._call_count}:")
            logger.debug(f"  contacts (first 2): {contacts[:2]}")
            logger.debug(f"  detected_class_names: {detected_class_names}")

    # --- Priority Hierarchy ---
    # Prefer face (Blowjob) when both face and lower-body classes are detected to
    # avoid mislabeling obvious BJ scenes as penetration positions.
    priority_order = [
        ('face', 'Blowjob'),
        ('pussy', 'Cowgirl / Missionary'),
        ('butt', 'Rev. Cowgirl / Doggy'),
        ('hand', 'Handjob'),
        ('breast', 'Boobjob'),
        ('navel', 'Not Relevant'),  # Navel contact is not a distinct action
        ('foot', 'Footjob'),
        ('anus', 'Rev. Cowgirl / Doggy')  # Anus implies a Doggy-style position
    ]

    for class_name, position in priority_order:
        if class_name in detected_class_names:
            if _assign_frame_position._call_count <= 10:
                if logger:
                    logger.debug(f"  MATCH: class_name='{class_name}' -> position='{position}' (type: {type(position)})")
            return position

    if _assign_frame_position._call_count <= 10:
        if logger:
            logger.debug("  NO MATCH -> returning 'Not Relevant'")
    return "Not Relevant"  # Default if no priority classes are found



def _aggregate_segments(frame_objects: List[FrameObject], fps: float, min_segment_duration_frames: int,
                            logger: logging.Logger) -> List[Segment]:
    """
    Aggregates frame data into a final, stable list of segments using
    a master iterative loop to ensure all merging rules are exhaustively applied.
    """
    if not frame_objects:
        return []

    # --- Config for Stability ---
    SHORT_FLICKER_DURATION = int(fps * 1.0)  # Anything less than 1s is a potential flicker to be merged.

    # --- Pass 1: Create an initial, granular list of segments. ---
    # We create a segment for every single change in position. This list will be messy
    # but provides the raw material for our robust merging loop.
    segments: List[Segment] = []
    if frame_objects:
        current_pos = frame_objects[0].assigned_position
        start_frame = frame_objects[0].frame_id
        
        # DEBUG: Log first segment creation
        if logger:
            logger.debug(f"_aggregate_segments: First frame position = '{current_pos}' (type: {type(current_pos)})")
        
        for i in range(1, len(frame_objects)):
            if frame_objects[i].assigned_position != current_pos:
                # DEBUG: Log segment creation
                if logger:
                    logger.debug(f"_aggregate_segments: Creating segment {start_frame}-{frame_objects[i - 1].frame_id} with position '{current_pos}' (type: {type(current_pos)})")
                segments.append(Segment(start_frame, frame_objects[i - 1].frame_id, current_pos))
                current_pos = frame_objects[i].assigned_position
                start_frame = frame_objects[i].frame_id
                
        # DEBUG: Log final segment creation
        if logger:
            logger.debug(f"_aggregate_segments: Creating final segment {start_frame}-{frame_objects[-1].frame_id} with position '{current_pos}' (type: {type(current_pos)})")
        segments.append(Segment(start_frame, frame_objects[-1].frame_id, current_pos))

    if logger:
        logger.debug(f"Pass 1 (Initial Creation) generated {len(segments)} granular segments.")

    # --- Pass 2: The Master Iterative Merging Loop ---
    # This loop continues until a full pass over all rules results in zero merges.
    # This guarantees that the list is fully stabilized.
    while True:
        merges_made_in_pass = 0

        # --- Rule A: Merge short "Not Relevant" segments between identical neighbors ---
        i = 0
        while i < len(segments) - 2:
            seg1, seg_gap, seg2 = segments[i], segments[i + 1], segments[i + 2]
            is_short_gap = seg_gap.major_position == "Not Relevant" and seg_gap.duration < (fps * 10)
            if seg1.major_position == seg2.major_position and seg1.major_position != "Not Relevant" and is_short_gap:
                if logger:
                    logger.debug(f"Rule A: Merging {seg1.major_position} across short NR gap.")
                seg1.end_frame_id = seg2.end_frame_id
                seg1.update_duration()
                segments.pop(i + 2)
                segments.pop(i + 1)
                merges_made_in_pass += 1
                i = 0  # Restart scan after a modification
                continue
            i += 1

        # --- Rule A2: Merge short penetration misclassifications between identical oral neighbors ---
        # Example: BJ | short CG | BJ -> merge into a single BJ segment (targeted; avoids collapsing real short BJs)
        i = 0
        while i < len(segments) - 2:
            seg1, seg_mid, seg2 = segments[i], segments[i + 1], segments[i + 2]
            is_short_mid = (seg_mid.major_position != "Not Relevant") and (seg_mid.duration < (fps * 5))
            oral_positions = {'Blowjob', 'Handjob'}
            penetration_positions = {'Cowgirl / Missionary', 'Rev. Cowgirl / Doggy'}
            neighbors_oral_and_identical = (seg1.major_position == seg2.major_position) and (seg1.major_position in oral_positions)
            mid_is_penetration = seg_mid.major_position in penetration_positions

            if neighbors_oral_and_identical and mid_is_penetration and is_short_mid:
                if logger:
                    logger.debug(
                        f"Rule A2: Merging short penetration '{seg_mid.major_position}' between oral segments '{seg1.major_position}'."
                )
                seg1.end_frame_id = seg2.end_frame_id
                seg1.update_duration()
                segments.pop(i + 2)
                segments.pop(i + 1)
                merges_made_in_pass += 1
                i = 0
                continue
            i += 1

        # --- Rule B: Merge adjacent "Handjob" and "Blowjob" segments into "Blowjob" ---
        i = 0
        while i < len(segments) - 1:
            pos1, pos2 = segments[i].major_position, segments[i + 1].major_position
            if {pos1, pos2} <= {'Handjob', 'Blowjob'}:  # Use set for commutative check
                if logger:
                    logger.debug(f"Rule B: Merging adjacent '{pos1}' and '{pos2}' into 'Blowjob'.")
                segments[i].major_position = 'Blowjob'
                segments[i].end_frame_id = segments[i + 1].end_frame_id
                segments[i].update_duration()
                segments.pop(i + 1)
                merges_made_in_pass += 1
                i = 0  # Restart scan
                continue
            i += 1

        # --- Rule C: Merge any remaining identical adjacent segments ---
        i = 0
        while i < len(segments) - 1:
            if segments[i].major_position == segments[i + 1].major_position:
                if logger:
                    logger.debug(f"Rule C: Merging adjacent identicals: {segments[i].major_position}")
                segments[i].end_frame_id = segments[i + 1].end_frame_id
                segments[i].update_duration()
                segments.pop(i + 1)
                merges_made_in_pass += 1
                i = 0  # Restart scan
                continue
            i += 1

        # If a full pass of all rules resulted in no changes, the list is stable.
        if merges_made_in_pass == 0:
            if logger:
                logger.debug("Pass 2 (Iterative Merging) stabilized.")
            break

    # --- Pass 3: Final Cleanup & Gap Filling ---

    # First, cleanup any remaining short segments (<10s) by merging them into their longest neighbor.
    min_duration_10s = int(fps * 10)
    i = 0
    while i < len(segments):
        if segments[i].duration < min_duration_10s:
            prev_dur = segments[i - 1].duration if i > 0 else -1
            next_dur = segments[i + 1].duration if i < len(segments) - 1 else -1

            if prev_dur == -1 and next_dur == -1:  # It's the only segment
                break

            # Prefer merging short segments into a Blowjob neighbor if present to reduce BJ -> CG mislabels
            prev_is_bj = (i > 0 and segments[i - 1].major_position == 'Blowjob')
            next_is_bj = (i < len(segments) - 1 and segments[i + 1].major_position == 'Blowjob')

            if prev_is_bj and not next_is_bj:
                target = 'prev'
            elif next_is_bj and not prev_is_bj:
                target = 'next'
            else:
                # Fall back to longest neighbor
                target = 'prev' if prev_dur >= next_dur else 'next'

            if target == 'prev':  # Merge into previous neighbor
                if logger:
                    logger.debug(f"Pass 3: Cleaning up short segment ({segments[i].major_position}) by merging into previous.")
                segments[i - 1].end_frame_id = segments[i].end_frame_id
                segments[i - 1].update_duration()
                segments.pop(i)
                i = 0  # Restart scan from the beginning after a merge
                continue
            else:  # Merge into next neighbor
                if logger:
                    logger.debug(f"Pass 3: Cleaning up short segment ({segments[i].major_position}) by merging into next.")
                segments[i + 1].start_frame_id = segments[i].start_frame_id
                segments[i + 1].update_duration()
                segments.pop(i)
                i = 0  # Restart scan
                continue
        i += 1

    # Second, fill any remaining gaps with "Not Relevant" segments for a gapless timeline.
    final_segments: List[Segment] = []
    last_end_frame = -1
    for seg in segments:
        if seg.start_frame_id > last_end_frame + 1:
            final_segments.append(Segment(last_end_frame + 1, seg.start_frame_id - 1, "Not Relevant"))
        final_segments.append(seg)
        last_end_frame = seg.end_frame_id

    # --- Pass 4: Final merging of adjacent segments of same type or HJ/BJ ---
    while True:
        merges_made = 0
        i = 0
        while i < len(final_segments) - 1:
            pos1, pos2 = final_segments[i].major_position, final_segments[i + 1].major_position

            # Check if segments are same type or HJ/BJ combination
            if pos1 == pos2 or {pos1, pos2} <= {'Handjob', 'Blowjob'}:
                if logger:
                    logger.debug(f"Pass 4: Merging adjacent '{pos1}' and '{pos2}' segments.")
                # For HJ/BJ combination, use 'Blowjob' as the merged type
                if {pos1, pos2} <= {'Handjob', 'Blowjob'}:
                    final_segments[i].major_position = 'Blowjob'
                final_segments[i].end_frame_id = final_segments[i + 1].end_frame_id
                final_segments[i].update_duration()
                final_segments.pop(i + 1)
                merges_made += 1
                i = 0  # Restart scan after modification
                continue
            i += 1

        if merges_made == 0:
            if logger:
                logger.debug("Pass 4 (Final Merging) stabilized.")
            break

    if logger:
        logger.debug(f"Final, clean segment count: {len(final_segments)}")
    return final_segments

def _calculate_normalized_distance_to_base(locked_penis_box_coords: Tuple[float, float, float, float],
                                               class_name: str,
                                               class_box_coords: Tuple[float, float, float, float],
                                               max_distance_ref: float) -> float:
    """
    Calculates normalized distance, with a crucial refinement for hand/face interaction.
    """
    penis_base_y = locked_penis_box_coords[3]  # y2 (bottom of the conceptual full-stroke box)

    # --- IMPROVED LOGIC ---
    # Use optimal reference points for each interaction type
    if class_name == 'face':
        box_y_ref = class_box_coords[3]  # Bottom of the face
    elif class_name == 'hand':
        box_y_ref = (class_box_coords[1] + class_box_coords[3]) / 2  # Center Y of the hand
    elif class_name == 'pussy':
        # Use center of pussy box for better penetration depth correlation
        box_y_ref = (class_box_coords[1] + class_box_coords[3]) / 2  # Center Y of pussy
    elif class_name == 'butt':
        box_y_ref = (9 * class_box_coords[3] + class_box_coords[1]) / 10  # Mostly bottom of butt
    else:  # breast, foot, etc.
        box_y_ref = (class_box_coords[1] + class_box_coords[3]) / 2  # Center of other parts

    # --- TEST LOGIC ---
    #box_y_ref = (class_box_coords[1] + class_box_coords[3]) / 2

    raw_distance = penis_base_y - box_y_ref
    if max_distance_ref <= 0:
        return 50.0
    normalized_distance = (raw_distance / max_distance_ref) * 100.0
    return np.clip(normalized_distance, 0, 100)


def _calculate_fallback_absolute_distance(locked_penis_box_coords: Tuple[float, float, float, float],
                                            fallback_class_name: str,
                                            fallback_box_coords: Tuple[float, float, float, float],
                                            primary_class_name: str,
                                            rolling_distances: List[float],
                                            max_window_size: int = 10) -> float:
    """
    Calculate absolute distance for fallback classes to maintain signal continuity.
    
    Instead of using the fallback's own reference system, this measures absolute distance
    from fallback to penis base and maps it to the primary class's expected signal range
    using rolling window statistics.
    """
    penis_base_y = locked_penis_box_coords[3]  # Bottom of penis conceptual box
    
    # Always use center point for fallback classes for consistency
    fallback_y_ref = (fallback_box_coords[1] + fallback_box_coords[3]) / 2
    
    # Calculate absolute distance from fallback to penis base
    absolute_distance = abs(penis_base_y - fallback_y_ref)
    
    # Add to rolling window
    rolling_distances.append(absolute_distance)
    if len(rolling_distances) > max_window_size:
        rolling_distances.pop(0)
    
    # Use rolling window statistics to map to primary class signal range
    if len(rolling_distances) >= 3:
        # Use median of recent distances for stability
        median_distance = np.median(rolling_distances)
        # Map to 0-100 range based on rolling statistics
        min_dist = min(rolling_distances)
        max_dist = max(rolling_distances)
        
        if max_dist > min_dist:
            # Normalize to 0-100 range based on rolling window
            normalized = ((median_distance - min_dist) / (max_dist - min_dist)) * 100.0
        else:
            normalized = 50.0  # Default middle value if no variation
    else:
        # Not enough history, use simple normalization
        normalized = min(100.0, (absolute_distance / 200.0) * 100.0)  # Assume 200px max distance
    
    return np.clip(normalized, 0, 100)


def _normalize_funscript_sparse_per_segment(app, frame_objects: List[FrameObject], segments: List[Segment], logger: Optional[logging.Logger]):
    """ Normalizes funscript distances (funscript_distance on FrameObject) per Segment.
    """
    # Vectorize operations using NumPy where possible
    for seg in segments:
        if not seg.segment_frame_objects: 
            continue

        # Convert to NumPy array for vectorized operations
        values = np.array([fo.funscript_distance for fo in seg.segment_frame_objects])
        if values.size == 0: 
            continue

        # Calculate percentiles once
        p01, p99 = np.percentile(values, [5, 95])
        
        # Vectorized filtering
        mask = (values >= p01) & (values <= p99)
        filtered_values = values[mask]
        
        # Calculate min/max once
        min_val_in_segment, max_val_in_segment = values.min(), values.max()
        
        if filtered_values.size > 0:
            filtered_min, filtered_max = filtered_values.min(), filtered_values.max()
        else:
            filtered_min, filtered_max = min_val_in_segment, max_val_in_segment
            
        scale_range = filtered_max - filtered_min
        
        # Vectorized normalization
        if seg.major_position in ['Not Relevant', 'Close up']:
            normalized_values = np.full_like(values, 100.0)
        else:
            # Vectorized outlier handling
            normalized_values = np.empty_like(values, dtype=float)
            
            # Low outliers
            low_mask = values <= p01
            if np.any(low_mask):
                denominator = p01 - min_val_in_segment
                if denominator > 1e-6:
                    normalized_values[low_mask] = ((values[low_mask] - min_val_in_segment) / denominator) * 5.0
                else:
                    normalized_values[low_mask] = 0.0
            
            # High outliers
            high_mask = values >= p99
            if np.any(high_mask):
                denominator = max_val_in_segment - p99
                if denominator > 1e-6:
                    normalized_values[high_mask] = 95.0 + ((values[high_mask] - p99) / denominator) * 5.0
                else:
                    normalized_values[high_mask] = 100.0
            
            # Normal values
            normal_mask = ~low_mask & ~high_mask
            if np.any(normal_mask):
                if scale_range < 1e-6:
                    normalized_values[normal_mask] = 50.0
                else:
                    normalized_values[normal_mask] = 5.0 + ((values[normal_mask] - filtered_min) / scale_range) * 90.0
        
        # Apply normalized values back to frame objects
        clipped_values = np.clip(np.round(normalized_values), 0, 100).astype(int)
        for i, fo in enumerate(seg.segment_frame_objects):
            fo.funscript_distance = clipped_values[i]

def _apply_signal_enhancement(app, frame_objects: List[FrameObject], logger: Optional[logging.Logger] = None):
    """
    Apply Stage 2 signal enhancement using frame difference analysis.
    
    This function enhances Stage 2 funscript signals by:
    - Suppressing false strokes (signal change without motion)
    - Adding missing strokes (motion without signal change) 
    - Reinforcing valid strokes (signal and motion agree)
    """
    # Check if enhancement is enabled in app settings
    if not getattr(app.app_settings, 'enable_signal_enhancement', True):
        if logger:
            logger.debug("Signal enhancement disabled, skipping")
        return
    
    n_frames = len(frame_objects)
    if not frame_objects or n_frames < 2:
        if logger:
            logger.debug("Not enough frames for signal enhancement")
        return
    
    if logger:
        logger.debug(f"Applying signal enhancement to {n_frames} frames")
    
    enhanced_count = 0
    
    # Cache property access and precompute where possible
    frame_data = []
    for frame_obj in frame_objects:
        if frame_obj.locked_penis_state.active and frame_obj.locked_penis_state.box:
            box = frame_obj.locked_penis_state.box
            # Precompute center coordinates
            cx = (box[0] + box[2]) / 2
            cy = (box[1] + box[3]) / 2
            frame_data.append((frame_obj.funscript_distance, cx, cy))
        else:
            frame_data.append((frame_obj.funscript_distance, None, None))
    
    # Process frames in pairs to reduce redundant calculations
    for i in range(1, n_frames):
        curr_distance, curr_cx, curr_cy = frame_data[i]
        prev_distance, prev_cx, prev_cy = frame_data[i-1]
        
        # Check for motion
        motion_detected = False
        if curr_cx is not None and prev_cx is not None:
            # Use squared distance to avoid sqrt calculation
            dx = curr_cx - prev_cx
            dy = curr_cy - prev_cy
            squared_movement = dx*dx + dy*dy
            # Compare squared distances (25 = 5^2)
            motion_detected = squared_movement > 25.0
        
        # Apply enhancement logic
        signal_change = abs(curr_distance - prev_distance)
        enhanced_signal = curr_distance
        
        # False stroke suppression: large signal change but no motion
        if signal_change > 8 and not motion_detected:
            # Reduce signal change by 40%
            change_direction = 1 if curr_distance > prev_distance else -1
            reduced_change = int(signal_change * 0.6)
            enhanced_signal = prev_distance + (change_direction * reduced_change)
            enhanced_count += 1
        
        # Missing stroke detection: motion but small signal change
        elif motion_detected and signal_change < 5:
            # Add small boost based on motion
            boost = 8  # Small boost
            direction = 1 if i % 2 == 0 else -1  # Alternate direction for oscillation
            enhanced_signal = curr_distance + (direction * boost)
            enhanced_count += 1
        
        # Apply enhanced signal
        frame_objects[i].funscript_distance = int(np.clip(enhanced_signal, 0, 100))
    
    if logger:
        logger.debug(f"Enhanced {enhanced_count} frame signals out of {n_frames}")

def _get_dominant_pose(frame_obj: FrameObject, is_vr: bool, frame_width: int) -> Optional[PoseRecord]:
    if not frame_obj.poses:
        return None
    if is_vr:
        frame_center_x = frame_width / 2
        return min(frame_obj.poses, key=lambda p: abs(((p.bbox[0] + p.bbox[2]) / 2) - frame_center_x))
    else:
        return max(frame_obj.poses, key=lambda p: p.person_box_area)


# --- Step 0: Global Object Tracking (IoU Tracker) ---

def resilient_tracker_step0(app, frames: List, video_info: Dict, logger: Optional[logging.Logger]):
    """
    Enhanced resilient tracker with:
    - Velocity gating for sudden jumps
    - Class exclusivity (breast, butt, pussy, face cannot overlap)
    - Tentative → Active promotion
    - Recently-dead buffer to avoid ghost recreation
    - Stricter ReID thresholds
    """
    if not logger:
        logger = logging.getLogger("ResilientTracker")

    if logger:
        logger.debug("Starting Step 0: Enhanced Resilient Tracking")

    fps = video_info.get('fps', 30.0)

    IOU_THRESHOLD = 0.5
    MAX_FRAMES_TO_BECOME_LOST = int(fps * 1.5)
    MAX_FRAMES_IN_LOST_STATE = int(fps * 8)
    REID_DISTANCE_FACTOR = 0.75  # stricter than before
    MAX_NEW_TRACKS_PER_FRAME = 3

    CLASS_SPECIFIC_IOU = {
        'hand': 0.4,
        'finger': 0.35,
        'pussy': 0.6,
        'butt': 0.6,
        'face': 0.55,
        'breast': 0.5,
        'foot': 0.45
    }

    EXCLUSIVE_CLASSES = {'breast', 'butt', 'pussy', 'face'}

    # Tentative promotion rule
    TENTATIVE_FRAMES_REQUIRED = 2  # must be seen in N consecutive frames before becoming active

    # Recently-dead buffer
    RECENT_DEAD_FRAMES = int(fps * 0.5)  # 0.5 sec
    recently_dead = []  # list of (frame_id, bbox, class_name)

    next_global_track_id = 1
    active_tracks = {}    # id -> {box_rec, frames_unseen, class_name, status}
    lost_tracks = {}
    tentative_tracks = {} # id -> same structure as active_tracks

    for frame_obj in sorted(frames, key=lambda f: f.frame_id):
        current_detections = [b for b in frame_obj.boxes if not b.is_excluded]
        unmatched_detections = list(range(len(current_detections)))

        # --- 1. Age existing tracks ---
        to_lost = []
        for track_id, td in active_tracks.items():
            td["frames_unseen"] += 1
            if td["frames_unseen"] > MAX_FRAMES_TO_BECOME_LOST:
                to_lost.append(track_id)

        for tid in to_lost:
            lost_tracks[tid] = active_tracks.pop(tid)
            lost_tracks[tid]["status"] = "lost"
            recently_dead.append((frame_obj.frame_id, lost_tracks[tid]["box_rec"].bbox.copy(), lost_tracks[tid]["class_name"]))

        # Prune recently_dead buffer
        recently_dead = [(fid, bb, cn) for (fid, bb, cn) in recently_dead if frame_obj.frame_id - fid <= RECENT_DEAD_FRAMES]

        # Delete old lost tracks
        to_delete = []
        for tid, td in lost_tracks.items():
            td["frames_unseen"] += 1
            if td["frames_unseen"] > MAX_FRAMES_IN_LOST_STATE:
                to_delete.append(tid)
        for tid in to_delete:
            del lost_tracks[tid]

        # --- 2. Match Active Tracks ---
        for track_id, td in list(active_tracks.items()):
            best_idx, max_iou = -1, -1
            class_name = td["class_name"]
            thr = CLASS_SPECIFIC_IOU.get(class_name, IOU_THRESHOLD)

            for i in unmatched_detections:
                det = current_detections[i]
                if det.class_name != class_name:
                    continue
                iou = _calculate_iou(td["box_rec"].bbox, det.bbox)

                # Velocity gating
                jump_dist = np.linalg.norm([det.cx - td["box_rec"].cx, det.cy - td["box_rec"].cy])
                max_jump = max(td["box_rec"].width, td["box_rec"].height) * 1.5
                if jump_dist > max_jump:
                    continue

                # Size consistency
                size_ratio = min(det.width / td["box_rec"].width, det.height / td["box_rec"].height)
                if not (0.7 <= size_ratio <= 1.3):
                    continue

                if iou > thr and iou > max_iou:
                    max_iou = iou
                    best_idx = i

            if best_idx != -1:
                det = current_detections[best_idx]
                det.track_id = track_id
                td["box_rec"] = det
                td["frames_unseen"] = 0
                unmatched_detections.remove(best_idx)

        # --- 3. Re-Identify Lost Tracks ---
        for i in list(unmatched_detections):
            det = current_detections[i]
            best_lost_id, min_score = -1, float('inf')
            for tid, td in lost_tracks.items():
                if td["class_name"] != det.class_name:
                    continue
                dist = np.linalg.norm([det.cx - td["box_rec"].cx, det.cy - td["box_rec"].cy])
                reid_thr = REID_DISTANCE_FACTOR * min(td["box_rec"].width, td["box_rec"].height)
                if dist > reid_thr:
                    continue
                size_ratio = min(det.width / td["box_rec"].width, det.height / td["box_rec"].height)
                if not (0.7 <= size_ratio <= 1.3):
                    continue
                score = dist
                if score < min_score:
                    min_score = score
                    best_lost_id = tid

            if best_lost_id != -1:
                det.track_id = best_lost_id
                reactivated = lost_tracks.pop(best_lost_id)
                reactivated["box_rec"] = det
                reactivated["frames_unseen"] = 0
                active_tracks[best_lost_id] = reactivated
                unmatched_detections.remove(i)

        # --- 4. Update Tentative Tracks ---
        for tid, td in list(tentative_tracks.items()):
            matched = False
            for i in unmatched_detections:
                det = current_detections[i]
                if det.class_name != td["class_name"]:
                    continue
                iou = _calculate_iou(td["box_rec"].bbox, det.bbox)
                if iou > CLASS_SPECIFIC_IOU.get(td["class_name"], IOU_THRESHOLD):
                    det.track_id = tid
                    td["box_rec"] = det
                    td["frames_seen"] += 1
                    matched = True
                    unmatched_detections.remove(i)
                    break
            if not matched:
                td["frames_unseen"] += 1
                if td["frames_unseen"] > 2:
                    del tentative_tracks[tid]
            elif td["frames_seen"] >= TENTATIVE_FRAMES_REQUIRED:
                td["status"] = "active"
                active_tracks[tid] = tentative_tracks.pop(tid)

        # --- 5. Create New Tentative Tracks ---
        new_tracks = 0
        for i in unmatched_detections:
            det = current_detections[i]
            if new_tracks >= MAX_NEW_TRACKS_PER_FRAME:
                break
            if det.confidence < 0.4:
                continue
            # Prevent recreation from recently_dead
            if any(det.class_name == cn and _calculate_iou(det.bbox, bb) > 0.5 for (_, bb, cn) in recently_dead):
                continue
            det.track_id = next_global_track_id
            tentative_tracks[next_global_track_id] = {
                "box_rec": det,
                "frames_unseen": 0,
                "frames_seen": 1,
                "class_name": det.class_name,
                "status": "tentative"
            }
            next_global_track_id += 1
            new_tracks += 1

        # --- 6. Enforce Exclusive Class Overlaps ---
        # Resolve conflicts by keeping highest-confidence track
        active_tracks_list = sorted(active_tracks.items(), key=lambda x: x[1]["box_rec"].confidence, reverse=True)
        kept_ids = []
        for tid, td in active_tracks_list:
            if td["class_name"] in EXCLUSIVE_CLASSES:
                if any(_calculate_iou(td["box_rec"].bbox, active_tracks[k]["box_rec"].bbox) > 0.3
                       and active_tracks[k]["class_name"] in EXCLUSIVE_CLASSES for k in kept_ids):
                    continue
            kept_ids.append(tid)
        active_tracks = {tid: active_tracks[tid] for tid in kept_ids}

    if logger:
        logger.debug(f"Tracking complete. Final ID count: {next_global_track_id - 1}")


def prev_resilient_tracker_step0(app, frames: List, video_info: Dict, logger: Optional[logging.Logger]):
    """
    An improved IoU-based tracker with state persistence to handle occlusions.
    This replaces the previous simple tracker.
    """
    if not logger:
        logger = logging.getLogger("ResilientTracker")

    if logger:
        logger.debug("Starting Step 0: Resilient Object Tracking")

    # --- Configuration for the new tracker ---
    fps = video_info.get('fps', 30.0)
    IOU_THRESHOLD = 0.5  # Min overlap to be considered the same object in consecutive frames.
    MAX_FRAMES_TO_BECOME_LOST = int(fps * 1.5)  # A track is "lost" if unseen for 1.5s.
    MAX_FRAMES_IN_LOST_STATE = int(fps * 8)  # A "lost" track is permanently deleted after 8s.
    REID_DISTANCE_THRESHOLD_FACTOR = 1.0  # Search radius for re-identification is 1x the object's size.
    
    # Class-specific thresholds
    CLASS_SPECIFIC_IOU = {
        'hand': 0.4,  # Hands move quickly, more permissive
        'finger': 0.35,
        'pussy': 0.6,  # Body parts should be more stable
        'butt': 0.6,
        'face': 0.55,
        'breast': 0.5,
        'foot': 0.45
    }
    
    MAX_NEW_TRACKS_PER_FRAME = 3  # Limit new track creation per frame

    next_global_track_id = 1
    active_tracks: Dict[int, Dict[str, Any]] = {}
    lost_tracks: Dict[int, Dict[str, Any]] = {}

    for frame_obj in sorted(frames, key=lambda f: f.frame_id):
        current_detections = [b for b in frame_obj.boxes if not b.is_excluded]
        unmatched_detections = list(range(len(current_detections)))

        # --- 1. Update and Prune Tracks ---
        tracks_to_move_to_lost = []
        for track_id, track_data in active_tracks.items():
            track_data["frames_unseen"] += 1
            if track_data["frames_unseen"] > MAX_FRAMES_TO_BECOME_LOST:
                tracks_to_move_to_lost.append(track_id)

        for track_id in tracks_to_move_to_lost:
            lost_tracks[track_id] = active_tracks.pop(track_id)
            if logger:
                logger.debug(f"Track {track_id} ({lost_tracks[track_id]['class_name']}) moved to 'lost' state.")

        tracks_to_delete_permanently = []
        for track_id, track_data in lost_tracks.items():
            track_data["frames_unseen"] += 1
            if track_data["frames_unseen"] > MAX_FRAMES_IN_LOST_STATE:
                tracks_to_delete_permanently.append(track_id)

        for track_id in tracks_to_delete_permanently:
            del lost_tracks[track_id]
            if logger:
                logger.debug(f"Track {track_id} permanently deleted.")

        # --- 2. Match Active Tracks ---
        for track_id, track_data in active_tracks.items():
            best_match_idx, max_iou = -1, -1
            class_name = track_data["class_name"]
            class_iou_threshold = CLASS_SPECIFIC_IOU.get(class_name, IOU_THRESHOLD)
            
            for i in unmatched_detections:
                det_box = current_detections[i]
                if det_box.class_name == class_name:
                    iou = _calculate_iou(track_data["box_rec"].bbox, det_box.bbox)
                    # Additional validation: check size consistency
                    size_ratio = min(det_box.width / track_data["box_rec"].width, 
                                   det_box.height / track_data["box_rec"].height)
                    size_consistency = size_ratio > 0.5 and size_ratio < 2.0  # Allow 50%-200% size variation
                    
                    if iou > class_iou_threshold and iou > max_iou and size_consistency:
                        max_iou = iou
                        best_match_idx = i

            if best_match_idx != -1:
                det_box = current_detections[best_match_idx]
                det_box.track_id = track_id
                track_data["box_rec"] = det_box
                track_data["frames_unseen"] = 0
                unmatched_detections.remove(best_match_idx)

        # --- 3. Re-Identify and Match Lost Tracks ---
        for i in list(unmatched_detections):  # Iterate over a copy
            new_det = current_detections[i]
            best_lost_match_id, min_score = -1, float('inf')

            for track_id, track_data in lost_tracks.items():
                if new_det.class_name == track_data["class_name"]:
                    # Use center-to-center distance instead of top-left corner
                    old_center = (track_data["box_rec"].cx, track_data["box_rec"].cy)
                    new_center = (new_det.cx, new_det.cy)
                    center_dist = np.linalg.norm(np.array(new_center) - np.array(old_center))
                    
                    # Tighter re-identification threshold
                    reid_threshold = REID_DISTANCE_THRESHOLD_FACTOR * min(track_data["box_rec"].width,
                                                                          track_data["box_rec"].height)
                    
                    # Size consistency check for re-identification
                    size_ratio = min(new_det.width / track_data["box_rec"].width,
                                   new_det.height / track_data["box_rec"].height)
                    size_consistency = size_ratio > 0.4 and size_ratio < 2.5
                    
                    # Combined score: distance + size consistency
                    if center_dist < reid_threshold and size_consistency:
                        # Lower score is better (distance-based)
                        score = center_dist + (1.0 - min(size_ratio, 1.0/size_ratio)) * 50
                        if score < min_score:
                            min_score = score
                            best_lost_match_id = track_id

            if best_lost_match_id != -1:
                if logger:
                    logger.debug(f"Re-identified track {best_lost_match_id} with new detection (score: {min_score:.2f}).")
                new_det.track_id = best_lost_match_id
                # Reactivate the track
                reactivated_track = lost_tracks.pop(best_lost_match_id)
                reactivated_track["box_rec"] = new_det
                reactivated_track["frames_unseen"] = 0
                active_tracks[best_lost_match_id] = reactivated_track
                unmatched_detections.remove(i)

        # --- 4. Create New Tracks (with limits and validation) ---
        new_tracks_created_this_frame = 0
        for i in unmatched_detections:
            if new_tracks_created_this_frame >= MAX_NEW_TRACKS_PER_FRAME:
                if logger:
                    logger.debug(f"Reached max new tracks limit ({MAX_NEW_TRACKS_PER_FRAME}) for frame {frame_obj.frame_id}")
                break
                
            det_box = current_detections[i]
            
            # Validate new track creation - avoid creating tracks for noise
            if det_box.confidence < 0.4:  # Minimum confidence for new tracks
                continue
                
            # Check for similarity to recently deleted tracks to avoid immediate recreation
            skip_creation = False
            recent_frame_threshold = max(1, int(fps * 0.25))  # Check last 0.25 seconds
            
            for recent_frame in range(max(0, frame_obj.frame_id - recent_frame_threshold), frame_obj.frame_id):
                # This would require tracking recently deleted tracks - simplified for now
                pass
            
            if not skip_creation:
                det_box.track_id = next_global_track_id
                active_tracks[next_global_track_id] = {
                    "box_rec": det_box,
                    "frames_unseen": 0,
                    "class_name": det_box.class_name,
                    "creation_frame": frame_obj.frame_id,
                    "creation_confidence": det_box.confidence
                }
                new_tracks_created_this_frame += 1
                next_global_track_id += 1

    if logger:
        logger.debug(f"Resilient tracking complete. Assigned up to global_track_id {next_global_track_id - 1}.")


# --- Stage 2 Analysis Steps ---
def pass_1_interpolate_boxes(app, frames: List, video_info: Dict, logger: Optional[logging.Logger]):
    if logger:
        logger.debug("Starting Stage 2 Pass 1: Interpolate Boxes (using S2 Tracker IDs)")
    track_history: Dict[int, Dict[str, Any]] = {}
    total_interpolated = 0
    MAX_GAP_FRAMES = int(video_info.get('fps', 30) * 1)  # Max 1 second gap for interpolation

    all_frames_sorted = sorted(frames, key=lambda f: f.frame_id)

    for frame_obj in all_frames_sorted:
        current_boxes_in_frame = list(frame_obj.boxes)

        for box_rec in current_boxes_in_frame:
            if box_rec.track_id is None or box_rec.track_id == -1:
                continue

            if box_rec.track_id in track_history:
                history_entry = track_history[box_rec.track_id]
                last_frame_obj: FrameObject = history_entry["last_frame_obj"]
                last_box_rec: BoxRecord = history_entry["last_box_rec"]

                frame_gap = frame_obj.frame_id - last_frame_obj.frame_id

                if 1 < frame_gap <= MAX_GAP_FRAMES:
                    # Interpolate for frames between last_frame_obj.frame_id+1 and frame_obj.frame_id-1
                    for i_frame_id in range(last_frame_obj.frame_id + 1, frame_obj.frame_id):
                        if i_frame_id >= len(all_frames_sorted):
                            continue

                        target_gap_frame_obj = all_frames_sorted[i_frame_id]

                        # Check if this track_id already exists in the target_gap_frame_obj (e.g. from other interpolation direction)
                        # This simplistic check might not be perfect for complex scenarios.
                        if any(b.track_id == box_rec.track_id for b in target_gap_frame_obj.boxes):
                            # Skip interpolation if track already exists in frame
                            continue

                        t = (i_frame_id - last_frame_obj.frame_id) / float(frame_gap)

                        interp_bbox_np = last_box_rec.bbox + t * (box_rec.bbox - last_box_rec.bbox)

                        # Basic check: distance moved by center point for interpolated box
                        center_interp_x = (interp_bbox_np[0] + interp_bbox_np[2]) / 2.0
                        center_interp_y = (interp_bbox_np[1] + interp_bbox_np[3]) / 2.0

                        delta_x_sq = (center_interp_x - last_box_rec.cx) ** 2
                        delta_y_sq = (center_interp_y - last_box_rec.cy) ** 2

                        dist_moved_center_sq = delta_x_sq + delta_y_sq

                        # Ensure the argument to sqrt is not negative due to potential float precision issues
                        if dist_moved_center_sq < 0:
                            dist_moved_center_sq = 0

                        dist_moved_center = math.sqrt(dist_moved_center_sq)

                        max_interp_move_thresh = max(last_box_rec.width,
                                                     last_box_rec.height) * frame_gap * 0.75  # Increased multiplier slightly for flexibility

                        if dist_moved_center < max_interp_move_thresh:
                            interpolated_br = BoxRecord(
                                frame_id=i_frame_id,
                                bbox=interp_bbox_np,
                                confidence=min(last_box_rec.confidence, box_rec.confidence) * 0.8,  # Reduced conf
                                class_id=box_rec.class_id,  # Assume class is consistent
                                class_name=box_rec.class_name,
                                status=constants.STATUS_INTERPOLATED,  # Mark as interpolated
                                yolo_input_size=frame_obj.yolo_input_size,
                                track_id=box_rec.track_id
                            )
                            target_gap_frame_obj.boxes.append(interpolated_br)
                            total_interpolated += 1

            track_history[box_rec.track_id] = {"last_frame_obj": frame_obj, "last_box_rec": box_rec}
    if logger:
        logger.debug(f"Stage 2 Pass 1: Interpolated {total_interpolated} boxes.")


def smooth_single_track_worker_numpy(track_data_tuple: Tuple[np.ndarray, int]) -> Optional[np.ndarray]:
    """
    Enhanced NumPy-native worker with improved temporal size smoothing.
    It receives a NumPy array for a track, smoothes it with RTS and temporal size smoothing,
    and returns the smoothed data along with the original track ID.
    """
    track_array, track_id = track_data_tuple

    try:
        # Run the high-performance RTS smoother using the new class
        smoother = RTSSmoother()
        smoothed_coords = smoother.smooth_trajectory(track_array)

        # Apply enhanced temporal size smoothing with frame-rate adaptation
        frame_ids = track_array[:, 0]
        final_coords = smoother.apply_temporal_size_smoothing(
            smoothed_coords, frame_ids, fps=30.0  # TODO: Get FPS from state
        )

        # Return the final smoothed cx, cy, w, h and the track_id to map it back
        return final_coords, track_id
    except Exception:
        # If smoothing fails for any reason, return None
        return None

# numpy
def pass_1b_smooth_all_tracks(app, frames: List, video_info: Dict, logger: Optional[logging.Logger]):
    """
    FULLY OPTIMIZED: Uses a NumPy-centric data structure and multiprocessing
    to smooth all tracks at maximum speed.
    """
    return

    if logger:
        logger.debug("Starting FULLY OPTIMIZED Stage 2 Pass 1b: Smooth All Tracked Boxes")

    # --- 1. One-time conversion to a unified NumPy array ---
    box_list_for_np = []
    for frame in frames:
        for box in frame.boxes:
            if box.track_id is not None:
                box_list_for_np.append([frame.frame_id, box.track_id, box.cx, box.cy, box.width, box.height])

    if not box_list_for_np:
        if logger:
            logger.debug("No tracked boxes to smooth.")
        return

    master_array = np.array(box_list_for_np, dtype=np.float32)
    unique_track_ids = np.unique(master_array[:, 1])

    # --- 2. Prepare arguments for parallel workers ---
    worker_args = []
    for track_id in unique_track_ids:
        track_array = master_array[master_array[:, 1] == track_id]
        # Ensure track is sorted by frame ID
        track_array = track_array[track_array[:, 0].argsort()]
        if len(track_array) < 5: continue
        worker_args.append((track_array, int(track_id)))

    # --- 3. Execute in parallel ---
    num_workers = psutil.cpu_count(logical=False)
    if logger:
        logger.debug(f"Distributing {len(worker_args)} tracks to {num_workers} NumPy-native workers...")

    all_smoothed_data = []
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance with uneven task times
        for result in pool.imap_unordered(smooth_single_track_worker_numpy, worker_args):
            if result is not None:
                all_smoothed_data.append(result)

    # --- 4. Integrate results back into the original FrameObjects ---
    if logger:
        logger.debug("Integrating smoothed results...")
    # Create a fast lookup: {track_id: {frame_id: smoothed_box_data}}
    results_map = {}
    for smoothed_track, track_id in all_smoothed_data:
        frame_ids_for_track = worker_args[track_id - 1][0][:, 0].astype(int)  # Get frame_ids from original args
        results_map[track_id] = {fid: smoothed_track[i] for i, fid in enumerate(frame_ids_for_track)}

    for frame in frames:
        for box in frame.boxes:
            if box.track_id in results_map and frame.frame_id in results_map[box.track_id]:
                smoothed_data = results_map[box.track_id][frame.frame_id]
                cx, cy, w, h = smoothed_data
                new_bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                box.update_bbox(new_bbox, new_status=constants.STATUS_SMOOTHED)

    if logger:
        logger.debug("Fully optimized smoothing pass complete.")

def pass_2_preliminary_height_estimation(app, frames: List, video_info: Dict, vr_vertical_third_filter: bool, logger: Optional[logging.Logger]):
    """
    A new, lightweight pass to provide a preliminary max_height estimate for each frame.
    This is crucial for the Kalman filter pass (Pass 3) to function correctly before
    the final, per-chapter height is calculated later in Pass 5.
    """
    if logger:
        logger.debug("Starting Stage 2 Pass 2: Preliminary Height Estimation")

    # We use a simple moving average to provide a slightly more stable preliminary height
    # than just the raw detection of a single frame.
    window_size = 5  # A small 5-frame window
    heights_buffer = []

    for frame_obj in frames:
        # Find the best penis detection for the current frame
        selected_penis = frame_obj.get_preferred_penis_box(
            video_info.get('actual_video_type', '2D'),
            vr_vertical_third_filter
        )

        current_height = 0.0
        if selected_penis:
            current_height = selected_penis.height

        # Add the current height (or 0 if no detection) to the buffer
        heights_buffer.append(current_height)
        if len(heights_buffer) > window_size:
            heights_buffer.pop(0)

        # Calculate the average of non-zero heights in the buffer
        non_zero_heights = [h for h in heights_buffer if h > 0]
        if non_zero_heights:
            avg_height = sum(non_zero_heights) / len(non_zero_heights)
            frame_obj.locked_penis_state.max_height = avg_height
        else:
            # If no detections in the window, keep the last known average or default to 0
            if not hasattr(frame_obj.locked_penis_state, 'max_height'):
                frame_obj.locked_penis_state.max_height = 0.0

    if logger:
        logger.debug("Preliminary height estimation complete.")


def pass_3_kalman_and_lock_state(app, frames: List, video_info: Dict, yolo_input_size: int, vr_vertical_third_filter: bool, logger: Optional[logging.Logger]):
    """
    REFACTORED: This pass now only manages the lock state (active/inactive) and
    calculates the Kalman-filtered VISIBLE penis box. It no longer creates the
    full conceptual box, as the final max_height is not yet known.
    """
    if logger:
        logger.debug("Starting Stage 2 Pass 3: Kalman Filter and Lock State")
    fps, yolo_size = video_info.get('fps', 30.0), yolo_input_size
    # --- State variables for the loop ---
    current_lp_active = False
    current_lp_last_raw_box_coords: Optional[Tuple[float, ...]] = None
    current_lp_consecutive_detections = 0
    current_lp_consecutive_non_detections = 0
    last_known_interaction_type = 'unknown'  # Tracks context: 'penetration', 'other', or 'unknown'
    last_known_penis_box: Optional[BoxRecord] = None

    # --- Kalman Filter Setup ---
    kf_height = cv2.KalmanFilter(2, 1)
    kf_height.measurementMatrix = np.array([[1, 0]], np.float32)
    kf_height.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
    kf_height.processNoiseCov = np.eye(2, dtype=np.float32) * 0.01
    kf_height.measurementNoiseCov = np.eye(1, dtype=np.float32) * 0.1
    kf_height.errorCovPost = np.eye(2, dtype=np.float32) * 1.0
    kf_height.statePost = np.array([[yolo_size * 0.1], [0]], dtype=np.float32)

    last_frame_dominant_pose: Optional[PoseRecord] = None
    is_vr = video_info.get('actual_video_type', '2D') == 'VR'
    pelvis_zone_indices = [11, 12]  # Left and Right Hip

    # --- Global state for locked penis ---
    locked_penis_tracker = {'box_rec': None, 'unseen_frames': 0}
    PENIS_PATIENCE = int(fps * 0.5)  # How many frames to hold onto a lock without seeing it

    for frame_obj in frames:
        dominant_pose_this_frame = _get_dominant_pose(frame_obj, is_vr, yolo_input_size)
        if dominant_pose_this_frame:
            frame_obj.dominant_pose_id = dominant_pose_this_frame.id

        penis_candidates = [b for b in frame_obj.boxes if
                            b.class_name == constants.PENIS_CLASS_NAME and not b.is_excluded]
        selected_penis_box_rec = None

        # --- Stable Penis Selection Logic ---
        # 1. Try to find the previously locked penis via IoU
        if locked_penis_tracker['box_rec']:
            best_iou = 0
            best_candidate = None
            for cand in penis_candidates:
                iou = _calculate_iou(locked_penis_tracker['box_rec'].bbox, cand.bbox)
                if iou > best_iou:
                    best_iou = iou
                    best_candidate = cand
            if best_iou > 0.1:  # Generous IoU threshold to maintain lock
                selected_penis_box_rec = best_candidate
                locked_penis_tracker['box_rec'] = selected_penis_box_rec
                locked_penis_tracker['unseen_frames'] = 0
            else:
                locked_penis_tracker['unseen_frames'] += 1

        # 2. If no IoU match, get the preferred (new) penis
        if not selected_penis_box_rec:
            preferred_penis = frame_obj.get_preferred_penis_box(video_info.get('actual_video_type', '2D'),
                                                                vr_vertical_third_filter)
            if preferred_penis:
                # Only switch if the old lock is lost or this new one is significantly better
                if locked_penis_tracker['unseen_frames'] > PENIS_PATIENCE or not locked_penis_tracker['box_rec']:
                    selected_penis_box_rec = preferred_penis
                    locked_penis_tracker['box_rec'] = selected_penis_box_rec
                    locked_penis_tracker['unseen_frames'] = 0

        # Reset lock if patient has run out
        if locked_penis_tracker['box_rec'] and locked_penis_tracker['unseen_frames'] > PENIS_PATIENCE:
            locked_penis_tracker['box_rec'] = None

        # --- CONTEXT-AWARE LOCK AND INFERENCE LOGIC (uses the now stable selected_penis_box_rec) ---
        if selected_penis_box_rec:
            last_known_penis_box = selected_penis_box_rec
            current_lp_consecutive_detections += 1
            current_lp_consecutive_non_detections = 0
            current_lp_last_raw_box_coords = tuple(selected_penis_box_rec.bbox)
            current_raw_height = selected_penis_box_rec.height

            # Update the interaction context based on current contact
            has_penetration_contact = False
            has_other_contact = False
            penetration_classes = {'pussy', 'butt', 'anus'}
            other_classes = {'hand', 'face', 'breast', 'foot'}
            for box_rec in frame_obj.boxes:
                if _calculate_iou(selected_penis_box_rec.bbox, box_rec.bbox) > 0.001:
                    if box_rec.class_name in penetration_classes:
                        has_penetration_contact = True
                        break
                    elif box_rec.class_name in other_classes:
                        has_other_contact = True

            if has_penetration_contact:
                last_known_interaction_type = 'penetration'
            elif has_other_contact:
                last_known_interaction_type = 'other'

            # Activate lock if needed
            if not current_lp_active and current_lp_consecutive_detections >= fps / 5:
                current_lp_active = True
                kf_height.statePost = np.array([[current_raw_height], [0]], dtype=np.float32)
                if logger:
                    logger.debug(f"Stage 2 LP Lock ACTIVATE at frame {frame_obj.frame_id}")

            kf_height.correct(np.array([[current_raw_height]], dtype=np.float32))

        else:  # NO penis detected in this frame
            current_lp_consecutive_detections = 0
            pose_is_stable = False
            # Check for pose stability to hold the lock during brief occlusions
            if frame_obj.poses and current_lp_active and dominant_pose_this_frame and last_frame_dominant_pose:
                dissimilarity = dominant_pose_this_frame.calculate_zone_dissimilarity(last_frame_dominant_pose,
                                                                                      pelvis_zone_indices)
                if dissimilarity < constants.POSE_STABILITY_THRESHOLD:
                    pose_is_stable = True

            # If no box was inferred, this is a true non-detection frame.
            if not selected_penis_box_rec:
                if not pose_is_stable:  # Only increment non-detections if pose is also unstable
                    current_lp_consecutive_non_detections += 1

            if current_lp_active and current_lp_consecutive_non_detections >= (fps * 2):
                current_lp_active = False
                last_known_interaction_type = 'unknown'  # Reset context on lock deactivation

        # --- Update the frame state ---
        lp_state = frame_obj.locked_penis_state
        lp_state.active = current_lp_active
        # Use the box from the stable tracker for last raw coords
        if last_known_penis_box:
            lp_state.last_raw_coords = tuple(last_known_penis_box.bbox)

        if current_lp_active and lp_state.last_raw_coords:
            predicted_height_kalman = kf_height.predict()[0, 0]
            x1, _, x2, y2_raw = lp_state.last_raw_coords
            frame_obj.penis_box_kalman = (x1, y2_raw - predicted_height_kalman, x2, y2_raw)
        else:
            frame_obj.penis_box_kalman = None

        last_frame_dominant_pose = dominant_pose_this_frame


def _update_frames_with_aggregated_positions(frames: List[FrameObject], aggregated_segments: List[Segment], logger: Optional[logging.Logger]):
    """
    Update individual frames to reflect their final aggregated segment position.
    This ensures the chapter bar displays correctly based on final segments rather than individual frame detections.
    """
    if logger:
        logger.debug(f"Updating {len(frames)} frames with aggregated segment positions from {len(aggregated_segments)} segments")
    
    updated_count = 0
    for frame_obj in frames:
        frame_id = frame_obj.frame_id
        original_position = frame_obj.assigned_position
        
        # Find which aggregated segment this frame belongs to
        for segment in aggregated_segments:
            if segment.start_frame_id <= frame_id <= segment.end_frame_id:
                # Update frame position to match the aggregated segment
                frame_obj.assigned_position = segment.major_position
                
                # Debug logging for significant changes
                if original_position != segment.major_position:
                    if updated_count < 10:  # Limit debug spam - show first 10 changes
                        if logger:
                            logger.debug(f"Frame {frame_id}: updated position '{original_position}' -> '{segment.major_position}' (segment {segment.start_frame_id}-{segment.end_frame_id})")
                    updated_count += 1
                break
        else:
            # Frame not found in any segment - this shouldn't happen but log it
            if logger:
                logger.warning(f"Frame {frame_id} not found in any aggregated segment")
    
    if logger:
        logger.debug(f"Updated {updated_count} frames with aggregated positions")


def pass_4_assign_positions_and_segments(app, frames: List, segments: List, video_info: Dict, yolo_input_size: int, logger: Optional[logging.Logger]):
    """ Stage 2 Pass 4: Assign frame positions and aggregate into segments (with Sparse Pose Persistence). """
    if logger:
        logger.debug("Starting Stage 2 Pass 4: Assign Positions & Segments (with Sparse Pose Persistence)")
    fps = video_info.get('fps', 30.0)
    is_vr = video_info.get('actual_video_type', '2D') == 'VR'
    yolo_size = yolo_input_size

    # --- State variables for persistence with sparse data ---
    last_known_good_position: str = "Not Relevant"
    last_known_good_pose: Optional[PoseRecord] = None
    most_recent_dominant_pose: Optional[PoseRecord] = None
    # ---

    for frame_obj in frames:
        # 1. ALWAYS check for new pose data to carry forward
        if frame_obj.poses:
            most_recent_dominant_pose = _get_dominant_pose(frame_obj, is_vr, yolo_size)

        # 2. Run original logic based on YOLO contact boxes
        frame_obj.detected_contact_boxes.clear()
        assigned_pos_for_frame = "Not Relevant"

        # Use conceptual locked penis box for contact detection (full interaction range)
        if frame_obj.locked_penis_state.active and frame_obj.locked_penis_state.last_raw_coords:
            lp_state = frame_obj.locked_penis_state
            
            # Calculate preliminary conceptual box if max_height is available
            if hasattr(lp_state, 'max_height') and lp_state.max_height > 0:
                x1, _, x2, y2_raw = lp_state.last_raw_coords
                conceptual_penis_coords = (x1, y2_raw - lp_state.max_height, x2, y2_raw)
            else:
                # Fallback: use visible box if conceptual box can't be calculated
                if frame_obj.penis_box_kalman:
                    conceptual_penis_coords = frame_obj.penis_box_kalman
                else:
                    # Last resort: use raw coordinates with default height expansion
                    x1, y1, x2, y2 = lp_state.last_raw_coords
                    height = y2 - y1
                    expanded_height = height * 2.0  # Assume stroke is 2x visible penis height
                    conceptual_penis_coords = (x1, y2 - expanded_height, x2, y2)
            
            for box_rec in frame_obj.boxes:
                if box_rec.is_excluded or box_rec.class_name in [constants.PENIS_CLASS_NAME, constants.GLANS_CLASS_NAME]:
                    continue
                # IMPROVED: Use conceptual penis box and require meaningful contact
                iou = _calculate_iou(conceptual_penis_coords, box_rec.bbox)
                if iou > 0.05:  # Require at least 5% overlap for meaningful contact
                    frame_obj.detected_contact_boxes.append({"class_name": box_rec.class_name, "box_rec": box_rec})
            assigned_pos_for_frame = _assign_frame_position(frame_obj.detected_contact_boxes, logger)

        # 3. Apply pose persistence logic
        if assigned_pos_for_frame != "Not Relevant":
            # Confident detection from YOLO. Update our memory.
            last_known_good_position = assigned_pos_for_frame
            # Anchor the "good" pose to the most recent one we've seen.
            last_known_good_pose = most_recent_dominant_pose

        elif last_known_good_position != "Not Relevant" and most_recent_dominant_pose and last_known_good_pose:
            # YOLO detection failed. Try to use pose memory to fill the gap.
            zone_indices = constants.INTERACTION_ZONES.get(last_known_good_position, [])

            if zone_indices:
                # Compare the most recent pose we have against our "anchor" pose for this action.
                dissimilarity = most_recent_dominant_pose.calculate_zone_dissimilarity(
                    last_known_good_pose, zone_indices
                )

                if dissimilarity < constants.POSE_STABILITY_THRESHOLD:
                    # The overall posture hasn't changed. Maintain the classification.
                    assigned_pos_for_frame = last_known_good_position
                    if logger:
                        logger.debug(
                               f"Frame {frame_obj.frame_id}: Sparse Pose persistence applied. Position '{last_known_good_position}' maintained.")

        # 4. Finalize the frame's position
        # DEBUG: Log position assignment for first few frames
        if frame_obj.frame_id < 10:
            if logger:
                logger.debug(f"Frame {frame_obj.frame_id}: assigned_pos_for_frame = '{assigned_pos_for_frame}' (type: {type(assigned_pos_for_frame)})")
        
        frame_obj.assigned_position = assigned_pos_for_frame
        
        # DEBUG: Verify what was actually stored
        if frame_obj.frame_id < 10:
            if logger:
                logger.debug(f"Frame {frame_obj.frame_id}: stored assigned_position = '{frame_obj.assigned_position}' (type: {type(frame_obj.assigned_position)})")

    # 5. Aggregate segments (this part of the function is unchanged)
    BaseSegment._id_counter = 0
    segments[:] = _aggregate_segments(frames, fps, int(1.0 * fps), logger)
    
    # 5.1 Update individual frames to reflect their final aggregated segment position
    # This ensures the chapter bar displays correctly based on final segments
    _update_frames_with_aggregated_positions(frames, segments, logger)

def pass_5_recalculate_heights_post_aggregation(app, frames: List, segments: List, video_info: Dict, yolo_input_size: int, vr_vertical_third_filter: bool, logger: Optional[logging.Logger]):
    """
    Recalculates max_height for the penis based on the final, aggregated chapter segments.
    This ensures height is consistent across a logical action, fixing the "drop to zero" issue.
    This pass should run AFTER pass_4_assign_positions_and_segments.
    """
    if logger:
        logger.debug("Starting Stage 2 Pass: Recalculate Heights Post-Aggregation")

    # --- Reset penis lock at the start of each new major segment ---
    # This prevents the lock from carrying over across completely different scenes
    # which was a key part of the user request.
    for i in range(len(segments)):
        segment = segments[i]
        # On the first frame of any segment that isn't the very first one...
        if i > 0:
            first_frame_of_segment_id = segment.start_frame_id
            if first_frame_of_segment_id < len(frames):
                # This is a conceptual reset. The actual lock state is managed in Pass 3.
                # Here we ensure the logic in Pass 6, which depends on this, knows about the change.
                if logger:
                    logger.debug(f"Chapter changed at frame {first_frame_of_segment_id}. Resetting interactor context for segment '{segment.major_position}'.")

    # Create a quick lookup for raw penis heights from the original detections
    raw_penis_heights = {}
    video_type = video_info.get('actual_video_type', '2D')
    vr_filter = vr_vertical_third_filter
    for frame_obj in frames:
        # We need the original, non-interpolated penis box for its height
        penis_detections = [b for b in frame_obj.boxes if b.class_name == constants.PENIS_CLASS_NAME and b.status == constants.STATUS_DETECTED]
        if penis_detections:
            # Sort by confidence or area to get the most likely "real" penis
            penis_detections.sort(key=lambda d: d.confidence, reverse=True)
            raw_penis_heights[frame_obj.frame_id] = penis_detections[0].height

    # Create a frame-to-object map for quick updates
    frames_by_id = {f.frame_id: f for f in frames}

    # Iterate through the final, aggregated segments
    for segment in segments:
        if segment.major_position == "Not Relevant":
            chapter_max_h = 0.0
        else:
            # Collect all raw penis heights that fall within this final segment
            heights_in_segment = [h for fid, h in raw_penis_heights.items() if segment.start_frame_id <= fid <= segment.end_frame_id]

            if not heights_in_segment:
                # If no raw detections exist in this segment (e.g., fully interpolated),
                # we must fall back to a reasonable default or carry over from a previous segment.
                # For now, a fallback based on input size is safer than 0.
                chapter_max_h = yolo_input_size * 0.2
            else:
                # Use 95th percentile as a robust statistic for max height
                chapter_max_h = np.percentile(heights_in_segment, 95)

        if logger:
            logger.debug(f"Final Segment '{segment.major_position}' ({segment.start_frame_id}-{segment.end_frame_id}): setting max_height to {chapter_max_h:.2f}")

        # Apply this calculated height to every frame object within the segment
        for i in range(segment.start_frame_id, segment.end_frame_id + 1):
            if i in frames_by_id:
                lp_state = frames_by_id[i].locked_penis_state
                lp_state.max_height = chapter_max_h
                # Set a reasonable penetration height based on the final max height
                lp_state.max_penetration_height = chapter_max_h * 0.65


def pass_6_determine_distance(app, frames: List, segments: List, video_info: Dict, yolo_input_size: int, logger: Optional[logging.Logger]):
    if logger:
        logger.debug("Starting Stage 2 Pass 6: Determine Frame Distances (IMPROVED FALLBACK CONTINUITY)")
    fps = video_info.get('fps', 30.0)
    yolo_size = yolo_input_size
    is_vr = video_info.get('actual_video_type', '2D') == 'VR'

    # --- Pre-fetch raw penis heights for efficient lookup ---
    raw_penis_heights = {}
    for frame_obj in frames:
        penis_detections = [b for b in frame_obj.boxes if b.class_name == constants.PENIS_CLASS_NAME and b.status == constants.STATUS_DETECTED]
        if penis_detections:
            penis_detections.sort(key=lambda d: d.confidence, reverse=True)
            raw_penis_heights[frame_obj.frame_id] = penis_detections[0].height

    if not raw_penis_heights:
        if logger:
            logger.debug("No raw penis heights found to process. Skipping distance calculation.")
        return

    # Create sorted NumPy arrays for lightning-fast lookups
    sorted_fids = np.array(sorted(raw_penis_heights.keys()))
    sorted_heights = np.array([raw_penis_heights[fid] for fid in sorted_fids])

    ROLLING_WINDOW_SECONDS = 30
    rolling_window_frames = int(fps * ROLLING_WINDOW_SECONDS)

    last_known_box_positions = {}

    for segment in segments:
        # (The segment setup and interactor locking logic remains the same as your last version)
        primary_classes_for_segment: List[str] = []
        fallback_classes_for_segment: List[str] = []
        is_penetration_pos = False
        if segment.major_position == 'Cowgirl / Missionary':
            primary_classes_for_segment = ['pussy']
            fallback_classes_for_segment = ['navel', 'breast', 'face']
            is_penetration_pos = True
        elif segment.major_position == 'Rev. Cowgirl / Doggy':
            primary_classes_for_segment = ['butt']
            fallback_classes_for_segment = ['anus', 'face']
            is_penetration_pos = True
        elif segment.major_position == 'Blowjob':
            primary_classes_for_segment = ['face', 'hand']
            # fallback_classes_for_segment = ['hand']
        elif segment.major_position == 'Handjob':
            primary_classes_for_segment = ['hand']
        elif segment.major_position == 'Boobjob':
            primary_classes_for_segment = ['breast']
            fallback_classes_for_segment = ['hand']
        elif segment.major_position == 'Footjob':
            primary_classes_for_segment = ['foot']
        if logger:
            logger.debug(
                   f"Processing segment '{segment.major_position}': Primary={primary_classes_for_segment}, Fallback={fallback_classes_for_segment}")
        active_interactor_state = {'id': None, 'unseen_frames': 0}
        locked_fallback_interactors = {}
        FALLBACK_PATIENCE = int(fps * 0.5)
        INTERACTOR_PATIENCE = int(fps * 0.75)
        last_valid_distance = 100
        
        # IMPROVED: Rolling distance tracking for smooth fallback transitions
        fallback_rolling_distances = {}  # {class_name: [distances...]}
        primary_class = primary_classes_for_segment[0] if primary_classes_for_segment else 'unknown'
        
        segment.segment_frame_objects = [fo for fo in frames if segment.start_frame_id <= fo.frame_id <= segment.end_frame_id]

        for frame_obj in segment.segment_frame_objects:
            lp_state = frame_obj.locked_penis_state
            frame_obj.is_occluded = False
            frame_obj.active_interaction_track_id = None
            frame_obj.fallback_contributor_ids = []

            if not lp_state.active or not lp_state.last_raw_coords:
                frame_obj.funscript_distance = 100
                continue

            # (Conceptual box and interactor locking logic remains the same)
            x1, _, x2, y2_raw = lp_state.last_raw_coords
            conceptual_full_box = (x1, y2_raw - lp_state.max_height, x2, y2_raw)
            lp_state.box = conceptual_full_box
            if frame_obj.penis_box_kalman:
                conceptual_y1 = conceptual_full_box[1]
                kalman_x1, kalman_y1, kalman_x2, kalman_y2 = frame_obj.penis_box_kalman
                clamped_y1 = max(kalman_y1, conceptual_y1)
                clamped_y1 = min(clamped_y1, kalman_y2)
                frame_obj.penis_box_kalman = (kalman_x1, clamped_y1, kalman_x2, kalman_y2)
            if frame_obj.penis_box_kalman and lp_state.max_height > 0:
                visible_height = frame_obj.penis_box_kalman[3] - frame_obj.penis_box_kalman[1]
                lp_state.visible_part = (visible_height / lp_state.max_height) * 100.0
            else:
                lp_state.visible_part = 0.0
            contacting_boxes_all = [c['box_rec'] for c in frame_obj.detected_contact_boxes]
            active_box = None
            locked_id = active_interactor_state['id']
            if locked_id is not None:
                # IMPROVED: Check if the tracked object is still making meaningful contact
                robustly_found_interactor = next((b for b in contacting_boxes_all if b.track_id == locked_id), None)

                if robustly_found_interactor:
                    # Verify the contact is still meaningful (not just barely touching)
                    contact_iou = _calculate_iou(conceptual_full_box, robustly_found_interactor.bbox)
                    if contact_iou > 0.05:  # Require meaningful contact to maintain lock
                        active_box = robustly_found_interactor
                        active_interactor_state['unseen_frames'] = 0
                        frame_obj.is_occluded = False
                    else:
                        # Object is present but contact is too weak - consider it occluded
                        active_interactor_state['unseen_frames'] += 1
                        frame_obj.is_occluded = True
                else:
                    active_interactor_state['unseen_frames'] += 1
                    if active_interactor_state['unseen_frames'] > INTERACTOR_PATIENCE:
                        active_interactor_state['id'] = None
                        active_box = None
            if active_box is None:
                frame_obj.is_occluded = True
                primary_contacts = [b for b in contacting_boxes_all if b.class_name in primary_classes_for_segment]
                if primary_contacts:
                    def contact_quality_score(box_rec):
                        # Calculate contact quality based on IoU, confidence, and movement  
                        iou = _calculate_iou(conceptual_full_box, box_rec.bbox)
                        
                        # Movement bonus: prefer objects that are moving (indicates active interaction)
                        is_moving = (box_rec.track_id not in last_known_box_positions or
                                   (abs(box_rec.cx - last_known_box_positions[box_rec.track_id][0]) >= 1 or
                                    abs(box_rec.cy - last_known_box_positions[box_rec.track_id][1]) >= 1))
                        movement_bonus = 0.3 if is_moving else 0.0
                        
                        # Combined score: IoU (50%), confidence (30%), movement (20%)
                        return iou * 0.5 + box_rec.confidence * 0.3 + movement_bonus * 0.2
                    
                    active_box = max(primary_contacts, key=contact_quality_score)
                    if active_box and active_box.track_id != active_interactor_state.get('id'):
                        active_interactor_state['id'] = active_box.track_id
                        active_interactor_state['unseen_frames'] = 0
                        frame_obj.is_occluded = False
                        if logger:
                            logger.debug(
                                   f"Frame {frame_obj.frame_id}: Locking on to new primary interactor TID {active_box.track_id} ('{active_box.class_name}')")
            final_contributors = []
            for fb_class in fallback_classes_for_segment:
                found_current_frame = False
                if fb_class in locked_fallback_interactors:
                    lock_info = locked_fallback_interactors[fb_class]
                    locked_box = next((b for b in frame_obj.boxes if b.track_id == lock_info['id']), None)
                    if locked_box:
                        final_contributors.append(locked_box)
                        lock_info['unseen_frames'] = 0
                        found_current_frame = True
                    else:
                        lock_info['unseen_frames'] += 1
                        if lock_info['unseen_frames'] > FALLBACK_PATIENCE:
                            del locked_fallback_interactors[fb_class]
                if not found_current_frame:
                    dominant_pose = _get_dominant_pose(frame_obj, is_vr, yolo_size)
                    alignment_ref_cx = lp_state.last_raw_coords[0] + (lp_state.last_raw_coords[2] - lp_state.last_raw_coords[0]) / 2
                    max_alignment_offset = yolo_size * 0.35
                    potential_fallbacks = [b for b in frame_obj.boxes if
                                           b.class_name in fallback_classes_for_segment and b != active_box]
                    aligned_fallback_candidates = _get_aligned_fallback_boxes(potential_fallbacks, dominant_pose, alignment_ref_cx, max_alignment_offset)
                    penis_base_y = conceptual_full_box[3]
                    locked_ids = {info['id'] for info in locked_fallback_interactors.values()}
                    best_new_candidate = sorted([
                        b for b in aligned_fallback_candidates if
                        b.class_name == fb_class and b.track_id not in locked_ids
                    ], key=lambda b: abs(b.cy - penis_base_y))
                    if best_new_candidate:
                        new_box_to_lock = best_new_candidate[0]
                        final_contributors.append(new_box_to_lock)
                        locked_fallback_interactors[fb_class] = {'id': new_box_to_lock.track_id, 'unseen_frames': 0}
                        if logger:
                            logger.debug(
                                   f"Frame {frame_obj.frame_id}: Locked new fallback '{fb_class}' to TID {new_box_to_lock.track_id}")
            if final_contributors:
                frame_obj.fallback_contributor_ids = [c.track_id for c in final_contributors if
                                                          c.track_id is not None]

            # --- 2. OPTIMIZED DYNAMIC AMPLIFICATION ---
            current_frame_id = frame_obj.frame_id
            window_start_frame = max(0, current_frame_id - rolling_window_frames)

            # Use binary search to find the slice indices instantly
            start_idx = np.searchsorted(sorted_fids, window_start_frame, side='left')
            end_idx = np.searchsorted(sorted_fids, current_frame_id, side='right')

            heights_in_window = sorted_heights[start_idx:end_idx]

            dynamic_max_dist_ref = 0
            if heights_in_window.size > 0:
                dynamic_max_dist_ref = np.percentile(heights_in_window, 95)
            else:
                dynamic_max_dist_ref = lp_state.max_height if lp_state.max_height > 0 else yolo_size * 0.3

            # (Weighted distance calculation using dynamic_max_dist_ref is unchanged)
            total_weighted_distance = 0.0
            total_weight = 0.0
            max_dist_ref = dynamic_max_dist_ref # Use the newly calculated dynamic reference

            if active_box:
                dist = _calculate_normalized_distance_to_base(conceptual_full_box, active_box.class_name, active_box.bbox, max_dist_ref)
                total_weighted_distance += dist * 1.0
                total_weight += 1.0
                frame_obj.active_interaction_track_id = active_box.track_id

            for contributor in final_contributors:
                if active_box and contributor.track_id == active_box.track_id:
                    continue
                
                # IMPROVED: Use absolute distance for fallback classes to maintain signal continuity
                if contributor.class_name not in fallback_rolling_distances:
                    fallback_rolling_distances[contributor.class_name] = []
                
                dist = _calculate_fallback_absolute_distance(
                    conceptual_full_box, 
                    contributor.class_name, 
                    contributor.bbox,
                    primary_class,
                    fallback_rolling_distances[contributor.class_name]
                )
                
                # IMPROVED: Reduced fallback weights to prevent signal spikes
                if contributor.class_name == 'breast':
                    weight = 0.8
                elif contributor.class_name == 'anus':
                    weight = 0.2 
                elif contributor.class_name == 'face':
                    weight = 0.1
                elif contributor.class_name == 'navel':
                    weight = 1.0
                else:
                    weight = 0.05
                
                total_weighted_distance += dist * weight
                total_weight += weight

            if total_weight > 0:
                final_distance = total_weighted_distance / total_weight
            else:
                final_distance = last_valid_distance

            # Store current distance for amplitude analysis
            current_distances = getattr(frame_obj, '_amplitude_window', [])
            current_distances.append(final_distance)
            
            # Keep a rolling window of distances for amplitude calculation
            AMPLITUDE_WINDOW = int(fps * 2.0)  # 2 seconds
            if len(current_distances) > AMPLITUDE_WINDOW:
                current_distances = current_distances[-AMPLITUDE_WINDOW:]
            frame_obj._amplitude_window = current_distances

            # IMPROVED: Amplitude-based signal enhancement for pussy interactions
            if primary_class == 'pussy' and active_box and active_box.class_name == 'pussy' and len(current_distances) >= 10:
                contact_iou = _calculate_iou(conceptual_full_box, active_box.bbox)
                if contact_iou > 0.1:  # Good pussy contact detected
                    # Calculate current amplitude (range of movement)
                    min_dist = min(current_distances)
                    max_dist = max(current_distances)
                    current_amplitude = max_dist - min_dist
                    
                    # If amplitude is too low, enhance it while preserving the center point
                    MIN_DESIRED_AMPLITUDE = 25  # Minimum range we want to see
                    if current_amplitude < MIN_DESIRED_AMPLITUDE and current_amplitude > 5:
                        center_point = (min_dist + max_dist) / 2
                        enhancement_factor = MIN_DESIRED_AMPLITUDE / current_amplitude
                        enhancement_factor = min(enhancement_factor, 2.0)  # Cap at 2x enhancement
                        
                        # Expand around center point
                        deviation_from_center = final_distance - center_point
                        enhanced_deviation = deviation_from_center * enhancement_factor
                        final_distance = center_point + enhanced_deviation
                        
                        # Ensure we stay within valid bounds
                        final_distance = max(0, min(100, final_distance))

            frame_obj.funscript_distance = int(np.clip(round(final_distance), 0, 100))
            last_valid_distance = frame_obj.funscript_distance

            for box in (contacting_boxes_all + final_contributors):
                if box and box.track_id:
                    last_known_box_positions[box.track_id] = (box.cx, box.cy)

def pass_7_smooth_and_normalize_distances(app, frames: List, funscript_frames: List, funscript_distances: List, logger: Optional[logging.Logger]):
    """ Stage 2 Pass 7 (denoising with SG) and Pass 9 (amplifying/normalizing per segment) combined """
    if logger:
        logger.debug("Starting Stage 2 Pass 7: Smooth and Normalize Distances")

    # Vectorized implementation for better performance
    all_raw_distances = np.array([fo.funscript_distance for fo in frames])
    if not all_raw_distances.size or len(all_raw_distances) < 11:  # Savgol needs enough points
        if logger:
            logger.debug("Not enough data points for Savitzky-Golay filter.")
    else:
        # Apply Savitzky-Golay filter with vectorized operations
        smoothed_distances = savgol_filter(all_raw_distances, window_length=11, polyorder=2)
        # Vectorized clipping and rounding
        clipped_distances = np.clip(np.round(smoothed_distances), 0, 100).astype(int)
        
        # Apply smoothed values back to frame objects
        for i, fo in enumerate(frames):
            fo.funscript_distance = clipped_distances[i]
            
        if logger:
            logger.debug("Applied Savitzky-Golay filter to distances.")
    
    # Apply signal enhancement if enabled
    _apply_signal_enhancement(app, frames, logger)

    # Now apply Stage 2 per-segment normalization (_normalize_funscript_sparse equivalent)
    _normalize_funscript_sparse_per_segment(app, frames, [], logger)
    if logger:
        logger.debug("Applied per-segment normalization to distances.")

    # Vectorized extraction of frame data
    if frames:
        full_script_data_np = np.array(
            [[fo.frame_id, fo.funscript_distance] for fo in frames],
            dtype=np.float64
        )
        funscript_frames[:] = full_script_data_np[:, 0].astype(int).tolist()
        funscript_distances[:] = full_script_data_np[:, 1].astype(int).tolist()
    else:
        funscript_frames.clear()
        funscript_distances.clear()



def pass_8_simplify_signal(app, frames: List, funscript_frames: List, funscript_distances: List, funscript_distances_lr: List, video_info: Dict, logger: Optional[logging.Logger]):
    """
    ULTIMATE SOLUTION: A fully vectorized, high-performance signal simplification pipeline
    that minimizes data conversions and uses optimized algorithms.
    """
    if logger:
        logger.debug("Starting ULTIMATE Stage 2 Pass 8: High-Performance Signal Simplification")

    if not frames:
        if logger:
            logger.debug("No data to simplify.")
        return

    # --- Step 1: One-time conversion to a NumPy array ---
    full_script_data_np = np.array(
        [[fo.frame_id, fo.funscript_distance] for fo in frames],
        dtype=np.float64
    )

    if full_script_data_np.shape[0] < 3:
        funscript_frames[:] = full_script_data_np[:, 0].astype(int).tolist()
        funscript_distances[:] = full_script_data_np[:, 1].astype(int).tolist()
        return

    positions = full_script_data_np[:, 1]

    # --- Step 2: Peak/Valley Detection (Vectorized) ---

    peaks, _ = find_peaks(positions, prominence=1.0)
    valleys, _ = find_peaks(-positions, prominence=1.0) # For minima

    # Combine peak, valley, start, and end indices. `np.unique` also sorts the indices.
    key_indices = np.unique(np.concatenate(([0, len(positions) - 1], peaks, valleys)))

    # Extract only the keyframes based on these indices. The data remains a NumPy array.
    keyframes_np = full_script_data_np[key_indices]
    if logger:
        logger.debug(f"Pass 1 (Peak/Valley): Simplified to {len(keyframes_np)} points.")

    # --- Step 3: High-Performance RDP Simplification ---
    # Apply RDP simplification. The `simplify_coords_vw` is a C-based implementation
    # from the `simplification` library, which is very fast. Epsilon=2.0 is a reasonable value.
    if len(keyframes_np) > 2:
        final_simplified_np = simplify_coords_vw(keyframes_np, 2.0)
    else:
        final_simplified_np = keyframes_np
    if logger:
        logger.debug(f"Pass 2 (RDP): Simplified to {len(final_simplified_np)} points.")

    # --- Step 4: Final Conversion to Output Lists ---
    final_frames = final_simplified_np[:, 0].astype(int)
    final_positions = final_simplified_np[:, 1].astype(int)

    funscript_frames[:] = final_frames.tolist()
    funscript_distances[:] = final_positions.tolist()

    if logger:
        logger.debug(f"High-performance simplification complete. Final points: {len(funscript_frames)}")

def load_yolo_results_stage2(msgpack_file_path: str, stop_event: threading.Event, logger: logging.Logger) -> Optional[List]:
    if logger:
        logger.debug(f"Loading YOLO results from: {msgpack_file_path}")

    if stop_event.is_set():
        if logger:
            logger.info("Load YOLO stopped by event.")
        else:
            logger.warning("Load YOLO stopped by event.")
        return None
    try:
        with open(msgpack_file_path, 'rb') as f:
            packed_data = f.read()
        all_frames_raw_detections = msgpack.unpackb(packed_data, raw=False)
        # Loaded frames' raw detections
        if logger:
            logger.debug(f"Loaded {len(all_frames_raw_detections)} frames' raw detections.")

        return all_frames_raw_detections
    except Exception as e:
        if logger:
            logger.error(f"Error loading/unpacking msgpack {msgpack_file_path}: {e}", exc_info=True)
        else:
            logger.error(f"Error loading/unpacking msgpack {msgpack_file_path}: {e}")
        return None


def _pre_scan_for_interactor_ids(frames: List, video_info: Dict, vr_vertical_third_filter: bool, logger: Optional[logging.Logger]) -> set:
    """
    Performs a read-only pre-scan of all frames to identify the track IDs of all
    objects that ever interact with the penis. This avoids running expensive
    recovery on irrelevant tracks.
    """
    if logger:
        logger.debug("Pre-scanning to identify all potential interactor track IDs...")
    interactor_ids = set()

    # This scan simulates the lock-on logic from Pass 3 to accurately find interactors.
    # It does not modify any frame state.
    current_lp_active = False
    current_lp_consecutive_detections = 0
    current_lp_consecutive_non_detections = 0
    fps = video_info.get('fps', 30.0)

    for frame in frames:
        penis_box = frame.get_preferred_penis_box(
            video_info.get('actual_video_type', '2D'),
            vr_vertical_third_filter
        )

        # Simulate lock state
        if penis_box:
            current_lp_consecutive_detections += 1
            current_lp_consecutive_non_detections = 0
            if not current_lp_active and current_lp_consecutive_detections >= fps / 5:
                current_lp_active = True
        else:
            current_lp_consecutive_detections = 0
            current_lp_consecutive_non_detections += 1
            if current_lp_active and current_lp_consecutive_non_detections >= (fps * 2):
                current_lp_active = False

        # If locked, find contacts and add their track IDs
        if current_lp_active and penis_box:
            for other_box in frame.boxes:
                if other_box.track_id is not None and other_box.class_name != constants.PENIS_CLASS_NAME:
                    if _calculate_iou(penis_box.bbox, other_box.bbox) > 0:
                        interactor_ids.add(other_box.track_id)

    if logger:
        logger.debug(f"Pre-scan found {len(interactor_ids)} unique interactor track IDs.")
    return interactor_ids


def _recover_gap_worker(args: dict) -> Tuple[List[BoxRecord], int]:
    """
    A worker function designed to be run in a separate process to recover a single gap.
    It now processes a precise range from start_frame to end_frame, as determined by
    the parent process, which handles the interruption logic.
    """
    gap_info = args['gap_info']
    preprocessed_video_path = args['preprocessed_video_path']
    yolo_input_size = args['yolo_input_size']

    recovered_boxes = []
    frames_processed_in_worker = 0

    # Each worker needs its own VideoProcessor instance
    class DummyAppForVP:
        def __init__(self):
            # Create a logger unique to the worker process for easier debugging
            self.logger = logging.getLogger(f"S2_OF_Worker_{os.getpid()}")
            self.hardware_acceleration_method = 'none'
            self.available_ffmpeg_hwaccels = ['none']

    # Create a logger for validation
    worker_logger = logging.getLogger(f"S2_OF_Worker_{os.getpid()}")

    # Validate preprocessed video before using it
    try:
        from detection.cd.stage_1_cd import _validate_preprocessed_video_completeness
        # Use gap info to get frame count and estimated fps
        expected_frames = gap_info["end_frame"] - gap_info["start_frame"] + 1
        estimated_fps = args.get('fps', 30.0)  # Get fps from args or use default

        if not _validate_preprocessed_video_completeness(preprocessed_video_path, expected_frames, estimated_fps, worker_logger):
            worker_logger.warning(f"Preprocessed video validation failed in OF worker: {os.path.basename(preprocessed_video_path)}")
            return [], 0
    except Exception as e:
        worker_logger.error(f"Error validating preprocessed video in OF worker: {e}")
        return [], 0

    vp = VideoProcessor(app_instance=DummyAppForVP(), yolo_input_size=yolo_input_size)
    if not vp.open_video(preprocessed_video_path):
        return [], 0

    try:
        # The start frame for OF is the last known good frame
        initial_frame_img = vp._get_specific_frame(gap_info["start_frame"] - 1)
        if initial_frame_img is None: return [], 0

        prev_gray = cv2.cvtColor(initial_frame_img, cv2.COLOR_BGR2GRAY)
        current_tracked_box = np.array(gap_info["last_known_box"].bbox, dtype=np.float32)
        if current_tracked_box.shape[0] != 4 or not np.all(np.isfinite(current_tracked_box)):
            # Invalid starting box; skip recovery for this gap
            return [], 0
        # Use cached DIS optical flow instance (ultrafast)
        flow_dense = _get_dis_flow_ultrafast()

        # PERFORMANCE FIX: Use streaming frame access instead of individual frame requests
        # This eliminates the FFmpeg process creation/destruction overhead per frame
        start_frame = gap_info["start_frame"]
        end_frame = gap_info["end_frame"]
        num_frames_in_gap = end_frame - start_frame + 1
        
        # Stream consecutive frames for the entire gap range
        for actual_frame_id, current_frame_img in vp.stream_frames_for_segment(start_frame, num_frames_in_gap):
            frames_processed_in_worker += 1
            if current_frame_img is None: break

            current_gray = cv2.cvtColor(current_frame_img, cv2.COLOR_BGR2GRAY)

            # Compute flow on a padded ROI around the current tracked box to reduce cost
            h, w = prev_gray.shape
            if not np.all(np.isfinite(current_tracked_box)):
                break
            x1f, y1f, x2f, y2f = map(float, current_tracked_box)
            # Clamp current box to valid image bounds to avoid NaN/invalid ROIs
            x1f = max(0.0, min(x1f, float(w - 1)))
            x2f = max(0.0, min(x2f, float(w)))
            y1f = max(0.0, min(y1f, float(h - 1)))
            y2f = max(0.0, min(y2f, float(h)))
            bw = max(1.0, x2f - x1f)
            bh = max(1.0, y2f - y1f)
            if not np.isfinite(bw) or not np.isfinite(bh):
                break
            pad_x = int(0.25 * bw)
            pad_y = int(0.25 * bh)
            rx1 = max(0, int(x1f) - pad_x)
            ry1 = max(0, int(y1f) - pad_y)
            rx2 = min(w, int(x2f) + pad_x)
            ry2 = min(h, int(y2f) + pad_y)

            if rx2 <= rx1 or ry2 <= ry1:
                break

            prev_roi = prev_gray[ry1:ry2, rx1:rx2]
            curr_roi = current_gray[ry1:ry2, rx1:rx2]
            # Ensure memory is contiguous for DIS optical flow
            prev_roi_c = np.ascontiguousarray(prev_roi)
            curr_roi_c = np.ascontiguousarray(curr_roi)
            try:
                flow = flow_dense.calc(prev_roi_c, curr_roi_c, None)
            except cv2.error:
                # If DIS requires contiguous memory or encounters a failure, stop recovery for this gap
                break
            if flow is None:
                break

            # Median flow within the original (unpadded) box area mapped into ROI coords
            bx1 = max(0, int(int(x1f) - rx1))
            by1 = max(0, int(int(y1f) - ry1))
            bx2 = min(flow.shape[1], int(int(x2f) - rx1))
            by2 = min(flow.shape[0], int(int(y2f) - ry1))
            if bx2 <= bx1 or by2 <= by1:
                break

            dx = float(np.median(flow[by1:by2, bx1:bx2, 0]))
            dy = float(np.median(flow[by1:by2, bx1:bx2, 1]))
            if not np.isfinite(dx) or not np.isfinite(dy):
                break
            # Apply a small damping factor to reduce overshoot
            damping = 0.85
            dx *= damping
            dy *= damping
            # Update and clamp within image bounds
            current_tracked_box += [dx, dy, dx, dy]
            x1f, y1f, x2f, y2f = map(float, current_tracked_box)
            x1f = max(0.0, min(x1f, float(w - 1)))
            x2f = max(x1f + 1.0, min(x2f, float(w)))
            y1f = max(0.0, min(y1f, float(h - 1)))
            y2f = max(y1f + 1.0, min(y2f, float(h)))
            current_tracked_box = np.array([x1f, y1f, x2f, y2f], dtype=np.float32)

            new_box = BoxRecord(
                frame_id=actual_frame_id, bbox=current_tracked_box,
                confidence=gap_info["last_known_box"].confidence * 0.75,
                class_id=gap_info["last_known_box"].class_id, class_name=gap_info["last_known_box"].class_name,
                status=constants.STATUS_OF_RECOVERED, yolo_input_size=yolo_input_size,
                track_id=gap_info["track_id"]
            )
            recovered_boxes.append(new_box)
            prev_gray = current_gray
    finally:
        vp.stop_processing(join_thread=False)
        del vp

    return recovered_boxes, frames_processed_in_worker


def pass_1c_recover_lost_tracks_with_of(
        app, frames: List, video_info: Dict, yolo_input_size: int, vr_vertical_third_filter: bool,
        preprocessed_video_path_arg: str,
        logger: Optional[logging.Logger],
        progress_callback: callable,
        main_step_info: tuple,
        num_workers: int
):
    """
    MODIFIED: Identifies gaps and provides an 'interrupt_frame' to the worker if a
    high-confidence re-detection appears within the gap.
    """
    if logger:
        logger.debug("Starting Stage 2 Pass 1c: Recover Lost Tracks with INTERRUPTIBLE Optical Flow")

    if not preprocessed_video_path_arg or not os.path.exists(preprocessed_video_path_arg):
        message = "Skipped - Preprocessed file not found."
        logger.warning(f"Optical flow recovery: {message} Path: {preprocessed_video_path_arg}")
        if progress_callback: progress_callback(main_step_info, {"message": message, "current": 1, "total": 1}, True)
        return

    # Validate preprocessed video before using it
    try:
        from detection.cd.stage_1_cd import _validate_preprocessed_video_completeness
        expected_frames = len(frames) if frames else 0
        fps = video_info.get('fps', 30.0)

        if expected_frames > 0 and not _validate_preprocessed_video_completeness(preprocessed_video_path_arg, expected_frames, fps, logger):
            message = "Skipped - Preprocessed video validation failed."
            logger.warning(f"Optical flow recovery: {message} Path: {preprocessed_video_path_arg}")
            if progress_callback: progress_callback(main_step_info, {"message": message, "current": 1, "total": 1}, True)
            return
    except Exception as e:
        logger.error(f"Error validating preprocessed video in Stage 2: {e}")
        message = "Skipped - Preprocessed video validation error."
        if progress_callback: progress_callback(main_step_info, {"message": message, "current": 1, "total": 1}, True)
        return

    # Gap identification remains the same (it's very fast)
    interactor_track_ids = _pre_scan_for_interactor_ids(frames, video_info, vr_vertical_third_filter, logger)
    if not interactor_track_ids:
        if progress_callback: progress_callback(main_step_info, {"message": "Skipped - No interactors", "current": 1, "total": 1}, True)
        return
    fps = video_info.get('fps', 30.0)
    # Max gap to attempt OF recovery on (shorter window for performance)
    MAX_RECOVERY_GAP = int(fps * 3)
    # Min gap to trigger OF instead of simple interpolation
    MIN_RECOVERY_GAP = int(fps * 1)
    # Confidence threshold for a detection to be considered a valid "interrupter"
    INTERRUPT_CONFIDENCE_THRESHOLD = 0.6

    frames_map = {f.frame_id: f for f in frames}

    # Create a map for quick lookup of all boxes by track_id
    # {track_id: {frame_id: box_record}}
    track_box_map = {}
    for frame in frames:
        for box in frame.boxes:
            if box.track_id not in track_box_map:
                track_box_map[box.track_id] = {}
            track_box_map[box.track_id][frame.frame_id] = box

    gaps_to_recover = []
    for track_id in interactor_track_ids:
        if track_id not in track_box_map: continue

        present_frames = sorted(track_box_map[track_id].keys())
        if len(present_frames) < 2: continue

        for i in range(len(present_frames) - 1):
            last_frame_id = present_frames[i]
            next_frame_id = present_frames[i + 1]
            gap_size = next_frame_id - last_frame_id - 1

            # Gate by configured thresholds
            if MIN_RECOVERY_GAP < gap_size <= MAX_RECOVERY_GAP:
                last_box = track_box_map[track_id][last_frame_id]

                # --- NEW LOGIC: Check for interrupter frames ---
                interrupt_frame = None
                # We search for a re-detection starting from the *next* known presence.
                # The tracker might have re-identified the object at 'next_frame_id'.
                # We check if that re-detection is high-confidence.
                next_box = track_box_map[track_id][next_frame_id]
                if next_box.status == constants.STATUS_DETECTED and next_box.confidence >= INTERRUPT_CONFIDENCE_THRESHOLD:
                    interrupt_frame = next_frame_id

                # The effective end of the gap is either the interrupt frame or the original end.
                effective_end_frame = interrupt_frame -1 if interrupt_frame else next_frame_id - 1

                gaps_to_recover.append({
                    "track_id": track_id,
                    "start_frame": last_frame_id + 1,
                    # The worker will run up to this frame
                    "end_frame": effective_end_frame,
                    "last_known_box": last_box,
                    # The original full gap size, for progress reporting
                    "gap_size": gap_size
                })


    if not gaps_to_recover:
        if progress_callback: progress_callback(main_step_info, {"message": "Completed (No gaps)", "current": 1, "total": 1}, True)
        return

    # --- Parallel Processing Setup ---
    logger.info(f"Distributing {len(gaps_to_recover)} recovery gaps to {num_workers} worker processes.")

    worker_args = [{
        'gap_info': gap,
        'preprocessed_video_path': preprocessed_video_path_arg,
        'yolo_input_size': yolo_input_size,
        'fps': fps
    } for gap in gaps_to_recover]

    total_frames_to_recover = sum(g['end_frame'] - g['start_frame'] + 1 for g in gaps_to_recover)
    frames_processed_count = 0
    total_boxes_added = 0
    start_time = time.time()

    with multiprocessing.Pool(processes=num_workers) as pool:
        # Use imap_unordered for better responsiveness as jobs finish
        for result_boxes, frames_in_worker in pool.imap_unordered(_recover_gap_worker, worker_args):
            frames_processed_count += frames_in_worker
            if result_boxes:
                for box in result_boxes:
                    frames_map[box.frame_id].boxes.append(box)
                total_boxes_added += len(result_boxes)

            # --- Update Progress Bar ---
            if progress_callback:
                time_elapsed = time.time() - start_time
                # Calculate true FPS and ETA based on frames.
                fps_val = frames_processed_count / time_elapsed if time_elapsed > 0 else 0
                eta = (total_frames_to_recover - frames_processed_count) / fps_val if fps_val > 0 else 0

                progress_data = {
                    "current": frames_processed_count, "total": total_frames_to_recover,
                    "message": "Recovering gaps...", "time_elapsed": time_elapsed,
                    "fps": fps_val, "eta": eta
                }
                progress_callback(main_step_info, progress_data, False)

    if progress_callback:
        progress_callback(main_step_info, {"message": "Completed", "current": total_frames_to_recover, "total": total_frames_to_recover}, True)

    logger.info(f"Parallel OF recovery finished. Added {total_boxes_added} new boxes across {len(gaps_to_recover)} gaps.")


def perform_contact_analysis(
        video_path_arg: str, msgpack_file_path_arg: str,
        preprocessed_video_path_arg: Optional[str],
        progress_callback: callable, stop_event: threading.Event,
        app=None,  # App (AppLogic) instance for state access
        ml_model_dir_path_arg: Optional[str] = None,
        parent_logger_arg: Optional[logging.Logger] = None,
        output_overlay_msgpack_path: Optional[str] = None,
        yolo_input_size_arg: int = 640,
        video_type_arg: str = 'auto',
        vr_input_format_arg: str = 'he',
        vr_fov_arg: int = 190,
        vr_pitch_arg: int = 0,
        vr_vertical_third_filter_arg: bool = True,
        enable_of_debug_prints: bool = False,
        discarded_classes_runtime_arg: Optional[List[str]] = None,
        scripting_range_active_arg: bool = False,
        scripting_range_start_frame_arg: Optional[int] = None,
        scripting_range_end_frame_arg: Optional[int] = None,
        generate_funscript_actions_arg: bool = True,
        is_ranged_data_source: bool = False,
        num_workers_stage2_of_arg: int = constants.DEFAULT_S2_OF_WORKERS,
        use_sqlite_storage: bool = True,
        output_folder_path: Optional[str] = None
):
    global _of_debug_prints_stage2
    _of_debug_prints_stage2 = enable_of_debug_prints

    logger = parent_logger_arg
    if not logger:  # Fallback logger
        logger = logging.getLogger("Stage2_Fallback")
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.info("Stage 2 using fallback logger.")

    FrameObject._id_counter = 0  # Reset static counters
    BaseSegment._id_counter = 0
    logger.info(f"--- Starting Stage 2 Analysis ---")

    def progress_wrapper(main_step_tuple, sub_info_from_module, force=False):
        """ This wrapper now passes the sub_info_from_module (which can be a tuple or dict) directly. """
        if progress_callback:
            # The original callback expects: main_info, sub_info, force
            progress_callback(main_step_tuple, sub_info_from_module, force)

    # 1. Initialize VideoProcessor (Simplified, primarily for video_info)
    vp_logger = logger.getChild("VideoProcessor_Stage2") if logger else logging.getLogger(
        "VideoProcessor_Stage2_Fallback")

    # Create a dummy app_instance proxy if app_logic_instance is None for VP
    class DummyAppForVP:
        pass

    dummy_app_vp = DummyAppForVP()
    dummy_app_vp.logger = vp_logger
    dummy_app_vp.hardware_acceleration_method = "none"  # Not critical for info

    vp = VideoProcessor(app_instance=dummy_app_vp, tracker=None, yolo_input_size=yolo_input_size_arg,
                        video_type=video_type_arg, vr_input_format=vr_input_format_arg,
                        vr_fov=vr_fov_arg, vr_pitch=vr_pitch_arg,
                        fallback_logger_config={'logger_instance': vp_logger})  # Pass its own logger
    if not vp.open_video(video_path_arg):
        logger.critical("VideoProcessor failed to open video or get info for Stage 2.")
        return {"error": "VideoProcessor failed to initialize for Stage 2"}
    if stop_event.is_set(): return {"error": "Processing stopped during VP init (Stage 2)."}

    video_info_dict = vp.video_info.copy()
    video_info_dict['actual_video_type'] = vp.determined_video_type  # Store determined type
    logger.info(f"Stage 2 VP Info: {video_info_dict}")
    vp.reset(close_video=True)
    del vp  # Release VP resources after getting info

    # 2. Load YOLO results (now with track_id)
    all_raw_detections = load_yolo_results_stage2(msgpack_file_path_arg, stop_event, logger)
    if stop_event.is_set() or not all_raw_detections:
        logger.warning("No YOLO detections loaded or process stopped (Stage 2).")
        return {"error": "Failed to load YOLO data or process stopped (Stage 2)"}

    num_video_frames = video_info_dict.get('total_frames', 0)
    if num_video_frames > 0 and len(all_raw_detections) != num_video_frames:
        logger.warning(
            f"Mismatch msgpack frames {len(all_raw_detections)} vs video frames {num_video_frames}.")
        if len(all_raw_detections) < num_video_frames:
            all_raw_detections.extend([[] for _ in range(num_video_frames - len(all_raw_detections))])
        else:
            all_raw_detections = all_raw_detections[:num_video_frames]
        logger.info(f"Adjusted raw detections to {len(all_raw_detections)} frames.")
    if not all_raw_detections: return {"error": "No detection data after adjustment."}

    # 3. Initialize frame objects and processing data
    try:
        # Initialize frame processing without AppStateContainer
        effective_discard_classes = set(constants.CLASSES_TO_DISCARD_BY_DEFAULT)
        if discarded_classes_runtime_arg:
            effective_discard_classes.update(discarded_classes_runtime_arg)
        
        # Create frame objects from raw data
        frame_id_offset = scripting_range_start_frame_arg if is_ranged_data_source and scripting_range_start_frame_arg is not None else 0
        frame_objects = []
        
        for i, raw_frame_data_dict in enumerate(all_raw_detections):
            absolute_frame_id = i + frame_id_offset
            fo = FrameObject(frame_id=absolute_frame_id, yolo_input_size=yolo_input_size_arg,
                             raw_frame_data=raw_frame_data_dict,
                             classes_to_discard_runtime_set=effective_discard_classes)

            # VR Filter for NON-PENIS boxes
            if video_info_dict.get('actual_video_type') == 'VR' and vr_vertical_third_filter_arg:
                for box_rec in fo.boxes:
                    if box_rec.class_name != constants.PENIS_CLASS_NAME and not (
                            yolo_input_size_arg / 3 <= box_rec.cx <= 2 * yolo_input_size_arg / 3):
                        box_rec.is_excluded = True
                        box_rec.status = "Excluded_VR_Filter_Peripheral"
            frame_objects.append(fo)
        
        # Initialize processing results storage
        segments = []
        funscript_frames = []
        funscript_distances = []
        funscript_distances_lr = []

        # SQLite storage handling (simplified for now)
        sqlite_storage = None
        sqlite_db_path = None
        if use_sqlite_storage and output_folder_path:
            try:
                from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
                sqlite_storage = Stage2SQLiteStorage(None, logger)
                
                # Generate database filename based on video name
                video_filename = os.path.splitext(os.path.basename(video_path_arg))[0]
                
                # Handle preprocessed videos: use original filename for database consistency  
                if video_filename.endswith('_preprocessed'):
                    video_filename = video_filename[:-len('_preprocessed')]
                    logger.debug(f"Using original video stem '{video_filename}' for SQLite database naming")
                
                db_filename = f"{video_filename}_stage2_data.db"
                sqlite_db_path = os.path.join(output_folder_path, db_filename)
                sqlite_storage.set_db_path(sqlite_db_path)
                logger.info(f"SQLite storage initialized in output folder: {sqlite_db_path}")
            except ImportError as e:
                logger.warning(f"SQLite storage not available, falling back to memory: {e}")
                use_sqlite_storage = False
        
        # Initialize processing data structures directly without AppStateContainer
        segments = []
        funscript_frames = []
        funscript_distances = []
        funscript_distances_lr = []
        
        # Initialize SQLite storage if enabled
        use_sqlite = use_sqlite_storage
        sqlite_storage = None
        sqlite_db_path = None
        
        if use_sqlite:
            try:
                from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
                sqlite_storage = Stage2SQLiteStorage(None, logger)
                if logger:
                    logger.info("SQLite storage module loaded successfully")
            except ImportError as e:
                if logger:
                    logger.warning(f"SQLite storage not available, falling back to memory: {e}")
                use_sqlite = False
        
        # Initialize SQLite database path in output folder if enabled
        if use_sqlite and sqlite_storage and output_folder_path:
            # Generate database filename based on video name
            video_filename = os.path.splitext(os.path.basename(video_path_arg))[0]
            
            # Handle preprocessed videos: use original filename for database consistency  
            if video_filename.endswith('_preprocessed'):
                video_filename = video_filename[:-len('_preprocessed')]
                logger.debug(f"Using original video stem '{video_filename}' for SQLite database naming")
            
            db_filename = f"{video_filename}_stage2_data.db"
            sqlite_db_path = os.path.join(output_folder_path, db_filename)

            # Set the database path and initialize
            sqlite_storage.set_db_path(sqlite_db_path)
            logger.info(f"SQLite storage initialized in output folder: {sqlite_db_path}")

    except Exception as e:
        logger.error(f"Error initializing Stage 2 processing: {e}", exc_info=True)
        return {"error": f"Stage 2 initialization failed: {e}"}
    if stop_event.is_set(): return {"error": "Processing stopped after initialization."}

    # --- Processing Steps ---
    # These steps will fill data into frame objects and segments list

    # Define main steps for progress reporting
    # Base steps for segmentation
    main_steps_list_base = [
        ("Step 1: Tracking Objects", resilient_tracker_step0),
        ("Step 2: Interpolate Short Gaps", pass_1_interpolate_boxes),
        # ("Step 3: Recover Long Gaps (OF)", pass_1c_recover_lost_tracks_with_of),  # TEMPORARILY DISABLED
        ("Step 4: Smooth All Tracked Boxes", pass_1b_smooth_all_tracks),
        ("Step 5: Kalman Filter & Lock State", pass_3_kalman_and_lock_state),
        ("Step 6: Assign Positions & Segments", pass_4_assign_positions_and_segments),
        ("Step 7: Recalculate Chapter Heights", pass_5_recalculate_heights_post_aggregation),
        ("Step 8: Determine Frame Distances", pass_6_determine_distance),
        ("Step 9: Smooth & Normalize Distances", pass_7_smooth_and_normalize_distances),
    ]
    main_steps_list_funscript_gen = [
    #    #("Step 8: Smooth & Normalize Distances", pass_7_smooth_and_normalize_distances),
        ("Step 10: Simplify Signal", pass_8_simplify_signal)
    ]
    main_steps_list = main_steps_list_base
    # if generate_funscript_actions_arg:
    #     main_steps_list.extend(main_steps_list_funscript_gen)

    num_main_steps = len(main_steps_list)

    for i, (main_step_name, step_func) in enumerate(main_steps_list):
        logger.info(f"Starting {main_step_name} (Stage 2)")
        main_step_tuple_for_callback = (i + 1, num_main_steps, main_step_name)

        progress_wrapper(main_step_tuple_for_callback, (0, 1, "Initializing..."), True)

        # --- Special handling for the OF recovery pass ---
        if step_func == pass_1c_recover_lost_tracks_with_of:
            step_func(app, frame_objects, video_info_dict, yolo_input_size_arg, vr_vertical_third_filter_arg, 
                          preprocessed_video_path_arg, logger, progress_wrapper,
                          main_step_tuple_for_callback, num_workers_stage2_of_arg)
        elif step_func == resilient_tracker_step0:
            step_func(app, frame_objects, video_info_dict, logger)
        elif step_func == pass_1_interpolate_boxes:
            step_func(app, frame_objects, video_info_dict, logger)
        elif step_func == pass_1b_smooth_all_tracks:
            step_func(app, frame_objects, video_info_dict, logger)
        elif step_func == pass_2_preliminary_height_estimation:
            step_func(app, frame_objects, video_info_dict, vr_vertical_third_filter_arg, logger)
        elif step_func == pass_3_kalman_and_lock_state:
            step_func(app, frame_objects, video_info_dict, yolo_input_size_arg, vr_vertical_third_filter_arg, logger)
        elif step_func == pass_4_assign_positions_and_segments:
            step_func(app, frame_objects, segments, video_info_dict, yolo_input_size_arg, logger)
        elif step_func == pass_5_recalculate_heights_post_aggregation:
            step_func(app, frame_objects, segments, video_info_dict, yolo_input_size_arg, vr_vertical_third_filter_arg, logger)
        elif step_func == pass_6_determine_distance:
            step_func(app, frame_objects, segments, video_info_dict, yolo_input_size_arg, logger)
        elif step_func == pass_7_smooth_and_normalize_distances:
            step_func(app, frame_objects, funscript_frames, funscript_distances, logger)
        elif step_func == pass_8_simplify_signal:
            step_func(app, frame_objects, funscript_frames, funscript_distances, funscript_distances_lr, video_info_dict, logger)

        if stop_event.is_set():
            logger.info(f"Stage 2 stopped during {main_step_name}.")
            progress_wrapper(main_step_tuple_for_callback, "Aborted", 1, 1, True)
            return {"error": f"Processing stopped during {main_step_name} (Stage 2)"}

        if step_func != pass_1c_recover_lost_tracks_with_of:

            progress_wrapper(main_step_tuple_for_callback, (1, 1, "Completed"), True)

    # --- Prepare overlay data BEFORE clearing frames from memory ---
    frame_data = []
    overlay_data_with_segments = None

    if output_overlay_msgpack_path:
        logger.info("Preparing overlay data before memory optimization...")
        frame_data = [frame.to_overlay_dict() for frame in frame_objects if not stop_event.is_set()]
        if stop_event.is_set(): return {"error": "Processing stopped during overlay data prep (Stage 2)."}
        
        #all_frames_overlay_data = frame_data  # Fix: assign frame_data to all_frames_overlay_data

        try:
            segments_for_overlay = [seg.to_dict() for seg in segments]
        except Exception:
            segments_for_overlay = []
        overlay_data_with_segments = {"frames": frame_data, "segments": segments_for_overlay, "metadata": {"schema": "v1.1"}}

    # --- Store processed data to SQLite for Stage 3 memory optimization ---
    if use_sqlite and sqlite_storage:
        logger.info("Storing processed frame data to SQLite database...")
        try:
            # Store frame objects to database
            sqlite_storage.store_frame_objects_batch(frame_objects, batch_size=2000)
            if logger:
                logger.info(f"Stored {len(frame_objects)} frame objects to SQLite")

            # Store Stage 2 segments to database
            sqlite_storage.store_segments(segments)
            if logger:
                logger.info(f"Stored {len(segments)} segments to SQLite")

            # Clear frames from memory to save RAM for Stage 3
            original_count = len(frame_objects)
            frame_objects.clear()
            if logger:
                logger.info(f"Cleared {original_count} frame objects from memory")

            logger.info("Successfully stored Stage 2 data to SQLite and cleared memory")
        except Exception as e:
            logger.error(f"Error storing data to SQLite: {e}", exc_info=True)
            # Continue with in-memory processing as fallback
            use_sqlite = False

    # --- Funscript Data Population from Stage 2 results ---
    if funscript_frames:
        funscript_distances_lr[:] = [50] * len(funscript_frames)
    else:  # If no frames (e.g. very short video or error), ensure lists are empty
        funscript_distances_lr.clear()
        funscript_distances.clear()

    if stop_event.is_set():
        logger.info("Stage 2 stopped before final data packaging.")
        return {"error": "Processing stopped before final data packaging (Stage 2)."}

    # Video segments for GUI from Segments
    # Get the full list of generated segments and funscript points
    video_segments_for_gui = [seg.to_dict() for seg in segments if not stop_event.is_set()]
    funscript_frames_full = funscript_frames
    funscript_distances_full = funscript_distances
    funscript_distances_lr_full = [50] * len(funscript_frames_full) if funscript_frames_full else []

    # These will hold the final, possibly filtered, results
    final_video_segments = video_segments_for_gui
    final_funscript_frames = funscript_frames_full
    final_funscript_distances = funscript_distances_full
    final_funscript_distances_lr = funscript_distances_lr_full

    if scripting_range_active_arg:
        logger.info(
            f"Filtering final S2 results for active range: {scripting_range_start_frame_arg} - {scripting_range_end_frame_arg}")
        start_f = scripting_range_start_frame_arg
        end_f = scripting_range_end_frame_arg
        if end_f is None or end_f == -1:
            end_f = len(frame_objects) - 1

        # Filter the video segments/chapters
        final_video_segments = [
            seg_dict for seg_dict in video_segments_for_gui
            if max(seg_dict['start_frame_id'], start_f) <= min(seg_dict['end_frame_id'], end_f)
        ]

        # Filter the funscript points
        if funscript_frames_full:
            filtered_frames, filtered_distances, filtered_distances_lr = [], [], []
            for i, frame_id in enumerate(funscript_frames_full):
                if start_f <= frame_id <= end_f:
                    filtered_frames.append(frame_id)
                    filtered_distances.append(funscript_distances_full[i])
                    filtered_distances_lr.append(funscript_distances_lr_full[i])

            final_funscript_frames = filtered_frames
            final_funscript_distances = filtered_distances
            final_funscript_distances_lr = filtered_distances_lr

    # Create funscript object instead of raw actions
    funscript_obj = None
    primary_actions_final = []  # Keep for backward compatibility if needed
    secondary_actions_final = []
    
    if generate_funscript_actions_arg:
        current_video_fps = video_info_dict.get('fps', 0)
        if current_video_fps > 0 and final_funscript_frames:
            # Create DualAxisFunscript object
            funscript_obj = DualAxisFunscript(logger=logger)
            
            # Add actions to the funscript object
            for fid, pos_primary, pos_secondary in zip(final_funscript_frames, final_funscript_distances,
                                                       final_funscript_distances_lr):
                if stop_event.is_set(): break
                timestamp_ms = int(round((fid / current_video_fps) * 1000))
                funscript_obj.add_action(timestamp_ms, int(pos_primary), int(pos_secondary))
            
            # log the final video segments
            logger.info(f"Final video segments: {final_video_segments}")

            # Set chapters from video segments
            if final_video_segments and not stop_event.is_set():
                funscript_obj.set_chapters_from_segments(final_video_segments, current_video_fps)

            # log the chapters of the funscript
            logger.info(f"Chapters of the funscript: {funscript_obj.chapters}")
            
            # Extract actions for backward compatibility
            if not stop_event.is_set():
                primary_actions_final = funscript_obj.primary_actions.copy()
                secondary_actions_final = funscript_obj.secondary_actions.copy()

    # Build the final return dictionary with funscript as primary return
    # The calling orchestrator is responsible for picking what it needs based on the mode.
    return_dict = {
        "success": True,
        "funscript": funscript_obj,  # Primary return - funscript object with actions and chapters
        "total_frames_processed": len(frame_objects),
        "processing_method": "contact_analysis",
        # Stage 3 compatibility data
        "segments_objects": segments,
        "all_s2_frame_objects_list": frame_objects,
        "sqlite_db_path": sqlite_db_path,
        # Legacy compatibility (for 2-stage mode only)
        "video_segments": final_video_segments
    }

    if output_overlay_msgpack_path and (overlay_data_with_segments or all_frames_overlay_data):
        logger.info(f"Saving Stage 2 overlay data to: {output_overlay_msgpack_path}")
        try:
            def numpy_default_handler(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable for msgpack")

            to_write = overlay_data_with_segments if overlay_data_with_segments is not None else all_frames_overlay_data
            with open(output_overlay_msgpack_path, 'wb') as f:
                f.write(msgpack.packb(to_write, use_bin_type=True, default=numpy_default_handler))
            if overlay_data_with_segments:
                logger.info(f"Successfully saved Stage 2 overlay package with {len(overlay_data_with_segments.get('frames', []))} frames and {len(overlay_data_with_segments.get('segments', []))} segments to {output_overlay_msgpack_path}.")
            else:
                logger.info(f"Successfully saved Stage 2 overlay data for {len(all_frames_overlay_data)} frames to {output_overlay_msgpack_path}.")

            if os.path.exists(output_overlay_msgpack_path):
                return_dict["overlay_msgpack_path"] = output_overlay_msgpack_path
        except Exception as e:
            logger.error(f"Error saving Stage 2 overlay msgpack to {output_overlay_msgpack_path}: {e}", exc_info=True)

    logger.info(f"--- Stage 2 Analysis Finished. Segments: {len(final_video_segments)} ---")
    return return_dict

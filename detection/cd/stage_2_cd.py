import numpy as np
import msgpack
import threading
import argparse
import math
import os
from collections import deque, defaultdict  # Added defaultdict
from typing import List, Dict, Any, Optional, Tuple, Union
import bisect
import logging
import cv2
from scipy.signal import savgol_filter, find_peaks
from simplification.cutil import simplify_coords_vw
from copy import deepcopy

from video.video_processor import VideoProcessor
from config.constants import *

_of_debug_prints_stage2 = False


def _debug_log(logger_instance: Optional[logging.Logger], message: str):
    """Helper for conditional debug logging."""
    if _of_debug_prints_stage2 and logger_instance:
        logger_instance.debug(f"[S2 DEBUG] {message}")
    elif _of_debug_prints_stage2 and not logger_instance:  # Fallback if no logger but debug is on
        print(f"[S2 DEBUG - NO LOGGER] {message}")


def _progress_update(callback, task_name, current, total, force_update=False):
    if callback:
        if force_update or current == 0 or current == total or (total > 0 and (current % (max(1, total // 20))) == 0):
            callback(task_name, current, total)


class BaseSegment:
    _id_counter = 0

    def __init__(self, start_frame_id: int, end_frame_id: Optional[int] = None):
        self.id = BaseSegment._id_counter
        BaseSegment._id_counter += 1
        self.start_frame_id = start_frame_id
        self.end_frame_id = end_frame_id if end_frame_id is not None else start_frame_id
        self.frames: List['FrameObject'] = []
        self.duration = 0
        self.update_duration()

    def update_duration(self):
        if self.frames:
            self.duration = self.end_frame_id - self.start_frame_id + 1
        elif self.end_frame_id >= self.start_frame_id:
            self.duration = self.end_frame_id - self.start_frame_id + 1
        else:
            self.duration = 0

    def add_frame(self, frame: 'FrameObject'):
        if not self.frames or frame.frame_id > self.frames[-1].frame_id:
            self.frames.append(frame)
            if frame.frame_id > self.end_frame_id: self.end_frame_id = frame.frame_id
            self.update_duration()
        elif frame.frame_id < self.start_frame_id:
            pass
        else:
            inserted = False
            for i, f_obj in enumerate(self.frames):
                if frame.frame_id < f_obj.frame_id:
                    self.frames.insert(i, frame); inserted = True; break
                elif frame.frame_id == f_obj.frame_id:
                    inserted = True; break
            if not inserted: self.frames.append(frame)
            self.update_duration()

    def get_occlusion_info(self, box_attribute_name: str = "tracked_box") -> List[Dict[str, Any]]:
        occlusions = []
        if not self.frames: return occlusions
        in_occlusion_block = False
        block_start_frame = -1
        block_status = ""
        for frame in sorted(self.frames, key=lambda f: f.frame_id):
            box = getattr(frame, box_attribute_name, None)
            is_synthesized = box and box.status not in [STATUS_DETECTED, STATUS_SMOOTHED]
            if is_synthesized:
                if not in_occlusion_block:
                    in_occlusion_block = True
                    block_start_frame = frame.frame_id
                    block_status = box.status
                elif block_status != box.status:
                    occlusions.append(
                        {"start_frame": block_start_frame, "end_frame": frame.frame_id - 1, "status": block_status})
                    block_start_frame = frame.frame_id
                    block_status = box.status
            else:
                if in_occlusion_block:
                    occlusions.append(
                        {"start_frame": block_start_frame, "end_frame": frame.frame_id - 1, "status": block_status})
                    in_occlusion_block = False
                    block_start_frame = -1
                    block_status = ""
        if in_occlusion_block:
            occlusions.append(
                {"start_frame": block_start_frame, "end_frame": self.frames[-1].frame_id, "status": block_status})
        return occlusions


class BoxRecord:
    def __init__(self, frame_id: int, bbox: Union[np.ndarray, List[float], Tuple[float, float, float, float]],
                 confidence: float, class_id: int, class_name: str,
                 status: str = STATUS_DETECTED, yolo_input_size: int = 640,
                 track_id: Optional[int] = None):  # track_id will be populated by Stage 2 tracker
        self.frame_id = int(frame_id)
        self.bbox = np.array(bbox, dtype=np.float32)
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.status = str(status)
        self.track_id = track_id  # Initialize to None or allow it to be set

        if not (len(self.bbox) == 4 and self.bbox[2] > self.bbox[0] and self.bbox[3] > self.bbox[1]):
            self.bbox = np.array([0, 0, 0, 0], dtype=np.float32)

        self._update_dims()
        self.yolo_input_size = yolo_input_size
        self.area_perc = (self.area / (yolo_input_size * yolo_input_size)) * 100 if yolo_input_size > 0 else 0
        self.is_excluded: bool = False
        self.is_tracked: bool = False

    def _update_dims(self):
        self.width = self.bbox[2] - self.bbox[0]
        self.height = self.bbox[3] - self.bbox[1]
        self.area = self.width * self.height
        self.cx = self.bbox[0] + self.width / 2
        self.cy = self.bbox[1] + self.height / 2
        self.x1, self.y1, self.x2, self.y2 = self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]
        self.box = tuple(self.bbox)

    def update_bbox(self, new_bbox: np.ndarray, new_status: Optional[str] = None):
        self.bbox = np.array(new_bbox, dtype=np.float32)
        self._update_dims()
        if new_status:
            self.status = new_status

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame_id": self.frame_id,
            "bbox": self.bbox.tolist() if isinstance(self.bbox, np.ndarray) else list(self.bbox),
            "confidence": float(self.confidence),
            "class_id": self.class_id,
            "class_name": self.class_name,
            "status": self.status,
            "track_id": self.track_id,
            "width": float(self.width),
            "height": float(self.height),
            "cx": float(self.cx),
            "cy": float(self.cy),
            "area_perc": float(self.area_perc),
            "is_excluded": self.is_excluded,
            "is_tracked": self.is_tracked,
        }

    def __repr__(self):
        return (f"BoxRecord(fid={self.frame_id}, cls='{self.class_name}', track_id={self.track_id}, "
                f"conf={self.confidence:.2f}, status='{self.status}', "
                f"bbox=[{self.bbox[0]:.0f},{self.bbox[1]:.0f},{self.bbox[2]:.0f},{self.bbox[3]:.0f}], "
                f"tracked={self.is_tracked}, excluded={self.is_excluded})")


class ATRLockedPenisState:
    def __init__(self):
        self.box: Optional[Tuple[float, float, float, float]] = None  # (x1,y1,x2,y2)
        self.active: bool = False
        self.max_height: float = 0.0
        self.max_penetration_height: float = 0.0
        self.area: float = 0.0
        self.consecutive_detections: int = 0
        self.consecutive_non_detections: int = 0
        self.visible_part: float = 100.0  # Percentage
        self.glans_detected: bool = False


class FrameObject:
    _id_counter = 0

    def __init__(self, frame_id: int, yolo_input_size: int, raw_detections_input: Optional[list] = None,
                 classes_to_discard_runtime_set: Optional[set] = None):
        self.id = FrameObject._id_counter
        FrameObject._id_counter += 1
        self.frame_pos = int(frame_id)
        self.frame_id = int(frame_id)
        self.yolo_input_size = yolo_input_size
        self.boxes: List[BoxRecord] = []
        # Store the effective discard set for use in parse_raw_detections
        self._effective_discard_classes_for_parse = classes_to_discard_runtime_set if classes_to_discard_runtime_set is not None else set(
            CLASSES_TO_DISCARD_CONST)
        self.parse_raw_detections(raw_detections_input or [])

        # Original Stage 2 attributes
        self.pref_penis: Optional[BoxRecord] = None
        self.atr_penis_box_kalman: Optional[Tuple[float, float, float, float]] = None
        self.atr_locked_penis_state = ATRLockedPenisState()
        self.atr_detected_contact_boxes: List[Dict] = []
        self.atr_distances_to_penis: List[Dict] = []
        self.atr_assigned_position: str = "Not Relevant"
        self.atr_funscript_distance: int = 50
        self.pos_0_100: int = 50
        self.pos_lr_0_100: int = 50

    def parse_raw_detections(self, raw_dets_for_frame: list):
        for det_data in raw_dets_for_frame:
            class_name = det_data.get('name', 'unknown_class')
            if class_name in self._effective_discard_classes_for_parse:
                continue
            bbox_raw = det_data.get('bbox')
            if not isinstance(bbox_raw, (list, tuple)) or len(bbox_raw) != 4:
                continue
            try:
                bbox_float = [float(b) for b in bbox_raw]
            except ValueError:
                continue

            box_rec = BoxRecord(frame_id=self.frame_id, bbox=bbox_float,
                                confidence=det_data.get('confidence', 0.0),
                                class_id=det_data.get('class', -1), class_name=class_name,
                                yolo_input_size=self.yolo_input_size,
                                track_id=None)
            self.boxes.append(box_rec)

    def get_preferred_penis_box(self, actual_video_type: str = '2D', vr_vertical_third_filter: bool = False) -> \
    Optional[BoxRecord]:
        penis_detections = [b for b in self.boxes if b.class_name == PENIS_CLASS_NAME_CONST and not b.is_excluded]
        if not penis_detections: return None
        penis_detections.sort(key=lambda d: (d.bbox[3], d.area), reverse=True)
        selected_penis = penis_detections[0]
        if vr_vertical_third_filter and actual_video_type == 'VR':
            if not (self.yolo_input_size / 3 <= selected_penis.cx <= 2 * self.yolo_input_size / 3):
                for p_det in penis_detections:
                    if (self.yolo_input_size / 3 <= p_det.cx <= 2 * self.yolo_input_size / 3):
                        return p_det
                return None  # No penis in central third

        return selected_penis

    def to_overlay_dict(self) -> Dict[str, Any]:
        frame_data = {
            "frame_id": self.frame_id,
            "pos_0_100": self.pos_0_100,
            "pos_lr_0_100": self.pos_lr_0_100,
            "atr_assigned_position": self.atr_assigned_position,
            "atr_funscript_distance": self.atr_funscript_distance,
            "yolo_boxes": []
        }

        # Map ATR's locked penis state to an existing role for coloring
        if self.atr_locked_penis_state.active and self.atr_locked_penis_state.box:
            lp_state = self.atr_locked_penis_state
            box_dims = lp_state.box
            w = box_dims[2] - box_dims[0]
            h = box_dims[3] - box_dims[1]
            overlay_lp_box = {
                "frame_id": self.frame_id,
                "bbox": list(box_dims),
                "confidence": 1.0,  # Confidence is high as it's a "locked" concept
                "class_id": -1,
                "class_name": PENIS_CLASS_NAME_CONST,  # Use a common class name
                "status": "ATR_LOCKED",  # Custom status
                "track_id": None,  # ATR locked penis is a derived concept, not directly tracked object
                "width": w, "height": h,
                "cx": box_dims[0] + w / 2,
                "cy": box_dims[1] + h / 2,
                "area_perc": (lp_state.area / (self.yolo_input_size ** 2)) * 100 if self.yolo_input_size > 0 else 0,
                "is_excluded": False,
                "role_in_frame": "locked_penis_box",  # <--- USE EXISTING ROLE FOR DISTINCT COLOR
                # ATR specific debug info can still be passed if get_box_style doesn't use it
                "atr_lp_max_h": lp_state.max_height,
                "atr_lp_max_pen_h": lp_state.max_penetration_height,
                "atr_lp_visible": lp_state.visible_part,
                "atr_lp_glans": lp_state.glans_detected
            }
            frame_data["yolo_boxes"].append(overlay_lp_box)

        # Map ATR's Kalman (visible) penis to an existing role
        if self.atr_penis_box_kalman:
            kp_box = self.atr_penis_box_kalman
            w = kp_box[2] - kp_box[0]
            h = kp_box[3] - kp_box[1]
            overlay_kp_box = {
                "frame_id": self.frame_id,
                "bbox": list(kp_box),
                "confidence": 0.9,  # High confidence as it's a processed box
                "class_id": -1,
                "class_name": PENIS_CLASS_NAME_CONST,  # Use a common class name
                "status": "ATR_VISIBLE_P",  # Custom status
                "track_id": None,  # Derived concept
                "width": w, "height": h,
                "cx": kp_box[0] + w / 2,
                "cy": kp_box[1] + h / 2,
                "area_perc": 0,  # Can calculate if needed
                "is_excluded": False,
                "role_in_frame": "pref_penis",  # <--- USE EXISTING ROLE FOR DISTINCT COLOR
            }
            frame_data["yolo_boxes"].append(overlay_kp_box)

        # General detections (including those marked 'is_tracked' by ATR)
        for box_obj in self.boxes:
            if not box_obj.is_excluded:  # Process only non-excluded boxes
                box_data = box_obj.to_dict()  # This includes 'is_tracked'
                box_data["role_in_frame"] = "general_detection"

                # Add ATR debug info if available (this part is fine)
                for atr_contact in self.atr_detected_contact_boxes:
                    # Ensure atr_contact['box_rec'] exists and its track_id matches
                    if 'box_rec' in atr_contact and atr_contact['box_rec'].track_id == box_obj.track_id and \
                            atr_contact.get('class_name') == box_obj.class_name:
                        box_data['atr_debug_info'] = atr_contact.get('position', '')
                        box_data['atr_iou_w_lp'] = atr_contact.get('iou', 0)
                        box_data['atr_speed_val'] = atr_contact.get('speed', 0)
                        break
                frame_data["yolo_boxes"].append(box_data)
        return frame_data

    def __repr__(self):
        return (f"FrameObject(id={self.frame_id}, num_boxes={len(self.boxes)}, "
                f"atr_pos='{self.atr_assigned_position}', atr_dist={self.atr_funscript_distance}, "
                f"atr_lp_active={'Yes' if self.atr_locked_penis_state.active else 'No'})")


class ATRSegment(BaseSegment):  # Replacing PenisSegment and SexActSegment with a single type from ATR
    def __init__(self, start_frame_id: int, end_frame_id: int, major_position: str):
        super().__init__(start_frame_id, end_frame_id)
        self.major_position = major_position  # From ATR's _aggregate_segments
        # Store frame objects belonging to this segment for easy access
        self.segment_frame_objects: List[FrameObject] = []

    def to_dict(self) -> Dict[str, Any]:
        # Mimic SexActSegment.to_dict() structure for GUI compatibility
        # Occlusion info might need to be re-evaluated based on ATR's continuous distance
        occlusion_info = []  # Placeholder, ATR logic doesn't directly define occlusions like S2

        # Map ATR major_position to position_long_name and short_name if possible
        # This requires POSITION_INFO_MAPPING_CONST to understand ATR's position strings
        # For now, use major_position directly or a generic mapping.

        # 1. position_long_name_val is straightforward:
        position_long_name_val = self.major_position

        # 2. To find position_short_name_val, we need to search the dictionary:
        position_short_name_val = "N/A"  # Default if not found

        for key, info in POSITION_INFO_MAPPING_CONST.items():
            if info["long_name"] == self.major_position:
                position_short_name_val = info["short_name"]
                break  # Found it, no need to continue

        # ATR's funscript generation is 1D. Range/offset might not directly map.
        # These can be calculated from the self.atr_funscript_distance values within this segment's frames.
        raw_range_val_ud = 0
        raw_range_offset_ud = 0
        if self.segment_frame_objects:
            distances_in_segment = [fo.atr_funscript_distance for fo in self.segment_frame_objects]
            if distances_in_segment:
                min_d, max_d = min(distances_in_segment), max(distances_in_segment)
                raw_range_val_ud = max_d - min_d
                # Offset isn't directly analogous from ATR's logic in a simple way for U/D.

        return {'start_frame_id': self.start_frame_id, 'end_frame_id': self.end_frame_id,
                'class_name': self.major_position,  # Using major_position as class_name
                'position_long_name': position_long_name_val,
                'position_short_name': position_short_name_val,
                'segment_type': "ATR_Segment", 'duration': self.duration,  # Use a new type
                'occlusions': occlusion_info,  # Placeholder
                'raw_range_val_ud': raw_range_val_ud,
                'raw_range_offset_ud': raw_range_offset_ud,
                'raw_range_val_lr': 0  # ATR logic is primarily single axis
                }

    def __repr__(self):
        return (f"ATRSegment(id={self.id}, frames {self.start_frame_id}-{self.end_frame_id}, "
                f"pos='{self.major_position}', duration={self.duration})")


class AppStateContainer:
    def __init__(self, video_info: Dict, yolo_input_size: int, vr_filter: bool,
                 all_frames_raw_detections: list,
                 logger: Optional[logging.Logger],
                 discarded_classes_runtime_arg: Optional[List[str]] = None,
                 scripting_range_active: bool = False,
                 scripting_range_start_frame: Optional[int] = None,
                 scripting_range_end_frame: Optional[int] = None,
                 is_ranged_data_source: bool = False):

        self.video_info = video_info
        self.yolo_input_size = yolo_input_size
        self.vr_vertical_third_filter = vr_filter  # This is the general VR filter for non-penis boxes
        self.frames: List[FrameObject] = []
        FrameObject._id_counter = 0

        self.effective_discard_classes = set(CLASSES_TO_DISCARD_CONST)
        if discarded_classes_runtime_arg:
            self.effective_discard_classes.update(discarded_classes_runtime_arg)
        if logger and discarded_classes_runtime_arg:
            logger.info(f"Stage 2 effective discarded classes: {sorted(list(self.effective_discard_classes))}")

        self.scripting_range_active = scripting_range_active
        self.scripting_range_start_frame = scripting_range_start_frame
        self.scripting_range_end_frame = scripting_range_end_frame
        if logger and self.scripting_range_active:
            logger.info(
                f"AppStateContainer initialized with active scripting range: Start={self.scripting_range_start_frame}, End={self.scripting_range_end_frame}")

        # --- LOGIC for frame_id offset ---
        frame_id_offset = 0
        # The offset is only applied if the data source itself is a ranged file.
        # If we are processing a range from a full data source, the offset must be 0.
        if is_ranged_data_source and self.scripting_range_start_frame is not None:
            frame_id_offset = self.scripting_range_start_frame
            if logger:
                logger.info(f"AppStateContainer applying frame ID offset of {frame_id_offset} (ranged data source).")

        for i, raw_dets_for_frame in enumerate(all_frames_raw_detections):
            absolute_frame_id = i + frame_id_offset
            fo = FrameObject(frame_id=absolute_frame_id, yolo_input_size=yolo_input_size,
                             raw_detections_input=raw_dets_for_frame,
                             classes_to_discard_runtime_set=self.effective_discard_classes)

            # VR Filter for NON-PENIS boxes (from original Stage 2)
            # ATR logic has a similar concept of "central_boxes"
            if video_info.get('actual_video_type') == 'VR' and vr_filter:
                for box_rec in fo.boxes:
                    # ATR used frame_width/3 to 2*frame_width/3. YOLO input size is the frame_width here.
                    center_x_cond = (yolo_input_size / 3 <= box_rec.cx <= 2 * yolo_input_size / 3)
                    # ATR's central third also included a y-check: y1 <= cy <= y2
                    # center_y_cond = (yolo_input_size / 3 <= box_rec.cy <= 2 * yolo_input_size / 3)
                    if box_rec.class_name != PENIS_CLASS_NAME_CONST and not center_x_cond:  # Only X filter for now
                        box_rec.is_excluded = True
                        box_rec.status = "Excluded_VR_Filter_Peripheral"
            self.frames.append(fo)

        self.atr_segments: List[ATRSegment] = []  # Store segments from ATR logic

        self.funscript_frames: List[int] = []
        self.funscript_distances: List[int] = []
        self.funscript_distances_lr: List[int] = []


# --- ATR Helper Functions (to be moved into this file) ---
def _atr_calculate_iou(box1: Tuple[float, float, float, float], box2: Tuple[float, float, float, float]) -> float:
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
    return inter_area / union_area if union_area > 0 else 0


def _atr_assign_frame_position(contacts: List[Dict]) -> str:
    if not contacts:
        return "Not Relevant"

    detected_class_names = {contact["class_name"] for contact in contacts} # Use a set for faster lookups

    # More explicit prioritization
    if 'pussy' in detected_class_names:
        return 'Cowgirl / Missionary' # Highest priority if pussy is involved
    if 'butt' in detected_class_names:
        return 'Rev. Cowgirl / Doggy' # Next highest

    # If neither pussy nor butt, then check for other combinations
    has_face = 'face' in detected_class_names
    has_hand = 'hand' in detected_class_names
    has_breast = 'breast' in detected_class_names
    has_foot = 'foot' in detected_class_names

    if has_face and has_hand: return 'Handjob / Blowjob'
    if has_face: return 'Handjob / Blowjob'

    # If only hand is primary after pussy/butt/face checks
    if has_hand and has_breast: return 'Boobjob'
    if has_hand: return 'Handjob / Blowjob' # Default hand interaction

    if has_foot: return 'Footjob'
    if has_breast: return 'Boobjob' # If only breast contact (e.g. paizuri without hands visible)

    return "Not Relevant"


def _atr_aggregate_segments(frame_objects: List[FrameObject], fps: float, min_segment_duration_frames: int,
                            logger: logging.Logger) -> List[ATRSegment]:
    segments_internal = []  # list of dicts: {"start_frame":int, "end_frame":int, "positions_in_segment":List[str]}
    current_segment_internal = None

    for fo in frame_objects:
        position = fo.atr_assigned_position  # Assumes this is pre-filled
        if current_segment_internal is None:
            current_segment_internal = {"start_frame": fo.frame_id, "end_frame": fo.frame_id,
                                        "positions_in_segment": [position]}
        else:
            # Determine major position of current_segment_internal
            if not current_segment_internal["positions_in_segment"]:  # Should not happen if segment exists
                # Fallback: treat current frame's position as the segment's major
                segment_major_position = position
            else:
                segment_major_position = max(set(current_segment_internal["positions_in_segment"]),
                                             key=current_segment_internal["positions_in_segment"].count)

            if position == segment_major_position:
                current_segment_internal["end_frame"] = fo.frame_id
                current_segment_internal["positions_in_segment"].append(position)
            else:
                # Check for consistent change (ATR's 1-second rule)
                consistent_change_frames = int(fps)
                consistent_change = True
                for i in range(1, consistent_change_frames + 1):
                    next_frame_idx = fo.frame_id + i
                    if next_frame_idx >= len(frame_objects) or frame_objects[
                        next_frame_idx].atr_assigned_position != position:
                        consistent_change = False
                        break

                if consistent_change:
                    if current_segment_internal["end_frame"] - current_segment_internal[
                        "start_frame"] + 1 >= min_segment_duration_frames:
                        segments_internal.append(deepcopy(current_segment_internal))
                    current_segment_internal = {"start_frame": fo.frame_id, "end_frame": fo.frame_id,
                                                "positions_in_segment": [position]}
                else:
                    current_segment_internal["end_frame"] = fo.frame_id
                    current_segment_internal["positions_in_segment"].append(position)

    if current_segment_internal and current_segment_internal["end_frame"] - current_segment_internal[
        "start_frame"] + 1 >= min_segment_duration_frames:
        segments_internal.append(deepcopy(current_segment_internal))

    # Convert to ATRSegment objects and merge
    atr_segments_final: List[ATRSegment] = []
    if not segments_internal: return []

    for seg_dict in segments_internal:
        major_pos = max(set(seg_dict["positions_in_segment"]), key=seg_dict["positions_in_segment"].count) if seg_dict[
            "positions_in_segment"] else "Not Relevant"
        new_atr_seg = ATRSegment(seg_dict["start_frame"], seg_dict["end_frame"], major_pos)
        new_atr_seg.segment_frame_objects = [fobj for fobj in frame_objects if
                                             new_atr_seg.start_frame_id <= fobj.frame_id <= new_atr_seg.end_frame_id]

        if atr_segments_final and new_atr_seg.major_position == atr_segments_final[-1].major_position:
            atr_segments_final[-1].end_frame_id = new_atr_seg.end_frame_id
            atr_segments_final[-1].segment_frame_objects.extend(new_atr_seg.segment_frame_objects)  # Append frames
            atr_segments_final[-1].update_duration()
        else:
            atr_segments_final.append(new_atr_seg)

    # ATR's final filter for short segments (merge with previous)
    final_filtered_segments: List[ATRSegment] = []
    for atr_seg in atr_segments_final:
        if atr_seg.duration >= min_segment_duration_frames:
            final_filtered_segments.append(atr_seg)
        elif final_filtered_segments:  # If short and there's a previous one
            final_filtered_segments[-1].end_frame_id = atr_seg.end_frame_id
            final_filtered_segments[-1].segment_frame_objects.extend(atr_seg.segment_frame_objects)
            final_filtered_segments[-1].update_duration()
        # else: short segment is the first one, gets discarded if below threshold

    _debug_log(logger,
               f"ATR Segments: initial={len(segments_internal)}, merged={len(atr_segments_final)}, final_filtered={len(final_filtered_segments)}")
    return final_filtered_segments


def _atr_calculate_normalized_distance_to_base(locked_penis_box_coords: Tuple[float, float, float, float],
                                               class_name: str,
                                               class_box_coords: Tuple[float, float, float, float],
                                               max_distance_ref: float) -> float:
    penis_base_y = locked_penis_box_coords[3]  # y2

    box_y_ref = 0
    if class_name == 'face':
        box_y_ref = class_box_coords[3]  # y2 (bottom of face)
    elif class_name == 'hand':
        box_y_ref = class_box_coords[1]  # y1 (top of hand)
    elif class_name == 'butt':
        box_y_ref = (9 * class_box_coords[3] + class_box_coords[1]) / 10  # Mostly bottom of butt
    else:
        box_y_ref = (class_box_coords[3] + class_box_coords[1]) / 2  # Center of other parts like pussy, breast, foot

    raw_distance = penis_base_y - box_y_ref  # Larger means penis is further up (less penetration)

    if max_distance_ref <= 0: return 50.0  # Avoid division by zero, default to mid

    normalized_distance = (raw_distance / max_distance_ref) * 100.0
    return np.clip(normalized_distance, 0, 100)


def _atr_normalize_funscript_sparse_per_segment(state: AppStateContainer, logger: Optional[logging.Logger]):
    """ Normalizes funscript distances (atr_funscript_distance on FrameObject) per ATRSegment.
        Similar to ATR's _normalize_funscript_sparse.
    """
    for atr_seg in state.atr_segments:
        if not atr_seg.segment_frame_objects: continue

        values = [fo.atr_funscript_distance for fo in atr_seg.segment_frame_objects]
        if not values: continue

        p01 = np.percentile(values, 5)
        p99 = np.percentile(values, 95)

        filtered_values = [v for v in values if p01 <= v <= p99]
        min_val_in_segment = min(values)  # True min for scaling outliers
        max_val_in_segment = max(values)  # True max for scaling outliers

        filtered_min = min(filtered_values) if filtered_values else min_val_in_segment
        filtered_max = max(filtered_values) if filtered_values else max_val_in_segment

        scale_range = filtered_max - filtered_min

        for fo in atr_seg.segment_frame_objects:
            val = float(fo.atr_funscript_distance)
            original_val_for_debug = val

            if atr_seg.major_position in ['Not Relevant', 'Close up']:
                val = 100.0
            else:
                if val <= p01:  # Outlier low
                    # Map to 0-5 based on position relative to true min_val_in_segment
                    # Avoid division by zero if p01 is the same as min_val_in_segment
                    denominator = p01 - min_val_in_segment
                    val = ((val - min_val_in_segment) / denominator) * 5.0 if denominator > 1e-6 else 0.0
                elif val >= p99:  # Outlier high
                    # Map to 95-100 based on position relative to true max_val_in_segment
                    denominator = max_val_in_segment - p99
                    val = 95.0 + ((val - p99) / denominator) * 5.0 if denominator > 1e-6 else 100.0
                else:  # Non-outlier
                    if scale_range < 1e-6:  # No variation in non-outliers
                        val = 50.0
                    else:  # Scale to 5-95
                        val = 5.0 + ((val - filtered_min) / scale_range) * 90.0

            fo.atr_funscript_distance = int(np.clip(round(val), 0, 100))


# --- New Step 0: Global Object Tracking (Simple IoU Tracker) ---
def simple_iou_tracker_step0(state: AppStateContainer, logger: Optional[logging.Logger]):
    _debug_log(logger, "Starting Step 0: Simple IoU Object Tracking")

    next_global_track_id = 1  # Start global track IDs from 1
    active_tracks: Dict[int, Dict[str, Any]] = {}  # {global_track_id: {"box_rec": BoxRecord, "frames_unseen": 0}}

    # Tracking parameters (can be tuned or moved to constants)
    iou_threshold = 0.3  # Min IoU to associate a detection with an existing track
    max_frames_unseen_to_kill_track = int(
        state.video_info.get('fps', 30) * 0.5)  # How many frames a track can be unseen

    for frame_obj in sorted(state.frames, key=lambda f: f.frame_id):
        current_detections = [b for b in frame_obj.boxes if not b.is_excluded]  # Use non-excluded boxes

        # Sort detections (e.g., by confidence or size) for more stable matching if needed
        # current_detections.sort(key=lambda b: b.confidence, reverse=True)

        matched_detection_indices_this_frame = [False] * len(current_detections)

        # Try to match active tracks with current detections
        track_ids_to_delete = []
        for track_id, track_data in active_tracks.items():
            last_box_rec = track_data["box_rec"]
            best_match_idx = -1
            max_iou = -1

            for i, det_box_rec in enumerate(current_detections):
                if matched_detection_indices_this_frame[i]: continue  # This detection already matched
                if det_box_rec.class_name != last_box_rec.class_name: continue  # Match only same class

                iou = _atr_calculate_iou(last_box_rec.bbox, det_box_rec.bbox)
                if iou > iou_threshold and iou > max_iou:
                    max_iou = iou
                    best_match_idx = i

            if best_match_idx != -1:  # Found a match
                matched_det = current_detections[best_match_idx]
                matched_det.track_id = track_id  # Assign global track_id
                active_tracks[track_id]["box_rec"] = matched_det  # Update track with new box
                active_tracks[track_id]["frames_unseen"] = 0
                matched_detection_indices_this_frame[best_match_idx] = True
            else:  # No match for this active track
                active_tracks[track_id]["frames_unseen"] += 1
                if active_tracks[track_id]["frames_unseen"] > max_frames_unseen_to_kill_track:
                    track_ids_to_delete.append(track_id)

        # Remove lost tracks
        for track_id in track_ids_to_delete:
            del active_tracks[track_id]

        # Create new tracks for unmatched detections
        for i, det_box_rec in enumerate(current_detections):
            if not matched_detection_indices_this_frame[i]:
                det_box_rec.track_id = next_global_track_id
                active_tracks[next_global_track_id] = {"box_rec": det_box_rec, "frames_unseen": 0}
                next_global_track_id += 1

    _debug_log(logger, f"Step 0: Tracking complete. Assigned up to global_track_id {next_global_track_id - 1}.")


# --- ATR Analysis Steps (modified to use FrameObject and BoxRecord) ---
def atr_pass_1_interpolate_boxes(state: AppStateContainer, logger: Optional[logging.Logger]):
    _debug_log(logger, "Starting ATR Pass 1: Interpolate Boxes (using S2 Tracker IDs)")
    track_history: Dict[int, Dict[str, Any]] = {}
    total_interpolated = 0
    MAX_GAP_FRAMES = int(state.video_info.get('fps', 30) * 0.3)  # Max 0.3 second gap for interpolation

    all_frames_sorted = sorted(state.frames, key=lambda f: f.frame_id)

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
                            # _debug_log(logger, f"Skipping interpolation for track {box_rec.track_id} in frame {i_frame_id}, already exists.")
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

                        # ATR had a np.linalg.norm(np.array(interpolated_box[:2]) - np.array(last_box[:2])) < 50
                        # This means top-left corner movement. Let's use center movement relative to width/height.
                        # Heuristic threshold: allow movement proportional to the box size and gap length
                        max_interp_move_thresh = max(last_box_rec.width,
                                                     last_box_rec.height) * frame_gap * 0.75  # Increased multiplier slightly for flexibility

                        if dist_moved_center < max_interp_move_thresh:
                            interpolated_br = BoxRecord(
                                frame_id=i_frame_id,
                                bbox=interp_bbox_np,
                                confidence=min(last_box_rec.confidence, box_rec.confidence) * 0.8,  # Reduced conf
                                class_id=box_rec.class_id,  # Assume class is consistent
                                class_name=box_rec.class_name,
                                status=STATUS_GENERATED_LINEAR,  # Mark as interpolated
                                yolo_input_size=frame_obj.yolo_input_size,
                                track_id=box_rec.track_id
                            )
                            target_gap_frame_obj.boxes.append(interpolated_br)
                            total_interpolated += 1

            track_history[box_rec.track_id] = {"last_frame_obj": frame_obj, "last_box_rec": box_rec}
    _debug_log(logger, f"ATR Pass 1: Interpolated {total_interpolated} boxes.")


def atr_pass_3_build_locked_penis(state: AppStateContainer, logger: Optional[logging.Logger]):
    """ ATR Pass 3: Build the locked penis box concept for each frame.
        Operates on state.frames. Modifies frame_obj.atr_locked_penis_state and frame_obj.atr_penis_box_kalman
    """
    _debug_log(logger, "Starting ATR Pass 3: Build Locked Penis")
    fps = state.video_info.get('fps', 30.0)
    yolo_size = state.yolo_input_size

    # Initialize overall locked_penis state (carries over frames within a continuous presence)
    # This is different from ATR's `locked_penis` dict that seemed to reset less often.
    # We'll manage a running state here.
    current_lp_active = False
    current_lp_last_raw_box_coords: Optional[
        Tuple[float, float, float, float]] = None  # Store the raw detected/selected penis box
    current_lp_max_height = 0.0
    current_lp_max_penetration_height = 0.0
    current_lp_area = 0.0
    current_lp_consecutive_detections = 0
    current_lp_consecutive_non_detections = 0
    # current_lp_glans_detected = False # This is per-frame

    # ATR Kalman for height (2 state vars: height, d_height; 1 measurement: height)
    kf_height = cv2.KalmanFilter(2, 1)
    kf_height.measurementMatrix = np.array([[1, 0]], np.float32)
    kf_height.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)  # Assumes dt=1 frame for simplicity
    kf_height.processNoiseCov = np.eye(2, dtype=np.float32) * 0.01
    kf_height.measurementNoiseCov = np.eye(1, dtype=np.float32) * 0.1
    kf_height.errorCovPost = np.eye(2, dtype=np.float32) * 1.0  # Initial uncertainty
    # Initialize state (e.g., height=50, d_height=0) - will be updated by first good detection
    kf_height.statePost = np.array([[yolo_size * 0.1], [0]], dtype=np.float32)

    for frame_obj in state.frames:
        # Determine preferred penis for this frame (can use FrameObject's existing method)
        # ATR Pass 3 used "central_boxes" then selected penis with max y2.
        # Let's use frame_obj.get_preferred_penis_box() for simplicity and consistency.
        # VR filter for central third is handled by AppStateContainer init for all boxes.
        # get_preferred_penis_box also has a VR filter if needed.

        # Filter boxes to central third for ATR's logic if VR (already done in AppStateContainer init)
        # For non-VR, ATR still used "central_boxes" but the definition was for the whole frame if not VR.
        # We can just use frame_obj.boxes and let get_preferred_penis_box handle selection.

        selected_penis_box_rec = frame_obj.get_preferred_penis_box(
            actual_video_type=state.video_info.get('actual_video_type', '2D'),
            vr_vertical_third_filter=state.vr_vertical_third_filter  # This is the general one
        )

        # Per-frame glans detection state
        frame_glans_detected = False

        if selected_penis_box_rec:
            current_lp_consecutive_detections += 1
            current_lp_consecutive_non_detections = 0
            current_lp_last_raw_box_coords = tuple(selected_penis_box_rec.bbox)  # Store raw box for position

            current_raw_height = selected_penis_box_rec.height

            # Update max_height (ATR logic: only increase unless glans detected)
            if current_raw_height > current_lp_max_height:
                current_lp_max_height = current_raw_height
                # ATR set max_penetration_height as 0.65 * current_height when max_height was updated
                current_lp_max_penetration_height = 0.65 * current_raw_height

            # Check for glans contact with this selected_penis_box_rec
            glans_boxes_in_frame = [b for b in frame_obj.boxes if
                                    b.class_name == GLANS_CLASS_NAME_CONST and not b.is_excluded]
            for glans_br in glans_boxes_in_frame:
                if _atr_calculate_iou(selected_penis_box_rec.box, glans_br.box) > 0.05:  # Small IOU to confirm overlap
                    frame_glans_detected = True
                    # ATR: Adjust height if glans detected
                    # glans_bottom_height = selected_penis_box_rec.bbox[3] - glans_br.bbox[3] (dist from penis bottom to glans bottom)
                    # glans_top_height = selected_penis_box_rec.bbox[3] - glans_br.bbox[1] (dist from penis bottom to glans top)
                    # ATR logic for height update with glans:
                    # current_lp_max_penetration_height = max(0, min(current_lp_max_height, glans_bottom_height))
                    # current_lp_max_height = max(0, min(current_lp_max_height, current_raw_height), glans_top_height)
                    # Simplified: if glans is detected, the currently visible height is a better estimate for "full extension" for that moment
                    # This part of ATR logic is tricky and might need tuning.
                    # For now, let's say glans detection means the current_raw_height is considered more authoritative
                    # and might refine max_height downwards if current raw height is smaller.
                    if current_raw_height < current_lp_max_height:  # If glans visible and current height is less than established max
                        current_lp_max_height = current_raw_height
                    current_lp_max_penetration_height = current_raw_height * 0.65  # Penetration depth based on current visible glans

                    break  # Found a glans associated with this penis

            # ATR: Navel check to limit "giant penis" error
            # This can be added if navel detection is reliable.

            if not current_lp_active and current_lp_consecutive_detections >= fps / 5:  # Activate lock
                current_lp_active = True
                # Initialize Kalman filter state with current height if it's the first activation in a sequence
                kf_height.statePost = np.array([[current_raw_height], [0]], dtype=np.float32)
                _debug_log(logger, f"ATR LP Lock ACTIVATE at frame {frame_obj.frame_id}, h={current_raw_height:.0f}")

            # Update Kalman filter with observed current_raw_height
            kf_height.correct(np.array([[current_raw_height]], dtype=np.float32))
            frame_obj.atr_penis_box_kalman = None  # Will be set below if active

        else:  # No penis detection this frame
            current_lp_consecutive_detections = 0
            current_lp_consecutive_non_detections += 1

            # ATR: Deactivate lock after 3*fps non-detections AND if not in "penetration"
            # "penetration" status isn't determined until Pass 4/5.
            # For now, use a simpler rule or rely on a general timeout for non-detection.
            if current_lp_active and current_lp_consecutive_non_detections >= 3 * fps:
                _debug_log(logger,
                           f"ATR LP Lock DEACTIVATE (timeout) at frame {frame_obj.frame_id} after {current_lp_consecutive_non_detections} non-det")
                current_lp_active = False
                # Reset max_height when lock deactivates? ATR didn't explicitly reset it, it carried over.
                # This could lead to a large locked box if a new penis appears much smaller.
                # Consider resetting max_height or having a decay mechanism if lock is lost for long.
                # For now, keep ATR's behavior of carrying over max_height.

        # Store ATR locked penis state for the frame
        lp_state = frame_obj.atr_locked_penis_state
        lp_state.active = current_lp_active
        lp_state.consecutive_detections = current_lp_consecutive_detections
        lp_state.consecutive_non_detections = current_lp_consecutive_non_detections
        lp_state.max_height = current_lp_max_height
        lp_state.max_penetration_height = current_lp_max_penetration_height
        lp_state.glans_detected = frame_glans_detected  # Per-frame glans status

        if current_lp_active and current_lp_last_raw_box_coords:
            predicted_kalman_state = kf_height.predict()
            predicted_height_kalman = predicted_kalman_state[0, 0]

            # Constrain predicted height
            predicted_height_final = np.clip(predicted_height_kalman, 0, current_lp_max_height)

            # The "visible" penis box based on Kalman prediction for height
            # Position (x1, x2, y2) comes from last raw detection. y1 is derived from predicted_height_final.
            x1, _, x2, y2_raw = current_lp_last_raw_box_coords
            kalman_penis_y1 = y2_raw - predicted_height_final
            frame_obj.atr_penis_box_kalman = (x1, kalman_penis_y1, x2, y2_raw)

            # The "locked_penis_box" for contact checks uses max_height
            locked_penis_y1 = y2_raw - current_lp_max_height
            lp_state.box = (x1, locked_penis_y1, x2, y2_raw)

            lp_state.area = (x2 - x1) * current_lp_max_height if current_lp_max_height > 0 else 0
            lp_state.visible_part = (
                                                predicted_height_final / current_lp_max_height) * 100.0 if current_lp_max_height > 0 else 0.0
        else:
            # If not active, or no last_raw_box, no specific ATR box for this frame
            lp_state.box = None
            lp_state.area = 0.0
            lp_state.visible_part = 0.0  # Or 100.0 if no detection implies full? ATR implies 0 if not active.
            frame_obj.atr_penis_box_kalman = None


def atr_pass_4_assign_positions_and_segments(state: AppStateContainer, logger: Optional[logging.Logger]):
    """ ATR Pass 4: Assign frame positions and aggregate into segments.
        Modifies frame_obj.atr_assigned_position and populates state.atr_segments.
    """
    _debug_log(logger, "Starting ATR Pass 4: Assign Positions & Segments")
    fps = state.video_info.get('fps', 30.0)
    yolo_size = state.yolo_input_size
    frame_area = yolo_size * yolo_size

    for frame_obj in state.frames:
        # Reset is_tracked for all boxes in the frame first
        for box in frame_obj.boxes:
            box.is_tracked = False

        assigned_pos_for_frame = "Not Relevant"  # Default
        frame_obj.atr_detected_contact_boxes = []  # Clear previous contacts for this frame

        if frame_obj.atr_locked_penis_state.active and frame_obj.atr_locked_penis_state.box:
            lp_box_coords = frame_obj.atr_locked_penis_state.box
            lp_area = frame_obj.atr_locked_penis_state.area

            contacts_for_frame_determination = []  # For _atr_assign_frame_position

            for box_rec in frame_obj.boxes:
                if box_rec.is_excluded or box_rec.class_name == PENIS_CLASS_NAME_CONST or box_rec.class_name == GLANS_CLASS_NAME_CONST:
                    continue

                iou_with_lp = _atr_calculate_iou(lp_box_coords, box_rec.bbox)
                if iou_with_lp > 0.05:
                    valid_contact_for_pos_assignment = True
                    if box_rec.class_name in ["hand", "foot"]:
                        if not (0.5 * lp_area <= box_rec.area <= 3.0 * lp_area) and lp_area > 1e-6:
                            valid_contact_for_pos_assignment = False

                    if valid_contact_for_pos_assignment:
                        # Add to list for position assignment
                        contacts_for_frame_determination.append({"class_name": box_rec.class_name, "box_rec": box_rec})
                        # Store all valid contacts for potential use in Pass 5 (distance calc)
                        # The 'position' and 'iou' debug info will be added in Pass 5 logic if needed
                        frame_obj.atr_detected_contact_boxes.append(
                            {"class_name": box_rec.class_name, "box_rec": box_rec, "iou": iou_with_lp})

            assigned_pos_for_frame = _atr_assign_frame_position(contacts_for_frame_determination)

            # Now, mark the specific boxes that CONTRIBUTED to this assigned_pos_for_frame as 'is_tracked'
            # This is a heuristic. If 'Cowgirl / Missionary', the 'pussy' box is tracked.
            # If 'Handjob / Blowjob', 'face' and 'hand' boxes are tracked.
            if assigned_pos_for_frame != "Not Relevant" and assigned_pos_for_frame != "Close Up":
                contributing_classes = []
                if assigned_pos_for_frame == 'Cowgirl / Missionary':
                    contributing_classes = ['pussy']
                elif assigned_pos_for_frame == 'Rev. Cowgirl / Doggy':
                    contributing_classes = ['butt']
                elif assigned_pos_for_frame == 'Handjob / Blowjob':
                    contributing_classes = ['face', 'hand']
                elif assigned_pos_for_frame == 'Boobjob':
                    contributing_classes = ['breast', 'hand']  # Or just breast
                elif assigned_pos_for_frame == 'Footjob':
                    contributing_classes = ['foot']

                for contact_dict in frame_obj.atr_detected_contact_boxes:
                    if contact_dict["class_name"] in contributing_classes:
                        contact_dict["box_rec"].is_tracked = True

        elif not frame_obj.atr_locked_penis_state.active:
            for box_rec in frame_obj.boxes:
                if box_rec.is_excluded: continue
                if box_rec.class_name in ["pussy", "butt"]:
                    if box_rec.area > 0.07 * frame_area:
                        assigned_pos_for_frame = "Close Up"
                        break
        frame_obj.atr_assigned_position = assigned_pos_for_frame

    min_segment_duration_sec = 1.0
    min_segment_duration_frames = int(min_segment_duration_sec * fps)

    BaseSegment._id_counter = 0
    state.atr_segments = _atr_aggregate_segments(state.frames, fps, min_segment_duration_frames, logger)
    _debug_log(logger,
               f"ATR Pass 4: Assigned positions. Created {len(state.atr_segments)} ATR segments after extended merging.")


def atr_pass_5_determine_distance(state: AppStateContainer, logger: Optional[logging.Logger]):
    _debug_log(logger, "Starting ATR Pass 5: Determine Frame Distances")
    fps = state.video_info.get('fps', 30.0)

    # ATR's transition logic states (not used in this simplified port yet)
    # transition_active = False
    # transition_frames_total = 2 * int(fps)
    # remaining_transition_frames = 0

    for frame_obj in state.frames:
        current_pos_for_frame = frame_obj.atr_assigned_position
        comp_dist_for_frame = 0.0  # Raw calculated distance
        num_touching_relevant_for_frame = 0

        # Find which ATRSegment this frame belongs to (if any)
        # This is redundant if we iterate through segments then frames.
        # Simpler: get position directly from frame_obj.atr_assigned_position

        if not frame_obj.atr_locked_penis_state.active or not frame_obj.atr_locked_penis_state.box:
            comp_dist_for_frame = 100.0  # Default if no active locked penis
        else:
            lp_state = frame_obj.atr_locked_penis_state
            lp_box_coords = lp_state.box

            relevant_classes_for_pos = []
            is_penetration_pos = False

            if current_pos_for_frame == "Cowgirl / Missionary":
                relevant_classes_for_pos = ["pussy"]
                if not lp_state.glans_detected: is_penetration_pos = True
            elif current_pos_for_frame == "Rev. Cowgirl / Doggy":
                relevant_classes_for_pos = ["butt"]
                if not lp_state.glans_detected: is_penetration_pos = True
            elif current_pos_for_frame == "Handjob / Blowjob":
                relevant_classes_for_pos = ["face", "hand"]
            elif current_pos_for_frame == "Boobjob":
                relevant_classes_for_pos = ["breast"]
            elif current_pos_for_frame == "Footjob":
                relevant_classes_for_pos = ["foot"]

            if not relevant_classes_for_pos:  # Includes "Not Relevant" and "Close Up"
                comp_dist_for_frame = 100.0
            else:
                # Use pre-stored atr_detected_contact_boxes from Pass 4
                # This list should already be filtered by IOU and area constraints.

                # ATR's logic for Handjob/Blowjob used weighted average by speed.
                # This requires storing speed per track_id, which is complex here.
                # Simplified: average distance from relevant contacts.

                contacting_relevant_boxes_distances = []
                for contact_dict in frame_obj.atr_detected_contact_boxes:
                    class_name = contact_dict["class_name"]
                    box_rec: BoxRecord = contact_dict["box_rec"]  # Assumes box_rec was stored

                    if class_name in relevant_classes_for_pos:
                        max_dist_ref = lp_state.max_penetration_height if is_penetration_pos else lp_state.max_height
                        if max_dist_ref <= 0: max_dist_ref = lp_state.max_height  # fallback if penetration height is zero
                        if max_dist_ref <= 0: max_dist_ref = state.yolo_input_size * 0.2  # ultimate fallback for ref

                        dist_val = _atr_calculate_normalized_distance_to_base(
                            lp_box_coords, class_name, box_rec.bbox, max_dist_ref
                        )
                        contacting_relevant_boxes_distances.append(dist_val)
                        num_touching_relevant_for_frame += 1

                if contacting_relevant_boxes_distances:
                    comp_dist_for_frame = sum(contacting_relevant_boxes_distances) / len(
                        contacting_relevant_boxes_distances)
                elif is_penetration_pos:  # Penetration pos but no direct relevant class contact, use visible part
                    comp_dist_for_frame = lp_state.visible_part  # visible_part is 0-100, higher means less penetrated
                else:  # No relevant contacts, and not a penetration pos
                    comp_dist_for_frame = 100.0

        frame_obj.atr_funscript_distance = int(np.clip(round(comp_dist_for_frame), 0, 100))

    _debug_log(logger, "ATR Pass 5: Determined raw frame distances.")


def atr_pass_6_smooth_and_normalize_distances(state: AppStateContainer, logger: Optional[logging.Logger]):
    """ ATR Pass 6 (denoising with SG) and Pass 9 (amplifying/normalizing per segment) combined """
    _debug_log(logger, "Starting ATR Pass 6 & 9: Smooth and Normalize Distances")

    all_raw_distances = [fo.atr_funscript_distance for fo in state.frames]
    if not all_raw_distances or len(all_raw_distances) < 11:  # Savgol needs enough points
        _debug_log(logger, "Not enough data points for Savitzky-Golay filter.")
    else:
        # ATR used (11,2) for Savgol
        smoothed_distances = savgol_filter(all_raw_distances, window_length=11, polyorder=2)
        for i, fo in enumerate(state.frames):
            fo.atr_funscript_distance = int(np.clip(round(smoothed_distances[i]), 0, 100))
        _debug_log(logger, "Applied Savitzky-Golay filter to distances.")

    # Now apply ATR's per-segment normalization (_normalize_funscript_sparse equivalent)
    _atr_normalize_funscript_sparse_per_segment(state, logger)
    _debug_log(logger, "Applied per-segment normalization to distances.")


def atr_pass_7_8_simplify_signal(state: AppStateContainer, logger: Optional[logging.Logger]):
    """ ATR Pass 7 & 8: Simplify the funscript signal using peak/valley and RDP.
        This modifies state.funscript_frames and state.funscript_distances.
        It assumes atr_funscript_distance on FrameObjects is the signal to simplify.
    """
    _debug_log(logger, "Starting ATR Pass 7 & 8: Simplify Signal")

    # Initial data: (frame_id, position)
    full_script_data = [(fo.frame_id, fo.atr_funscript_distance) for fo in state.frames]
    if not full_script_data:
        _debug_log(logger, "No data to simplify.")
        return

    # Skip simplification
    skip_simplification = True

    if skip_simplification:
        # Store the final simplified actions in state
        state.funscript_frames = [action[0] for action in full_script_data]
        state.funscript_distances = [int(action[1]) for action in full_script_data]
        return

    frames_np, positions_np = zip(*full_script_data)
    positions_np = np.array(positions_np, dtype=float)

    # ATR Pass 7: Peak/Valley detection
    peaks, _ = find_peaks(positions_np, prominence=0.1)
    valleys, _ = find_peaks(-positions_np, prominence=0.1)  # For minima

    keep_indices_pv = sorted(list(set(peaks).union(set(valleys))))

    # ATR's intermediate point removal (if prev < current < next or prev > current > next)
    # This seems to remove points on monotonic slopes, which might be too aggressive
    # depending on the Savgol output. For now, we follow ATR's logic.
    # Let's make it optional or tunable. Using a less aggressive filter:
    # Keep only points that are true local extremum or endpoints of flat segments.

    if not keep_indices_pv:  # If no peaks/valleys (e.g. flat line)
        # Keep first and last point, and points where value changes if flat for a while
        simplified_data_pv = []
        if len(full_script_data) > 0:
            simplified_data_pv.append(full_script_data[0])
            if len(full_script_data) > 1:
                for i in range(1, len(full_script_data) - 1):
                    if full_script_data[i][1] != simplified_data_pv[-1][1]:
                        simplified_data_pv.append(full_script_data[i])
                if full_script_data[-1][1] != simplified_data_pv[-1][1] or len(
                        simplified_data_pv) == 1:  # ensure last point is diff or if only one point so far
                    simplified_data_pv.append(full_script_data[-1])
        _debug_log(logger, f"ATR Pass 7 (Peak/Valley): Kept {len(simplified_data_pv)} points (mostly flat signal).")

    else:  # Peaks/Valleys found
        simplified_data_pv = [full_script_data[i] for i in keep_indices_pv]
        # Ensure first and last frames are included for RDP
        if not simplified_data_pv or simplified_data_pv[0][0] != full_script_data[0][0]:
            simplified_data_pv.insert(0, full_script_data[0])
        if not simplified_data_pv or simplified_data_pv[-1][0] != full_script_data[-1][0]:
            simplified_data_pv.append(full_script_data[-1])
        # Remove duplicates that might have been added if first/last were already peaks/valleys
        temp_unique_data = []
        seen_frames = set()
        for item in simplified_data_pv:
            if item[0] not in seen_frames:
                temp_unique_data.append(item)
                seen_frames.add(item[0])
        simplified_data_pv = sorted(temp_unique_data, key=lambda x: x[0])
        _debug_log(logger, f"ATR Pass 7 (Peak/Valley): Kept {len(simplified_data_pv)} points.")

    # ATR Pass 8: RDP (simplify_coords_vw from simplification lib)
    # ATR used epsilon=2.0. This is applied to the (frame_id, position) pairs.
    # The 'vw' variant is Visvalingam-Whyatt, which is generally good.
    if len(simplified_data_pv) > 2:
        simplified_data_rdp = simplify_coords_vw(simplified_data_pv, 2.0)
        _debug_log(logger, f"ATR Pass 8 (RDP): Simplified to {len(simplified_data_rdp)} points.")
    else:
        simplified_data_rdp = simplified_data_pv  # Not enough points for RDP
        _debug_log(logger, f"ATR Pass 8 (RDP): Skipped, not enough points ({len(simplified_data_rdp)}).")

    # ATR Pass 9's final cleanup (variation_threshold)
    # This can be aggressive. Let's make it gentle or optional.
    # For now, apply it as ATR did. Variation threshold = 10.
    if len(simplified_data_rdp) > 2:
        cleaned_data_final_pass = [simplified_data_rdp[0]]  # Keep first
        for i in range(1, len(simplified_data_rdp) - 1):
            prev_pos = cleaned_data_final_pass[-1][1]  # Use the last *kept* point's position
            current_pos = simplified_data_rdp[i][1]
            next_pos = simplified_data_rdp[i + 1][1]  # Look ahead to the next point in RDP output

            # Keep if it's a significant change from last kept, or if it's an extremum relative to next
            if abs(current_pos - prev_pos) >= 10 or \
                    (current_pos > prev_pos and current_pos > next_pos) or \
                    (current_pos < prev_pos and current_pos < next_pos):
                cleaned_data_final_pass.append(simplified_data_rdp[i])

        if len(simplified_data_rdp) > 1:  # Ensure last point is added if not already
            if not cleaned_data_final_pass or simplified_data_rdp[-1][0] != cleaned_data_final_pass[-1][0]:
                cleaned_data_final_pass.append(simplified_data_rdp[-1])
        _debug_log(logger, f"ATR Pass 9 (Final Cleanup): Reduced to {len(cleaned_data_final_pass)} points.")
        final_simplified_actions = cleaned_data_final_pass
    else:
        final_simplified_actions = simplified_data_rdp

    # Store the final simplified actions in state
    state.funscript_frames = [action[0] for action in final_simplified_actions]
    state.funscript_distances = [int(action[1]) for action in final_simplified_actions]

def load_yolo_results_stage2(msgpack_file_path: str, stop_event: threading.Event, logger: logging.Logger) -> Optional[List]:
    # _debug_log(logger, f"Loading YOLO results from: {msgpack_file_path}")
    if logger: # Use logger directly if _debug_log isn't available or needed here
        logger.debug(f"[S2 Load] Loading YOLO results from: {msgpack_file_path}")
    else:
        print(f"[S2 Load - NO LOGGER] Loading YOLO results from: {msgpack_file_path}")

    if stop_event.is_set():
        if logger:
            logger.info("Load YOLO stopped by event.")
        else:
            print("Load YOLO stopped by event.")
        return None
    try:
        with open(msgpack_file_path, 'rb') as f:
            packed_data = f.read()
        all_frames_raw_detections = msgpack.unpackb(packed_data, raw=False)
        # _debug_log(logger, f"Loaded {len(all_frames_raw_detections)} frames' raw detections.")
        if logger:
            logger.debug(f"[S2 Load] Loaded {len(all_frames_raw_detections)} frames' raw detections.")
        else:
            print(f"[S2 Load - NO LOGGER] Loaded {len(all_frames_raw_detections)} frames' raw detections.")

        return all_frames_raw_detections
    except Exception as e:
        if logger:
            logger.error(f"Error loading/unpacking msgpack {msgpack_file_path}: {e}", exc_info=True)
        else:
            print(f"Error loading/unpacking msgpack {msgpack_file_path}: {e}")
        return None

def perform_contact_analysis(  # Renamed from original Stage 2's perform_contact_analysis
        video_path_arg: str, msgpack_file_path_arg: str,
        progress_callback: callable, stop_event: threading.Event,
        app_logic_instance=None,  # To access AppStateContainer-like features
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
        is_ranged_data_source: bool = False
):
    global _of_debug_prints_stage2
    _of_debug_prints_stage2 = enable_of_debug_prints

    logger = parent_logger_arg
    if not logger:  # Fallback logger
        logger = logging.getLogger("ATR_Stage2_Fallback")
        if not logger.hasHandlers():
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        logger.info("ATR Stage 2 using fallback logger.")

    FrameObject._id_counter = 0  # Reset static counters
    BaseSegment._id_counter = 0
    logger.info(f"--- Starting ATR-based Stage 2 Analysis ---")

    # Progress updates will need to be mapped to the sub_step_progress_wrapper
    # For now, use a simplified callback for ATR steps.
    def atr_progress_wrapper(main_step_tuple, sub_task_name, current, total, force=False):
        # Example: ("Step X: ATR Y", current, total)
        # This requires the calling code to manage main_step_tuple
        if progress_callback:  # Check if the main callback exists
            # The original progress_callback from AppStageProcessor expects:
            # main_info_from_module, sub_info_from_module, force_update=False
            sub_info = (current, total, sub_task_name)
            progress_callback(main_step_tuple, sub_info, force)

    # 1. Initialize VideoProcessor (Simplified, primarily for video_info)
    vp_logger = logger.getChild("VideoProcessor_ATR_S2") if logger else logging.getLogger(
        "VideoProcessor_ATR_S2_Fallback")

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
        logger.critical("VideoProcessor failed to open video or get info for ATR Stage 2.")
        return {"error": "VideoProcessor failed to initialize for ATR Stage 2"}
    if stop_event.is_set(): return {"error": "Processing stopped during VP init (ATR S2)."}

    video_info_dict = vp.video_info.copy()
    video_info_dict['actual_video_type'] = vp.determined_video_type  # Store determined type
    logger.info(f"ATR S2 VP Info: {video_info_dict}")
    vp.reset(close_video=True);
    del vp  # Release VP resources after getting info

    # 2. Load YOLO results (now with track_id)
    all_raw_detections = load_yolo_results_stage2(msgpack_file_path_arg, stop_event, logger)
    if stop_event.is_set() or not all_raw_detections:
        logger.warning("No YOLO detections loaded or process stopped (ATR S2).")
        return {"error": "Failed to load YOLO data or process stopped (ATR S2)"}

    num_video_frames = video_info_dict.get('total_frames', 0)
    if num_video_frames > 0 and len(all_raw_detections) != num_video_frames:
        logger.warning(
            f"Mismatch msgpack frames {len(all_raw_detections)} vs video frames {num_video_frames} (ATR S2).")
        if len(all_raw_detections) < num_video_frames:
            all_raw_detections.extend([[] for _ in range(num_video_frames - len(all_raw_detections))])
        else:
            all_raw_detections = all_raw_detections[:num_video_frames]
        logger.info(f"Adjusted raw detections to {len(all_raw_detections)} frames (ATR S2).")
    if not all_raw_detections: return {"error": "No detection data after adjustment (ATR S2)."}

    # 3. Initialize AppStateContainer
    try:
        state = AppStateContainer(video_info_dict, yolo_input_size_arg, vr_vertical_third_filter_arg,
                                  all_raw_detections, logger,
                                  discarded_classes_runtime_arg=discarded_classes_runtime_arg,
                                  scripting_range_active=scripting_range_active_arg,
                                  scripting_range_start_frame=scripting_range_start_frame_arg,
                                  scripting_range_end_frame=scripting_range_end_frame_arg,
                                  is_ranged_data_source=is_ranged_data_source)
    except Exception as e:
        logger.error(f"Error creating AppStateContainer (ATR S2): {e}", exc_info=True)
        return {"error": f"AppStateContainer init failed (ATR S2): {e}"}
    if stop_event.is_set(): return {"error": "Processing stopped after AppState init (ATR S2)."}

    # --- ATR Processing Steps ---
    # These steps will fill data into state.frames[...].atr_... attributes and state.atr_segments

    # Define main steps for progress reporting
    # Base steps for segmentation
    atr_main_steps_list_base = [
        ("Step 1: Interpolate Boxes", atr_pass_1_interpolate_boxes),
        ("Step 2: Build Locked Penis", atr_pass_3_build_locked_penis),
        ("Step 3: Assign Positions & Segments", atr_pass_4_assign_positions_and_segments),
        ("Step 4: Determine Frame Distances", atr_pass_5_determine_distance), # Raw distances might still be useful context
    ]
    atr_main_steps_list_funscript_gen = [
        ("Step 5: Smooth & Normalize Distances", atr_pass_6_smooth_and_normalize_distances),
        ("Step 6: Simplify Signal", atr_pass_7_8_simplify_signal)
    ]
    atr_main_steps_list = atr_main_steps_list_base
    if generate_funscript_actions_arg:
        atr_main_steps_list.extend(atr_main_steps_list_funscript_gen)

    num_main_atr_steps = len(atr_main_steps_list)

    for i, (main_step_name_atr, step_func_atr) in enumerate(atr_main_steps_list):
        logger.info(f"Starting {main_step_name_atr} (ATR S2)")
        main_step_tuple_for_callback = (i + 1, num_main_atr_steps, main_step_name_atr)

        # Signal start of this main ATR step (sub-progress will be 0/1 initially)
        atr_progress_wrapper(main_step_tuple_for_callback, "Initializing...", 0, 1, True)

        step_func_atr(state, logger)  # Most ATR passes don't have internal progress loops for callback

        if stop_event.is_set():
            logger.info(f"ATR S2 stopped during {main_step_name_atr}.")
            atr_progress_wrapper(main_step_tuple_for_callback, "Aborted", 1, 1, True)
            return {"error": f"Processing stopped during {main_step_name_atr} (ATR S2)"}

        # Signal completion of this main ATR step
        atr_progress_wrapper(main_step_tuple_for_callback, "Completed", 1, 1, True)

    # --- Funscript Data Population from ATR results ---
    # state.funscript_frames and state.funscript_distances are already populated by atr_pass_7_8_simplify_signal
    # state.funscript_distances_lr will be neutral as ATR is single-axis
    if state.funscript_frames:
        state.funscript_distances_lr = [50] * len(state.funscript_frames)
    else:  # If no frames (e.g. very short video or error), ensure lists are empty
        state.funscript_distances_lr = []
        state.funscript_distances = []

    if stop_event.is_set():
        logger.info("ATR S2 stopped before final data packaging.")
        return {"error": "Processing stopped before final data packaging (ATR S2)."}

    # Video segments for GUI from ATRSegments
    # Get the full list of generated segments and funscript points
    video_segments_for_gui = [atr_seg.to_dict() for atr_seg in state.atr_segments if not stop_event.is_set()]
    funscript_frames_full = state.funscript_frames
    funscript_distances_full = state.funscript_distances
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
            end_f = len(state.frames) - 1

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

    # Convert the (now correctly filtered) frames/distances to funscript actions
    primary_actions_final = []
    secondary_actions_final = []
    if generate_funscript_actions_arg:
        current_video_fps = state.video_info.get('fps', 0)
        if current_video_fps > 0 and final_funscript_frames:
            # ... (logic to convert frames to ms and create actions list remains the same, but now uses `final_...` lists) ...
            temp_primary_actions = {}
            temp_secondary_actions = {}
            for fid, pos_primary, pos_secondary in zip(final_funscript_frames, final_funscript_distances,
                                                       final_funscript_distances_lr):
                if stop_event.is_set(): break
                timestamp_ms = int(round((fid / current_video_fps) * 1000))
                temp_primary_actions[timestamp_ms] = {"at": timestamp_ms, "pos": int(pos_primary)}
                temp_secondary_actions[timestamp_ms] = {"at": timestamp_ms, "pos": int(pos_secondary)}

            if not stop_event.is_set():
                primary_actions_final = sorted(temp_primary_actions.values(), key=lambda x: x["at"])
                secondary_actions_final = sorted(temp_secondary_actions.values(), key=lambda x: x["at"])

    # Build the final return dictionary with the filtered results
    return_dict = {
        "video_segments": final_video_segments,
        "primary_actions": primary_actions_final,
        "secondary_actions": secondary_actions_final,
    }
    if not generate_funscript_actions_arg:
        return_dict["atr_segments_objects"] = state.atr_segments
        return_dict["all_s2_frame_objects_list"] = state.frames



    if output_overlay_msgpack_path:
        logger.info(f"Preparing to save ATR Stage 2 overlay data to: {output_overlay_msgpack_path}")
        try:
            all_frames_overlay_data = [frame.to_overlay_dict() for frame in state.frames if not stop_event.is_set()]
            if stop_event.is_set(): return {"error": "Processing stopped during overlay data prep (ATR S2)."}

            # Add a handler for NumPy types
            def numpy_default_handler(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable for msgpack")

            with open(output_overlay_msgpack_path, 'wb') as f:
                # Use the numpy_default_handler in packb
                f.write(msgpack.packb(all_frames_overlay_data, use_bin_type=True, default=numpy_default_handler))
            logger.info(f"Successfully saved ATR overlay data for {len(all_frames_overlay_data)} frames to {output_overlay_msgpack_path}.")
            if os.path.exists(output_overlay_msgpack_path):
                return_dict["overlay_msgpack_path"] = output_overlay_msgpack_path
        except Exception as e:
            logger.error(f"Error saving ATR Stage 2 overlay msgpack to {output_overlay_msgpack_path}: {e}", exc_info=True)

    logger.info(f"--- ATR-based Stage 2 Analysis Finished. Segments: {len(video_segments_for_gui)} ---")
    return return_dict


if __name__ == "__main__":
    print("ATR-based Stage 2 - Standalone Execution Mode")
    parser = argparse.ArgumentParser(description="ATR-based Stage 2: Contact Analysis & Funscript Gen")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video")
    parser.add_argument("--msgpack_file", type=str, required=True,
                        help="Path to .msgpack from Stage 1 (with track_ids)")
    parser.add_argument("--yolo_input_size", type=int, default=640, help="YOLO input size used in Stage 1")
    # Add other relevant CLI args from original Stage 2 if needed (video_type, vr_format, etc.)
    parser.add_argument("--debug_prints", action='store_true', help="Enable detailed debug prints for ATR S2")
    parser.add_argument("--log_level", type=str, default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    parser.add_argument("--output_dir", type=str, default=".", help="Directory to save output files (e.g., overlay)")

    args = parser.parse_args()

    main_cli_logger = logging.getLogger("ATR_S2_CLI_Test")
    main_cli_logger.setLevel(getattr(logging, args.log_level.upper(), logging.INFO))
    if not main_cli_logger.hasHandlers():
        cli_handler = logging.StreamHandler();
        cli_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        cli_handler.setFormatter(cli_formatter);
        main_cli_logger.addHandler(cli_handler)


    class MockProgressCallbackForATRTest:  # Simplified for ATR steps
        def __init__(self, logger_instance): self.logger = logger_instance

        def __call__(self, main_step_tuple, sub_step_tuple, force_update=False):
            main_curr, main_total, main_name = main_step_tuple
            sub_curr, sub_total, sub_name = sub_step_tuple
            main_prog_perc = (main_curr / main_total * 100) if main_total > 0 else 0
            sub_prog_perc = (sub_curr / sub_total * 100) if sub_total > 0 else 0

            progress_msg = (f"Progress: {main_name} ({main_curr}/{main_total} - {main_prog_perc:.1f}%) => "
                            f"{sub_name} ({sub_curr}/{sub_total} - {sub_prog_perc:.1f}%)")
            self.logger.info(progress_msg)


    mock_stop_event = threading.Event()
    os.makedirs(args.output_dir, exist_ok=True)

    if not os.path.exists(args.video_path):
        main_cli_logger.error(f"Video file not found: {args.video_path}")
    elif not os.path.exists(args.msgpack_file):
        main_cli_logger.error(f"Msgpack file not found: {args.msgpack_file}")
    else:
        main_cli_logger.info(f"Starting ATR Stage 2 standalone test with: {args.video_path}")
        base_name = os.path.splitext(os.path.basename(args.video_path))[0]
        cli_output_overlay_path = os.path.join(args.output_dir, f"{base_name}_atr_stage2_overlay_CLI.msgpack")

        results = perform_contact_analysis(  # Call the ATR-infused version
            video_path_arg=args.video_path,
            msgpack_file_path_arg=args.msgpack_file,
            progress_callback=MockProgressCallbackForATRTest(main_cli_logger),  # Use the ATR-style mock
            stop_event=mock_stop_event,
            app_logic_instance=None,  # No full app logic in standalone
            parent_logger_arg=main_cli_logger,
            output_overlay_msgpack_path=cli_output_overlay_path,
            yolo_input_size_arg=args.yolo_input_size,
            enable_of_debug_prints=args.debug_prints,
            # Pass other necessary args, e.g. video_type, vr settings if they affect ATR logic.
            # For now, defaults in perform_contact_analysis will be used if not provided.
        )
        if results:
            main_cli_logger.info("\n--- ATR Stage 2 Standalone Test Results ---")
            if "error" in results: main_cli_logger.error(f"Error: {results['error']}")
            video_segs = results.get('video_segments', [])
            main_cli_logger.info(f"Video Segments (ATR): {len(video_segs)}")
            primary_actions = results.get('primary_actions', [])
            secondary_actions = results.get('secondary_actions', [])  # Should be neutral
            main_cli_logger.info(f"Primary Actions generated (ATR): {len(primary_actions)}")
            main_cli_logger.info(f"Secondary Actions generated (ATR): {len(secondary_actions)}")
            if results.get("overlay_msgpack_path"):
                main_cli_logger.info(f"Overlay data saved to: {results['overlay_msgpack_path']}")
        else:
            main_cli_logger.error("ATR Stage 2 Standalone test returned None or an error occurred.")

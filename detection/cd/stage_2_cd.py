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
from copy import deepcopy

from video.video_processor import VideoProcessor
from config import constants

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
            is_synthesized = box and box.status not in [constants.STATUS_DETECTED, constants.STATUS_SMOOTHED]
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
                 status: str = constants.STATUS_DETECTED, yolo_input_size: int = 640,
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


class PoseRecord:
    _id_counter = 0

    def __init__(self, frame_id: int, bbox: list, keypoints_data: list):
        self.id = PoseRecord._id_counter
        PoseRecord._id_counter += 1
        self.frame_id = int(frame_id)
        self.bbox = np.array(bbox, dtype=np.float32)
        self.keypoints = np.array(keypoints_data, dtype=np.float32)
        self.keypoint_confidence_threshold = 0.3

    @property
    def person_box_area(self) -> float:
        if self.bbox is None or len(self.bbox) < 4: return 0.0
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def calculate_zone_dissimilarity(self, other_pose: 'PoseRecord', zone_keypoint_indices: List[int]) -> float:
        """Calculates dissimilarity based on a specific set of keypoints (e.g., pelvis, arms)."""
        if self.keypoints.shape != other_pose.keypoints.shape or not zone_keypoint_indices:
            return 999.0

        total_distance = 0.0
        valid_points_count = 0
        for i in zone_keypoint_indices:
            if i >= len(self.keypoints):
                continue  # Index out of bounds
            kp1, kp2 = self.keypoints[i], other_pose.keypoints[i]
            if kp1[2] > self.keypoint_confidence_threshold and kp2[2] > self.keypoint_confidence_threshold:
                total_distance += np.linalg.norm(kp1[:2] - kp2[:2])
                valid_points_count += 1

        if valid_points_count == 0: return 999.0  # No common points in the zone to compare
        if valid_points_count < len(zone_keypoint_indices) * 0.5: return 999.0  # Require at least half the zone points

        normalization_factor = np.sqrt(self.person_box_area) if self.person_box_area > 0 else 1.0
        return (
                           total_distance / valid_points_count) / normalization_factor * 100.0 if normalization_factor > 0 else 999.0

    def to_dict(self):
        return {"id": self.id, "bbox": self.bbox.tolist(), "keypoints": self.keypoints.tolist()}


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

    def __init__(self, frame_id: int, yolo_input_size: int, raw_frame_data: Optional[dict] = None,
                 classes_to_discard_runtime_set: Optional[set] = None):
        self.id = FrameObject._id_counter
        FrameObject._id_counter += 1
        self.frame_pos = int(frame_id)
        self.frame_id = int(frame_id)
        self.yolo_input_size = yolo_input_size
        self.boxes: List[BoxRecord] = []
        self.poses: List[PoseRecord] = []
        self._effective_discard_classes = classes_to_discard_runtime_set or set(constants.CLASSES_TO_DISCARD_BY_DEFAULT)
        PoseRecord._id_counter = 0
        self.parse_raw_frame_data(raw_frame_data or {})

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
        self.dominant_pose_id: Optional[int] = None
        self.is_occluded: bool = False
        self.active_interaction_track_id: Optional[int] = None

    def parse_raw_frame_data(self, raw_frame_data: dict):
        if not isinstance(raw_frame_data, dict): return
        raw_detections = raw_frame_data.get("detections", []);
        raw_poses = raw_frame_data.get("poses", [])
        for det_data in raw_detections:
            if det_data.get('name') in self._effective_discard_classes: continue
            self.boxes.append(
                BoxRecord(self.frame_id, det_data.get('bbox'), det_data.get('confidence'), det_data.get('class'),
                          det_data.get('name'), yolo_input_size=self.yolo_input_size))
        for pose_data in raw_poses:
            self.poses.append(PoseRecord(self.frame_id, pose_data.get('bbox'), pose_data.get('keypoints')))

    def get_preferred_penis_box(self, actual_video_type: str = '2D', vr_vertical_third_filter: bool = False) -> \
    Optional[BoxRecord]:
        penis_detections = [b for b in self.boxes if b.class_name == constants.PENIS_CLASS_NAME and not b.is_excluded]
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
            "atr_assigned_position": self.atr_assigned_position,
            "dominant_pose_id": self.dominant_pose_id,
            "active_interaction_track_id": self.active_interaction_track_id,  # <-- Add this for highlighting
            "is_occluded": self.is_occluded,
            "yolo_boxes": [b.to_dict() for b in self.boxes if not b.is_excluded],
            "poses": [p.to_dict() for p in self.poses]
        }
        if self.atr_locked_penis_state.active and self.atr_locked_penis_state.box:
            lp_state = self.atr_locked_penis_state
            box_dims = lp_state.box
            w = box_dims[2] - box_dims[0]
            h = box_dims[3] - box_dims[1]
            frame_data["yolo_boxes"].append(
                {"frame_id": self.frame_id, "bbox": list(box_dims), "confidence": 1.0, "class_id": -1,
                 "class_name": "locked_penis", "status": "ATR_LOCKED", "width": w, "height": h,
                 "cx": box_dims[0] + w / 2, "cy": box_dims[1] + h / 2})
        return frame_data

    def __repr__(self):
        return f"FrameObject(id={self.frame_id}, #boxes={len(self.boxes)}, #poses={len(self.poses)}, atr_pos='{self.atr_assigned_position}')"


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
        position_short_name_key_val = "NR"  # Default if not found

        for key, info in constants.POSITION_INFO_MAPPING.items():
            if info["long_name"] == self.major_position:
                position_short_name_key_val = key
                break


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
                'position_short_name': position_short_name_key_val,
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
    def __init__(self, video_info: Dict, yolo_input_size: int, vr_filter: bool, all_frames_raw_data: list,
                 logger: Optional[logging.Logger], discarded_classes_runtime_arg: Optional[List[str]] = None,
                 scripting_range_active: bool = False, scripting_range_start_frame: Optional[int] = None,
                 scripting_range_end_frame: Optional[int] = None, is_ranged_data_source: bool = False):
        self.video_info = video_info
        self.yolo_input_size = yolo_input_size
        self.vr_vertical_third_filter = vr_filter  # This is the general VR filter for non-penis boxes
        self.frames: List[FrameObject] = []
        FrameObject._id_counter = 0

        self.effective_discard_classes = set(constants.CLASSES_TO_DISCARD_BY_DEFAULT)
        if discarded_classes_runtime_arg: self.effective_discard_classes.update(discarded_classes_runtime_arg)
        frame_id_offset = scripting_range_start_frame if is_ranged_data_source and scripting_range_start_frame is not None else 0
        for i, raw_frame_data_dict in enumerate(all_frames_raw_data):
            absolute_frame_id = i + frame_id_offset
            fo = FrameObject(frame_id=absolute_frame_id, yolo_input_size=yolo_input_size,
                             raw_frame_data=raw_frame_data_dict,
                             classes_to_discard_runtime_set=self.effective_discard_classes)

            # VR Filter for NON-PENIS boxes (from original Stage 2)
            # ATR logic has a similar concept of "central_boxes"
            if video_info.get('actual_video_type') == 'VR' and vr_filter:
                for box_rec in fo.boxes:
                    if box_rec.class_name != constants.PENIS_CLASS_NAME and not (
                            yolo_input_size / 3 <= box_rec.cx <= 2 * yolo_input_size / 3):
                        box_rec.is_excluded = True
                        box_rec.status = "Excluded_VR_Filter_Peripheral"
            self.frames.append(fo)

        self.atr_segments: List[ATRSegment] = []
        self.funscript_frames: List[int] = []
        self.funscript_distances: List[int] = []
        self.funscript_distances_lr: List[int] = []


# --- ATR Helper Functions (to be moved into this file) ---
def _atr_calculate_iou(box1: Tuple[float, ...], box2: Tuple[float, ...]) -> float:
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


def _atr_assign_frame_position(contacts: List[Dict]) -> str:
    if not contacts:
        return "Not Relevant"

    detected_class_names = {contact["class_name"] for contact in contacts} # Use a set for faster lookups

    # Explicit prioritization
    if 'pussy' in detected_class_names:
        return 'Cowgirl / Missionary' # Highest priority if pussy is involved
    if 'butt' in detected_class_names:
        return 'Rev. Cowgirl / Doggy' # Next highest

    # Check for face and hand for more specific acts
    has_face = 'face' in detected_class_names
    has_hand = 'hand' in detected_class_names

    # If face is involved, it's a Blowjob, regardless of hand presence.
    if has_face:
        return 'Blowjob'

    # If no face, but hand is present, it's a Handjob.
    if has_hand:
        return 'Handjob'

    # Check for other positions if the above are not met
    if 'breast' in detected_class_names:
        return 'Boobjob'
    if 'foot' in detected_class_names:
        return 'Footjob'

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
    """
    Calculates normalized distance, with a crucial refinement for hand/face interaction.
    """
    penis_base_y = locked_penis_box_coords[3]  # y2 (bottom of the conceptual full-stroke box)

    # --- REFINED LOGIC ---
    # Use the center of the hand/face for a more stable reference point during rotation.
    if class_name == 'face':
        box_y_ref = class_box_coords[3]  # Bottom of the face
    elif class_name == 'hand':
        box_y_ref = (class_box_coords[1] + class_box_coords[3]) / 2  # Center Y of the hand/face
    elif class_name == 'butt':
        box_y_ref = (9 * class_box_coords[3] + class_box_coords[1]) / 10  # Mostly bottom of butt
    else:  # pussy, breast, foot, etc.
        box_y_ref = (class_box_coords[1] + class_box_coords[3]) / 2  # Center of other parts

    raw_distance = penis_base_y - box_y_ref
    if max_distance_ref <= 0: return 50.0
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

def _get_dominant_pose(frame_obj: FrameObject, is_vr: bool, frame_width: int) -> Optional[PoseRecord]:
    if not frame_obj.poses:
        return None
    if is_vr:
        frame_center_x = frame_width / 2
        return min(frame_obj.poses, key=lambda p: abs(((p.bbox[0] + p.bbox[2]) / 2) - frame_center_x))
    else:
        return max(frame_obj.poses, key=lambda p: p.person_box_area)


def _infer_penis_box_from_pose(frame_obj: FrameObject) -> Optional[BoxRecord]:
    dominant_pose = _get_dominant_pose(frame_obj, False, frame_obj.yolo_input_size)
    if not dominant_pose: return None
    keypoints = dominant_pose.keypoints
    l_hip_idx, r_hip_idx = 11, 12
    if len(keypoints) > max(l_hip_idx, r_hip_idx) and keypoints[l_hip_idx][2] > 0.3 and keypoints[r_hip_idx][2] > 0.3:
        l_hip, r_hip = keypoints[l_hip_idx], keypoints[r_hip_idx]
        hip_center_x, hip_center_y = (l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2
        person_height = dominant_pose.bbox[3] - dominant_pose.bbox[1]
        inferred_box_width, inferred_box_height = person_height * 0.1, person_height * 0.2
        x1, y1 = hip_center_x - inferred_box_width / 2, hip_center_y - inferred_box_height / 4
        x2, y2 = x1 + inferred_box_width, y1 + inferred_box_height
        return BoxRecord(frame_id=frame_obj.frame_id, bbox=[x1, y1, x2, y2], confidence=0.5, class_id=-1,
                         class_name=constants.PENIS_CLASS_NAME, status=constants.STATUS_POSE_INFERRED,
                         yolo_input_size=frame_obj.yolo_input_size)
    return None


def _infer_interaction_box_from_pose(pose: PoseRecord, class_name: str) -> Optional[Tuple[float, float, float, float]]:
    keypoints = pose.keypoints
    l_hip_idx, r_hip_idx = 11, 12
    l_sho_idx, r_sho_idx = 5, 6
    if len(keypoints) <= max(l_hip_idx, r_hip_idx) or keypoints[l_hip_idx][2] < 0.3 or keypoints[r_hip_idx][
        2] < 0.3: return None
    l_hip, r_hip = keypoints[l_hip_idx], keypoints[r_hip_idx]
    hip_center_x, hip_center_y = (l_hip[0] + r_hip[0]) / 2, (l_hip[1] + r_hip[1]) / 2
    hip_width = np.linalg.norm(l_hip[:2] - r_hip[:2])
    if hip_width < 1: return None
    if class_name in ['pussy', 'butt']:
        box_h, box_w = hip_width * 0.8, hip_width * 1.2
        x1, y1 = hip_center_x - box_w / 2, hip_center_y - box_h / 4
        return (x1, y1, x1 + box_w, y1 + box_h)
    elif class_name == 'face':
        if len(keypoints) > max(l_sho_idx, r_sho_idx) and keypoints[l_sho_idx][2] > 0.3 and keypoints[r_sho_idx][
            2] > 0.3:
            l_sho, r_sho = keypoints[l_sho_idx], keypoints[r_sho_idx]
            sho_center_x = (l_sho[0] + r_sho[0]) / 2
            sho_width = np.linalg.norm(l_sho[:2] - r_sho[:2])
            if sho_width > 1: box_h = box_w = sho_width; x1, y1 = sho_center_x - box_w / 2, l_sho[1] - box_h; return (
                x1, y1, x1 + box_w, y1 + box_h)
    return None


# --- Step 0: Global Object Tracking (Simple IoU Tracker) ---
def simple_iou_tracker_step0(state: AppStateContainer, logger: Optional[logging.Logger]):
    _debug_log(logger, "Starting Step 0: Simple IoU Object Tracking")
    next_global_track_id = 1
    active_tracks: Dict[int, Dict[str, Any]] = {}
    iou_threshold = 0.3
    max_frames_unseen = int(state.video_info.get('fps', 30) * 0.5)

    for frame_obj in sorted(state.frames, key=lambda f: f.frame_id):
        current_detections = [b for b in frame_obj.boxes if not b.is_excluded]
        matched_indices = [False] * len(current_detections)
        track_ids_to_delete = []

        for track_id, track_data in active_tracks.items():
            best_match_idx, max_iou = -1, -1
            for i, det_box in enumerate(current_detections):
                if matched_indices[i] or det_box.class_name != track_data["class_name"]: continue
                iou = _atr_calculate_iou(track_data["box_rec"].bbox, det_box.bbox)
                if iou > iou_threshold and iou > max_iou: max_iou, best_match_idx = iou, i
            if best_match_idx != -1:
                current_detections[best_match_idx].track_id = track_id
                active_tracks[track_id].update({"box_rec": current_detections[best_match_idx], "frames_unseen": 0})
                matched_indices[best_match_idx] = True
            else:
                active_tracks[track_id]["frames_unseen"] += 1
                if active_tracks[track_id]["frames_unseen"] > max_frames_unseen: track_ids_to_delete.append(track_id)

        for track_id in track_ids_to_delete: del active_tracks[track_id]

        for i, det_box in enumerate(current_detections):
            if not matched_indices[i]:
                det_box.track_id = next_global_track_id
                active_tracks[next_global_track_id] = {"box_rec": det_box, "frames_unseen": 0,
                                                       "class_name": det_box.class_name}
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
                                status=constants.STATUS_INTERPOLATED,  # Mark as interpolated
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
    fps, yolo_size = state.video_info.get('fps', 30.0), state.yolo_input_size
    current_lp_active, current_lp_last_raw_box_coords, current_lp_max_height, current_lp_max_penetration_height, current_lp_area, current_lp_consecutive_detections, current_lp_consecutive_non_detections = False, None, 0.0, 0.0, 0.0, 0, 0
    kf_height = cv2.KalmanFilter(2, 1)
    kf_height.measurementMatrix = np.array([[1, 0]], np.float32)
    kf_height.transitionMatrix = np.array([[1, 1], [0, 1]], np.float32)
    kf_height.processNoiseCov = np.eye(2, dtype=np.float32) * 0.01
    kf_height.measurementNoiseCov = np.eye(1, dtype=np.float32) * 0.1
    kf_height.errorCovPost = np.eye(2, dtype=np.float32) * 1.0
    kf_height.statePost = np.array([[yolo_size * 0.1], [0]], dtype=np.float32)
    last_frame_dominant_pose: Optional[PoseRecord] = None
    is_vr = state.video_info.get('actual_video_type', '2D') == 'VR'
    pelvis_zone_indices = [11, 12]  # Left and Right Hip

    for frame_obj in state.frames:
        dominant_pose_this_frame = _get_dominant_pose(frame_obj, is_vr, state.yolo_input_size)
        if dominant_pose_this_frame:
            frame_obj.dominant_pose_id = dominant_pose_this_frame.id

        selected_penis_box_rec = frame_obj.get_preferred_penis_box(state.video_info.get('actual_video_type', '2D'),
                                                                   state.vr_vertical_third_filter)

        frame_glans_detected = False

        if selected_penis_box_rec:
            current_lp_consecutive_detections += 1
            current_lp_consecutive_non_detections = 0
            current_lp_last_raw_box_coords = tuple(selected_penis_box_rec.bbox)
            current_raw_height = selected_penis_box_rec.height

            # Update max_height (ATR logic: only increase unless glans detected)
            if current_raw_height > current_lp_max_height:
                current_lp_max_height = current_raw_height
                # ATR set max_penetration_height as 0.65 * current_height when max_height was updated
                current_lp_max_penetration_height = 0.65 * current_raw_height

            # Check for glans contact with this selected_penis_box_rec
            glans_boxes_in_frame = [b for b in frame_obj.boxes if
                                    b.class_name == constants.GLANS_CLASS_NAME and not b.is_excluded]
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
            pose_is_stable_in_interaction_zone = False

            # If the lock was active, check if the person's pelvis is still where the penis was
            # Check if pose data is available before using it
            if frame_obj.poses and current_lp_active and dominant_pose_this_frame and last_frame_dominant_pose and current_lp_last_raw_box_coords:
                person_iou_with_last_penis = _atr_calculate_iou(dominant_pose_this_frame.bbox, current_lp_last_raw_box_coords)
                if person_iou_with_last_penis > 0.1:
                    pelvis_dissimilarity = dominant_pose_this_frame.calculate_zone_dissimilarity(
                        last_frame_dominant_pose, pelvis_zone_indices)
                    if pelvis_dissimilarity < 2.0:
                        pose_is_stable_in_interaction_zone = True

                        _debug_log(logger,
                                   f"Frame {frame_obj.frame_id}: Penis lock held by stable pelvis (IoU: {person_iou_with_last_penis:.2f}, Dissim: {pelvis_dissimilarity:.2f}).")

            # Only increment non-detection counter if pose is NOT stable
            if not pose_is_stable_in_interaction_zone:
                current_lp_consecutive_non_detections += 1

            # Deactivation logic is now naturally robust to flicker
            if current_lp_active and current_lp_consecutive_non_detections >= (state.video_info.get('fps', 30) * 3):
                current_lp_active = False


        lp_state = frame_obj.atr_locked_penis_state
        lp_state.active = current_lp_active
        lp_state.consecutive_detections = current_lp_consecutive_detections
        lp_state.consecutive_non_detections = current_lp_consecutive_non_detections
        lp_state.max_height = current_lp_max_height
        lp_state.max_penetration_height = current_lp_max_penetration_height
        lp_state.glans_detected = frame_glans_detected
        if current_lp_active and current_lp_last_raw_box_coords:
            predicted_height_kalman = kf_height.predict()[0, 0]
            predicted_height_final = np.clip(predicted_height_kalman, 0, current_lp_max_height)

            # The "visible" penis box based on Kalman prediction for height
            # Position (x1, x2, y2) comes from last raw detection. y1 is derived from predicted_height_final.
            x1, _, x2, y2_raw = current_lp_last_raw_box_coords
            frame_obj.atr_penis_box_kalman = (x1, y2_raw - predicted_height_final, x2, y2_raw)
            lp_state.box = (x1, y2_raw - current_lp_max_height, x2, y2_raw)
            lp_state.area = (x2 - x1) * current_lp_max_height if current_lp_max_height > 0 else 0
            lp_state.visible_part = (
                                                predicted_height_final / current_lp_max_height) * 100.0 if current_lp_max_height > 0 else 0.0
        else:
            # If not active, or no last_raw_box, no specific ATR box for this frame
            lp_state.box = None
            lp_state.area = 0.0
            lp_state.visible_part = 0.0  # Or 100.0 if no detection implies full? ATR implies 0 if not active.
            frame_obj.atr_penis_box_kalman = None

        last_frame_dominant_pose = dominant_pose_this_frame

def atr_pass_4_assign_positions_and_segments(state: AppStateContainer, logger: Optional[logging.Logger]):
    """ ATR Pass 4: Assign frame positions and aggregate into segments.
        Modifies frame_obj.atr_assigned_position and populates state.atr_segments.
    """
    _debug_log(logger, "Starting ATR Pass 4: Assign Positions & Segments")
    fps, yolo_size, frame_area = state.video_info.get('fps', 30.0), state.yolo_input_size, state.yolo_input_size ** 2
    last_frame_position = "Not Relevant"
    last_frame_dominant_pose: Optional[PoseRecord] = None
    is_vr = state.video_info.get('actual_video_type', '2D') == 'VR'

    for frame_obj in state.frames:
        dominant_pose_this_frame = _get_dominant_pose(frame_obj, is_vr, yolo_size)
        frame_obj.atr_detected_contact_boxes.clear()  # Clear any previous data
        assigned_pos_for_frame = "Not Relevant"
        if frame_obj.atr_locked_penis_state.active and frame_obj.atr_locked_penis_state.box:
            lp_box_coords = frame_obj.atr_locked_penis_state.box

            # This list is now correctly populated on the frame object itself.
            for box_rec in frame_obj.boxes:
                if box_rec.is_excluded or box_rec.class_name in [constants.PENIS_CLASS_NAME,
                                                                 constants.GLANS_CLASS_NAME]:
                    continue
                if _atr_calculate_iou(lp_box_coords, box_rec.bbox) > 0.05:
                    frame_obj.atr_detected_contact_boxes.append({"class_name": box_rec.class_name, "box_rec": box_rec})

            assigned_pos_for_frame = _atr_assign_frame_position(frame_obj.atr_detected_contact_boxes)

        # --- NEW CONTEXT-AWARE LOGIC ---
        if assigned_pos_for_frame == "Not Relevant" and last_frame_position != "Not Relevant":
            pose_is_stable_in_interaction_zone = False
            if dominant_pose_this_frame and last_frame_dominant_pose:
                # Define the interaction zone based on the LAST known position
                interaction_zone_indices = []
                if "Cowgirl" in last_frame_position or "Doggy" in last_frame_position:
                    interaction_zone_indices = [11, 12, 13, 14]
                elif "Blowjob" in last_frame_position:
                    interaction_zone_indices = [0, 1, 2, 5, 6]
                elif "Handjob" in last_frame_position:
                    interaction_zone_indices = [5, 6, 7, 8, 9, 10]
                elif "Boobjob" in last_frame_position:
                    interaction_zone_indices = [5, 6]
                elif "Footjob" in last_frame_position:
                    interaction_zone_indices = [13, 14, 15, 16]

                if interaction_zone_indices:
                    if dominant_pose_this_frame.calculate_zone_dissimilarity(last_frame_dominant_pose,
                                                                             interaction_zone_indices) < 2.5:
                        pose_is_stable_in_interaction_zone = True

            if pose_is_stable_in_interaction_zone:
                assigned_pos_for_frame = last_frame_position  # Carry over the previous classification

        frame_obj.atr_assigned_position = assigned_pos_for_frame
        last_frame_position = assigned_pos_for_frame
        last_frame_dominant_pose = dominant_pose_this_frame

    BaseSegment._id_counter = 0
    state.atr_segments = _atr_aggregate_segments(state.frames, fps, int(1.0 * fps), logger)


def atr_pass_5_determine_distance(state: AppStateContainer, logger: Optional[logging.Logger]):
    _debug_log(logger, "Starting ATR Pass 5: Determine Frame Distances (Corrected Logic)")
    is_vr = state.video_info.get('actual_video_type', '2D') == 'VR'

    # Initialize previous distance tracker
    prev_valid_distance = 100  # Default starting value

    for frame_obj in state.frames:
        frame_obj.is_occluded = False
        frame_obj.active_interaction_track_id = None

        if not frame_obj.atr_locked_penis_state.active or not frame_obj.atr_locked_penis_state.box:
            frame_obj.atr_funscript_distance = 100
            continue

        # --- ENSURE WE USE THE LOCKED PENIS BOX CONSISTENTLY ---
        lp_state = frame_obj.atr_locked_penis_state
        lp_box_coords = lp_state.box

        current_pos = frame_obj.atr_assigned_position
        relevant_classes, is_penetration_pos, secondary_classes = [], False, []

        # Define primary and secondary interaction classes based on position
        if "Cowgirl" in current_pos or "Doggy" in current_pos:
            relevant_classes, is_penetration_pos, secondary_classes = ["pussy", "butt"], not lp_state.glans_detected, [
                "breast", "navel"]
        elif "Blowjob" in current_pos:
            relevant_classes, is_penetration_pos, secondary_classes = ["face", "hand"], False, ["breast"]
        elif "Handjob" in current_pos:
            relevant_classes, is_penetration_pos, secondary_classes = ["hand"], False, ["breast"]

        #if not relevant_classes:
        #    frame_obj.atr_funscript_distance = 100
        #    continue

        contacting_boxes = [c['box_rec'] for c in frame_obj.atr_detected_contact_boxes]
        primary_contacts = [b for b in contacting_boxes if b.class_name in relevant_classes]

        #comp_dist_for_frame = 100.0
        active_box = None

        # --- UNIFIED LOGIC FOR ALL POSITIONS ---
        if primary_contacts:
            # Use the most confident primary contact as the active one
            active_box = max(primary_contacts, key=lambda b: b.confidence)
        else:  # Occlusion Logic
            dominant_pose = _get_dominant_pose(frame_obj, is_vr, state.yolo_input_size)
            if dominant_pose:
                secondary_contacts = [b for b in contacting_boxes if
                                      b.class_name in secondary_classes and _atr_calculate_iou(dominant_pose.bbox, b.bbox) > 0]

                if secondary_contacts:
                    active_box = max(secondary_contacts, key=lambda b: b.confidence)
                    frame_obj.is_occluded = True

        if active_box:
            frame_obj.active_interaction_track_id = active_box.track_id  # Tag for highlighting

            # --- UNIFIED DISTANCE LOGIC USING THE NEW BOTTOM-TO-BOTTOM CALCULATION ---
            max_dist_ref = lp_state.max_height  # Use the full penis height as the normalization reference
            if max_dist_ref <= 0: max_dist_ref = state.yolo_input_size * 0.3  # Fallback

            comp_dist_for_frame = _atr_calculate_normalized_distance_to_base(lp_box_coords, active_box.class_name,
                                                                             active_box.bbox, max_dist_ref)
            prev_valid_distance = comp_dist_for_frame  # Update the previous valid distance

        elif frame_obj.is_occluded and is_penetration_pos:
            # Fallback to penis visibility only if occluded and in a penetration scene with no secondary motion
            comp_dist_for_frame = lp_state.visible_part
            prev_valid_distance = comp_dist_for_frame  # Update the previous valid distance
        else:
            # No active box found and not occluded - use previous valid distance
            comp_dist_for_frame = prev_valid_distance

        frame_obj.atr_funscript_distance = int(np.clip(round(comp_dist_for_frame), 0, 100))


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
    skip_simplification = False

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

def perform_contact_analysis(
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
    vp.reset(close_video=True)
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
        ("Step 1: Tracking Objects", simple_iou_tracker_step0),
        ("Step 2: Interpolate Boxes", atr_pass_1_interpolate_boxes),
        ("Step 3: Build Locked Penis", atr_pass_3_build_locked_penis),
        ("Step 4: Assign Positions & Segments", atr_pass_4_assign_positions_and_segments),
        ("Step 5: Determine Frame Distances", atr_pass_5_determine_distance),
    ]
    atr_main_steps_list_funscript_gen = [
        ("Step 6: Smooth & Normalize Distances", atr_pass_6_smooth_and_normalize_distances),
        ("Step 7: Simplify Signal", atr_pass_7_8_simplify_signal)
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
    # The calling orchestrator is responsible for picking what it needs based on the mode.
    return_dict = {
        "video_segments": final_video_segments,
        "primary_actions": primary_actions_final,
        "secondary_actions": secondary_actions_final,
        "atr_segments_objects": state.atr_segments,
        "all_s2_frame_objects_list": state.frames
    }

    if output_overlay_msgpack_path:
        logger.info(f"Preparing to save ATR Stage 2 overlay data to: {output_overlay_msgpack_path}")
        try:
            all_frames_overlay_data = [frame.to_overlay_dict() for frame in state.frames if not stop_event.is_set()]
            if stop_event.is_set(): return {"error": "Processing stopped during overlay data prep (ATR S2)."}

            def numpy_default_handler(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable for msgpack")

            with open(output_overlay_msgpack_path, 'wb') as f:
                f.write(msgpack.packb(all_frames_overlay_data, use_bin_type=True, default=numpy_default_handler))
            logger.info(f"Successfully saved ATR overlay data for {len(all_frames_overlay_data)} frames to {output_overlay_msgpack_path}.")
            if os.path.exists(output_overlay_msgpack_path):
                return_dict["overlay_msgpack_path"] = output_overlay_msgpack_path
        except Exception as e:
            logger.error(f"Error saving ATR Stage 2 overlay msgpack to {output_overlay_msgpack_path}: {e}", exc_info=True)

    logger.info(f"--- ATR-based Stage 2 Analysis Finished. Segments: {len(final_video_segments)} ---")
    return return_dict


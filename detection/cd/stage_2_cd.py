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

# --- Global Debug Flag ---
_of_debug_prints_stage2 = False
CONTACT_ONLY_FALLBACKS = {'hand', 'pussy', 'butt'}

def _debug_log(logger_instance: Optional[logging.Logger], message: str):
    """Helper for conditional debug logging."""
    if _of_debug_prints_stage2 and logger_instance:
        logger_instance.debug(f"[S2 DEBUG] {message}")
    elif _of_debug_prints_stage2 and not logger_instance:
        print(f"[S2 DEBUG - NO LOGGER] {message}")


def _progress_update(callback, task_name, current, total, force_update=False):
    if callback:
        if force_update or current == 0 or current == total or (total > 0 and (current % (max(1, total // 20))) == 0):
            callback(task_name, current, total)


# --- Data Structures ---

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
                    self.frames.insert(i, frame)
                    inserted = True
                    break
                elif frame.frame_id == f_obj.frame_id:
                    inserted = True
                    break
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

        if valid_points_count == 0:
            return 999.0  # No common points in the zone to compare
        if valid_points_count < len(zone_keypoint_indices) * 0.5:
            return 999.0  # Require at least half the zone points

        normalization_factor = np.sqrt(self.person_box_area) if self.person_box_area > 0 else 1.0
        return (total_distance / valid_points_count) / normalization_factor * 100.0 if normalization_factor > 0 else 999.0

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
        self.last_raw_coords: Optional[Tuple[float, float, float, float]] = None


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
        # List to hold all unique-class fallback contributors.
        self.atr_fallback_contributor_ids: List[int] = []
        self.atr_assigned_position: str = "Not Relevant"
        self.atr_funscript_distance: int = 50
        self.pos_0_100: int = 50
        self.pos_lr_0_100: int = 50
        self.dominant_pose_id: Optional[int] = None
        self.is_occluded: bool = False
        self.active_interaction_track_id: Optional[int] = None
        self.motion_mode: Optional[str] = None

    def parse_raw_frame_data(self, raw_frame_data: dict):
        if not isinstance(raw_frame_data, dict): return
        raw_detections = raw_frame_data.get("detections", [])
        raw_poses = raw_frame_data.get("poses", [])
        for det_data in raw_detections:
            if det_data.get('class_name') in self._effective_discard_classes: continue
            self.boxes.append(
                BoxRecord(self.frame_id, det_data.get('bbox'), det_data.get('confidence'), det_data.get('class'),
                          det_data.get('class_name'), yolo_input_size=self.yolo_input_size))
        for pose_data in raw_poses:
            self.poses.append(PoseRecord(self.frame_id, pose_data.get('bbox'), pose_data.get('keypoints')))

    def get_preferred_penis_box(self, actual_video_type: str = '2D', vr_vertical_third_filter: bool = False) -> Optional[BoxRecord]:
        penis_detections = [b for b in self.boxes if b.class_name == constants.PENIS_CLASS_NAME and not b.is_excluded]
        if not penis_detections: return None

        # --- MODIFICATION: Prioritize low-center boxes for VR/POV ---
        if actual_video_type == 'VR':
            # Score based on a combination of being low (high y2) and centered (low distance to center x)
            center_x = self.yolo_input_size / 2
            penis_detections.sort(key=lambda d: (abs(d.cx - center_x), -d.bbox[3]), reverse=False)
        else:
            # Original logic: sort by lowness (y2) then area
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
            "active_interaction_track_id": self.active_interaction_track_id,
            "is_occluded": self.is_occluded,
            "motion_mode": self.motion_mode,
            "atr_aligned_fallback_candidate_ids": self.atr_fallback_contributor_ids,
            "yolo_boxes": [b.to_dict() for b in self.boxes if not b.is_excluded],
            "poses": [p.to_dict() for p in self.poses]
        }
        if self.atr_locked_penis_state.active and self.atr_locked_penis_state.box:
            lp_state = self.atr_locked_penis_state
            box_dims = lp_state.box
            w = box_dims[2] - box_dims[0]
            h = box_dims[3] - box_dims[1]
            locked_penis_dict = {
                "frame_id": self.frame_id,
                "bbox": list(box_dims),
                "confidence": 1.0,
                "class_id": -1,
                "class_name": "locked_penis",
                "status": "ATR_LOCKED",
                "width": w,
                "height": h,
                "cx": box_dims[0] + w / 2,
                "cy": box_dims[1] + h / 2
            }
            if self.atr_penis_box_kalman:
                locked_penis_dict["visible_bbox"] = list(self.atr_penis_box_kalman)
            frame_data["yolo_boxes"].append(locked_penis_dict)
        return frame_data

    def __repr__(self):
        return f"FrameObject(id={self.frame_id}, #boxes={len(self.boxes)}, #poses={len(self.poses)}, atr_pos='{self.atr_assigned_position}')"


class ATRSegment(BaseSegment):  # Replacing PenisSegment and SexActSegment with a single type from ATR
    def __init__(self, start_frame_id: int, end_frame_id: int, major_position: str):
        super().__init__(start_frame_id, end_frame_id)
        self.major_position = major_position
        self.segment_frame_objects: List[FrameObject] = []
        # List to hold different analysis sub-segments
        # Each entry is a dict: {'start': int, 'end': int, 'mode': str, 'roi': tuple}
        self.sub_segments = []

    def add_sub_segment(self, start_frame, end_frame, mode, roi=None):
        # mode can be 'YOLO' or 'OPTICAL_FLOW'
        # roi is the (x,y,w,h) tuple for optical flow to use
        self.sub_segments.append({
            'start': start_frame,
            'end': end_frame,
            'mode': mode,
            'roi_hint': roi
        })

    def to_dict(self) -> Dict[str, Any]:
        occlusion_info = []  # Placeholder

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
                 scripting_range_end_frame: Optional[int] = None, is_ranged_data_source: bool = False,
                 use_sqlite: bool = True):
        self.video_info = video_info
        self.yolo_input_size = yolo_input_size
        self.vr_vertical_third_filter = vr_filter  # This is the general VR filter for non-penis boxes
        self.frames: List[FrameObject] = []
        self.logger = logger
        self.use_sqlite = use_sqlite
        self.sqlite_storage = None
        self.sqlite_db_path = None

        # Initialize SQLite storage if enabled
        if self.use_sqlite:
            # Generate SQLite database path in output folder
            self.sqlite_db_path = None
            try:
                from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
                self.sqlite_storage = Stage2SQLiteStorage(None, logger)  # Will be set later
                if logger:
                    logger.info("SQLite storage module loaded successfully")
            except ImportError as e:
                if logger:
                    logger.warning(f"SQLite storage not available, falling back to memory: {e}")
                self.use_sqlite = False

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

    def store_frames_to_sqlite(self, batch_size: int = 2000):
        """Store frame objects to SQLite database in batches."""
        if not self.use_sqlite or not self.sqlite_storage or not self.frames:
            return

        self.sqlite_storage.store_frame_objects_batch(self.frames, batch_size)
        if self.logger:
            self.logger.info(f"Stored {len(self.frames)} frame objects to SQLite")

    def store_segments_to_sqlite(self):
        """Store ATR segments to SQLite database."""
        if not self.use_sqlite or not self.sqlite_storage or not self.atr_segments:
            return

        self.sqlite_storage.store_atr_segments(self.atr_segments)
        if self.logger:
            self.logger.info(f"Stored {len(self.atr_segments)} ATR segments to SQLite")

    def clear_memory_frames(self):
        """Clear frame objects from memory after storing to SQLite."""
        if self.use_sqlite and self.sqlite_storage:
            original_count = len(self.frames)
            self.frames.clear()
            if self.logger:
                self.logger.info(f"Cleared {original_count} frame objects from memory")

    def cleanup_sqlite(self, remove_db_file: bool = True):
        """Clean up SQLite resources."""
        if self.sqlite_storage:
            self.sqlite_storage.close()
            self.sqlite_storage.cleanup_temp_files()
            if remove_db_file and self.sqlite_db_path and os.path.exists(self.sqlite_db_path):
                try:
                    os.remove(self.sqlite_db_path)
                    if self.logger:
                        self.logger.info(f"Removed SQLite database file: {self.sqlite_db_path}")
                except OSError:
                    pass

    def __del__(self):
        """Cleanup when object is destroyed - but preserve DB file for Stage 3."""
        # Don't remove the database file during garbage collection
        # Stage 3 will clean it up when it's done
        if self.sqlite_storage:
            self.sqlite_storage.close()


# --- Kalman Smoother Helpers ---
q_ext = np.diag(np.array([0.03, 0.03, 100.0, 100.0, 0.03, 0.03, 50.0, 50.0], dtype=np.float32))
r_ext = np.diag(np.array([0.1, 0.1, 0.1, 0.1], dtype=np.float32))
h_ext = np.array(
    [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
    dtype=np.float32)

def run_rts_smoother_numpy(track_array: np.ndarray) -> np.ndarray:
    """
    A high-performance, NumPy-native implementation of the RTS smoother.
    It operates directly on a NumPy array of box data for a single track.

    Args:
        track_array: A 2D NumPy array for a single track with columns:
                     [frame_id, cx, cy, width, height].

    Returns:
        A 2D NumPy array with the smoothed [cx, cy, width, height] data.
    """
    n = track_array.shape[0]
    frame_ids = track_array[:, 0]
    measurements = track_array[:, 1:5]  # cx, cy, w, h

    # Kalman filter matrices (constants)
    q_ext = np.diag([0.03, 0.03, 100.0, 100.0, 0.03, 0.03, 50.0, 50.0], dtype=np.float32)
    r_ext = np.diag([0.1, 0.1, 0.1, 0.1], dtype=np.float32)
    h_ext = np.array(
        [[1, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0]],
        dtype=np.float32)

    # Vectorized calculation of time deltas (dt)
    dts = np.diff(frame_ids, prepend=frame_ids[0])
    dts[dts <= 0] = 1.0

    # Initialize lists for storing states
    x_filtered = np.zeros((n, 8, 1), dtype=np.float32)
    P_filtered = np.zeros((n, 8, 8), dtype=np.float32)
    x_pred_list = np.zeros((n, 8, 1), dtype=np.float32)
    P_pred_list = np.zeros((n, 8, 8), dtype=np.float32)

    # Initial state
    initial_v = (measurements[1] - measurements[0]) / dts[1] if n > 1 else np.zeros(4)
    x_filtered[0] = np.hstack([measurements[0], initial_v]).reshape(8, 1)
    P_filtered[0] = np.eye(8, dtype=np.float32) * 100.0

    # --- Forward Pass (Kalman Filter) ---
    A = np.eye(8, dtype=np.float32)
    for i in range(1, n):
        dt = dts[i]
        A[0, 4] = A[1, 5] = A[2, 6] = A[3, 7] = dt

        x_pred = A @ x_filtered[i - 1]
        P_pred = A @ P_filtered[i - 1] @ A.T + q_ext

        x_pred_list[i] = x_pred
        P_pred_list[i] = P_pred

        z = measurements[i].reshape(4, 1)
        y = z - (h_ext @ x_pred)
        S = h_ext @ P_pred @ h_ext.T + r_ext
        K = P_pred @ h_ext.T @ np.linalg.pinv(S)

        x_filtered[i] = x_pred + K @ y
        P_filtered[i] = (np.eye(8) - K @ h_ext) @ P_pred

    # --- Backward Pass (RTS Smoother) ---
    x_smoothed = np.copy(x_filtered)
    P_smoothed = np.copy(P_filtered)

    for i in range(n - 2, -1, -1):
        dt = dts[i + 1]
        A[0, 4] = A[1, 5] = A[2, 6] = A[3, 7] = dt

        P_pred_inv = np.linalg.pinv(P_pred_list[i + 1])
        J = P_filtered[i] @ A.T @ P_pred_inv

        x_smoothed[i] += J @ (x_smoothed[i + 1] - x_pred_list[i + 1])
        # P_smoothed is not needed for the final output, so we can skip its calculation

    return x_smoothed[:, :4, 0]  # Return only the smoothed [cx, cy, w, h]

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

def _atr_assign_frame_position(contacts: List[Dict]) -> str:
    if not contacts:
        return "Not Relevant"

    # Use a set for faster 'in' checks
    detected_class_names = {contact["class_name"] for contact in contacts}

    # --- Strict Priority Hierarchy ---
    # This order is now the single source of truth for classification.
    priority_order = [
        ('pussy', 'Cowgirl / Missionary'),
        ('butt', 'Rev. Cowgirl / Doggy'),
        ('face', 'Blowjob'),
        ('hand', 'Handjob'),
        ('breast', 'Boobjob'),
        ('navel', 'Not Relevant'),  # Navel contact is not a distinct action
        ('foot', 'Footjob'),
        ('anus', 'Rev. Cowgirl / Doggy')  # Anus implies a Doggy-style position
    ]

    for class_name, position in priority_order:
        if class_name in detected_class_names:
            return position

    return "Not Relevant"  # Default if no priority classes are found



def _atr_aggregate_segments(frame_objects: List[FrameObject], fps: float, min_segment_duration_frames: int,
                            logger: logging.Logger) -> List[ATRSegment]:
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
    segments: List[ATRSegment] = []
    if frame_objects:
        current_pos = frame_objects[0].atr_assigned_position
        start_frame = frame_objects[0].frame_id
        for i in range(1, len(frame_objects)):
            if frame_objects[i].atr_assigned_position != current_pos:
                segments.append(ATRSegment(start_frame, frame_objects[i - 1].frame_id, current_pos))
                current_pos = frame_objects[i].atr_assigned_position
                start_frame = frame_objects[i].frame_id
        segments.append(ATRSegment(start_frame, frame_objects[-1].frame_id, current_pos))

    _debug_log(logger, f"Pass 1 (Initial Creation) generated {len(segments)} granular segments.")

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
                _debug_log(logger, f"Rule A: Merging {seg1.major_position} across short NR gap.")
                seg1.end_frame_id = seg2.end_frame_id
                seg1.update_duration()
                segments.pop(i + 2)
                segments.pop(i + 1)
                merges_made_in_pass += 1
                i = 0  # Restart scan after a modification
                continue
            i += 1

        # --- Rule B: Merge adjacent "Handjob" and "Blowjob" segments into "Blowjob" ---
        i = 0
        while i < len(segments) - 1:
            pos1, pos2 = segments[i].major_position, segments[i + 1].major_position
            if {pos1, pos2} <= {'Handjob', 'Blowjob'}:  # Use set for commutative check
                _debug_log(logger, f"Rule B: Merging adjacent '{pos1}' and '{pos2}' into 'Blowjob'.")
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
                _debug_log(logger, f"Rule C: Merging adjacent identicals: {segments[i].major_position}")
                segments[i].end_frame_id = segments[i + 1].end_frame_id
                segments[i].update_duration()
                segments.pop(i + 1)
                merges_made_in_pass += 1
                i = 0  # Restart scan
                continue
            i += 1

        # If a full pass of all rules resulted in no changes, the list is stable.
        if merges_made_in_pass == 0:
            _debug_log(logger, "Pass 2 (Iterative Merging) stabilized.")
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

            if prev_dur >= next_dur:  # Merge into previous neighbor
                _debug_log(logger,
                           f"Pass 3: Cleaning up short segment ({segments[i].major_position}) by merging into previous.")
                segments[i - 1].end_frame_id = segments[i].end_frame_id
                segments[i - 1].update_duration()
                segments.pop(i)
                i = 0  # Restart scan from the beginning after a merge
                continue
            else:  # Merge into next neighbor
                _debug_log(logger,
                           f"Pass 3: Cleaning up short segment ({segments[i].major_position}) by merging into next.")
                segments[i + 1].start_frame_id = segments[i].start_frame_id
                segments[i + 1].update_duration()
                segments.pop(i)
                i = 0  # Restart scan
                continue
        i += 1

    # Second, fill any remaining gaps with "Not Relevant" segments for a gapless timeline.
    final_segments: List[ATRSegment] = []
    last_end_frame = -1
    for seg in segments:
        if seg.start_frame_id > last_end_frame + 1:
            final_segments.append(ATRSegment(last_end_frame + 1, seg.start_frame_id - 1, "Not Relevant"))
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
                _debug_log(logger, f"Pass 4: Merging adjacent '{pos1}' and '{pos2}' segments.")
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
            _debug_log(logger, "Pass 4 (Final Merging) stabilized.")
            break

    _debug_log(logger, f"Final, clean segment count: {len(final_segments)}")
    return final_segments

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
        box_y_ref = (class_box_coords[1] + class_box_coords[3]) / 2  # Center Y of the hand
    elif class_name in ['butt', 'pussy']:
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


# --- Step 0: Global Object Tracking (IoU Tracker) ---

def resilient_tracker_step0(state: AppStateContainer, logger: Optional[logging.Logger]):
    """
    An improved IoU-based tracker with state persistence to handle occlusions.
    This replaces the previous simple tracker.
    """
    if not logger:
        logger = logging.getLogger("ResilientTracker")

    _debug_log(logger, "Starting Step 0: Resilient Object Tracking")

    # --- Configuration for the new tracker ---
    fps = state.video_info.get('fps', 30.0)
    IOU_THRESHOLD = 0.3  # Min overlap to be considered the same object in consecutive frames.
    MAX_FRAMES_TO_BECOME_LOST = int(fps * 0.75)  # A track is "lost" if unseen for 1s.
    MAX_FRAMES_IN_LOST_STATE = int(fps * 10)  # A "lost" track is permanently deleted after 10s.
    REID_DISTANCE_THRESHOLD_FACTOR = 2.0  # Search radius for re-identification is 2x the object's size.

    next_global_track_id = 1
    active_tracks: Dict[int, Dict[str, Any]] = {}
    lost_tracks: Dict[int, Dict[str, Any]] = {}

    for frame_obj in sorted(state.frames, key=lambda f: f.frame_id):
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
            _debug_log(logger, f"Track {track_id} ({lost_tracks[track_id]['class_name']}) moved to 'lost' state.")

        tracks_to_delete_permanently = []
        for track_id, track_data in lost_tracks.items():
            track_data["frames_unseen"] += 1
            if track_data["frames_unseen"] > MAX_FRAMES_IN_LOST_STATE:
                tracks_to_delete_permanently.append(track_id)

        for track_id in tracks_to_delete_permanently:
            del lost_tracks[track_id]
            _debug_log(logger, f"Track {track_id} permanently deleted.")

        # --- 2. Match Active Tracks ---
        for track_id, track_data in active_tracks.items():
            best_match_idx, max_iou = -1, -1
            for i in unmatched_detections:
                det_box = current_detections[i]
                if det_box.class_name == track_data["class_name"]:
                    iou = _atr_calculate_iou(track_data["box_rec"].bbox, det_box.bbox)
                    if iou > IOU_THRESHOLD and iou > max_iou:
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
            best_lost_match_id, min_dist = -1, float('inf')

            for track_id, track_data in lost_tracks.items():
                if new_det.class_name == track_data["class_name"]:
                    dist = np.linalg.norm(np.array(new_det.bbox[:2]) - np.array(track_data["box_rec"].bbox[:2]))
                    reid_threshold = REID_DISTANCE_THRESHOLD_FACTOR * max(track_data["box_rec"].width,
                                                                          track_data["box_rec"].height)

                    if dist < reid_threshold and dist < min_dist:
                        min_dist = dist
                        best_lost_match_id = track_id

            if best_lost_match_id != -1:
                _debug_log(logger, f"Re-identified track {best_lost_match_id} with new detection.")
                new_det.track_id = best_lost_match_id
                # Reactivate the track
                reactivated_track = lost_tracks.pop(best_lost_match_id)
                reactivated_track["box_rec"] = new_det
                reactivated_track["frames_unseen"] = 0
                active_tracks[best_lost_match_id] = reactivated_track
                unmatched_detections.remove(i)

        # --- 4. Create New Tracks ---
        for i in unmatched_detections:
            det_box = current_detections[i]
            det_box.track_id = next_global_track_id
            active_tracks[next_global_track_id] = {
                "box_rec": det_box,
                "frames_unseen": 0,
                "class_name": det_box.class_name
            }
            next_global_track_id += 1

    _debug_log(logger, f"Resilient tracking complete. Assigned up to global_track_id {next_global_track_id - 1}.")


# --- ATR Analysis Steps ---
def atr_pass_1_interpolate_boxes(state: AppStateContainer, logger: Optional[logging.Logger]):
    _debug_log(logger, "Starting ATR Pass 1: Interpolate Boxes (using S2 Tracker IDs)")
    track_history: Dict[int, Dict[str, Any]] = {}
    total_interpolated = 0
    MAX_GAP_FRAMES = int(state.video_info.get('fps', 30) * 1)  # Max 1 second gap for interpolation

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


def smooth_single_track_worker_numpy(track_data_tuple: Tuple[np.ndarray, int]) -> Optional[np.ndarray]:
    """
    NumPy-native worker. It receives a NumPy array for a track, smoothes it,
    and returns the smoothed data along with the original track ID.
    """
    track_array, track_id = track_data_tuple

    try:
        # Run the high-performance NumPy-native smoother
        smoothed_coords = run_rts_smoother_numpy(track_array)

        # Apply additional size smoothing (temporally)
        widths, heights = smoothed_coords[:, 2], smoothed_coords[:, 3]
        alpha = 0.1  # Simple exponential moving average for size
        for i in range(1, len(widths)):
            widths[i] = widths[i - 1] + alpha * (widths[i] - widths[i - 1])
            heights[i] = heights[i - 1] + alpha * (heights[i] - heights[i - 1])

        # Return the final smoothed cx, cy, w, h and the track_id to map it back
        return np.hstack([smoothed_coords[:, :2], widths[:, np.newaxis], heights[:, np.newaxis]]), track_id
    except Exception:
        # If smoothing fails for any reason, return None
        return None

# numpy
def atr_pass_1b_smooth_all_tracks(state: AppStateContainer, logger: Optional[logging.Logger]):
    """
    FULLY OPTIMIZED: Uses a NumPy-centric data structure and multiprocessing
    to smooth all tracks at maximum speed.
    """
    return

    _debug_log(logger, "Starting FULLY OPTIMIZED ATR Pass 1b: Smooth All Tracked Boxes")

    # --- 1. One-time conversion to a unified NumPy array ---
    box_list_for_np = []
    for frame in state.frames:
        for box in frame.boxes:
            if box.track_id is not None:
                box_list_for_np.append([frame.frame_id, box.track_id, box.cx, box.cy, box.width, box.height])

    if not box_list_for_np:
        _debug_log(logger, "No tracked boxes to smooth.")
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
    _debug_log(logger, f"Distributing {len(worker_args)} tracks to {num_workers} NumPy-native workers...")

    all_smoothed_data = []
    with Pool(processes=num_workers) as pool:
        # Use imap_unordered for better performance with uneven task times
        for result in pool.imap_unordered(smooth_single_track_worker_numpy, worker_args):
            if result is not None:
                all_smoothed_data.append(result)

    # --- 4. Integrate results back into the original FrameObjects ---
    _debug_log(logger, "Integrating smoothed results...")
    # Create a fast lookup: {track_id: {frame_id: smoothed_box_data}}
    results_map = {}
    for smoothed_track, track_id in all_smoothed_data:
        frame_ids_for_track = worker_args[track_id - 1][0][:, 0].astype(int)  # Get frame_ids from original args
        results_map[track_id] = {fid: smoothed_track[i] for i, fid in enumerate(frame_ids_for_track)}

    for frame in state.frames:
        for box in frame.boxes:
            if box.track_id in results_map and frame.frame_id in results_map[box.track_id]:
                smoothed_data = results_map[box.track_id][frame.frame_id]
                cx, cy, w, h = smoothed_data
                new_bbox = np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2])
                box.update_bbox(new_bbox, new_status=constants.STATUS_SMOOTHED)

    _debug_log(logger, "Fully optimized smoothing pass complete.")

def atr_pass_2_preliminary_height_estimation(state: AppStateContainer, logger: Optional[logging.Logger]):
    """
    A new, lightweight pass to provide a preliminary max_height estimate for each frame.
    This is crucial for the Kalman filter pass (Pass 3) to function correctly before
    the final, per-chapter height is calculated later in Pass 5.
    """
    _debug_log(logger, "Starting ATR Pass 2: Preliminary Height Estimation")

    # We use a simple moving average to provide a slightly more stable preliminary height
    # than just the raw detection of a single frame.
    window_size = 5  # A small 5-frame window
    heights_buffer = []

    for frame_obj in state.frames:
        # Find the best penis detection for the current frame
        selected_penis = frame_obj.get_preferred_penis_box(
            state.video_info.get('actual_video_type', '2D'),
            state.vr_vertical_third_filter
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
            frame_obj.atr_locked_penis_state.max_height = avg_height
        else:
            # If no detections in the window, keep the last known average or default to 0
            if not hasattr(frame_obj.atr_locked_penis_state, 'max_height'):
                frame_obj.atr_locked_penis_state.max_height = 0.0

    _debug_log(logger, "Preliminary height estimation complete.")


def atr_pass_3_kalman_and_lock_state(state: AppStateContainer, logger: Optional[logging.Logger]):
    """
    REFACTORED: This pass now only manages the lock state (active/inactive) and
    calculates the Kalman-filtered VISIBLE penis box. It no longer creates the
    full conceptual box, as the final max_height is not yet known.
    """
    _debug_log(logger, "Starting ATR Pass 3: Kalman Filter and Lock State")
    fps, yolo_size = state.video_info.get('fps', 30.0), state.yolo_input_size
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
    is_vr = state.video_info.get('actual_video_type', '2D') == 'VR'
    pelvis_zone_indices = [11, 12]  # Left and Right Hip

    # --- Global state for locked penis ---
    locked_penis_tracker = {'box_rec': None, 'unseen_frames': 0}
    PENIS_PATIENCE = int(fps * 0.5)  # How many frames to hold onto a lock without seeing it

    for frame_obj in state.frames:
        dominant_pose_this_frame = _get_dominant_pose(frame_obj, is_vr, state.yolo_input_size)
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
                iou = _atr_calculate_iou(locked_penis_tracker['box_rec'].bbox, cand.bbox)
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
            preferred_penis = frame_obj.get_preferred_penis_box(state.video_info.get('actual_video_type', '2D'),
                                                                state.vr_vertical_third_filter)
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
                if _atr_calculate_iou(selected_penis_box_rec.bbox, box_rec.bbox) > 0.001:
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
                _debug_log(logger, f"ATR LP Lock ACTIVATE at frame {frame_obj.frame_id}")

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
        lp_state = frame_obj.atr_locked_penis_state
        lp_state.active = current_lp_active
        # Use the box from the stable tracker for last raw coords
        if last_known_penis_box:
            lp_state.last_raw_coords = tuple(last_known_penis_box.bbox)

        if current_lp_active and lp_state.last_raw_coords:
            predicted_height_kalman = kf_height.predict()[0, 0]
            x1, _, x2, y2_raw = lp_state.last_raw_coords
            frame_obj.atr_penis_box_kalman = (x1, y2_raw - predicted_height_kalman, x2, y2_raw)
        else:
            frame_obj.atr_penis_box_kalman = None

        last_frame_dominant_pose = dominant_pose_this_frame


def atr_pass_4_assign_positions_and_segments(state: AppStateContainer, logger: Optional[logging.Logger]):
    """ ATR Pass 4: Assign frame positions and aggregate into segments (with Sparse Pose Persistence). """
    _debug_log(logger, "Starting ATR Pass 4: Assign Positions & Segments (with Sparse Pose Persistence)")
    fps = state.video_info.get('fps', 30.0)
    is_vr = state.video_info.get('actual_video_type', '2D') == 'VR'
    yolo_size = state.yolo_input_size

    # --- State variables for persistence with sparse data ---
    last_known_good_position: str = "Not Relevant"
    last_known_good_pose: Optional[PoseRecord] = None
    most_recent_dominant_pose: Optional[PoseRecord] = None
    # ---

    for frame_obj in state.frames:
        # 1. ALWAYS check for new pose data to carry forward
        if frame_obj.poses:
            most_recent_dominant_pose = _get_dominant_pose(frame_obj, is_vr, yolo_size)

        # 2. Run original logic based on YOLO contact boxes
        frame_obj.atr_detected_contact_boxes.clear()
        assigned_pos_for_frame = "Not Relevant"

        # Use the kalman-filtered visible box for contact detection, not the conceptual box.
        if frame_obj.atr_locked_penis_state.active and frame_obj.atr_penis_box_kalman:
            visible_penis_coords = frame_obj.atr_penis_box_kalman
            for box_rec in frame_obj.boxes:
                if box_rec.is_excluded or box_rec.class_name in [constants.PENIS_CLASS_NAME, constants.GLANS_CLASS_NAME]:
                    continue
                if _atr_calculate_iou(visible_penis_coords, box_rec.bbox) > 0:
                    frame_obj.atr_detected_contact_boxes.append({"class_name": box_rec.class_name, "box_rec": box_rec})
            assigned_pos_for_frame = _atr_assign_frame_position(frame_obj.atr_detected_contact_boxes)

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
                    _debug_log(logger,
                               f"Frame {frame_obj.frame_id}: Sparse Pose persistence applied. Position '{last_known_good_position}' maintained.")

        # 4. Finalize the frame's position
        frame_obj.atr_assigned_position = assigned_pos_for_frame

    # 5. Aggregate segments (this part of the function is unchanged)
    BaseSegment._id_counter = 0
    state.atr_segments = _atr_aggregate_segments(state.frames, fps, int(1.0 * fps), logger)

def atr_pass_5_recalculate_heights_post_aggregation(state: AppStateContainer, logger: Optional[logging.Logger]):
    """
    Recalculates max_height for the penis based on the final, aggregated chapter segments.
    This ensures height is consistent across a logical action, fixing the "drop to zero" issue.
    This pass should run AFTER atr_pass_4_assign_positions_and_segments.
    """
    _debug_log(logger, "Starting ATR Pass: Recalculate Heights Post-Aggregation")

    # --- Reset penis lock at the start of each new major segment ---
    # This prevents the lock from carrying over across completely different scenes
    # which was a key part of the user request.
    for i in range(len(state.atr_segments)):
        segment = state.atr_segments[i]
        # On the first frame of any segment that isn't the very first one...
        if i > 0:
            first_frame_of_segment_id = segment.start_frame_id
            if first_frame_of_segment_id < len(state.frames):
                # This is a conceptual reset. The actual lock state is managed in Pass 3.
                # Here we ensure the logic in Pass 6, which depends on this, knows about the change.
                _debug_log(logger, f"Chapter changed at frame {first_frame_of_segment_id}. Resetting interactor context for segment '{segment.major_position}'.")

    # Create a quick lookup for raw penis heights from the original detections
    raw_penis_heights = {}
    video_type = state.video_info.get('actual_video_type', '2D')
    vr_filter = state.vr_vertical_third_filter
    for frame_obj in state.frames:
        # We need the original, non-interpolated penis box for its height
        penis_detections = [b for b in frame_obj.boxes if b.class_name == constants.PENIS_CLASS_NAME and b.status == constants.STATUS_DETECTED]
        if penis_detections:
            # Sort by confidence or area to get the most likely "real" penis
            penis_detections.sort(key=lambda d: d.confidence, reverse=True)
            raw_penis_heights[frame_obj.frame_id] = penis_detections[0].height

    # Create a frame-to-object map for quick updates
    frames_by_id = {f.frame_id: f for f in state.frames}

    # Iterate through the final, aggregated segments
    for segment in state.atr_segments:
        if segment.major_position == "Not Relevant":
            chapter_max_h = 0.0
        else:
            # Collect all raw penis heights that fall within this final segment
            heights_in_segment = [h for fid, h in raw_penis_heights.items() if segment.start_frame_id <= fid <= segment.end_frame_id]

            if not heights_in_segment:
                # If no raw detections exist in this segment (e.g., fully interpolated),
                # we must fall back to a reasonable default or carry over from a previous segment.
                # For now, a fallback based on input size is safer than 0.
                chapter_max_h = state.yolo_input_size * 0.2
            else:
                # Use 95th percentile as a robust statistic for max height
                chapter_max_h = np.percentile(heights_in_segment, 95)

        _debug_log(logger, f"Final Segment '{segment.major_position}' ({segment.start_frame_id}-{segment.end_frame_id}): setting max_height to {chapter_max_h:.2f}")

        # Apply this calculated height to every frame object within the segment
        for i in range(segment.start_frame_id, segment.end_frame_id + 1):
            if i in frames_by_id:
                lp_state = frames_by_id[i].atr_locked_penis_state
                lp_state.max_height = chapter_max_h
                # Set a reasonable penetration height based on the final max height
                lp_state.max_penetration_height = chapter_max_h * 0.65


def initial_atr_pass_6_determine_distance(state: AppStateContainer, logger: Optional[logging.Logger]):
    _debug_log(logger, "Starting ATR Pass 6: Determine Frame Distances (Multi-Fallback Heuristic with LOCKING)")
    fps = state.video_info.get('fps', 30.0)
    yolo_size = state.yolo_input_size
    is_vr = state.video_info.get('actual_video_type', '2D') == 'VR'

    # --- Keep track of last bbox position to detect movement ---
    last_known_box_positions = {}  # {track_id: (cx, cy)}

    for segment in state.atr_segments:
        # --- Define context for the entire segment based on its final, stable classification ---
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
            primary_classes_for_segment = ['face']
            fallback_classes_for_segment = ['hand']
        elif segment.major_position == 'Handjob':
            primary_classes_for_segment = ['hand']
            fallback_classes_for_segment = ['breast']
        elif segment.major_position == 'Boobjob':
            primary_classes_for_segment = ['breast']
            fallback_classes_for_segment = ['hand']
        elif segment.major_position == 'Footjob':
            primary_classes_for_segment = ['foot']

        _debug_log(logger,
                   f"Processing segment '{segment.major_position}': Primary={primary_classes_for_segment}, Fallback={fallback_classes_for_segment}")

        # --- Initialize state for this segment's processing ---
        # The lock/stickiness is now reset for each segment.
        active_interactor_state = {'id': None, 'unseen_frames': 0}
        # --- NEW: State for locking fallback interactors per-segment ---
        locked_fallback_interactors = {}  # Format: {class_name: {'id': track_id, 'unseen_frames': 0}}
        FALLBACK_PATIENCE = int(fps * 0.5) # 0.5 seconds of patience
        INTERACTOR_PATIENCE = int(fps * 0.75)
        last_valid_distance = 100  # Start at 'out' position for each segment

        # Assign all frame objects to their segment for easy lookup
        segment.segment_frame_objects = [fo for fo in state.frames if segment.start_frame_id <= fo.frame_id <= segment.end_frame_id]

        for frame_obj in segment.segment_frame_objects:
            lp_state = frame_obj.atr_locked_penis_state
            frame_obj.is_occluded = False
            frame_obj.active_interaction_track_id = None
            frame_obj.atr_fallback_contributor_ids = []  # Reset for each frame

            if not lp_state.active or not lp_state.last_raw_coords:
                frame_obj.atr_funscript_distance = 100
                continue

            # --- Calculate conceptual box and CLAMP the visible box ---
            x1, _, x2, y2_raw = lp_state.last_raw_coords
            conceptual_full_box = (x1, y2_raw - lp_state.max_height, x2, y2_raw)
            lp_state.box = conceptual_full_box

            # Clamp the visible kalman box to ensure it's within the conceptual box's bounds
            if frame_obj.atr_penis_box_kalman:
                conceptual_y1 = conceptual_full_box[1]  # Top of the conceptual box
                kalman_x1, kalman_y1, kalman_x2, kalman_y2 = frame_obj.atr_penis_box_kalman

                # The top of the visible box (kalman_y1) cannot be higher than the conceptual top
                clamped_y1 = max(kalman_y1, conceptual_y1)
                clamped_y1 = min(clamped_y1, kalman_y2)  # Ensure y1 <= y2

                # Update the box in-place so all subsequent logic uses the corrected value
                frame_obj.atr_penis_box_kalman = (kalman_x1, clamped_y1, kalman_x2, kalman_y2)

            # --- Now calculate visible part using the corrected (clamped) box ---
            if frame_obj.atr_penis_box_kalman and lp_state.max_height > 0:
                # This calculation now uses the clamped coordinates, ensuring a correct result
                visible_height = frame_obj.atr_penis_box_kalman[3] - frame_obj.atr_penis_box_kalman[1]
                lp_state.visible_part = (visible_height / lp_state.max_height) * 100.0
            else:
                lp_state.visible_part = 0.0

            # --- Get all contacting objects for the frame ---
            contacting_boxes_all = [c['box_rec'] for c in frame_obj.atr_detected_contact_boxes]
            active_box = None
            comp_dist_for_frame = -1

            # --- STATEFUL & HIERARCHICAL INTERACTOR SELECTION ---

            # 1. Check if the locked interactor is still valid and PRIMARY for this segment
            locked_id = active_interactor_state['id']
            if locked_id is not None:
                found_locked_interactor = next((b for b in contacting_boxes_all if b.track_id == locked_id and b.class_name in primary_classes_for_segment), None)
                if found_locked_interactor:
                    active_box = found_locked_interactor
                    active_interactor_state['unseen_frames'] = 0
                else:
                    active_interactor_state['unseen_frames'] += 1
                    if active_interactor_state['unseen_frames'] > INTERACTOR_PATIENCE:
                        active_interactor_state['id'] = None  # Lost the lock

            # 2. If no valid lock, find the best NEW PRIMARY interactor
            if active_box is None:
                primary_contacts = [b for b in contacting_boxes_all if b.class_name in primary_classes_for_segment]

                # --- Prefer moving contacts, but accept static ones if no moving ones exist ---
                if primary_contacts:
                    moving_contacts = [b for b in primary_contacts if
                                     b.track_id not in last_known_box_positions or
                                     (abs(b.cx - last_known_box_positions[b.track_id][0]) >= 1 or
                                      abs(b.cy - last_known_box_positions[b.track_id][1]) >= 1)]

                    if moving_contacts:
                        active_box = max(moving_contacts, key=lambda b: b.confidence)
                    else:
                        active_box = max(primary_contacts, key=lambda b: b.confidence)

                    if active_box and active_box.track_id != active_interactor_state.get('id'):
                        active_interactor_state['id'] = active_box.track_id
                        _debug_log(logger, f"Frame {frame_obj.frame_id}: Locking on to new primary interactor TID {active_box.track_id} ('{active_box.class_name}')")
                    active_interactor_state['unseen_frames'] = 0

            # --- NEW: STATEFUL & HIERARCHICAL FALLBACK SELECTION ---
            final_contributors = []
            dominant_pose = _get_dominant_pose(frame_obj, is_vr, yolo_size)
            alignment_ref_cx = lp_state.last_raw_coords[0] + (lp_state.last_raw_coords[2] - lp_state.last_raw_coords[0]) / 2
            max_alignment_offset = yolo_size * 0.25

            potential_fallbacks = [b for b in frame_obj.boxes if b.class_name in fallback_classes_for_segment and b != active_box]
            aligned_fallback_candidates = _get_aligned_fallback_boxes(potential_fallbacks, dominant_pose, alignment_ref_cx, max_alignment_offset)

            # Iterate through the possible fallback classes for this segment type
            for fb_class in fallback_classes_for_segment:
                found_current_frame = False
                # 1. Check if we have a lock for this class
                if fb_class in locked_fallback_interactors:
                    lock_info = locked_fallback_interactors[fb_class]
                    locked_box = next((b for b in aligned_fallback_candidates if b.track_id == lock_info['id']), None)
                    if locked_box:
                        final_contributors.append(locked_box)
                        lock_info['unseen_frames'] = 0
                        found_current_frame = True
                    else:
                        lock_info['unseen_frames'] += 1
                        if lock_info['unseen_frames'] > FALLBACK_PATIENCE:
                            del locked_fallback_interactors[fb_class]

                # 2. If no lock or lock expired, find the best new candidate
                if not found_current_frame:
                    penis_base_y = conceptual_full_box[3]
                    # Get candidates of the right class that aren't already locked by another class
                    locked_ids = {info['id'] for info in locked_fallback_interactors.values()}
                    best_new_candidate = sorted([
                        b for b in aligned_fallback_candidates if b.class_name == fb_class and b.track_id not in locked_ids
                    ], key=lambda b: abs(b.cy - penis_base_y))

                    if best_new_candidate:
                        new_box_to_lock = best_new_candidate[0]
                        final_contributors.append(new_box_to_lock)
                        locked_fallback_interactors[fb_class] = {'id': new_box_to_lock.track_id, 'unseen_frames': 0}
                        _debug_log(logger, f"Frame {frame_obj.frame_id}: Locked new fallback '{fb_class}' to TID {new_box_to_lock.track_id}")

            if final_contributors:
                frame_obj.atr_fallback_contributor_ids = [c.track_id for c in final_contributors if c.track_id is not None]

            # --- WEIGHTED DISTANCE CALCULATION ---
            total_weighted_distance = 0.0
            total_weight = 0.0
            max_dist_ref = lp_state.max_height if lp_state.max_height > 0 else state.yolo_input_size * 0.3

            # 1. Primary Contributor (Weight: 1.0)
            if active_box:
                dist = _atr_calculate_normalized_distance_to_base(conceptual_full_box, active_box.class_name, active_box.bbox, max_dist_ref)
                total_weighted_distance += dist * 1.0
                total_weight += 1.0
                frame_obj.active_interaction_track_id = active_box.track_id
                frame_obj.is_occluded = False
            else:
                 frame_obj.is_occluded = True # Occluded if no primary contact

            # 2. Visible Penis Part (Weight: 0.05)
            #if lp_state.visible_part > 0 and is_penetration_pos:
            #    # The visible part is already a 0-100 value, representing "out". We invert it for distance.
            #    total_weighted_distance += (100.0 - lp_state.visible_part) * 0.5
            #    total_weight += 0.05

            # 3. Best-of-Class Fallback Contributors (Weight: 0.2 each)
            for contributor in final_contributors:
                dist = _atr_calculate_normalized_distance_to_base(conceptual_full_box, contributor.class_name, contributor.bbox, max_dist_ref)
                total_weighted_distance += dist * 0.5
                total_weight += 0.5

            # --- Finalize Frame Distance ---
            if total_weight > 0:
                final_distance = total_weighted_distance / total_weight
            else:
                # If no contributors at all, hold the last known value
                final_distance = last_valid_distance
                _debug_log(logger, f"Frame {frame_obj.frame_id}: No contributors found. Holding last valid distance ({final_distance:.1f}).")

            frame_obj.atr_funscript_distance = int(np.clip(round(final_distance), 0, 100))
            last_valid_distance = frame_obj.atr_funscript_distance

            # Update positions for next frame's movement check
            for box in contacting_boxes_all:
                if box.track_id:
                    last_known_box_positions[box.track_id] = (box.cx, box.cy)


def prev_atr_pass_6_determine_distance(state: AppStateContainer, logger: Optional[logging.Logger]):
    _debug_log(logger, "Starting ATR Pass 6: Determine Frame Distances (Multi-Fallback Heuristic with ROBUST LOCKING)")
    fps = state.video_info.get('fps', 30.0)
    yolo_size = state.yolo_input_size
    is_vr = state.video_info.get('actual_video_type', '2D') == 'VR'

    last_known_box_positions = {}

    for segment in state.atr_segments:
        # --- Define context for the entire segment ---
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
            primary_classes_for_segment = ['face']
            fallback_classes_for_segment = ['hand']
        elif segment.major_position == 'Handjob':
            primary_classes_for_segment = ['hand']
            fallback_classes_for_segment = ['breast']
        elif segment.major_position == 'Boobjob':
            primary_classes_for_segment = ['breast']
            fallback_classes_for_segment = ['hand']
        elif segment.major_position == 'Footjob':
            primary_classes_for_segment = ['foot']

        _debug_log(logger,
                   f"Processing segment '{segment.major_position}': Primary={primary_classes_for_segment}, Fallback={fallback_classes_for_segment}")

        # --- Initialize state for this segment's processing ---
        active_interactor_state = {'id': None, 'unseen_frames': 0}
        locked_fallback_interactors = {}
        FALLBACK_PATIENCE = int(fps * 0.5)
        INTERACTOR_PATIENCE = int(fps * 0.75)
        last_valid_distance = 100

        segment.segment_frame_objects = [fo for fo in state.frames if segment.start_frame_id <= fo.frame_id <= segment.end_frame_id]

        for frame_obj in segment.segment_frame_objects:
            lp_state = frame_obj.atr_locked_penis_state
            frame_obj.is_occluded = False  # Default to not occluded
            frame_obj.active_interaction_track_id = None
            frame_obj.atr_fallback_contributor_ids = []

            if not lp_state.active or not lp_state.last_raw_coords:
                frame_obj.atr_funscript_distance = 100
                continue

            # --- Calculate conceptual box and clamp visible box ---
            x1, _, x2, y2_raw = lp_state.last_raw_coords
            conceptual_full_box = (x1, y2_raw - lp_state.max_height, x2, y2_raw)
            lp_state.box = conceptual_full_box

            if frame_obj.atr_penis_box_kalman:
                conceptual_y1 = conceptual_full_box[1]
                kalman_x1, kalman_y1, kalman_x2, kalman_y2 = frame_obj.atr_penis_box_kalman
                clamped_y1 = max(kalman_y1, conceptual_y1)
                clamped_y1 = min(clamped_y1, kalman_y2)
                frame_obj.atr_penis_box_kalman = (kalman_x1, clamped_y1, kalman_x2, kalman_y2)

            if frame_obj.atr_penis_box_kalman and lp_state.max_height > 0:
                visible_height = frame_obj.atr_penis_box_kalman[3] - frame_obj.atr_penis_box_kalman[1]
                lp_state.visible_part = (visible_height / lp_state.max_height) * 100.0
            else:
                lp_state.visible_part = 0.0

            contacting_boxes_all = [c['box_rec'] for c in frame_obj.atr_detected_contact_boxes]
            active_box = None

            # --- ROBUST Primary Interactor Selection (Stateful) ---
            locked_id = active_interactor_state['id']
            if locked_id is not None:
                # First, check if the tracked object is present ANYWHERE in the frame, making the lock robust.
                robustly_found_interactor = next((b for b in frame_obj.boxes if b.track_id == locked_id and b.class_name in primary_classes_for_segment), None)

                if robustly_found_interactor:
                    active_box = robustly_found_interactor
                    active_interactor_state['unseen_frames'] = 0
                    # Check if the robustly found interactor is actually touching now. This determines occlusion, not the lock.
                    if not any(b.track_id == locked_id for b in contacting_boxes_all):
                        frame_obj.is_occluded = True  # It's locked but not touching, so it's occluded.
                else:
                    # The track ID is truly gone from the frame.
                    active_interactor_state['unseen_frames'] += 1
                    if active_interactor_state['unseen_frames'] > INTERACTOR_PATIENCE:
                        active_interactor_state['id'] = None
                        active_box = None

            # --- Find a NEW Primary Interactor if no lock is active ---
            if active_box is None:
                frame_obj.is_occluded = True  # Occluded if there's no primary contact
                primary_contacts = [b for b in contacting_boxes_all if b.class_name in primary_classes_for_segment]
                if primary_contacts:
                    moving_contacts = [b for b in primary_contacts if b.track_id not in last_known_box_positions or (
                            abs(b.cx - last_known_box_positions[b.track_id][0]) >= 1 or abs(
                        b.cy - last_known_box_positions[b.track_id][1]) >= 1)]
                    active_box = max(moving_contacts, key=lambda b: b.confidence) if moving_contacts else max(primary_contacts, key=lambda b: b.confidence)

                    if active_box and active_box.track_id != active_interactor_state.get('id'):
                        active_interactor_state['id'] = active_box.track_id
                        active_interactor_state['unseen_frames'] = 0
                        frame_obj.is_occluded = False  # A new primary contact was found
                        _debug_log(logger, f"Frame {frame_obj.frame_id}: Locking on to new primary interactor TID {active_box.track_id} ('{active_box.class_name}')")

            # --- ROBUST, STATEFUL & HIERARCHICAL FALLBACK SELECTION ---
            final_contributors = []
            for fb_class in fallback_classes_for_segment:
                found_current_frame = False
                if fb_class in locked_fallback_interactors:
                    lock_info = locked_fallback_interactors[fb_class]
                    # This lock is robust: it just checks for the ID's presence in the whole frame.
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
                    max_alignment_offset = yolo_size * 0.25

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
                        _debug_log(logger, f"Frame {frame_obj.frame_id}: Locked new fallback '{fb_class}' to TID {new_box_to_lock.track_id}")

            if final_contributors:
                frame_obj.atr_fallback_contributor_ids = [c.track_id for c in final_contributors if c.track_id is not None]

            # --- WEIGHTED DISTANCE CALCULATION ---
            total_weighted_distance = 0.0
            total_weight = 0.0
            max_dist_ref = lp_state.max_height if lp_state.max_height > 0 else state.yolo_input_size * 0.3

            # Primary Contributor (Weight: 1.0)
            if active_box:
                dist = _atr_calculate_normalized_distance_to_base(conceptual_full_box, active_box.class_name, active_box.bbox, max_dist_ref)
                total_weighted_distance += dist * 1.0
                total_weight += 1.0
                frame_obj.active_interaction_track_id = active_box.track_id

            # Fallback Contributors (Weight: 0.2 each)
            for contributor in final_contributors:
                # Ensure we don't double-count if a primary is also somehow a fallback
                if active_box and contributor.track_id == active_box.track_id:
                    continue
                dist = _atr_calculate_normalized_distance_to_base(conceptual_full_box, contributor.class_name, contributor.bbox, max_dist_ref)
                total_weighted_distance += dist * 0.5
                total_weight += 0.5

            # Finalize Frame Distance
            if total_weight > 0:
                final_distance = total_weighted_distance / total_weight
            else:
                final_distance = last_valid_distance
                _debug_log(logger, f"Frame {frame_obj.frame_id}: No contributors found. Holding last valid distance ({final_distance:.1f}).")

            frame_obj.atr_funscript_distance = int(np.clip(round(final_distance), 0, 100))
            last_valid_distance = frame_obj.atr_funscript_distance

            # Update positions for next frame's movement check
            for box in (contacting_boxes_all + final_contributors):
                if box and box.track_id:
                    last_known_box_positions[box.track_id] = (box.cx, box.cy)

def prev2_atr_pass_6_determine_distance(state: AppStateContainer, logger: Optional[logging.Logger]):
    _debug_log(logger, "Starting ATR Pass 6: Determine Frame Distances (DYNAMIC AMPLIFICATION)")
    fps = state.video_info.get('fps', 30.0)
    yolo_size = state.yolo_input_size
    is_vr = state.video_info.get('actual_video_type', '2D') == 'VR'

    # --- Pre-fetch raw penis heights for efficient lookup ---
    raw_penis_heights = {}
    for frame_obj in state.frames:
        penis_detections = [b for b in frame_obj.boxes if b.class_name == constants.PENIS_CLASS_NAME and b.status == constants.STATUS_DETECTED]
        if penis_detections:
            penis_detections.sort(key=lambda d: d.confidence, reverse=True)
            raw_penis_heights[frame_obj.frame_id] = penis_detections[0].height

    # --- 💡 CONFIGURATION FOR DYNAMIC AMPLIFICATION 💡 ---
    # This is the parameter to tweak. It sets the look-back duration in seconds.
    ROLLING_WINDOW_SECONDS = 20
    rolling_window_frames = int(fps * ROLLING_WINDOW_SECONDS)

    last_known_box_positions = {}

    for segment in state.atr_segments:
        # (Segment setup logic remains the same)
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
            primary_classes_for_segment = ['face']
            fallback_classes_for_segment = ['hand']
        elif segment.major_position == 'Handjob':
            primary_classes_for_segment = ['hand']
            fallback_classes_for_segment = ['breast']
        elif segment.major_position == 'Boobjob':
            primary_classes_for_segment = ['breast']
            fallback_classes_for_segment = ['hand']
        elif segment.major_position == 'Footjob':
            primary_classes_for_segment = ['foot']
        _debug_log(logger,
                   f"Processing segment '{segment.major_position}': Primary={primary_classes_for_segment}, Fallback={fallback_classes_for_segment}")
        active_interactor_state = {'id': None, 'unseen_frames': 0}
        locked_fallback_interactors = {}
        FALLBACK_PATIENCE = int(fps * 0.5)
        INTERACTOR_PATIENCE = int(fps * 0.75)
        last_valid_distance = 100
        segment.segment_frame_objects = [fo for fo in state.frames if
                                         segment.start_frame_id <= fo.frame_id <= segment.end_frame_id]


        for frame_obj in segment.segment_frame_objects:
            lp_state = frame_obj.atr_locked_penis_state
            frame_obj.is_occluded = False
            frame_obj.active_interaction_track_id = None
            frame_obj.atr_fallback_contributor_ids = []

            if not lp_state.active or not lp_state.last_raw_coords:
                frame_obj.atr_funscript_distance = 100
                continue

            # (Conceptual box and interactor locking logic remains the same)
            x1, _, x2, y2_raw = lp_state.last_raw_coords
            conceptual_full_box = (x1, y2_raw - lp_state.max_height, x2, y2_raw)
            lp_state.box = conceptual_full_box
            if frame_obj.atr_penis_box_kalman:
                conceptual_y1 = conceptual_full_box[1]
                kalman_x1, kalman_y1, kalman_x2, kalman_y2 = frame_obj.atr_penis_box_kalman
                clamped_y1 = max(kalman_y1, conceptual_y1)
                clamped_y1 = min(clamped_y1, kalman_y2)
                frame_obj.atr_penis_box_kalman = (kalman_x1, clamped_y1, kalman_x2, kalman_y2)
            if frame_obj.atr_penis_box_kalman and lp_state.max_height > 0:
                visible_height = frame_obj.atr_penis_box_kalman[3] - frame_obj.atr_penis_box_kalman[1]
                lp_state.visible_part = (visible_height / lp_state.max_height) * 100.0
            else:
                lp_state.visible_part = 0.0
            contacting_boxes_all = [c['box_rec'] for c in frame_obj.atr_detected_contact_boxes]
            active_box = None
            locked_id = active_interactor_state['id']
            if locked_id is not None:
                robustly_found_interactor = next((b for b in frame_obj.boxes if b.track_id == locked_id and b.class_name in primary_classes_for_segment), None)
                if robustly_found_interactor:
                    active_box = robustly_found_interactor
                    active_interactor_state['unseen_frames'] = 0
                    if not any(b.track_id == locked_id for b in contacting_boxes_all):
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
                    moving_contacts = [b for b in primary_contacts if b.track_id not in last_known_box_positions or (
                                abs(b.cx - last_known_box_positions[b.track_id][0]) >= 1 or abs(
                            b.cy - last_known_box_positions[b.track_id][1]) >= 1)]
                    active_box = max(moving_contacts, key=lambda b: b.confidence) if moving_contacts else max(
                        primary_contacts, key=lambda b: b.confidence)
                    if active_box and active_box.track_id != active_interactor_state.get('id'):
                        active_interactor_state['id'] = active_box.track_id
                        active_interactor_state['unseen_frames'] = 0
                        frame_obj.is_occluded = False
                        _debug_log(logger,
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
                    alignment_ref_cx = lp_state.last_raw_coords[0] + (
                                lp_state.last_raw_coords[2] - lp_state.last_raw_coords[0]) / 2
                    max_alignment_offset = yolo_size * 0.25
                    potential_fallbacks = [b for b in frame_obj.boxes if
                                           b.class_name in fallback_classes_for_segment and b != active_box]
                    aligned_fallback_candidates = _get_aligned_fallback_boxes(potential_fallbacks, dominant_pose,
                                                                              alignment_ref_cx, max_alignment_offset)
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
                        _debug_log(logger,
                                   f"Frame {frame_obj.frame_id}: Locked new fallback '{fb_class}' to TID {new_box_to_lock.track_id}")
            if final_contributors:
                frame_obj.atr_fallback_contributor_ids = [c.track_id for c in final_contributors if
                                                          c.track_id is not None]


            # --- DYNAMIC AMPLIFICATION: Calculate max_dist_ref on the fly ---
            current_frame_id = frame_obj.frame_id
            window_start_frame = max(0, current_frame_id - rolling_window_frames)

            heights_in_window = [
                h for fid, h in raw_penis_heights.items()
                if window_start_frame <= fid <= current_frame_id
            ]

            dynamic_max_dist_ref = 0
            if heights_in_window:
                dynamic_max_dist_ref = np.percentile(heights_in_window, 95)

            if dynamic_max_dist_ref <= 0:
                dynamic_max_dist_ref = lp_state.max_height if lp_state.max_height > 0 else yolo_size * 0.3


            # --- WEIGHTED DISTANCE CALCULATION (using dynamic reference) ---
            total_weighted_distance = 0.0
            total_weight = 0.0
            max_dist_ref = dynamic_max_dist_ref # Use the newly calculated dynamic reference

            if active_box:
                dist = _atr_calculate_normalized_distance_to_base(conceptual_full_box, active_box.class_name, active_box.bbox, max_dist_ref)
                total_weighted_distance += dist * 1.0
                total_weight += 1.0
                frame_obj.active_interaction_track_id = active_box.track_id

            for contributor in final_contributors:
                if active_box and contributor.track_id == active_box.track_id:
                    continue
                dist = _atr_calculate_normalized_distance_to_base(conceptual_full_box, contributor.class_name, contributor.bbox, max_dist_ref)
                if contributor.class_name == 'face':
                    total_weighted_distance += dist * 0.2
                    total_weight += 0.2
                elif contributor.class_name == 'breast':
                    total_weighted_distance += dist * 0.4
                    total_weight += 0.4
                else:
                    total_weighted_distance += dist * 0.7
                    total_weight += 0.7

            if total_weight > 0:
                final_distance = total_weighted_distance / total_weight
            else:
                final_distance = last_valid_distance

            frame_obj.atr_funscript_distance = int(np.clip(round(final_distance), 0, 100))
            last_valid_distance = frame_obj.atr_funscript_distance

            for box in (contacting_boxes_all + final_contributors):
                if box and box.track_id:
                    last_known_box_positions[box.track_id] = (box.cx, box.cy)

def atr_pass_6_determine_distance(state: AppStateContainer, logger: Optional[logging.Logger]):
    _debug_log(logger, "Starting ATR Pass 6: Determine Frame Distances (DYNAMIC AMPLIFICATION)")
    fps = state.video_info.get('fps', 30.0)
    yolo_size = state.yolo_input_size
    is_vr = state.video_info.get('actual_video_type', '2D') == 'VR'

    # --- Pre-fetch raw penis heights for efficient lookup ---
    raw_penis_heights = {}
    for frame_obj in state.frames:
        penis_detections = [b for b in frame_obj.boxes if b.class_name == constants.PENIS_CLASS_NAME and b.status == constants.STATUS_DETECTED]
        if penis_detections:
            penis_detections.sort(key=lambda d: d.confidence, reverse=True)
            raw_penis_heights[frame_obj.frame_id] = penis_detections[0].height

    if not raw_penis_heights:
        _debug_log(logger, "No raw penis heights found to process. Skipping distance calculation.")
        return

    # Create sorted NumPy arrays for lightning-fast lookups
    sorted_fids = np.array(sorted(raw_penis_heights.keys()))
    sorted_heights = np.array([raw_penis_heights[fid] for fid in sorted_fids])

    ROLLING_WINDOW_SECONDS = 30
    rolling_window_frames = int(fps * ROLLING_WINDOW_SECONDS)

    last_known_box_positions = {}

    for segment in state.atr_segments:
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
        _debug_log(logger,
                   f"Processing segment '{segment.major_position}': Primary={primary_classes_for_segment}, Fallback={fallback_classes_for_segment}")
        active_interactor_state = {'id': None, 'unseen_frames': 0}
        locked_fallback_interactors = {}
        FALLBACK_PATIENCE = int(fps * 0.5)
        INTERACTOR_PATIENCE = int(fps * 0.75)
        last_valid_distance = 100
        segment.segment_frame_objects = [fo for fo in state.frames if segment.start_frame_id <= fo.frame_id <= segment.end_frame_id]

        for frame_obj in segment.segment_frame_objects:
            lp_state = frame_obj.atr_locked_penis_state
            frame_obj.is_occluded = False
            frame_obj.active_interaction_track_id = None
            frame_obj.atr_fallback_contributor_ids = []

            if not lp_state.active or not lp_state.last_raw_coords:
                frame_obj.atr_funscript_distance = 100
                continue

            # (Conceptual box and interactor locking logic remains the same)
            x1, _, x2, y2_raw = lp_state.last_raw_coords
            conceptual_full_box = (x1, y2_raw - lp_state.max_height, x2, y2_raw)
            lp_state.box = conceptual_full_box
            if frame_obj.atr_penis_box_kalman:
                conceptual_y1 = conceptual_full_box[1]
                kalman_x1, kalman_y1, kalman_x2, kalman_y2 = frame_obj.atr_penis_box_kalman
                clamped_y1 = max(kalman_y1, conceptual_y1)
                clamped_y1 = min(clamped_y1, kalman_y2)
                frame_obj.atr_penis_box_kalman = (kalman_x1, clamped_y1, kalman_x2, kalman_y2)
            if frame_obj.atr_penis_box_kalman and lp_state.max_height > 0:
                visible_height = frame_obj.atr_penis_box_kalman[3] - frame_obj.atr_penis_box_kalman[1]
                lp_state.visible_part = (visible_height / lp_state.max_height) * 100.0
            else:
                lp_state.visible_part = 0.0
            contacting_boxes_all = [c['box_rec'] for c in frame_obj.atr_detected_contact_boxes]
            active_box = None
            locked_id = active_interactor_state['id']
            if locked_id is not None:
                robustly_found_interactor = next((b for b in frame_obj.boxes if b.track_id == locked_id and b.class_name in primary_classes_for_segment), None)
                if robustly_found_interactor:
                    active_box = robustly_found_interactor
                    active_interactor_state['unseen_frames'] = 0
                    if not any(b.track_id == locked_id for b in contacting_boxes_all):
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
                    moving_contacts = [b for b in primary_contacts if b.track_id not in last_known_box_positions or (
                                abs(b.cx - last_known_box_positions[b.track_id][0]) >= 1 or abs(
                            b.cy - last_known_box_positions[b.track_id][1]) >= 1)]
                    active_box = max(moving_contacts, key=lambda b: b.confidence) if moving_contacts else max(
                        primary_contacts, key=lambda b: b.confidence)
                    if active_box and active_box.track_id != active_interactor_state.get('id'):
                        active_interactor_state['id'] = active_box.track_id
                        active_interactor_state['unseen_frames'] = 0
                        frame_obj.is_occluded = False
                        _debug_log(logger,
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
                    max_alignment_offset = yolo_size * 0.25
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
                        _debug_log(logger,
                                   f"Frame {frame_obj.frame_id}: Locked new fallback '{fb_class}' to TID {new_box_to_lock.track_id}")
            if final_contributors:
                frame_obj.atr_fallback_contributor_ids = [c.track_id for c in final_contributors if
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
                dist = _atr_calculate_normalized_distance_to_base(conceptual_full_box, active_box.class_name, active_box.bbox, max_dist_ref)
                total_weighted_distance += dist * 1.0
                total_weight += 1.0
                frame_obj.active_interaction_track_id = active_box.track_id

            for contributor in final_contributors:
                if active_box and contributor.track_id == active_box.track_id:
                    continue
                dist = _atr_calculate_normalized_distance_to_base(conceptual_full_box, contributor.class_name, contributor.bbox, max_dist_ref)
                if contributor.class_name == 'breast':
                    weight = 0.2
                elif contributor.class_name == 'anus':
                    weight = 0.3
                elif contributor.class_name == 'face':
                    weight = 0.1
                elif contributor.class_name == 'navel':
                    weight = 0.8
                else:
                    weight = 0.05
                total_weighted_distance += dist * weight
                total_weight += weight

            if total_weight > 0:
                final_distance = total_weighted_distance / total_weight
            else:
                final_distance = last_valid_distance

            frame_obj.atr_funscript_distance = int(np.clip(round(final_distance), 0, 100))
            last_valid_distance = frame_obj.atr_funscript_distance

            for box in (contacting_boxes_all + final_contributors):
                if box and box.track_id:
                    last_known_box_positions[box.track_id] = (box.cx, box.cy)

def atr_pass_7_smooth_and_normalize_distances(state: AppStateContainer, logger: Optional[logging.Logger]):
    """ ATR Pass 7 (denoising with SG) and Pass 9 (amplifying/normalizing per segment) combined """
    _debug_log(logger, "Starting ATR Pass 7: Smooth and Normalize Distances")

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

    full_script_data_np = np.array(
        [[fo.frame_id, fo.atr_funscript_distance] for fo in state.frames],
        dtype=np.float64
    )

    state.funscript_frames = full_script_data_np[:, 0].astype(int).tolist()
    state.funscript_distances = full_script_data_np[:, 1].astype(int).tolist()



def atr_pass_8_simplify_signal(state: AppStateContainer, logger: Optional[logging.Logger]):
    """
    ULTIMATE SOLUTION: A fully vectorized, high-performance signal simplification pipeline
    that minimizes data conversions and uses optimized algorithms.
    """
    _debug_log(logger, "Starting ULTIMATE ATR Pass 8: High-Performance Signal Simplification")

    if not state.frames:
        _debug_log(logger, "No data to simplify.")
        return

    # --- Step 1: One-time conversion to a NumPy array ---
    full_script_data_np = np.array(
        [[fo.frame_id, fo.atr_funscript_distance] for fo in state.frames],
        dtype=np.float64
    )

    if full_script_data_np.shape[0] < 3:
        state.funscript_frames = full_script_data_np[:, 0].astype(int).tolist()
        state.funscript_distances = full_script_data_np[:, 1].astype(int).tolist()
        return

    positions = full_script_data_np[:, 1]

    # --- Step 2: Peak/Valley Detection (Vectorized) ---

    peaks, _ = find_peaks(positions, prominence=1.0)
    valleys, _ = find_peaks(-positions, prominence=1.0) # For minima

    # Combine peak, valley, start, and end indices. `np.unique` also sorts the indices.
    key_indices = np.unique(np.concatenate(([0, len(positions) - 1], peaks, valleys)))

    # Extract only the keyframes based on these indices. The data remains a NumPy array.
    keyframes_np = full_script_data_np[key_indices]
    _debug_log(logger, f"Pass 1 (Peak/Valley): Simplified to {len(keyframes_np)} points.")

    # --- Step 3: High-Performance RDP Simplification ---
    # Apply RDP simplification. The `simplify_coords_vw` is a C-based implementation
    # from the `simplification` library, which is very fast. Epsilon=2.0 is a reasonable value.
    if len(keyframes_np) > 2:
        final_simplified_np = simplify_coords_vw(keyframes_np, 2.0)
    else:
        final_simplified_np = keyframes_np
    _debug_log(logger, f"Pass 2 (RDP): Simplified to {len(final_simplified_np)} points.")

    # --- Step 4: Final Conversion to Output Lists ---
    final_frames = final_simplified_np[:, 0].astype(int)
    final_positions = final_simplified_np[:, 1].astype(int)

    state.funscript_frames = final_frames.tolist()
    state.funscript_distances = final_positions.tolist()

    _debug_log(logger, f"High-performance simplification complete. Final points: {len(state.funscript_frames)}")

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


def _pre_scan_for_interactor_ids(state: AppStateContainer, logger: Optional[logging.Logger]) -> set:
    """
    Performs a read-only pre-scan of all frames to identify the track IDs of all
    objects that ever interact with the penis. This avoids running expensive
    recovery on irrelevant tracks.
    """
    _debug_log(logger, "Pre-scanning to identify all potential interactor track IDs...")
    interactor_ids = set()

    # This scan simulates the lock-on logic from Pass 3 to accurately find interactors.
    # It does not modify any frame state.
    current_lp_active = False
    current_lp_consecutive_detections = 0
    current_lp_consecutive_non_detections = 0
    fps = state.video_info.get('fps', 30.0)

    for frame in state.frames:
        penis_box = frame.get_preferred_penis_box(
            state.video_info.get('actual_video_type', '2D'),
            state.vr_vertical_third_filter
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
                    if _atr_calculate_iou(penis_box.bbox, other_box.bbox) > 0:
                        interactor_ids.add(other_box.track_id)

    _debug_log(logger, f"Pre-scan found {len(interactor_ids)} unique interactor track IDs.")
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
        flow_dense = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_FAST)

        # The loop now runs only for the precise, pre-calculated duration.
        # No extra 'interrupt' check is needed here.
        for frame_id in range(gap_info["start_frame"], gap_info["end_frame"] + 1):
            frames_processed_in_worker += 1
            current_frame_img = vp._get_specific_frame(frame_id)
            if current_frame_img is None: break

            current_gray = cv2.cvtColor(current_frame_img, cv2.COLOR_BGR2GRAY)
            flow = flow_dense.calc(prev_gray, current_gray, None)
            if flow is None: break

            x1, y1, x2, y2 = [max(0, int(c)) for c in current_tracked_box]
            h, w = prev_gray.shape
            x2, y2 = min(w, x2), min(h, y2)

            if x2 > x1 and y2 > y1:
                # Use median flow in the box to be robust to outliers
                dx, dy = np.median(flow[y1:y2, x1:x2, 0]), np.median(flow[y1:y2, x1:x2, 1])
                current_tracked_box += [dx, dy, dx, dy]

                new_box = BoxRecord(
                    frame_id=frame_id, bbox=current_tracked_box,
                    confidence=gap_info["last_known_box"].confidence * 0.75,  # Lower confidence for recovered boxes
                    class_id=gap_info["last_known_box"].class_id, class_name=gap_info["last_known_box"].class_name,
                    status=constants.STATUS_OF_RECOVERED, yolo_input_size=yolo_input_size,
                    track_id=gap_info["track_id"]
                )
                recovered_boxes.append(new_box)
                prev_gray = current_gray
            else:
                # Stop if the tracked box becomes invalid
                break
    finally:
        vp.stop_processing(join_thread=False)
        del vp

    return recovered_boxes, frames_processed_in_worker


def atr_pass_1c_recover_lost_tracks_with_of(
        state: AppStateContainer,
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
    _debug_log(logger, "Starting ATR Pass 1c: Recover Lost Tracks with INTERRUPTIBLE Optical Flow")

    if not preprocessed_video_path_arg or not os.path.exists(preprocessed_video_path_arg):
        message = "Skipped - Preprocessed file not found."
        logger.warning(f"Optical flow recovery: {message} Path: {preprocessed_video_path_arg}")
        if progress_callback: progress_callback(main_step_info, {"message": message, "current": 1, "total": 1}, True)
        return

    # Validate preprocessed video before using it
    try:
        from detection.cd.stage_1_cd import _validate_preprocessed_video_completeness
        expected_frames = len(state.frames) if state.frames else 0
        fps = state.video_info.get('fps', 30.0)

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
    interactor_track_ids = _pre_scan_for_interactor_ids(state, logger)
    if not interactor_track_ids:
        if progress_callback: progress_callback(main_step_info, {"message": "Skipped - No interactors", "current": 1, "total": 1}, True)
        return
    fps = state.video_info.get('fps', 30.0)
    # Max gap to attempt OF recovery on
    MAX_RECOVERY_GAP = int(fps * 5)
    # Min gap to trigger OF instead of simple interpolation
    MIN_RECOVERY_GAP = int(fps * 1)
    # Confidence threshold for a detection to be considered a valid "interrupter"
    INTERRUPT_CONFIDENCE_THRESHOLD = 0.6

    frames_map = {f.frame_id: f for f in state.frames}

    # Create a map for quick lookup of all boxes by track_id
    # {track_id: {frame_id: box_record}}
    track_box_map = {}
    for frame in state.frames:
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
        'yolo_input_size': state.yolo_input_size,
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
        app_logic_instance=None,  # To access AppStateContainer-like features
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

    def atr_progress_wrapper(main_step_tuple, sub_info_from_module, force=False):
        """ This wrapper now passes the sub_info_from_module (which can be a tuple or dict) directly. """
        if progress_callback:
            # The original callback expects: main_info, sub_info, force
            progress_callback(main_step_tuple, sub_info_from_module, force)

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

    # 3. Initialize AppStateContainer with SQLite storage
    try:
        state = AppStateContainer(video_info_dict, yolo_input_size_arg, vr_vertical_third_filter_arg,
                                  all_raw_detections, logger,
                                  discarded_classes_runtime_arg=discarded_classes_runtime_arg,
                                  scripting_range_active=scripting_range_active_arg,
                                  scripting_range_start_frame=scripting_range_start_frame_arg,
                                  scripting_range_end_frame=scripting_range_end_frame_arg,
                                  is_ranged_data_source=is_ranged_data_source,
                                  use_sqlite=use_sqlite_storage)

        # Initialize SQLite database path in output folder if enabled
        if state.use_sqlite and state.sqlite_storage and output_folder_path:
            # Generate database filename based on video name
            video_filename = os.path.splitext(os.path.basename(video_path_arg))[0]
            
            # Handle preprocessed videos: use original filename for database consistency  
            if video_filename.endswith('_preprocessed'):
                video_filename = video_filename[:-len('_preprocessed')]
                logger.debug(f"Using original video stem '{video_filename}' for SQLite database naming")
            
            db_filename = f"{video_filename}_stage2_data.db"
            state.sqlite_db_path = os.path.join(output_folder_path, db_filename)

            # Set the database path and initialize
            state.sqlite_storage.set_db_path(state.sqlite_db_path)
            logger.info(f"SQLite storage initialized in output folder: {state.sqlite_db_path}")

    except Exception as e:
        logger.error(f"Error creating AppStateContainer (ATR S2): {e}", exc_info=True)
        return {"error": f"AppStateContainer init failed (ATR S2): {e}"}
    if stop_event.is_set(): return {"error": "Processing stopped after AppState init (ATR S2)."}

    # --- ATR Processing Steps ---
    # These steps will fill data into state.frames[...].atr_... attributes and state.atr_segments

    # Define main steps for progress reporting
    # Base steps for segmentation
    atr_main_steps_list_base = [
        ("Step 1: Tracking Objects", resilient_tracker_step0),
        ("Step 2: Interpolate Short Gaps", atr_pass_1_interpolate_boxes),
        ("Step 3: Recover Long Gaps (OF)", atr_pass_1c_recover_lost_tracks_with_of),
        ("Step 4: Smooth All Tracked Boxes", atr_pass_1b_smooth_all_tracks),
        ("Step 5: Kalman Filter & Lock State", atr_pass_3_kalman_and_lock_state),
        ("Step 6: Assign Positions & Segments", atr_pass_4_assign_positions_and_segments),
        ("Step 7: Recalculate Chapter Heights", atr_pass_5_recalculate_heights_post_aggregation),
        ("Step 8: Determine Frame Distances", atr_pass_6_determine_distance),
        ("Step 9: Smooth & Normalize Distances", atr_pass_7_smooth_and_normalize_distances),
    ]
    atr_main_steps_list_funscript_gen = [
    #    #("Step 8: Smooth & Normalize Distances", atr_pass_7_smooth_and_normalize_distances),
        ("Step 10: Simplify Signal", atr_pass_8_simplify_signal)
    ]
    atr_main_steps_list = atr_main_steps_list_base
    # if generate_funscript_actions_arg:
    #     atr_main_steps_list.extend(atr_main_steps_list_funscript_gen)

    num_main_atr_steps = len(atr_main_steps_list)

    for i, (main_step_name_atr, step_func_atr) in enumerate(atr_main_steps_list):
        logger.info(f"Starting {main_step_name_atr} (ATR S2)")
        main_step_tuple_for_callback = (i + 1, num_main_atr_steps, main_step_name_atr)

        atr_progress_wrapper(main_step_tuple_for_callback, (0, 1, "Initializing..."), True)

        # --- Special handling for the OF recovery pass ---
        if step_func_atr == atr_pass_1c_recover_lost_tracks_with_of:
            step_func_atr(state, preprocessed_video_path_arg, logger, atr_progress_wrapper,
                          main_step_tuple_for_callback, num_workers_stage2_of_arg)
        else:
            # Standard call for all other passes
            step_func_atr(state, logger)

        if stop_event.is_set():
            logger.info(f"ATR S2 stopped during {main_step_name_atr}.")
            atr_progress_wrapper(main_step_tuple_for_callback, "Aborted", 1, 1, True)
            return {"error": f"Processing stopped during {main_step_name_atr} (ATR S2)"}

        if step_func_atr != atr_pass_1c_recover_lost_tracks_with_of:

            atr_progress_wrapper(main_step_tuple_for_callback, (1, 1, "Completed"), True)

    # --- Prepare overlay data BEFORE clearing frames from memory ---
    all_frames_overlay_data = []
    overlay_package = None
    if output_overlay_msgpack_path:
        logger.info("Preparing overlay data before memory optimization...")
        all_frames_overlay_data = [frame.to_overlay_dict() for frame in state.frames if not stop_event.is_set()]
        if stop_event.is_set(): return {"error": "Processing stopped during overlay data prep (ATR S2)."}
        try:
            segments_for_overlay = [seg.to_dict() for seg in state.atr_segments]
        except Exception:
            segments_for_overlay = []
        overlay_package = {"frames": all_frames_overlay_data, "segments": segments_for_overlay, "metadata": {"schema": "v1.1"}}

    # --- Store processed data to SQLite for Stage 3 memory optimization ---
    if state.use_sqlite and state.sqlite_storage:
        logger.info("Storing processed frame data to SQLite database...")
        try:
            # Store frame objects to database
            state.store_frames_to_sqlite(batch_size=2000)

            # Store ATR segments to database
            state.store_segments_to_sqlite()

            # Clear frames from memory to save RAM for Stage 3
            state.clear_memory_frames()

            logger.info("Successfully stored Stage 2 data to SQLite and cleared memory")
        except Exception as e:
            logger.error(f"Error storing data to SQLite: {e}", exc_info=True)
            # Continue with in-memory processing as fallback
            state.use_sqlite = False

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
        "all_s2_frame_objects_list": state.frames,
        "sqlite_db_path": getattr(state, 'sqlite_db_path', None)  # Include SQLite path if available
    }

    if output_overlay_msgpack_path and (overlay_package or all_frames_overlay_data):
        logger.info(f"Saving ATR Stage 2 overlay data to: {output_overlay_msgpack_path}")
        try:
            def numpy_default_handler(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable for msgpack")

            to_write = overlay_package if overlay_package is not None else all_frames_overlay_data
            with open(output_overlay_msgpack_path, 'wb') as f:
                f.write(msgpack.packb(to_write, use_bin_type=True, default=numpy_default_handler))
            if overlay_package:
                logger.info(f"Successfully saved ATR overlay package with {len(overlay_package.get('frames', []))} frames and {len(overlay_package.get('segments', []))} segments to {output_overlay_msgpack_path}.")
            else:
                logger.info(f"Successfully saved ATR overlay data for {len(all_frames_overlay_data)} frames to {output_overlay_msgpack_path}.")
            if os.path.exists(output_overlay_msgpack_path):
                return_dict["overlay_msgpack_path"] = output_overlay_msgpack_path
        except Exception as e:
            logger.error(f"Error saving ATR Stage 2 overlay msgpack to {output_overlay_msgpack_path}: {e}", exc_info=True)

    logger.info(f"--- ATR-based Stage 2 Analysis Finished. Segments: {len(final_video_segments)} ---")
    return return_dict

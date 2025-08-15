"""
Frame object and related state structures for Stage 2 processing.

This module contains the FrameObject class which represents all detection
data for a single frame, along with helper classes like LockedPenisState.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from config import constants
from .box_records import BoxRecord, PoseRecord


class LockedPenisState:
    """
    State information for locked penis tracking.
    
    This class maintains the state of penis tracking across frames, including
    position, detection statistics, and visibility information.
    """
    
    def __init__(self):
        """Initialize locked penis state with default values."""
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
    """
    Comprehensive representation of all detection data for a single video frame.
    
    This class aggregates YOLO detections, pose data, and Stage 2 analysis results
    for a single frame, serving as the primary data structure for Stage 2 processing.
    """
    
    _id_counter = 0

    def __init__(self, frame_id: int, yolo_input_size: int, raw_frame_data: Optional[dict] = None,
                 classes_to_discard_runtime_set: Optional[set] = None):
        """
        Initialize a FrameObject with detection data.
        
        Args:
            frame_id: Frame index in the video
            yolo_input_size: Size of YOLO input (typically 640)
            raw_frame_data: Raw detection data from YOLO processing
            classes_to_discard_runtime_set: Set of class names to exclude from processing
        """
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
        self.penis_box_kalman: Optional[Tuple[float, float, float, float]] = None
        self.locked_penis_state = LockedPenisState()
        self.detected_contact_boxes: List[Dict] = []
        self.distances_to_penis: List[Dict] = []
        # List to hold all unique-class fallback contributors.
        self.fallback_contributor_ids: List[int] = []
        self.assigned_position: str = "Not Relevant"
        self.funscript_distance: int = 50
        self.pos_0_100: int = 50
        self.pos_lr_0_100: int = 50
        self.dominant_pose_id: Optional[int] = None
        self.is_occluded: bool = False
        self.active_interaction_track_id: Optional[int] = None
        self.motion_mode: Optional[str] = None

    def parse_raw_frame_data(self, raw_frame_data: dict):
        """
        Parse raw YOLO detection data into BoxRecord and PoseRecord objects.
        
        Args:
            raw_frame_data: Dictionary containing 'detections' and 'poses' lists
        """
        if not isinstance(raw_frame_data, dict):
            return
            
        raw_detections = raw_frame_data.get("detections", [])
        raw_poses = raw_frame_data.get("poses", [])
        
        for det_data in raw_detections:
            if det_data.get('class_name') in self._effective_discard_classes:
                continue
            self.boxes.append(
                BoxRecord(self.frame_id, det_data.get('bbox'), det_data.get('confidence'), det_data.get('class'),
                          det_data.get('class_name'), yolo_input_size=self.yolo_input_size))
                          
        for pose_data in raw_poses:
            self.poses.append(PoseRecord(self.frame_id, pose_data.get('bbox'), pose_data.get('keypoints')))

    def get_preferred_penis_box(self, actual_video_type: str = '2D', vr_vertical_third_filter: bool = False) -> Optional[BoxRecord]:
        """
        Get the most suitable penis detection for this frame.
        
        Uses different selection criteria for VR vs 2D videos to account for
        different viewing perspectives and typical penis positions.
        
        Args:
            actual_video_type: '2D' or 'VR' video type
            vr_vertical_third_filter: If True, prefer boxes in central third for VR
            
        Returns:
            Best penis BoxRecord, or None if no suitable detection found
        """
        penis_detections = [b for b in self.boxes if b.class_name == constants.PENIS_CLASS_NAME and not b.is_excluded]
        if not penis_detections:
            return None

        # MODIFICATION: Prioritize low-center boxes for VR/POV
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
        """
        Convert frame data to dictionary format suitable for overlay visualization.
        
        This method creates a representation of the frame data that can be used
        for creating debug overlays or saving analysis results.
        
        Returns:
            Dictionary containing frame analysis data
        """
        frame_data = {
            "frame_id": self.frame_id,
            "assigned_position": self.assigned_position,
            "dominant_pose_id": self.dominant_pose_id,
            "active_interaction_track_id": self.active_interaction_track_id,
            "is_occluded": self.is_occluded,
            "motion_mode": self.motion_mode,
            "aligned_fallback_candidate_ids": self.fallback_contributor_ids,
            "yolo_boxes": [b.to_dict() for b in self.boxes if not b.is_excluded],
            "poses": [p.to_dict() for p in self.poses]
        }
        
        # Add locked penis state if active
        if self.locked_penis_state.active and self.locked_penis_state.box:
            lp_state = self.locked_penis_state
            box_dims = lp_state.box
            w = box_dims[2] - box_dims[0]
            h = box_dims[3] - box_dims[1]
            locked_penis_dict = {
                "frame_id": self.frame_id,
                "bbox": list(box_dims),
                "confidence": 1.0,
                "class_id": -1,
                "class_name": "locked_penis",
                "status": "LOCKED",
                "width": w,
                "height": h,
                "cx": box_dims[0] + w / 2,
                "cy": box_dims[1] + h / 2
            }
            if self.penis_box_kalman:
                locked_penis_dict["visible_bbox"] = list(self.penis_box_kalman)
            frame_data["yolo_boxes"].append(locked_penis_dict)
            
        return frame_data

    def __repr__(self):
        return f"FrameObject(id={self.frame_id}, #boxes={len(self.boxes)}, #poses={len(self.poses)}, pos='{self.assigned_position}')"
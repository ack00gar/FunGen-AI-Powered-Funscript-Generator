"""
Box and pose record data structures for Stage 2 processing.

This module contains the BoxRecord and PoseRecord classes that represent
individual detection results from YOLO processing.
"""

import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from config import constants


class BoxRecord:
    """
    Represents a single bounding box detection from YOLO processing.
    
    Contains both the original detection data and additional metadata
    added during Stage 2 processing like tracking IDs and status.
    """
    
    def __init__(self, frame_id: int, bbox: Union[np.ndarray, List[float], Tuple[float, float, float, float]],
                 confidence: float, class_id: int, class_name: str,
                 status: str = constants.STATUS_DETECTED, yolo_input_size: int = 640,
                 track_id: Optional[int] = None):
        """
        Initialize a BoxRecord.
        
        Args:
            frame_id: Frame index where this detection occurred
            bbox: Bounding box coordinates as [x1, y1, x2, y2]
            confidence: Detection confidence score (0.0-1.0)
            class_id: YOLO class ID number
            class_name: Human-readable class name
            status: Processing status (detected, smoothed, interpolated, etc.)
            yolo_input_size: Size of YOLO input (for area percentage calculation)
            track_id: Tracking ID assigned during Stage 2 processing
        """
        self.frame_id = int(frame_id)
        self.bbox = np.array(bbox, dtype=np.float32)
        self.confidence = float(confidence)
        self.class_id = int(class_id)
        self.class_name = str(class_name)
        self.status = str(status)
        self.track_id = track_id

        # Validate and fix invalid bounding boxes
        if not (len(self.bbox) == 4 and self.bbox[2] > self.bbox[0] and self.bbox[3] > self.bbox[1]):
            self.bbox = np.array([0, 0, 0, 0], dtype=np.float32)

        self._update_dims()
        self.yolo_input_size = yolo_input_size
        self.area_perc = (self.area / (yolo_input_size * yolo_input_size)) * 100 if yolo_input_size > 0 else 0
        self.is_excluded: bool = False
        self.is_tracked: bool = False

    def _update_dims(self):
        """Update derived dimensions and properties from bbox coordinates."""
        self.width = self.bbox[2] - self.bbox[0]
        self.height = self.bbox[3] - self.bbox[1]
        self.area = self.width * self.height
        self.cx = self.bbox[0] + self.width / 2
        self.cy = self.bbox[1] + self.height / 2
        self.x1, self.y1, self.x2, self.y2 = self.bbox[0], self.bbox[1], self.bbox[2], self.bbox[3]
        self.box = tuple(self.bbox)

    def update_bbox(self, new_bbox: np.ndarray, new_status: Optional[str] = None):
        """
        Update the bounding box coordinates and optionally the status.
        
        Args:
            new_bbox: New bounding box coordinates
            new_status: Optional new status to set
        """
        self.bbox = np.array(new_bbox, dtype=np.float32)
        self._update_dims()
        if new_status:
            self.status = new_status

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert BoxRecord to dictionary representation.
        
        Returns:
            Dictionary containing all BoxRecord data
        """
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
    """
    Represents a single pose detection with keypoints from YOLO pose estimation.
    
    Contains the person bounding box and keypoint data for pose-based analysis.
    """
    
    _id_counter = 0

    def __init__(self, frame_id: int, bbox: list, keypoints_data: list):
        """
        Initialize a PoseRecord.
        
        Args:
            frame_id: Frame index where this pose was detected
            bbox: Person bounding box coordinates
            keypoints_data: Array of keypoint coordinates and confidences
        """
        self.id = PoseRecord._id_counter
        PoseRecord._id_counter += 1
        self.frame_id = int(frame_id)
        self.bbox = np.array(bbox, dtype=np.float32)
        self.keypoints = np.array(keypoints_data, dtype=np.float32)
        self.keypoint_confidence_threshold = 0.3

    @property
    def person_box_area(self) -> float:
        """
        Calculate the area of the person bounding box.
        
        Returns:
            Area of the bounding box, or 0.0 if invalid
        """
        if self.bbox is None or len(self.bbox) < 4:
            return 0.0
        return (self.bbox[2] - self.bbox[0]) * (self.bbox[3] - self.bbox[1])

    def calculate_zone_dissimilarity(self, other_pose: 'PoseRecord', zone_keypoint_indices: List[int]) -> float:
        """
        Calculate dissimilarity between this pose and another based on specific keypoints.
        
        This is used to compare poses in specific body regions (e.g., pelvis, arms) for
        tracking and analysis purposes.
        
        Args:
            other_pose: Another PoseRecord to compare against
            zone_keypoint_indices: List of keypoint indices to compare
            
        Returns:
            Dissimilarity score (lower is more similar), or 999.0 if comparison fails
        """
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
        """
        Convert PoseRecord to dictionary representation.
        
        Returns:
            Dictionary containing pose data
        """
        return {
            "id": self.id,
            "bbox": self.bbox.tolist(),
            "keypoints": self.keypoints.tolist()
        }
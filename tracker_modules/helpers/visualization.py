#!/usr/bin/env python3
"""
Visualization Helpers for Tracker Modules

Provides reusable visualization utilities for all trackers to eliminate duplication
and ensure consistent UI rendering across the application.

Author: VR Funscript AI Generator
"""

import numpy as np
import cv2
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Import official class colors from constants
try:
    from config.constants_colors import RGBColors
    CLASS_COLORS = RGBColors.CLASS_COLORS
except ImportError:
    # Fallback colors if import fails
    CLASS_COLORS = {
        "penis": (255, 0, 0),
        "glans": (0, 128, 0),
        "pussy": (0, 0, 255),
        "butt": (255, 180, 0),
        "anus": (128, 0, 128),
        "breast": (255, 165, 0),
        "navel": (0, 255, 255),
        "hand": (255, 0, 255),
        "face": (0, 255, 0),
        "foot": (165, 42, 42),
        "hips center": (0, 0, 0),
        "locked_penis": (0, 255, 255),
    }


@dataclass
class BoundingBox:
    """Standard bounding box format for visualization."""
    x1: float
    y1: float
    x2: float
    y2: float
    class_name: str
    confidence: float
    track_id: Optional[int] = None
    status: Optional[str] = None
    color_override: Optional[Tuple[int, int, int]] = None
    thickness_override: Optional[float] = None
    label_suffix: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with VideoDisplayUI."""
        return {
            "bbox": [self.x1, self.y1, self.x2, self.y2],
            "class_name": self.class_name,
            "confidence": self.confidence,
            "track_id": self.track_id,
            "status": self.status,
            # Custom fields for enhanced visualization
            "color_override": self.color_override,
            "thickness_override": self.thickness_override,
            "label_suffix": self.label_suffix
        }


@dataclass
class PoseKeypoints:
    """Standard pose keypoints format for visualization."""
    person_id: int
    keypoints: List[Tuple[float, float, float]]  # (x, y, confidence)
    is_primary: bool = False
    center: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary format compatible with VideoDisplayUI."""
        return {
            "id": self.person_id,
            "keypoints": self.keypoints,
            "is_primary": self.is_primary,
            "center": self.center
        }


class TrackerVisualizationHelper:
    """
    Centralized visualization helper for all tracker modules.
    Converts tracker-specific data to standardized formats for VideoDisplayUI.
    """
    
    # Standard class priorities for YOLO detections
    CLASS_PRIORITIES = {
        'penis': 10, 'locked_penis': 10,
        'pussy': 9, 'vagina': 9,
        'ass': 8, 'anus': 8, 'butt': 8,
        'hand': 5, 'finger': 5,
        'mouth': 4, 'face': 3,
        'breast': 2, 'body': 1,
        'person': 1
    }
    
    # Standard colors for different detection types (using official class colors where available)
    COLORS = {
        # Object detection classes - use official colors from constants
        'locked_penis': CLASS_COLORS.get('locked_penis', (0, 255, 255)),
        'penis': CLASS_COLORS.get('penis', (255, 0, 0)),
        'pussy': CLASS_COLORS.get('pussy', (0, 0, 255)),
        'butt': CLASS_COLORS.get('butt', (255, 180, 0)),
        'anus': CLASS_COLORS.get('anus', (128, 0, 128)),
        'breast': CLASS_COLORS.get('breast', (255, 165, 0)),
        'navel': CLASS_COLORS.get('navel', (0, 255, 255)),
        'hand': CLASS_COLORS.get('hand', (255, 0, 255)),
        'face': CLASS_COLORS.get('face', (0, 255, 0)),
        'foot': CLASS_COLORS.get('foot', (165, 42, 42)),
        'glans': CLASS_COLORS.get('glans', (0, 128, 0)),
        'hips center': CLASS_COLORS.get('hips center', (0, 0, 0)),
        
        # Tracker-specific visualization colors
        'active_contact': (0, 255, 0),    # Green
        'change_region': (0, 180, 180),   # Dim teal
        'flow_vector': (0, 255, 0),       # Green
        'oscillation': (255, 255, 0),     # Yellow
        'person_primary': (255, 255, 255), # White
        'person_contact': (0, 255, 255),   # Cyan
        'anatomical_face': CLASS_COLORS.get('face', (0, 255, 0)),
        'anatomical_breast': CLASS_COLORS.get('breast', (255, 165, 0)),
        'anatomical_navel': CLASS_COLORS.get('navel', (0, 255, 255)),  # Cyan
        'anatomical_hands': CLASS_COLORS.get('hand', (255, 0, 255)),   # Use hand color
        # Contact-aware visualization colors
        'contact_high_priority': (255, 100, 0),    # Bright orange - direct contact
        'contact_medium_priority': (255, 200, 0),  # Yellow-orange - nearby/indirect
        'contact_low_priority': (100, 255, 100),   # Light green - distant
        'no_contact': (120, 120, 120)              # Gray - no contact
    }
    
    @classmethod
    def prepare_overlay_data(cls, 
                            yolo_boxes: Optional[List[BoundingBox]] = None,
                            poses: Optional[List[PoseKeypoints]] = None,
                            change_regions: Optional[List[Dict]] = None,
                            flow_vectors: Optional[List[Dict]] = None,
                            motion_mode: Optional[str] = None,
                            locked_penis_box: Optional[Dict] = None,
                            active_track_id: Optional[int] = None,
                            contact_info: Optional[Dict] = None,
                            **kwargs) -> Dict[str, Any]:
        """
        Convert tracker data to standard overlay format for VideoDisplayUI.
        
        Args:
            yolo_boxes: List of BoundingBox objects
            poses: List of PoseKeypoints objects  
            change_regions: List of change region dictionaries
            flow_vectors: List of optical flow vectors
            motion_mode: Current motion mode string
            locked_penis_box: Special locked penis detection
            active_track_id: Currently active interaction track
            contact_info: Contact detection information
            **kwargs: Additional overlay data
            
        Returns:
            Dictionary in VideoDisplayUI overlay format
        """
        overlay_data = {}
        
        # Convert YOLO boxes
        if yolo_boxes:
            overlay_data["yolo_boxes"] = [box.to_dict() for box in yolo_boxes]
        
        # Convert poses
        if poses:
            overlay_data["poses"] = [pose.to_dict() for pose in poses]
            # Find dominant pose
            primary_poses = [p for p in poses if p.is_primary]
            if primary_poses:
                overlay_data["dominant_pose_id"] = primary_poses[0].person_id
        
        # Add motion mode
        if motion_mode:
            overlay_data["motion_mode"] = motion_mode
        
        # Add locked penis box
        if locked_penis_box:
            overlay_data["locked_penis_box"] = locked_penis_box
        
        # Add active track
        if active_track_id is not None:
            overlay_data["active_interaction_track_id"] = active_track_id
        
        # Add contact info
        if contact_info:
            overlay_data["contact_info"] = contact_info
        
        # Add change regions
        if change_regions:
            overlay_data["change_regions"] = change_regions
        
        # Add flow vectors
        if flow_vectors:
            overlay_data["flow_vectors"] = flow_vectors
        
        # Add any additional custom data
        overlay_data.update(kwargs)
        
        return overlay_data
    
    @classmethod
    def create_debug_window_data(cls, 
                                tracker_name: str,
                                metrics: Dict[str, Any],
                                show_graphs: bool = False,
                                graphs: Optional[Dict[str, Any]] = None,
                                progress_bars: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Create standardized debug window data for tracker metrics.
        
        Args:
            tracker_name: Display name for the tracker
            metrics: Dictionary of metric name to value
            show_graphs: Whether to include graph data
            graphs: Dictionary of graph name to graph data for plotting
            progress_bars: Dictionary of bar name to value (0.0-1.0) for progress bars
            
        Returns:
            Dictionary with debug window configuration
        """
        debug_data = {
            "title": f"{tracker_name} Debug",
            "metrics": metrics,
            "show_graphs": show_graphs,
            "window_flags": ["no_collapse", "always_auto_resize"],
            "position": None,  # Will be set by UI based on preferences
            "size": None       # Auto-size by default
        }
        
        # Add graphical elements if provided
        if show_graphs and graphs:
            debug_data["graphs"] = graphs
            
        if progress_bars:
            debug_data["progress_bars"] = progress_bars
            
        return debug_data
    
    @classmethod
    def convert_semantic_regions_to_boxes(cls, semantic_regions: List[Any]) -> List[BoundingBox]:
        """
        Convert semantic regions from hybrid tracker to standard bounding boxes.
        
        Args:
            semantic_regions: List of semantic region objects (can be dict or dataclass)
            
        Returns:
            List of BoundingBox objects
        """
        boxes = []
        for region in semantic_regions:
            # Handle both dict and object/dataclass formats
            if hasattr(region, '__dict__'):
                # It's an object/dataclass, access attributes directly
                class_name = getattr(region, 'class_name', 'unknown')
                priority = getattr(region, 'priority', 0)
                bbox = getattr(region, 'bbox', (0, 0, 0, 0))
                confidence = getattr(region, 'confidence', 0.0)
            else:
                # It's a dictionary
                class_name = region.get('class_name', 'unknown')
                priority = region.get('priority', 0)
                bbox = region.get('bbox', (0, 0, 0, 0))
                confidence = region.get('confidence', 0.0)
            
            # Use class-specific colors from constants, fallback to priority-based
            class_key = class_name.lower()
            if class_key in cls.COLORS:
                color = cls.COLORS[class_key]
            elif class_key in CLASS_COLORS:
                color = CLASS_COLORS[class_key]
            else:
                # Fallback: priority-based color for unknown classes
                priority_ratio = min(priority / 10.0, 1.0)
                color = (
                    int(255 * priority_ratio),         # Red
                    int(128 * priority_ratio),         # Green  
                    int(255 * (1 - priority_ratio))   # Blue
                )
            
            # bbox is already extracted above
            box = BoundingBox(
                x1=bbox[0], y1=bbox[1], x2=bbox[2], y2=bbox[3],
                class_name=class_name,
                confidence=confidence,
                track_id=None,  # SemanticRegion doesn't have track_id
                color_override=color,
                thickness_override=2.0 if priority > 5 else 1.0
            )
            boxes.append(box)
        
        return boxes
    
    @classmethod
    def convert_pose_data_to_keypoints(cls, pose_data: Dict) -> List[PoseKeypoints]:
        """
        Convert pose estimation data to standard keypoint format.
        
        Args:
            pose_data: Dictionary with pose estimation results
            
        Returns:
            List of PoseKeypoints objects
        """
        keypoints_list = []
        
        # Handle hybrid tracker's data format
        persons = pose_data.get('persons', {})
        if not persons:
            # Try hybrid tracker format
            all_persons = pose_data.get('all_persons', [])
            primary_person = pose_data.get('primary_person')
            
            if primary_person:
                persons['primary'] = primary_person
            
            # Add other persons from all_persons if available
            if isinstance(all_persons, list):
                # Convert list of person data to dict format
                for i, person_data in enumerate(all_persons):
                    person_key = f'person_{i}'
                    persons[person_key] = person_data
            elif isinstance(all_persons, dict):
                persons.update(all_persons)
        
        primary_id = pose_data.get('primary_person_id', 'primary')
        
        for person_id, person_data in persons.items():
            # Extract keypoints - handle hybrid tracker's named keypoint format
            keypoints = person_data.get('keypoints', [])
            
            # Convert to list of (x, y, confidence) tuples in COCO order
            keypoint_tuples = []
            
            # COCO keypoint order
            coco_keypoint_names = [
                'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
            ]
            
            # If keypoints is a dict (hybrid tracker named format)
            if isinstance(keypoints, dict):
                # Extract in COCO order
                for kp_name in coco_keypoint_names:
                    if kp_name in keypoints:
                        kp = keypoints[kp_name]
                        keypoint_tuples.append((
                            kp.get('x', 0),
                            kp.get('y', 0),
                            kp.get('confidence', 0)
                        ))
                    else:
                        # Missing keypoint - add with zero confidence
                        keypoint_tuples.append((0, 0, 0))
            
            # If keypoints is a list (standard format)
            elif isinstance(keypoints, list):
                for kp in keypoints:
                    if isinstance(kp, dict):
                        keypoint_tuples.append((
                            kp.get('x', 0),
                            kp.get('y', 0),
                            kp.get('confidence', 0)
                        ))
                    elif isinstance(kp, (list, tuple)) and len(kp) >= 2:
                        conf = kp[2] if len(kp) > 2 else 1.0
                        keypoint_tuples.append((kp[0], kp[1], conf))
                    elif isinstance(kp, (int, float)) and len(keypoints) % 3 == 0:
                        # Handle flat list format: [x1, y1, c1, x2, y2, c2, ...]
                        for i in range(0, len(keypoints), 3):
                            if i + 2 < len(keypoints):
                                keypoint_tuples.append((keypoints[i], keypoints[i+1], keypoints[i+2]))
                        break  # Only process once for flat format
            
            # Create PoseKeypoints object
            pose_kp = PoseKeypoints(
                person_id=person_id,
                keypoints=keypoint_tuples,
                is_primary=(person_id == primary_id),
                center=person_data.get('center')
            )
            keypoints_list.append(pose_kp)
        
        return keypoints_list
    
    @classmethod
    def create_activity_panel_data(cls, 
                                  anatomical_activities: Dict[str, Dict],
                                  signal_components: Dict[str, float],
                                  contact_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create data for the activity panel overlay.
        
        Args:
            anatomical_activities: Activity data for each anatomical region
            signal_components: Signal strength components
            contact_info: Contact detection information
            
        Returns:
            Dictionary with activity panel data
        """
        return {
            "anatomical_activities": anatomical_activities,
            "signal_components": signal_components,
            "contact_info": contact_info,
            "panel_position": "top_right",  # Can be customized
            "panel_size": (250, 200)        # Width, height
        }
    
    @classmethod
    def calculate_iou(cls, box1: Tuple[float, float, float, float], 
                     box2: Tuple[float, float, float, float]) -> float:
        """
        Calculate Intersection over Union for two bounding boxes.
        
        Args:
            box1: (x1, y1, x2, y2) coordinates
            box2: (x1, y1, x2, y2) coordinates
            
        Returns:
            IoU value between 0 and 1
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = area1 + area2 - intersection_area
        
        if union_area == 0:
            return 0.0
        
        return intersection_area / union_area
    
    @classmethod
    def get_class_priority(cls, class_name: str) -> int:
        """Get the priority value for a given class name."""
        return cls.CLASS_PRIORITIES.get(class_name.lower(), 0)
    
    @classmethod
    def get_standard_color(cls, color_key: str) -> Tuple[int, int, int]:
        """Get a standard color by key."""
        return cls.COLORS.get(color_key, (255, 255, 255))
    
    @classmethod
    def get_class_color(cls, class_name: str) -> Tuple[int, int, int]:
        """
        Get the official color for a detection class.
        
        Args:
            class_name: Name of the detection class
            
        Returns:
            RGB color tuple (0-255 range)
        """
        return CLASS_COLORS.get(class_name.lower(), (128, 128, 128))  # Default to gray
    
    @classmethod
    def analyze_skeleton_penis_contact(cls, poses: List[Dict], locked_penis_box: Optional[Dict]) -> Dict[str, str]:
        """
        Analyze which skeleton parts are in contact with the locked penis.
        
        Args:
            poses: List of pose dictionaries with keypoints
            locked_penis_box: Locked penis bounding box dict with 'bbox' key
            
        Returns:
            Dictionary mapping body part names to contact levels ('high', 'medium', 'low', 'none')
        """
        if not poses or not locked_penis_box or 'bbox' not in locked_penis_box:
            return {}
        
        penis_bbox = locked_penis_box['bbox']  # (x1, y1, x2, y2)
        contact_analysis = {}
        
        # COCO keypoint indices for body parts we care about
        body_parts = {
            'hands': [9, 10],  # left_wrist, right_wrist
            'arms': [7, 8, 9, 10],  # elbows + wrists
            'face': [0, 1, 2, 3, 4],  # nose, eyes, ears
            'torso': [5, 6, 11, 12],  # shoulders + hips
            'legs': [13, 14, 15, 16],  # knees + ankles
            'core': [11, 12]  # hips only
        }
        
        for pose in poses:
            if not pose.get('is_primary', False):
                continue  # Only analyze primary person
                
            keypoints = pose.get('keypoints', [])
            if len(keypoints) < 17:
                continue
            
            for part_name, keypoint_indices in body_parts.items():
                contact_level = cls._calculate_keypoint_penis_contact(
                    keypoints, keypoint_indices, penis_bbox
                )
                contact_analysis[part_name] = contact_level
                
            break  # Only analyze first (primary) person
        
        return contact_analysis
    
    @classmethod
    def _calculate_keypoint_penis_contact(cls, keypoints: List[Tuple], 
                                        keypoint_indices: List[int], 
                                        penis_bbox: Tuple[float, float, float, float]) -> str:
        """Calculate contact level between keypoints and penis bounding box."""
        if not keypoints or not keypoint_indices:
            return 'none'
        
        px1, py1, px2, py2 = penis_bbox
        penis_center_x = (px1 + px2) / 2
        penis_center_y = (py1 + py2) / 2
        penis_width = px2 - px1
        penis_height = py2 - py1
        penis_size = max(penis_width, penis_height)
        
        min_distance = float('inf')
        max_confidence = 0.0
        overlap_count = 0
        
        for idx in keypoint_indices:
            if idx >= len(keypoints):
                continue
                
            kp = keypoints[idx]
            if len(kp) < 3:
                continue
                
            x, y, conf = kp[0], kp[1], kp[2]
            
            # Skip low-confidence keypoints
            if conf < 0.5:
                continue
                
            max_confidence = max(max_confidence, conf)
            
            # Check if keypoint is inside penis bbox
            if px1 <= x <= px2 and py1 <= y <= py2:
                overlap_count += 1
                min_distance = 0  # Direct overlap
            else:
                # Calculate distance to penis center
                dist = ((x - penis_center_x)**2 + (y - penis_center_y)**2)**0.5
                min_distance = min(min_distance, dist)
        
        # No valid keypoints found
        if max_confidence == 0.0:
            return 'none'
        
        # Direct overlap - highest priority
        if overlap_count > 0:
            return 'high'
        
        # Close proximity - scale by penis size
        if min_distance < penis_size * 0.5:  # Within half penis size
            return 'high' 
        elif min_distance < penis_size * 1.0:  # Within penis size
            return 'medium'
        elif min_distance < penis_size * 2.0:  # Within double penis size
            return 'low'
        else:
            return 'none'
    
    @classmethod 
    def apply_contact_aware_colors(cls, change_regions: List[Dict], 
                                 contact_analysis: Dict[str, str],
                                 poses: List[Dict]) -> List[Dict]:
        """
        Apply contact-aware colors to change regions based on skeleton proximity to penis.
        
        Args:
            change_regions: List of change region dictionaries
            contact_analysis: Result from analyze_skeleton_penis_contact
            poses: List of pose dictionaries for spatial analysis
            
        Returns:
            Modified change regions with contact-aware colors
        """
        if not change_regions or not poses:
            return change_regions
        
        # Get primary person keypoints
        primary_keypoints = None
        for pose in poses:
            if pose.get('is_primary', False):
                primary_keypoints = pose.get('keypoints', [])
                break
                
        if not primary_keypoints or len(primary_keypoints) < 17:
            return change_regions  # No valid pose data
        
        # COCO body part keypoint groups
        body_part_keypoints = {
            'hands': [9, 10],
            'arms': [7, 8, 9, 10], 
            'face': [0, 1, 2, 3, 4],
            'torso': [5, 6, 11, 12],
            'legs': [13, 14, 15, 16],
            'core': [11, 12]
        }
        
        enhanced_regions = []
        
        for region in change_regions:
            if 'bbox' not in region:
                enhanced_regions.append(region)
                continue
                
            bbox = region['bbox']  # (x1, y1, x2, y2)
            region_center_x = (bbox[0] + bbox[2]) / 2
            region_center_y = (bbox[1] + bbox[3]) / 2
            
            # Find closest body part to this region
            closest_part = None
            min_distance = float('inf')
            
            for part_name, keypoint_indices in body_part_keypoints.items():
                # Calculate average position of this body part
                valid_kps = []
                for idx in keypoint_indices:
                    if idx < len(primary_keypoints):
                        kp = primary_keypoints[idx]
                        if len(kp) >= 3 and kp[2] > 0.5:  # Good confidence
                            valid_kps.append((kp[0], kp[1]))
                
                if valid_kps:
                    import numpy as np
                    part_center = np.mean(valid_kps, axis=0)
                    dist = ((region_center_x - part_center[0])**2 + 
                           (region_center_y - part_center[1])**2)**0.5
                    
                    if dist < min_distance:
                        min_distance = dist
                        closest_part = part_name
            
            # Apply contact-aware color based on closest body part's contact level
            enhanced_region = region.copy()
            
            if closest_part and closest_part in contact_analysis:
                contact_level = contact_analysis[closest_part]
                
                if contact_level == 'high':
                    enhanced_region['contact_color'] = cls.COLORS['contact_high_priority']
                    enhanced_region['contact_priority'] = 3
                elif contact_level == 'medium':
                    enhanced_region['contact_color'] = cls.COLORS['contact_medium_priority'] 
                    enhanced_region['contact_priority'] = 2
                elif contact_level == 'low':
                    enhanced_region['contact_color'] = cls.COLORS['contact_low_priority']
                    enhanced_region['contact_priority'] = 1
                else:
                    enhanced_region['contact_color'] = cls.COLORS['no_contact']
                    enhanced_region['contact_priority'] = 0
                    
                enhanced_region['closest_body_part'] = closest_part
                enhanced_region['contact_level'] = contact_level
            else:
                enhanced_region['contact_color'] = cls.COLORS['no_contact']
                enhanced_region['contact_priority'] = 0
                enhanced_region['closest_body_part'] = 'unknown'
                enhanced_region['contact_level'] = 'none'
            
            enhanced_regions.append(enhanced_region)
        
        return enhanced_regions
    
    @classmethod
    def draw_filled_rectangle(cls, frame: np.ndarray, bbox: tuple, color: tuple):
        """
        Draw a filled rectangle with transparency support.
        
        Args:
            frame: Frame to draw on
            bbox: (x1, y1, x2, y2) coordinates 
            color: (R, G, B, A) color tuple
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            
            if len(color) == 4:  # RGBA
                r, g, b, alpha = color
                alpha_norm = alpha / 255.0
                
                # Create overlay for transparency
                overlay = frame.copy()
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (b, g, r), -1)  # BGR for OpenCV
                
                # Blend with original
                cv2.addWeighted(overlay, alpha_norm, frame, 1 - alpha_norm, 0, frame)
            else:  # RGB
                r, g, b = color
                cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), -1)  # BGR for OpenCV
                
        except Exception as e:
            print(f"Error drawing filled rectangle: {e}")
    
    @classmethod
    def draw_rectangle(cls, frame: np.ndarray, bbox: tuple, color: tuple, thickness: int = 2):
        """
        Draw a rectangle outline.
        
        Args:
            frame: Frame to draw on
            bbox: (x1, y1, x2, y2) coordinates
            color: (R, G, B) color tuple
            thickness: Line thickness
        """
        try:
            x1, y1, x2, y2 = map(int, bbox)
            r, g, b = color
            cv2.rectangle(frame, (x1, y1), (x2, y2), (b, g, r), thickness)  # BGR for OpenCV
        except Exception as e:
            print(f"Error drawing rectangle: {e}")
"""
Segment data structures for Stage 2 processing.

This module contains the BaseSegment and Segment classes used for
grouping and analyzing sequences of frames with similar characteristics.
"""

import logging
from typing import List, Dict, Any, Optional, TYPE_CHECKING
from config import constants

# Use TYPE_CHECKING to avoid circular imports
if TYPE_CHECKING:
    from .frame_objects import FrameObject

# Import the debug logging function
def _debug_log(message: str):
    """Centralized debug logging for Stage 2."""
    logger = logging.getLogger(__name__)
    logger.debug(f"[S2] {message}")


class BaseSegment:
    """
    Base class for video segments representing sequences of related frames.
    
    This class provides common functionality for managing frame sequences
    and calculating segment properties like duration and occlusion information.
    """
    
    _id_counter = 0

    def __init__(self, start_frame_id: int, end_frame_id: Optional[int] = None):
        """
        Initialize a BaseSegment.
        
        Args:
            start_frame_id: First frame ID in the segment
            end_frame_id: Last frame ID in the segment (defaults to start_frame_id)
        """
        self.id = BaseSegment._id_counter
        BaseSegment._id_counter += 1
        self.start_frame_id = start_frame_id
        self.end_frame_id = end_frame_id if end_frame_id is not None else start_frame_id
        self.frames: List['FrameObject'] = []
        self.duration = 0
        self.update_duration()

    def update_duration(self):
        """Update the segment duration based on frame range."""
        if self.frames:
            self.duration = self.end_frame_id - self.start_frame_id + 1
        elif self.end_frame_id >= self.start_frame_id:
            self.duration = self.end_frame_id - self.start_frame_id + 1
        else:
            self.duration = 0

    def add_frame(self, frame: 'FrameObject'):
        """
        Add a frame to this segment, maintaining frame order.
        
        Args:
            frame: FrameObject to add to the segment
        """
        if not self.frames or frame.frame_id > self.frames[-1].frame_id:
            self.frames.append(frame)
            if frame.frame_id > self.end_frame_id:
                self.end_frame_id = frame.frame_id
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
            if not inserted:
                self.frames.append(frame)
            self.update_duration()

    def get_occlusion_info(self, box_attribute_name: str = "tracked_box") -> List[Dict[str, Any]]:
        """
        Analyze occlusion patterns within this segment.
        
        This method identifies sequences of frames where the tracked object
        was occluded or lost, useful for understanding tracking quality.
        
        Args:
            box_attribute_name: Name of the box attribute to analyze for occlusions
            
        Returns:
            List of occlusion blocks with start/end frames and status
        """
        occlusions = []
        if not self.frames:
            return occlusions
            
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


class Segment(BaseSegment):
    """
    Segment for Stage 2 processing.
    
    This class extends BaseSegment with functionality for
    position classification and funscript generation.
    """

    def __init__(self, start_frame_id: int, end_frame_id: int, major_position: str):
        """
        Initialize a Segment.
        
        Args:
            start_frame_id: First frame ID in the segment
            end_frame_id: Last frame ID in the segment
            major_position: Primary position/activity for this segment
        """
        super().__init__(start_frame_id, end_frame_id)
        
        # DEBUG: Log constructor parameters
        if hasattr(Segment, '_instance_count'):
            Segment._instance_count += 1
        else:
            Segment._instance_count = 1
        
        if Segment._instance_count <= 5:
            _debug_log(f"Segment.__init__ #{Segment._instance_count}: major_position='{major_position}' (type: {type(major_position)})")
        
        self.major_position = major_position
        
        # DEBUG: Verify what was stored
        if Segment._instance_count <= 5:
            _debug_log(f"Segment.__init__ #{Segment._instance_count}: stored self.major_position='{self.major_position}' (type: {type(self.major_position)})")
        
        self.segment_frame_objects: List['FrameObject'] = []

        # List to hold different analysis sub-segments
        # Each entry is a dict: {'start': int, 'end': int, 'mode': str, 'roi': tuple}
        self.sub_segments = []

    @property
    def position_long_name(self) -> str:
        """
        Get the canonical long name for this segment's position.
        
        Returns:
            Long name of the position
        """
        return self.major_position

    @property
    def position_short_name(self) -> str:
        """
        Map the segment's long name to a standardized short code.
        
        Returns:
            Short code for the position (e.g., 'BJ' for 'Blow Job')
        """
        try:
            for short_code, info in constants.POSITION_INFO_MAPPING.items():
                if info.get("long_name") == self.major_position:
                    return short_code
        except Exception:
            pass
        # Fallback: if unknown mapping, return the long name
        return self.major_position

    def add_sub_segment(self, start_frame, end_frame, mode, roi=None):
        """
        Add a sub-segment for specialized analysis.
        
        Args:
            start_frame: Start frame for the sub-segment
            end_frame: End frame for the sub-segment
            mode: Analysis mode ('YOLO' or 'OPTICAL_FLOW')
            roi: ROI tuple (x,y,w,h) for optical flow processing
        """
        self.sub_segments.append({
            'start': start_frame,
            'end': end_frame,
            'mode': mode,
            'roi_hint': roi
        })

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert Segment to dictionary representation.
        
        This method creates a comprehensive dictionary representation of the segment
        suitable for serialization and analysis.
        
        Returns:
            Dictionary containing segment data
        """
        occlusion_info = []  # Placeholder

        # 1. position_long_name_val is straightforward:
        position_long_name_val = self.major_position

        # DEBUG: Log to_dict conversion
        if hasattr(Segment, '_to_dict_count'):
            Segment._to_dict_count += 1
        else:
            Segment._to_dict_count = 1
        
        if Segment._to_dict_count <= 5:
            _debug_log(f"Segment.to_dict #{Segment._to_dict_count}: self.major_position='{self.major_position}' (type: {type(self.major_position)})")

        # 2. To find position_short_name_val, we need to search the dictionary:
        position_short_name_key_val = "NR"  # Default if not found

        for key, info in constants.POSITION_INFO_MAPPING.items():
            if info["long_name"] == self.major_position:
                position_short_name_key_val = key
                if Segment._to_dict_count <= 5:
                    _debug_log(f"Segment.to_dict #{Segment._to_dict_count}: FOUND match - key='{key}', long_name='{info['long_name']}'")
                break
        else:
            if Segment._to_dict_count <= 5:
                _debug_log(f"Segment.to_dict #{Segment._to_dict_count}: NO MATCH found for '{self.major_position}'")
                _debug_log(f"Segment.to_dict #{Segment._to_dict_count}: Available long_names: {[info['long_name'] for info in constants.POSITION_INFO_MAPPING.values()]}")


        raw_range_val_ud = 0
        raw_range_offset_ud = 0
        if self.segment_frame_objects:
            # Filter out None values before calculating min/max
            distances_in_segment = [fo.funscript_distance for fo in self.segment_frame_objects 
                                  if fo.funscript_distance is not None]
            if distances_in_segment:
                min_d, max_d = min(distances_in_segment), max(distances_in_segment)
                raw_range_val_ud = max_d - min_d

        return {
            'start_frame_id': self.start_frame_id, 
            'end_frame_id': self.end_frame_id,
            'class_name': self.major_position,  # Using major_position as class_name
            'position_long_name': position_long_name_val,
            'position_short_name': position_short_name_key_val,
            'segment_type': "Segment", 
            'duration': self.duration,
            'occlusions': occlusion_info,  # Placeholder
            'raw_range_val_ud': raw_range_val_ud,
            'raw_range_offset_ud': raw_range_offset_ud,
            'raw_range_val_lr': 0  # Stage 2 logic is primarily single axis
        }

    def __repr__(self):
        return (f"Segment(id={self.id}, frames {self.start_frame_id}-{self.end_frame_id}, "
                f"pos='{self.major_position}', duration={self.duration})")
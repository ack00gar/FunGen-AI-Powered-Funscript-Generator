import uuid
import math
from typing import Optional, Tuple, List, Dict, Any
import random
from config import constants
from config.element_group_colors import SegmentColors

class VideoSegment:
    """
    Represents a video segment/chapter with timing, classification, and visual properties.
    """
    
    _POSITION_COLOR_MAP = {
        "BJ": SegmentColors.BJ,
        "HJ": SegmentColors.HJ,
        "NR": SegmentColors.NR,
        "CG/Miss.": SegmentColors.CG_MISS,
        "R.CG/Dog.": SegmentColors.REV_CG_DOG,
        "FootJ": SegmentColors.FOOTJ,
        "BoobJ": SegmentColors.BOOBJ,
        "C-Up": SegmentColors.CLOSEUP,
    }
    
    def __init__(self, start_frame_id: int, end_frame_id: int, class_id: Optional[int], 
                 class_name: str, segment_type: str, position_short_name: str,
                 position_long_name: str, duration: int = 0, occlusions: int = 0, 
                 color: Optional[Tuple[float, float, float, float]] = None, source: str = "manual",
                 user_roi_fixed: Optional[Tuple[int, int, int, int]] = None,
                 user_roi_initial_point_relative: Optional[Tuple[float, float]] = None,
                 refined_track_id: Optional[int] = None):
        """Initialize a VideoSegment with the given parameters."""
        self.start_frame_id = int(start_frame_id)
        self.end_frame_id = int(end_frame_id)
        self.class_id = class_id  # Can be int or None
        self.class_name = str(class_name)
        self.segment_type = str(segment_type)
        self.position_short_name = str(position_short_name)
        self.position_long_name = str(position_long_name)
        self.duration = duration  # In frames or seconds, clarify based on usage
        self.occlusions = occlusions
        self.source = source
        self.unique_id = f"segment_{uuid.uuid4()}"  # Always generate a new one initially

        self.user_roi_fixed = user_roi_fixed
        self.user_roi_initial_point_relative = user_roi_initial_point_relative
        self.refined_track_id = refined_track_id

        # Set color using centralized mapping
        self.color = tuple(color) if color else self._get_segment_color(self.position_short_name)

    @classmethod
    def _get_segment_color(cls, position_short_name: str) -> Tuple[float, float, float, float]:
        """Get the appropriate color for a segment based on position_short_name."""
        return cls._POSITION_COLOR_MAP.get(position_short_name, SegmentColors.DEFAULT)

    # ==================== TIMING CONVERSION METHODS ====================
    @staticmethod
    def _frames_to_timecode(frames: int, fps: float) -> str:
        """Convert frame number to timecode string (HH:MM:SS.mmm)."""
        if fps <= 0: return "00:00:00.000"
        if frames < 0: frames = 0  # Ensure frames are not negative for timecode calc

        # Ensure total_seconds is non-negative before further calculations
        total_seconds_float = max(0.0, frames / fps)
        hours = math.floor(total_seconds_float / 3600)
        minutes = math.floor((total_seconds_float % 3600) / 60)
        seconds = math.floor(total_seconds_float % 60)
        milliseconds = math.floor((total_seconds_float - math.floor(total_seconds_float)) * 1000)

        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    @staticmethod
    def _timecode_to_frames(timecode_str: str, fps: float) -> int:
        """Convert timecode string (HH:MM:SS.mmm) to frame number."""
        if fps <= 0: return 0

        try:
            time_parts = timecode_str.split(':')
            if len(time_parts) != 3: raise ValueError("Timecode must be HH:MM:SS.mmm")

            hours = int(time_parts[0])
            minutes = int(time_parts[1])

            sec_ms_parts = time_parts[2].split('.')
            if len(sec_ms_parts) not in [1, 2]: raise ValueError("Seconds part must be SS or SS.mmm")

            seconds = int(sec_ms_parts[0])
            milliseconds = int(sec_ms_parts[1]) if len(sec_ms_parts) > 1 else 0

            total_seconds = hours * 3600 + minutes * 60 + seconds + milliseconds / 1000.0
            return int(round(total_seconds * fps))
        except (ValueError, IndexError):
            return 0  # Return 0 for any parsing errors

    @staticmethod
    def ms_to_frame_idx(ms: int, total_frames: int, fps: float) -> int:
        """Convert milliseconds to frame index."""
        time_in_seconds = ms / 1000
        frame_idx = int(time_in_seconds * fps)
        return min(frame_idx, total_frames - 1)

    def to_dict(self) -> Dict[str, Any]:
        """Convert segment to dictionary for project saving."""
        return {
            'start_frame_id': self.start_frame_id,
            'end_frame_id': self.end_frame_id,
            'class_id': self.class_id,
            'class_name': self.class_name,
            'segment_type': self.segment_type,
            'position_short_name': self.position_short_name,
            'position_long_name': self.position_long_name,
            'duration': self.duration,
            'occlusions': self.occlusions,
            'source': self.source,
            'color': list(self.color) if isinstance(self.color, tuple) else self.color,
            'unique_id': self.unique_id,
            'user_roi_fixed': self.user_roi_fixed,
            'user_roi_initial_point_relative': self.user_roi_initial_point_relative,
            'refined_track_id': self.refined_track_id,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'VideoSegment':
        """Create segment from dictionary for project loading."""
        # Validate and correct position names if they're corrupted (e.g., "0.0")
        position_short_name = data.get('position_short_name', data.get('segment_type', 'default'))
        position_long_name = data.get('position_long_name', data.get('class_name', 'Unknown'))
        
        # Fix corrupted position names by using class_id as reference
        if position_short_name == "0.0" or position_long_name == "0.0":
            class_id = data.get('class_id', '')
            # Try to map from class_id to proper position names
            if class_id == "Cowgirl / Missionary":
                position_short_name = "CG/Miss."
                position_long_name = "Cowgirl / Missionary"
            elif class_id in constants.POSITION_INFO_MAPPING:
                position_info = constants.POSITION_INFO_MAPPING[class_id]
                position_short_name = position_info.get("short_name", class_id)
                position_long_name = position_info.get("long_name", class_id)
        
        segment = cls(
            start_frame_id=data.get('start_frame_id', 0),
            end_frame_id=data.get('end_frame_id', 0),
            class_id=data.get('class_id'),  # Allow None
            class_name=data.get('class_name', 'Unknown'),
            segment_type=data.get('segment_type', 'default'),
            position_short_name=position_short_name,
            position_long_name=position_long_name,
            duration=data.get('duration', 0),
            occlusions=data.get('occlusions', []),
            source=data.get('source', 'project_load')
        )
        # Restore color, ensuring it's a tuple
        color_data = data.get('color')
        if color_data is not None:
            segment.color = tuple(color_data) if isinstance(color_data, list) else color_data
            
        # Restore unique_id, or it's already generated by constructor
        segment.unique_id = data.get('unique_id', segment.unique_id)
        segment.user_roi_fixed = data.get('user_roi_fixed')
        segment.user_roi_initial_point_relative = data.get('user_roi_initial_point_relative')
        segment.refined_track_id = data.get('refined_track_id')

        return segment

    # ==================== FUNSCRIPT CONVERSION METHODS ====================
    def to_funscript_chapter_dict(self, fps: float) -> Dict[str, str]:
        """Converts the segment to the Funscript chapter metadata format."""
        if fps <= 0:
            return {
                "name": self.position_long_name,
                "startTime": "00:00:00.000",
                "endTime": "00:00:00.000"
            }
        return {
            "name": self.position_long_name,
            "startTime": self._frames_to_timecode(self.start_frame_id, fps),
            "endTime": self._frames_to_timecode(self.end_frame_id, fps)
        }

    @classmethod
    def from_funscript_chapter_dict(cls, data: Dict[str, str], fps: float) -> 'VideoSegment':
        """Create segment from Funscript chapter metadata format."""
        long_name = data.get("name", "Unnamed Chapter")
        startTime_str = data.get("startTime", "00:00:00.000")
        endTime_str = data.get("endTime", "00:00:00.000")

        REVERSE_POSITION_MAPPING = {
            info["long_name"]: info["short_name"]
            for info in constants.POSITION_INFO_MAPPING.values()
        }

        short_name = REVERSE_POSITION_MAPPING.get(long_name)

        LONG_NAME_TO_KEY = {
            info["long_name"]: key
            for key, info in constants.POSITION_INFO_MAPPING.items()
        }
        body_part = LONG_NAME_TO_KEY.get(long_name)

        start_frame = cls._timecode_to_frames(startTime_str, fps)
        end_frame = cls._timecode_to_frames(endTime_str, fps)

        return cls(
            start_frame_id=start_frame,
            end_frame_id=max(start_frame, end_frame),
            class_id=None,  # Not available in this format
            class_name=body_part,
            segment_type="SexAct",  # Default
            position_short_name=short_name,
            position_long_name=long_name,  # Use name as long name
            source="funscript_import"
        )

    # ==================== VALIDATION METHODS ====================
    @staticmethod
    def is_valid_dict(data_dict: Dict[str, Any]) -> bool:
        """Validate if dictionary contains required keys for segment creation."""
        if not isinstance(data_dict, dict):
            return False
            
        required_keys = ["start_frame_id", "end_frame_id", "class_name"]
        return all(key in data_dict for key in required_keys)

    # ==================== COLOR ASSIGNMENT METHODS ====================
    @classmethod
    def assign_colors_to_segments(cls, segments: List['VideoSegment']) -> None:
        """
        Assigns colors to a list of VideoSegment objects based on their position_short_name.
        Should be called whenever chapters are created, imported, or edited.
        """
        for seg in segments:
            seg.color = cls._get_segment_color(seg.position_short_name)

    @classmethod
    def assign_random_colors_to_segments(cls, segments: List['VideoSegment']) -> None:
        """
        Assign random colors to segments from the available palette,
        ensuring no segment gets the same color as the previous two.
        """
        palette = list(cls._POSITION_COLOR_MAP.values())
        last_colors = []
        for seg in segments:
            available_colors = [c for c in palette if c not in last_colors[-2:]] if len(palette) > 2 else palette
            color = random.choice(available_colors)
            seg.color = color
            last_colors.append(color)

    # ==================== UTILITY METHODS ====================
    def __repr__(self) -> str:
        """String representation of the segment."""
        return (f"<VideoSegment id:{self.unique_id} frames:{self.start_frame_id}-{self.end_frame_id} "
                f"name:'{self.class_name}' type:'{self.segment_type}' pos:'{self.position_short_name}'>")

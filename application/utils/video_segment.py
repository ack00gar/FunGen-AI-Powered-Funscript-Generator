import uuid
import math
from typing import Optional, Tuple
import random

from config import constants


class VideoSegment:
    def __init__(self, start_frame_id, end_frame_id, class_id, class_name, segment_type, position_short_name,
                 position_long_name, duration=0, occlusions=0, color=None, source="manual",
                 user_roi_fixed: Optional[Tuple[int, int, int, int]] = None,
                 user_roi_initial_point_relative: Optional[Tuple[float, float]] = None,
                 refined_track_id: Optional[int] = None):
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
        # Ensure unique_id is robust, especially if segments are created without explicit IDs
        self.unique_id = f"segment_{uuid.uuid4()}"  # Always generate a new one initially

        self.user_roi_fixed = user_roi_fixed
        self.user_roi_initial_point_relative = user_roi_initial_point_relative

        self.refined_track_id = refined_track_id

        default_colors = {
            "BJ": (0.9, 0.4, 0.4, 0.8),
            "HJ": (0.4, 0.9, 0.4, 0.8),
            "NR": (0.6, 0.6, 0.6, 0.7),
            "CG/Miss.": (0.4, 0.4, 0.9, 0.8),
            "Rev.CG/Doggy": (0.9, 0.6, 0.2, 0.8),
            "FootJ": (0.9, 0.9, 0.3, 0.8),
            "BoobJ": (0.9, 0.3, 0.6, 0.8),
            "CloseUp": (0.7, 0.7, 0.9, 0.8),
            "default": (0.5, 0.5, 0.5, 0.7)
        }

        if color:
            self.color = tuple(color)  # Ensure it's a tuple
        else:
            self.color = constants.DEFAULT_CHAPTER_COLORS.get(self.position_short_name, constants.DEFAULT_CHAPTER_COLORS["default"])

    @staticmethod
    def _frames_to_timecode(frames: int, fps: float) -> str:
        if fps <= 0: return "00:00:00.000"
        if frames < 0: frames = 0  # Ensure frames are not negative for timecode calc
        total_seconds_float = frames / fps

        # Ensure total_seconds is non-negative before further calculations
        total_seconds_float = max(0.0, total_seconds_float)

        hours = math.floor(total_seconds_float / 3600)
        minutes = math.floor((total_seconds_float % 3600) / 60)
        seconds = math.floor(total_seconds_float % 60)
        milliseconds = math.floor((total_seconds_float - math.floor(total_seconds_float)) * 1000)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}.{milliseconds:03d}"

    @staticmethod
    def _timecode_to_frames(timecode_str: str, fps: float) -> int:
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
        except ValueError as e:
            # Consider logging this error
            # print(f"Error parsing timecode '{timecode_str}': {e}")
            return 0  # Return 0 or raise error
        except Exception:  # Catch any other parsing errors
            # print(f"Generic error parsing timecode '{timecode_str}'")
            return 0

    @staticmethod
    def ms_to_frame_idx(ms: int, total_frames: int, fps: float) -> int:
        time_in_seconds = ms / 1000
        frame_idx = int(time_in_seconds * fps)
        return min(frame_idx, total_frames - 1)

    def to_dict(self):  # For project saving (full data)
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
    def from_dict(cls, data):  # For project loading (full data)
        # Create instance with basic data
        segment = cls(
            start_frame_id=data.get('start_frame_id', 0),
            end_frame_id=data.get('end_frame_id', 0),
            class_id=data.get('class_id'),  # Allow None
            class_name=data.get('class_name', 'Unknown'),
            segment_type=data.get('segment_type', 'default'),
            position_short_name=data.get('position_short_name', data.get('segment_type', 'default')),  # Fallback
            position_long_name=data.get('position_long_name', data.get('class_name', 'Unknown')),  # Fallback
            duration=data.get('duration', 0),
            occlusions=data.get('occlusions', []),
            source=data.get('source', 'project_load')
            # Color will be set/re-calculated based on other props or loaded
        )
        # Restore color, ensuring it's a tuple
        color_data = data.get('color')
        if color_data is not None:
            segment.color = tuple(color_data) if isinstance(color_data, list) else color_data
        # else, the constructor's default color logic based on type/name would have applied.

        # Restore unique_id, or it's already generated by constructor
        segment.unique_id = data.get('unique_id', segment.unique_id)

        segment.user_roi_fixed = data.get('user_roi_fixed')
        segment.user_roi_initial_point_relative = data.get('user_roi_initial_point_relative')

        segment.refined_track_id = data.get('refined_track_id')

        return segment

    def to_funscript_chapter_dict(self, fps: float) -> dict:
        """Converts the segment to the Funscript chapter metadata format."""
        if fps <= 0:
            # Handle cases where FPS is not available, perhaps by returning frame IDs or raising an error
            # For now, returning with a default timecode if FPS is invalid.
            # A better approach might be for the caller (ApplicationLogic) to check FPS before calling.
            return {
                "name": self.position_long_name,
                "startTime": "00:00:00.000",
                "endTime": "00:00:00.000"
            }
        return {
            "name": self.position_long_name,
            "startTime": VideoSegment._frames_to_timecode(self.start_frame_id, fps),
            "endTime": VideoSegment._frames_to_timecode(self.end_frame_id, fps)
        }

    @classmethod
    def from_funscript_chapter_dict(cls, data: dict, fps: float):
        """Creates a VideoSegment from the Funscript chapter metadata format."""
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

        start_frame = VideoSegment._timecode_to_frames(startTime_str, fps)
        end_frame = VideoSegment._timecode_to_frames(endTime_str, fps)

        if fps <= 0:  # If FPS is invalid, frames might be 0, leading to start_frame == end_frame
            if startTime_str != endTime_str:  # Only if times were meant to be different
                # This indicates an issue, perhaps log a warning.
                # End frame could be set to start_frame + some_default_duration_in_frames if desired.
                pass

        # Create a VideoSegment with defaults for fields not present in this simple format
        segment = cls(
            start_frame_id=start_frame,
            end_frame_id=max(start_frame, end_frame),
            class_id=None,  # Not available in this format
            class_name=body_part,
            segment_type="SexAct",  # Default
            position_short_name=short_name,
            position_long_name=long_name,  # Use name as long name
            source="funscript_import"
            # Color will be set by the constructor's default logic based on class_name/type
        )
        # The unique_id is already generated by the constructor.
        return segment

    @staticmethod
    def is_valid_dict(data_dict: dict) -> bool:  # This is for project file loading
        if not isinstance(data_dict, dict):
            return False
        required_keys = [  # These are for the richer project file format
            "start_frame_id", "end_frame_id", "class_name"
        ]
        # segment_type, position_long_name, position_short_name are good to have but might have defaults
        return all(key in data_dict for key in required_keys)

    def __repr__(self):  #
        return (f"<VideoSegment id:{self.unique_id} frames:{self.start_frame_id}-{self.end_frame_id} "  #
                f"name:'{self.class_name}' type:'{self.segment_type}' pos:'{self.position_short_name}'>")


    ###########################################################################################
    # TEMPORARY: This method assigns random colors to chapters for visual distinction when all chapters have the same position_short_name 'NR'.
    # Remove this logic once position detection is implemented for "Scene Detection without AI analysis"
    
    # AI Analysis = fixed colors per position
    # Scene Detection without AI analysis = returns all 'NR', will then assign random colors
    ###########################################################################################
    @staticmethod
    def assign_colors_to_segments(segments):
        """
        Assigns colors to a list of VideoSegment objects based on their position_short_name.
        Should be called whenever chapters are created, imported, or edited.
        """
        for seg in segments:
            seg.color = constants.DEFAULT_CHAPTER_COLORS.get(seg.position_short_name, constants.DEFAULT_CHAPTER_COLORS["default"])

    @staticmethod
    def assign_random_colors_to_segments(segments):
        """
        Assigns random colors to chapters from the DEFAULT_CHAPTER_COLORS palette (excluding 'default'),
        ensuring no chapter gets the same color as the previous two.
        """
        palette = [c for k, c in constants.DEFAULT_CHAPTER_COLORS.items() if k != "default"]
        last_colors = []
        for seg in segments:
            available_colors = [c for c in palette if c not in last_colors[-2:]] if len(palette) > 2 else palette
            color = random.choice(available_colors)
            seg.color = color
            last_colors.append(color)

    # END OF TEMPORARY COLOR SELECTION LOGIC
    ###########################################################################################


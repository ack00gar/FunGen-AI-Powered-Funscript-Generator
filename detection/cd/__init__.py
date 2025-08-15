"""
Detection CD package initialization.
"""

from .stage_1_cd import FFmpegEncoder, Stage1QueueMonitor
from .stage_2_cd import BaseSegment, BoxRecord, PoseRecord
from .data_structures import FrameObject, LockedPenisState, Segment
from .stage_2_sqlite_storage import Stage2SQLiteStorage 
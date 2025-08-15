"""
Data structures for Stage 2 Computer Detection processing.

This module contains all the core data structures used in Stage 2 processing,
extracted from the monolithic stage_2_cd.py for better maintainability.
"""

from .frame_objects import FrameObject, LockedPenisState
from .box_records import BoxRecord, PoseRecord
from .segments import BaseSegment, Segment

__all__ = [
    'FrameObject',
    'LockedPenisState', 
    'BoxRecord',
    'PoseRecord',
    'BaseSegment',
    'Segment',
    'AppStateContainer'
]
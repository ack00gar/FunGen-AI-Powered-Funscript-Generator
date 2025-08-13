"""
Data structures for Stage 2 Computer Detection processing.

This module contains all the core data structures used in Stage 2 processing,
extracted from the monolithic stage_2_cd.py for better maintainability.
"""

from .frame_objects import FrameObject, ATRLockedPenisState
from .box_records import BoxRecord, PoseRecord
from .segments import BaseSegment, ATRSegment
from .state_container import AppStateContainer

__all__ = [
    'FrameObject',
    'ATRLockedPenisState', 
    'BoxRecord',
    'PoseRecord',
    'BaseSegment',
    'ATRSegment',
    'AppStateContainer'
]
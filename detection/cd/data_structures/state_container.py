"""
Application state container for Stage 2 processing.

This module contains the AppStateContainer class which manages the overall
state and data for Stage 2 processing, including frame objects, segments,
and SQLite storage integration.
"""

import os
import logging
from typing import List, Dict, Any, Optional
from config import constants
from .frame_objects import FrameObject
from .segments import ATRSegment


class AppStateContainer:
    """
    Container class for managing all Stage 2 processing state and data.
    
    This class serves as the central data repository for Stage 2 processing,
    managing frame objects, segments, funscript data, and optional SQLite storage.
    """

    def __init__(self, video_info: Dict, yolo_input_size: int, vr_filter: bool, all_frames_raw_data: list,
                 logger: Optional[logging.Logger], discarded_classes_runtime_arg: Optional[List[str]] = None,
                 scripting_range_active: bool = False, scripting_range_start_frame: Optional[int] = None,
                 scripting_range_end_frame: Optional[int] = None, is_ranged_data_source: bool = False,
                 use_sqlite: bool = True):
        """
        Initialize the AppStateContainer.
        
        Args:
            video_info: Dictionary containing video metadata (fps, dimensions, etc.)
            yolo_input_size: Size of YOLO input processing (typically 640)
            vr_filter: Whether to apply VR-specific filtering
            all_frames_raw_data: List of raw frame detection data
            logger: Logger instance for debug/info messages
            discarded_classes_runtime_arg: Additional classes to discard during processing
            scripting_range_active: Whether processing a specific frame range
            scripting_range_start_frame: Start frame for range processing
            scripting_range_end_frame: End frame for range processing
            is_ranged_data_source: Whether the data source represents a frame range
            use_sqlite: Whether to use SQLite for data storage
        """
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

        # Reset FrameObject counter for consistent IDs
        FrameObject._id_counter = 0

        # Set up class filtering
        self.effective_discard_classes = set(constants.CLASSES_TO_DISCARD_BY_DEFAULT)
        if discarded_classes_runtime_arg:
            self.effective_discard_classes.update(discarded_classes_runtime_arg)
            
        # Create frame objects from raw data
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

        # Initialize processing results storage
        self.atr_segments: List[ATRSegment] = []
        self.funscript_frames: List[int] = []
        self.funscript_distances: List[int] = []
        self.funscript_distances_lr: List[int] = []

    def store_frames_to_sqlite(self, batch_size: int = 2000):
        """
        Store frame objects to SQLite database in batches.
        
        This method efficiently stores large numbers of frame objects to SQLite
        using batch processing to optimize performance.
        
        Args:
            batch_size: Number of frames to process in each batch
        """
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
        """
        Clear frame objects from memory after storing to SQLite.
        
        This method helps manage memory usage by clearing frame data from memory
        after it has been persisted to SQLite storage.
        """
        if self.use_sqlite and self.sqlite_storage:
            original_count = len(self.frames)
            self.frames.clear()
            if self.logger:
                self.logger.info(f"Cleared {original_count} frame objects from memory")

    def cleanup_sqlite(self, remove_db_file: bool = True):
        """
        Clean up SQLite resources.
        
        Args:
            remove_db_file: Whether to remove the database file from disk
        """
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
        """
        Cleanup when object is destroyed - but preserve DB file for Stage 3.
        
        This destructor ensures SQLite connections are properly closed while
        preserving the database file for use by Stage 3 processing.
        """
        # Don't remove the database file during garbage collection
        # Stage 3 will clean it up when it's done
        if self.sqlite_storage:
            self.sqlite_storage.close()
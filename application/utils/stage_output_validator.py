"""
Stage output validation utilities for FunGen.
Validates completeness of stage outputs to enable smart skipping of re-processing.
"""

import os
import logging
import msgpack
import sqlite3
from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path


class StageOutputValidator:
    """Validates completeness of processing stage outputs."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the validator.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def validate_stage1_output(self, msgpack_path: str, expected_frame_count: Optional[int] = None) -> bool:
        """
        Validate Stage 1 output msgpack file.
        
        Args:
            msgpack_path: Path to the stage 1 msgpack file
            expected_frame_count: Expected number of frames (optional)
            
        Returns:
            True if output is valid and complete
        """
        try:
            if not os.path.exists(msgpack_path):
                self.logger.debug(f"Stage 1 msgpack not found: {msgpack_path}")
                return False
            
            # Check file size (empty files are invalid)
            if os.path.getsize(msgpack_path) == 0:
                self.logger.debug(f"Stage 1 msgpack is empty: {msgpack_path}")
                return False
            
            # Try to load and validate msgpack content
            with open(msgpack_path, 'rb') as f:
                data = msgpack.unpack(f, raw=False)
            
            if not isinstance(data, (list, dict)):
                self.logger.debug(f"Stage 1 msgpack has invalid format: {msgpack_path}")
                return False
            
            # If it's a list, check it has frames
            if isinstance(data, list):
                frame_count = len(data)
                if frame_count == 0:
                    self.logger.debug(f"Stage 1 msgpack has no frames: {msgpack_path}")
                    return False
                
                if expected_frame_count and frame_count < expected_frame_count * 0.95:  # Allow 5% tolerance
                    self.logger.debug(f"Stage 1 msgpack has too few frames: {frame_count} < {expected_frame_count}")
                    return False
            
            # If it's a dict, check for required keys
            elif isinstance(data, dict):
                if 'frames' not in data and 'detections' not in data:
                    self.logger.debug(f"Stage 1 msgpack missing required keys: {msgpack_path}")
                    return False
            
            self.logger.debug(f"Stage 1 msgpack validated successfully: {msgpack_path}")
            return True
            
        except Exception as e:
            self.logger.debug(f"Stage 1 msgpack validation failed: {msgpack_path}, error: {e}")
            return False
    
    def validate_stage2_output(self, video_path: str, output_folder: Optional[str] = None) -> Tuple[bool, Dict[str, str]]:
        """
        Validate Stage 2 output files (database and overlay msgpack).
        
        Args:
            video_path: Path to the video file
            output_folder: Optional output folder path
            
        Returns:
            Tuple of (is_valid, file_paths_dict)
        """
        try:
            video_stem = Path(video_path).stem
            
            if output_folder:
                base_dir = output_folder
            else:
                base_dir = os.path.dirname(video_path)
            
            # Try to find Stage 2 files - first with the current video stem
            db_path = os.path.join(base_dir, f"{video_stem}_stage2_data.db")
            overlay_msgpack_path = os.path.join(base_dir, f"{video_stem}_stage2_overlay.msgpack")
            
            # Check if files exist with current stem
            db_exists = os.path.exists(db_path)
            overlay_exists = os.path.exists(overlay_msgpack_path)
            
            # If files don't exist and this is a preprocessed video, try original stem
            if not db_exists and not overlay_exists and video_stem.endswith('_preprocessed'):
                original_stem = video_stem[:-len('_preprocessed')]  # Remove '_preprocessed' suffix
                self.logger.debug(f"Stage 2 files not found with preprocessed stem '{video_stem}', trying original stem '{original_stem}'")
                
                db_path = os.path.join(base_dir, f"{original_stem}_stage2_data.db")
                overlay_msgpack_path = os.path.join(base_dir, f"{original_stem}_stage2_overlay.msgpack")
                
                db_exists = os.path.exists(db_path)
                overlay_exists = os.path.exists(overlay_msgpack_path)
                
                if db_exists or overlay_exists:
                    self.logger.debug(f"Found Stage 2 files using original stem '{original_stem}'")
            
            file_paths = {
                'database': db_path,
                'overlay_msgpack': overlay_msgpack_path
            }
            
            if not db_exists and not overlay_exists:
                self.logger.debug(f"Stage 2 outputs not found for: {video_stem}")
                return False, file_paths
            
            validation_results = []
            
            # Validate database if it exists
            if db_exists:
                db_valid = self._validate_stage2_database(db_path)
                validation_results.append(db_valid)
                if db_valid:
                    self.logger.debug(f"Stage 2 database validated: {db_path}")
                else:
                    self.logger.debug(f"Stage 2 database validation failed: {db_path}")
            
            # Validate overlay msgpack if it exists
            if overlay_exists:
                overlay_valid = self._validate_stage2_overlay_msgpack(overlay_msgpack_path)
                validation_results.append(overlay_valid)
                if overlay_valid:
                    self.logger.debug(f"Stage 2 overlay msgpack validated: {overlay_msgpack_path}")
                else:
                    self.logger.debug(f"Stage 2 overlay msgpack validation failed: {overlay_msgpack_path}")
            
            # Consider valid if at least one file exists and is valid
            is_valid = len(validation_results) > 0 and any(validation_results)
            
            if is_valid:
                self.logger.info(f"Stage 2 outputs validated successfully for: {video_stem}")
            else:
                self.logger.debug(f"Stage 2 outputs validation failed for: {video_stem}")
            
            return is_valid, file_paths
            
        except Exception as e:
            self.logger.error(f"Stage 2 validation error for {video_path}: {e}")
            return False, {}
    
    def _validate_stage2_database(self, db_path: str) -> bool:
        """
        Validate Stage 2 SQLite database.
        
        Args:
            db_path: Path to the database file
            
        Returns:
            True if database is valid and contains data
        """
        try:
            if os.path.getsize(db_path) == 0:
                return False
            
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Check if required tables exist
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Look for common stage 2 table names
                expected_tables = ['segments', 'frames', 'frame_objects', 's2_segments']
                has_required_tables = any(table in tables for table in expected_tables)
                
                if not has_required_tables:
                    self.logger.debug(f"Stage 2 database missing required tables: {db_path}")
                    return False
                
                # Check if tables have data
                for table in tables:
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table}")
                        count = cursor.fetchone()[0]
                        if count > 0:
                            return True  # At least one table has data
                    except sqlite3.Error:
                        continue  # Skip problematic tables
                
                self.logger.debug(f"Stage 2 database has no data: {db_path}")
                return False
                
        except Exception as e:
            self.logger.debug(f"Stage 2 database validation error: {db_path}, error: {e}")
            return False
    
    def _validate_stage2_overlay_msgpack(self, msgpack_path: str) -> bool:
        """
        Validate Stage 2 overlay msgpack file.
        
        Args:
            msgpack_path: Path to the overlay msgpack file
            
        Returns:
            True if msgpack is valid and contains data
        """
        try:
            if os.path.getsize(msgpack_path) == 0:
                return False
            
            with open(msgpack_path, 'rb') as f:
                data = msgpack.unpack(f, raw=False)
            
            if not isinstance(data, (list, dict)):
                return False
            
            # Check for data content
            if isinstance(data, list):
                return len(data) > 0
            elif isinstance(data, dict):
                return len(data) > 0
            
            return False
            
        except Exception as e:
            self.logger.debug(f"Stage 2 overlay msgpack validation error: {msgpack_path}, error: {e}")
            return False
    
    def should_skip_stage2(self, video_path: str, force_rerun: bool = False, 
                          output_folder: Optional[str] = None) -> Tuple[bool, Dict[str, str]]:
        """
        Determine if Stage 2 should be skipped based on existing outputs.
        
        Args:
            video_path: Path to the video file
            force_rerun: Whether to force re-running stage 2
            output_folder: Optional output folder path
            
        Returns:
            Tuple of (should_skip, file_paths_dict)
        """
        if force_rerun:
            self.logger.info("Stage 2 force rerun enabled, not skipping")
            return False, {}
        
        is_valid, file_paths = self.validate_stage2_output(video_path, output_folder)
        
        if is_valid:
            self.logger.info(f"Stage 2 outputs found and valid, skipping re-processing for: {Path(video_path).stem}")
            return True, file_paths
        else:
            self.logger.debug(f"Stage 2 outputs missing or invalid, will process: {Path(video_path).stem}")
            return False, file_paths
    
    def get_stage2_output_paths(self, video_path: str, output_folder: Optional[str] = None) -> Dict[str, str]:
        """
        Get expected Stage 2 output file paths.
        
        Args:
            video_path: Path to the video file
            output_folder: Optional output folder path
            
        Returns:
            Dictionary with expected file paths
        """
        video_stem = Path(video_path).stem
        
        if output_folder:
            base_dir = output_folder
        else:
            base_dir = os.path.dirname(video_path)
        
        return {
            'database': os.path.join(base_dir, f"{video_stem}_stage2_data.db"),
            'overlay_msgpack': os.path.join(base_dir, f"{video_stem}_stage2_overlay.msgpack")
        }
    
    def can_skip_stage2_for_stage3(self, video_path: str, force_rerun: bool = False, 
                                  output_folder: Optional[str] = None,
                                  project_db_path: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
        """
        Determine if Stage 2 can be skipped when running 3-stage methods, with additional
        validation for Stage 3 compatibility.
        
        Args:
            video_path: Path to the video file
            force_rerun: Whether to force re-running stage 2
            output_folder: Optional output folder path
            
        Returns:
            Tuple of (can_skip, stage2_data_info)
        """
        if force_rerun:
            self.logger.info("Stage 2 force rerun enabled for Stage 3, not skipping")
            return False, {}
        
        # Prioritize project-saved database path if provided and valid
        if project_db_path and os.path.exists(project_db_path):
            self.logger.info(f"Using project-saved database path: {project_db_path}")
            file_paths = {
                'database': project_db_path,
                'overlay_msgpack': self.get_stage2_output_paths(video_path, output_folder).get('overlay_msgpack')
            }
            can_skip = True  # We trust the project-saved path
        else:
            # Fall back to heuristic discovery
            can_skip, file_paths = self.should_skip_stage2(video_path, force_rerun, output_folder)
            
            if not can_skip:
                return False, {}
        
        # Additional validation for Stage 3 compatibility
        stage2_data_info = {
            'file_paths': file_paths,
            'frame_objects_available': False,
            'segments_available': False,
            'estimated_frame_count': 0
        }
        
        db_path = file_paths.get('database')
        overlay_path = file_paths.get('overlay_msgpack')
        
        # Try database validation first (preferred)
        if db_path and os.path.exists(db_path) and os.path.getsize(db_path) > 0:
            db_valid, db_info = self._validate_stage2_for_stage3_compatibility(db_path)
            if db_valid:
                stage2_data_info.update(db_info)
                self.logger.info(f"Stage 2 database validated for Stage 3 compatibility: {Path(video_path).stem}")
                return True, stage2_data_info
            else:
                self.logger.debug(f"Stage 2 database not compatible with Stage 3: {Path(video_path).stem}")
        
        # Fallback: validate using overlay msgpack if database is missing/empty
        if overlay_path and os.path.exists(overlay_path) and os.path.getsize(overlay_path) > 0:
            overlay_valid, overlay_info = self._validate_stage2_overlay_for_stage3(overlay_path)
            if overlay_valid:
                stage2_data_info.update(overlay_info)
                self.logger.info(f"Stage 2 overlay validated for Stage 3 compatibility (database unavailable): {Path(video_path).stem}")
                return True, stage2_data_info
            else:
                self.logger.debug(f"Stage 2 overlay not suitable for Stage 3: {Path(video_path).stem}")
        
        self.logger.debug(f"No suitable Stage 2 data found for Stage 3 validation: {Path(video_path).stem}")
        return False, stage2_data_info
    
    def _validate_stage2_for_stage3_compatibility(self, db_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate Stage 2 database for Stage 3 compatibility.
        
        Args:
            db_path: Path to Stage 2 database
            
        Returns:
            Tuple of (is_compatible, db_info_dict)
        """
        db_info = {
            'frame_objects_available': False,
            'segments_available': False,
            'estimated_frame_count': 0,
            'has_atr_data': False,
            'has_funscript_positions': False
        }
        
        try:
            with sqlite3.connect(db_path) as conn:
                cursor = conn.cursor()
                
                # Check available tables
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = [row[0] for row in cursor.fetchall()]
                
                # Check for frame_objects table (required for Stage 3)
                if 'frame_objects' in tables:
                    cursor.execute("SELECT COUNT(*) FROM frame_objects")
                    frame_count = cursor.fetchone()[0]
                    if frame_count > 0:
                        db_info['frame_objects_available'] = True
                        db_info['estimated_frame_count'] = frame_count
                        
                        # Check if frame objects have required Stage 3 data
                        # Check for funscript position data (different column names in different schemas)
                        position_columns = ['pos_0_100', 'funscript_distance', 'funscript_position']
                        for col in position_columns:
                            try:
                                cursor.execute(f"SELECT COUNT(*) FROM frame_objects WHERE {col} IS NOT NULL LIMIT 1")
                                if cursor.fetchone()[0] > 0:
                                    db_info['has_funscript_positions'] = True
                                    break
                            except sqlite3.Error:
                                continue
                        
                        # Check for Stage 2 data
                        # Check different possible column names for Stage 2 data
                        stage2_columns = ['locked_penis_active', 'locked_penis_state', 'locked_penis_state']
                        for col in stage2_columns:
                            try:
                                cursor.execute(f"SELECT COUNT(*) FROM frame_objects WHERE {col} IS NOT NULL LIMIT 1")
                                if cursor.fetchone()[0] > 0:
                                    db_info['has_atr_data'] = True
                                    break
                            except sqlite3.Error:
                                continue
                
                # Check for segments table
                if 'segments' in tables or 's2_segments' in tables:
                    segment_table = 's2_segments' if 's2_segments' in tables else 'segments'
                    cursor.execute(f"SELECT COUNT(*) FROM {segment_table}")
                    segment_count = cursor.fetchone()[0]
                    if segment_count > 0:
                        db_info['segments_available'] = True
                
                # For Stage 3 compatibility, we need frame objects with position data
                is_compatible = (db_info['frame_objects_available'] and 
                               db_info['has_funscript_positions'] and
                               db_info['estimated_frame_count'] > 10)
                
                if is_compatible:
                    self.logger.debug(f"Stage 2 database compatible with Stage 3: {frame_count} frames with position data")
                else:
                    self.logger.debug(f"Stage 2 database not compatible: frames={frame_count}, has_positions={db_info['has_funscript_positions']}")
                
                return is_compatible, db_info
                
        except Exception as e:
            self.logger.debug(f"Error validating Stage 2 database for Stage 3: {db_path}, error: {e}")
            return False, db_info
    
    def _validate_stage2_overlay_for_stage3(self, overlay_path: str) -> Tuple[bool, Dict[str, Any]]:
        """
        Validate Stage 2 overlay msgpack for Stage 3 compatibility.
        
        Args:
            overlay_path: Path to overlay msgpack file
            
        Returns:
            Tuple of (is_compatible, overlay_info_dict)
        """
        overlay_info = {
            'frame_objects_available': False,
            'segments_available': False,
            'estimated_frame_count': 0,
            'has_atr_data': False,
            'has_funscript_positions': False
        }
        
        try:
            import msgpack
            
            with open(overlay_path, 'rb') as f:
                data = msgpack.unpack(f, raw=False)
            
            if not isinstance(data, list):
                self.logger.debug(f"Stage 2 overlay has unexpected format: {overlay_path}")
                return False, overlay_info
            
            frame_count = len(data)
            if frame_count == 0:
                self.logger.debug(f"Stage 2 overlay is empty: {overlay_path}")
                return False, overlay_info
            
            overlay_info['estimated_frame_count'] = frame_count
            
            # Check first few frames for required data structure
            frames_with_data = 0
            frames_checked = min(10, frame_count)  # Check first 10 frames or all if less
            
            for i in range(frames_checked):
                frame_data = data[i]
                if isinstance(frame_data, dict):
                    # Look for Stage 2 specific data structures based on actual overlay format
                    has_boxes = 'yolo_boxes' in frame_data or 'boxes' in frame_data or 'detections' in frame_data
                    has_position_data = 'assigned_position' in frame_data or 'position' in frame_data
                    has_tracking_data = 'active_interaction_track_id' in frame_data or 'motion_mode' in frame_data
                    
                    if has_boxes or has_position_data or has_tracking_data:
                        frames_with_data += 1
                        if not overlay_info['has_atr_data'] and has_tracking_data:
                            overlay_info['has_atr_data'] = True
                        if not overlay_info['has_funscript_positions'] and has_position_data:
                            overlay_info['has_funscript_positions'] = True
            
            # Consider valid if most sampled frames have relevant data
            if frames_with_data >= (frames_checked * 0.7):  # 70% of sampled frames have data
                overlay_info['frame_objects_available'] = True
                overlay_info['segments_available'] = True  # Assume segments if we have frame data
                
                # For Stage 3 compatibility, we need frame data
                is_compatible = overlay_info['frame_objects_available'] and frame_count > 10
                
                if is_compatible:
                    self.logger.debug(f"Stage 2 overlay compatible with Stage 3: {frame_count} frames with data")
                else:
                    self.logger.debug(f"Stage 2 overlay not compatible: insufficient frame data")
                
                return is_compatible, overlay_info
            else:
                self.logger.debug(f"Stage 2 overlay has insufficient data: {frames_with_data}/{frames_checked} frames")
                return False, overlay_info
                
        except Exception as e:
            self.logger.debug(f"Error validating Stage 2 overlay for Stage 3: {overlay_path}, error: {e}")
            return False, overlay_info


# Convenience functions
def validate_stage1_output(msgpack_path: str, expected_frame_count: Optional[int] = None, 
                          logger: Optional[logging.Logger] = None) -> bool:
    """Convenience function to validate Stage 1 output."""
    validator = StageOutputValidator(logger)
    return validator.validate_stage1_output(msgpack_path, expected_frame_count)


def should_skip_stage2(video_path: str, force_rerun: bool = False, 
                      output_folder: Optional[str] = None,
                      logger: Optional[logging.Logger] = None) -> Tuple[bool, Dict[str, str]]:
    """Convenience function to check if Stage 2 should be skipped."""
    validator = StageOutputValidator(logger)
    return validator.should_skip_stage2(video_path, force_rerun, output_folder)


def get_stage2_output_paths(video_path: str, output_folder: Optional[str] = None) -> Dict[str, str]:
    """Convenience function to get Stage 2 output paths."""
    validator = StageOutputValidator()
    return validator.get_stage2_output_paths(video_path, output_folder)


def can_skip_stage2_for_stage3(video_path: str, force_rerun: bool = False, 
                              output_folder: Optional[str] = None,
                              logger: Optional[logging.Logger] = None,
                              project_db_path: Optional[str] = None) -> Tuple[bool, Dict[str, Any]]:
    """Convenience function to check if Stage 2 can be skipped for Stage 3."""
    validator = StageOutputValidator(logger)
    return validator.can_skip_stage2_for_stage3(video_path, force_rerun, output_folder, project_db_path)
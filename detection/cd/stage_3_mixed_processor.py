"""
Stage 3 Mixed Processor - Combines Stage 2 output with selective ROI tracking
Author: FunGen AI System
Version: 1.0.0

This module implements a "mixed" approach to Stage 3 processing:
- Uses Stage 2 signal as-is for most chapters
- Applies YOLO ROI tracking only for BJ/HJ chapters using Stage 2 detections as ROI input
- Maintains compatibility with existing 3-stage infrastructure
"""

import time
import logging
import cv2
import os
import numpy as np
from typing import Optional, List, Dict, Any, Tuple, Union
from multiprocessing import Event

from funscript import DualAxisFunscript
from tracker import ROITracker
from detection.cd.stage_2_cd import FrameObject
from application.utils.video_segment import VideoSegment
from config import constants


class MixedStageProcessor:
    """
    Processes Stage 3 using a mixed approach:
    - Stage 2 signal for non-BJ/HJ chapters
    - ROI tracking for BJ/HJ chapters using Stage 2 detection data
    """
    
    def __init__(self, tracker_model_path: str, pose_model_path: Optional[str] = None):
        """Initialize the mixed processor with model paths."""
        self.tracker_model_path = tracker_model_path
        self.pose_model_path = pose_model_path
        
        # Stage 2 data
        self.stage2_frame_objects: Dict[int, FrameObject] = {}
        self.stage2_segments: List[VideoSegment] = []
        
        # ROI Tracker (initialized when needed)
        self.roi_tracker: Optional[ROITracker] = None
        
        # Processing state
        self.current_roi: Optional[Tuple[int, int, int, int]] = None
        self.locked_penis_active: bool = False
        self.current_chapter_type: Optional[str] = None
        
        # Live tracker state for BJ/HJ chapters
        self.live_tracker_active: bool = False
        self.oscillation_intensity: float = 0.0
        
        # ROI adaptation control - don't update ROI every frame
        self.roi_update_counter: int = 0
        self.roi_update_frequency: int = 30  # Update ROI every 30 frames (~1 second at 30fps)
        self.last_used_roi: Optional[Tuple[int, int, int, int]] = None
        
        # Enhanced oscillation settings for mixed mode
        self.mixed_mode_settings = {
            'ema_alpha': 0.15,  # Less aggressive smoothing than default 0.3
            'base_sensitivity': 3.5,  # Higher sensitivity for better response
            'grid_size': 15,  # Smaller grid for more precise detection
            'hold_duration_ms': 150,  # Shorter hold for more responsiveness
        }
        
        # Debug info
        self.signal_source: str = "stage2"  # "stage2" or "roi_tracker"
        self.debug_data: Dict[int, Any] = {}  # Store debug info for msgpack (frame_id -> debug_info)
    
    def set_stage2_results(self, frame_objects: Dict[int, FrameObject], segments: List[VideoSegment]):
        """Set the Stage 2 results that will be used as input."""
        self.stage2_frame_objects = frame_objects
        self.stage2_segments = segments
        
        logging.info(f"Mixed processor initialized with {len(frame_objects)} frames and {len(segments)} segments")
    
    def _get_segment_position_short_name(self, segment) -> str:
        """
        Get the position short name from either VideoSegment or ATRSegment objects.
        Returns the standardized short name (e.g., 'HJ', 'BJ', 'CG/Miss', 'NR').
        """
        if hasattr(segment, 'position_short_name'):
            # VideoSegment object
            return segment.position_short_name
        elif hasattr(segment, 'major_position'):
            # ATRSegment object - map major_position to short name
            position_mapping = {
                'Handjob': 'HJ',
                'Blowjob': 'BJ',
                'Cowgirl / Missionary': 'CG/Miss',
                'Not Relevant': 'NR'
            }
            return position_mapping.get(segment.major_position, segment.major_position)
        else:
            return 'Other'
    
    def determine_chapter_type(self, frame_id: int) -> str:
        """
        Determine the chapter type for a given frame based on Stage 2 segments.
        Returns 'BJ', 'HJ', or 'Other'
        """
        # Find the segment containing this frame
        for segment in self.stage2_segments:
            if segment.start_frame_id <= frame_id <= segment.end_frame_id:
                position_short_name = self._get_segment_position_short_name(segment)
                if position_short_name in ['BJ', 'HJ']:
                    return position_short_name
                break
        return 'Other'
    
    def extract_roi_from_stage2(self, frame_id: int) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract ROI from Stage 2 locked penis state for the given frame.
        Returns (x1, y1, x2, y2) or None if no valid ROI.
        """
        frame_obj = self.stage2_frame_objects.get(frame_id)
        if not frame_obj:
            return None
        
        # Check if locked penis is active and has a valid box
        if (frame_obj.atr_locked_penis_state.active and 
            frame_obj.atr_locked_penis_state.box):
            box = frame_obj.atr_locked_penis_state.box
            try:
                # Debug: Log box data to understand the issue
                logging.debug(f"Frame {frame_id} box data: {box} (type: {type(box)})")
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    # Ensure all elements are numeric
                    coords = []
                    for i, coord in enumerate(box[:4]):
                        if isinstance(coord, (bytes, str)):
                            # Don't spam logs - just track corrupted frames
                            if not hasattr(self, '_corrupted_frames_logged'):
                                self._corrupted_frames_logged = set()
                                self._corrupted_frame_count = 0
                            
                            if frame_id not in self._corrupted_frames_logged:
                                self._corrupted_frames_logged.add(frame_id)
                                self._corrupted_frame_count += 1
                                
                                # Only log first few corrupted frames, then summarize
                                if self._corrupted_frame_count <= 3:
                                    logging.warning(f"Frame {frame_id} has corrupted box coordinates (binary data)")
                                elif self._corrupted_frame_count == 4:
                                    logging.warning(f"Multiple frames have corrupted box coordinates, suppressing further individual warnings...")
                            
                            return None
                        coords.append(float(coord))  # Convert to float first, then int
                    # Convert to integer coordinates
                    return (int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3]))
                else:
                    logging.error(f"Frame {frame_id} box is not a valid sequence: {box}")
                    return None
            except (ValueError, TypeError) as e:
                logging.error(f"Frame {frame_id} box conversion error: {e}, box: {box}")
                return None
        
        return None
    
    def get_stage2_signal(self, frame_id: int) -> float:
        """
        Get the Stage 2 funscript signal for the given frame.
        Returns a value between 0.0 and 1.0.
        """
        frame_obj = self.stage2_frame_objects.get(frame_id)
        if frame_obj and hasattr(frame_obj, 'atr_funscript_distance'):
            try:
                # Debug: Check if the distance value is valid
                distance = frame_obj.atr_funscript_distance
                if isinstance(distance, (bytes, str)):
                    logging.error(f"Frame {frame_id} atr_funscript_distance is not numeric: {distance} (type: {type(distance)})")
                    return 0.5
                return float(distance) / 100.0
            except (ValueError, TypeError) as e:
                logging.error(f"Frame {frame_id} signal conversion error: {e}, distance: {frame_obj.atr_funscript_distance}")
                return 0.5  # Default middle position
        else:
            return 0.5  # Default middle position
    
    def initialize_roi_tracker_if_needed(self, tracker_config: Dict[str, Any], 
                                       common_app_config: Dict[str, Any]) -> bool:
        """Initialize ROI tracker if it hasn't been initialized yet."""
        if self.roi_tracker is not None:
            return True
        
        try:
            # Mock app instance for ROI tracker initialization
            class MockApp:
                def __init__(self, det_model_path, pose_model_path, mixed_settings):
                    self.yolo_det_model_path = det_model_path
                    self.yolo_pose_model_path = pose_model_path
                    self.yolo_input_size = common_app_config.get('yolo_input_size', 640)
                    
                    # Create mock settings with optimized values for mixed mode
                    class MockSettings:
                        def get(self, key, default=None):
                            # Override specific settings for better mixed mode performance
                            if key == 'oscillation_detector_grid_size':
                                return mixed_settings['grid_size']
                            if key == 'oscillation_detector_sensitivity':
                                return tracker_config.get('oscillation_sensitivity', 1.5)
                            if key == 'live_oscillation_dynamic_amp_enabled':
                                return True
                            return tracker_config.get(key, default)
                    
                    self.app_settings = MockSettings()
            
            # Pass the model paths and mixed settings to the mock app
            mock_app = MockApp(self.tracker_model_path, self.pose_model_path, self.mixed_mode_settings)
            
            # Initialize ROI tracker for oscillation detection only
            # Suppress ROI tracker logging to reduce noise
            import logging
            roi_logger = logging.getLogger('tracker')
            original_level = roi_logger.level
            roi_logger.setLevel(logging.ERROR)  # Only show errors
            
            try:
                self.roi_tracker = ROITracker(
                    app_logic_instance=mock_app, 
                    tracker_model_path=None,  # No model needed for mixed mode
                    pose_model_path=None,     # No pose model needed
                    load_models_on_init=False  # Explicitly disable model loading
                )
            finally:
                # Restore original logging level
                roi_logger.setLevel(original_level)
            
            # Override oscillation settings for better mixed mode performance
            self.roi_tracker.oscillation_ema_alpha = self.mixed_mode_settings['ema_alpha']
            self.roi_tracker.oscillation_hold_duration_ms = self.mixed_mode_settings['hold_duration_ms']
            
            # Ensure oscillation attributes are initialized
            if not hasattr(self.roi_tracker, 'oscillation_funscript_pos'):
                self.roi_tracker.oscillation_funscript_pos = 50
            if not hasattr(self.roi_tracker, 'oscillation_last_known_pos'):
                self.roi_tracker.oscillation_last_known_pos = 50.0
            return True

        except Exception as e:
            logging.error(f"Failed to initialize ROI tracker: {e}", exc_info=True)
            return False
    
    def process_frame_mixed(self, frame_id: int, video_frame: np.ndarray,
                          tracker_config: Dict[str, Any], 
                          common_app_config: Dict[str, Any],
                          frame_time_ms: float) -> Tuple[float, Dict[str, Any]]:
        """
        Process a single frame using mixed approach.
        Returns: (funscript_position_0_1, debug_info)
        """
        debug_info = self.get_debug_info()
        debug_info['frame_id'] = frame_id
        
        # Determine chapter type for this frame
        chapter_type = self.determine_chapter_type(frame_id)
        self.current_chapter_type = chapter_type
        debug_info['current_chapter_type'] = chapter_type
        
        if chapter_type in ['BJ', 'HJ']:
            # Use ROI tracking for BJ/HJ chapters
            self.signal_source = "roi_tracker"
            
            # Extract ROI from Stage 2 data
            roi = self.extract_roi_from_stage2(frame_id)
            self.current_roi = roi
            debug_info['current_roi'] = roi
            
            if roi and self.initialize_roi_tracker_if_needed(tracker_config, common_app_config):
                # Process frame with ROI tracker
                try:
                    self.live_tracker_active = True
                    debug_info['live_tracker_active'] = True
                    
                    # Smart ROI adaptation - don't update every frame to allow tracking to work
                    should_update_roi = False
                    
                    if roi:
                        # Only update ROI periodically or when significantly different
                        if self.last_used_roi is None:
                            # First time setting ROI
                            should_update_roi = True
                        elif self.roi_update_counter >= self.roi_update_frequency:
                            # Periodic update
                            should_update_roi = True
                            self.roi_update_counter = 0
                        elif self.last_used_roi and roi:
                            # Check if ROI has moved significantly (more than 20% of box dimensions)
                            old_roi = self.last_used_roi
                            roi_moved = (
                                abs(roi[0] - old_roi[0]) > (old_roi[2] - old_roi[0]) * 0.2 or
                                abs(roi[1] - old_roi[1]) > (old_roi[3] - old_roi[1]) * 0.2
                            )
                            if roi_moved:
                                should_update_roi = True
                        
                        if should_update_roi:
                            # Set oscillation area based on Stage 2 locked penis detection
                            if hasattr(self.roi_tracker, 'set_oscillation_area'):
                                self.roi_tracker.set_oscillation_area(roi)
                            elif hasattr(self.roi_tracker, 'oscillation_area_fixed'):
                                self.roi_tracker.oscillation_area_fixed = roi
                            
                            self.last_used_roi = roi
                            debug_info['roi_updated'] = True
                        else:
                            debug_info['roi_updated'] = False
                    
                    self.roi_update_counter += 1
                    
                    # Process frame for oscillation/tracking
                    if hasattr(self.roi_tracker, 'process_frame_for_oscillation'):
                        self.roi_tracker.process_frame_for_oscillation(video_frame, frame_time_ms=int(frame_time_ms))
                        # The funscript position is stored in oscillation_funscript_pos attribute
                        position = getattr(self.roi_tracker, 'oscillation_funscript_pos', 50) / 100.0  # Convert to 0-1 range
                        # Calculate intensity from oscillation data
                        if hasattr(self.roi_tracker, 'oscillation_last_known_pos'):
                            self.oscillation_intensity = abs(self.roi_tracker.oscillation_last_known_pos - 50) / 50.0
                        else:
                            self.oscillation_intensity = 0.0
                        
                        # Collect debug data for msgpack
                        debug_info['oscillation_intensity'] = self.oscillation_intensity
                        debug_info['oscillation_pos'] = self.roi_tracker.oscillation_funscript_pos
                        debug_info['oscillation_last_known'] = getattr(self.roi_tracker, 'oscillation_last_known_pos', None)
                        debug_info['ema_alpha'] = self.roi_tracker.oscillation_ema_alpha
                        debug_info['roi_current'] = roi
                        debug_info['roi_update_counter'] = self.roi_update_counter
                        
                        # Store debug data for later msgpack creation
                        self.debug_data[frame_id] = debug_info.copy()
                        
                        return position, debug_info
                    
                except Exception as e:
                    logging.warning(f"ROI tracking failed for frame {frame_id}: {e}")
                    # Fall back to Stage 2 signal
                    self.live_tracker_active = False
        
        # Use Stage 2 signal for non-BJ/HJ chapters or when ROI tracking fails
        self.signal_source = "stage2"
        self.live_tracker_active = False
        debug_info['live_tracker_active'] = False
        debug_info['signal_source'] = self.signal_source
        
        stage2_position = self.get_stage2_signal(frame_id)
        return stage2_position, debug_info
    
    def get_debug_info(self) -> Dict[str, Any]:
        """Get current debug information for display."""
        return {
            'current_roi': self.current_roi,
            'locked_penis_active': self.locked_penis_active,
            'current_chapter_type': self.current_chapter_type,
            'live_tracker_active': self.live_tracker_active,
            'oscillation_intensity': self.oscillation_intensity,
            'signal_source': self.signal_source
        }
    
    def save_debug_msgpack(self, output_path: str) -> bool:
        """Save debug data to msgpack for visualization and troubleshooting."""
        try:
            import msgpack
            
            # Prepare debug data for serialization
            debug_export = {
                'metadata': {
                    'version': '1.0',
                    'processor_type': 'stage_3_mixed',
                    'mixed_mode_settings': self.mixed_mode_settings,
                    'roi_update_frequency': self.roi_update_frequency,
                    'total_frames': len(self.debug_data)
                },
                'frame_data': {}
            }
            
            # Convert debug data to serializable format
            for frame_id, debug_info in self.debug_data.items():
                serializable_info = {}
                for key, value in debug_info.items():
                    # Convert numpy types and other non-serializable types
                    if hasattr(value, 'tolist'):  # numpy arrays
                        serializable_info[key] = value.tolist()
                    elif isinstance(value, (int, float, str, bool, list, dict, type(None))):
                        serializable_info[key] = value
                    else:
                        serializable_info[key] = str(value)  # Convert to string as fallback
                
                debug_export['frame_data'][str(frame_id)] = serializable_info
            
            # Write to msgpack file
            with open(output_path, 'wb') as f:
                msgpack.pack(debug_export, f)
            
            logging.info(f"Stage 3 mixed debug data saved to: {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Failed to save debug msgpack: {e}")
            return False


def perform_mixed_stage_analysis(
    video_path: str,
    preprocessed_video_path_arg: Optional[str],
    atr_segments_list: List[VideoSegment],
    s2_frame_objects_map: Dict[int, FrameObject],
    tracker_config: Dict[str, Any],
    common_app_config: Dict[str, Any],
    progress_callback=None,
    stop_event: Optional[Event] = None,
    parent_logger: Optional[logging.Logger] = None,
    sqlite_db_path: Optional[str] = None
) -> Dict[str, Any]:
    """
    Perform mixed Stage 3 analysis on video segments.
    
    This function serves as the main entry point for mixed stage processing,
    compatible with the existing 3-stage infrastructure.
    """
    logger = parent_logger or logging.getLogger(__name__)
    logger.info("Starting mixed Stage 3 analysis")
    
    try:
        # Initialize mixed processor
        tracker_model_path = common_app_config.get('yolo_det_model_path', '')
        pose_model_path = common_app_config.get('yolo_pose_model_path')
        
        processor = MixedStageProcessor(tracker_model_path, pose_model_path)
        processor.set_stage2_results(s2_frame_objects_map, atr_segments_list)
        
        # Initialize results
        primary_actions = []
        secondary_actions = []
        
        # Process each relevant segment
        total_frames = 0
        total_roi_tracking_frames = 0  # Frames that need ROI processing (BJ/HJ)
        processed_frames = 0
        processed_roi_frames = 0  # Frames processed with ROI tracking
        
        # Count total frames to process and separate ROI tracking frames
        for segment in atr_segments_list:
            position_short_name = processor._get_segment_position_short_name(segment)
            if position_short_name not in ['NR', 'C-Up']:
                segment_frames = segment.end_frame_id - segment.start_frame_id + 1
                total_frames += segment_frames
                
                # Count frames that need ROI tracking (BJ/HJ only)
                if position_short_name in ['BJ', 'HJ']:
                    total_roi_tracking_frames += segment_frames
        
        logger.info(f"Mixed Stage 3: Processing {total_frames} frames across {len(atr_segments_list)} segments")
        
        
        # Track processing time
        start_time = time.time()
        
        # Open video for frame processing
        import cv2
        cap = cv2.VideoCapture(preprocessed_video_path_arg or video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        
        video_fps = cap.get(cv2.CAP_PROP_FPS) or common_app_config.get('video_fps', 30.0)
        try:
            video_fps = float(video_fps)
            if video_fps <= 0:
                video_fps = 30.0
        except (ValueError, TypeError):
            logger.error(f"Invalid video_fps value: {video_fps}, using default 30.0")
            video_fps = 30.0
        
        # Process segments
        for segment_idx, segment in enumerate(atr_segments_list):
            if stop_event and stop_event.is_set():
                break
            
            # Get position short name using helper function
            position_short_name = processor._get_segment_position_short_name(segment)
            
            # Get additional segment info for logging
            if hasattr(segment, 'position_long_name'):
                position_long_name = segment.position_long_name
                class_name = getattr(segment, 'class_name', 'N/A')
                segment_type = getattr(segment, 'segment_type', 'N/A')
            elif hasattr(segment, 'major_position'):
                position_long_name = segment.major_position
                class_name = segment.major_position
                segment_type = 'SexAct'
            else:
                logger.warning(f"Unknown segment type: {type(segment)} - skipping")
                continue
            
            # Skip non-relevant segments
            if position_short_name in ['NR', 'C-Up']:
                logger.info(f"Skipping non-relevant segment: {position_short_name}")
                continue
            
            is_tracking_chapter = position_short_name in ['BJ', 'HJ']
            
            processing_mode = 'ROI Tracking' if is_tracking_chapter else 'Stage 2 Signal'
            # Only log ROI tracking segments to reduce verbosity
            if is_tracking_chapter:
                logger.info(f"Processing segment {segment_idx + 1}/{len(atr_segments_list)}: "
                           f"{position_short_name} ({segment.start_frame_id}-{segment.end_frame_id}) "
                           f"- Using {processing_mode}")
            
            # Process frames in this segment
            try:
                start_frame = int(segment.start_frame_id)
                end_frame = int(segment.end_frame_id)
            except (ValueError, TypeError) as e:
                logger.error(f"Invalid segment frame IDs: start={segment.start_frame_id}, end={segment.end_frame_id}, error: {e}")
                continue
            
            for frame_id in range(start_frame, end_frame + 1):
                if stop_event and stop_event.is_set():
                    break
                
                if is_tracking_chapter:
                    # Use ROI tracking for BJ/HJ chapters
                    # Seek to frame
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ret, frame = cap.read()
                    if not ret:
                        logger.warning(f"Could not read frame {frame_id}, falling back to Stage 2 signal")
                        # Fall back to Stage 2 signal
                        position = processor.get_stage2_signal(frame_id)
                    else:
                        # Calculate frame timestamp in milliseconds
                        try:
                            frame_time_ms = int((frame_id / video_fps) * 1000.0)
                            
                            # Process frame with ROI tracking
                            position, debug_info = processor.process_frame_mixed(
                                frame_id, frame, tracker_config, common_app_config, frame_time_ms
                            )
                            processed_roi_frames += 1
                        except (ValueError, TypeError) as e:
                            logger.error(f"Frame {frame_id} time calculation error: frame_id={frame_id}, video_fps={video_fps}, error: {e}")
                            # Fall back to Stage 2 signal
                            position = processor.get_stage2_signal(frame_id)
                            processor.signal_source = "stage2"
                            processor.live_tracker_active = False
                            processor.current_chapter_type = position_short_name
                else:
                    # Use Stage 2 signal directly for non-BJ/HJ chapters (no video processing)
                    position = processor.get_stage2_signal(frame_id)
                    processor.signal_source = "stage2"
                    processor.live_tracker_active = False
                    processor.current_chapter_type = position_short_name
                
                # Convert to funscript action
                try:
                    timestamp_ms = int((frame_id / video_fps) * 1000)
                    pos_0_100 = int(position * 100)
                except (ValueError, TypeError) as e:
                    logger.error(f"Frame {frame_id} conversion error: frame_id={frame_id}, video_fps={video_fps}, position={position}, error: {e}")
                    continue  # Skip this frame
                
                primary_actions.append({
                    'at': timestamp_ms,
                    'pos': pos_0_100
                })
                
                processed_frames += 1
                
                # Update progress
                if progress_callback and processed_frames % 10 == 0:
                    # Calculate progress metrics
                    current_chapter_idx = segment_idx + 1
                    total_chapters = len(atr_segments_list)
                    chapter_name = position_short_name
                    current_chunk_idx = 1  # No chunks in mixed mode
                    total_chunks = 1
                    time_elapsed = time.time() - start_time
                    
                    # Use ROI tracking frames for ETA calculation when in tracking chapters
                    if is_tracking_chapter and total_roi_tracking_frames > 0:
                        # ETA based on ROI tracking frames (more accurate for BJ/HJ chapters)
                        roi_processing_fps = processed_roi_frames / max(1, time_elapsed)
                        remaining_roi_frames = total_roi_tracking_frames - processed_roi_frames
                        eta_seconds = remaining_roi_frames / max(1, roi_processing_fps)
                        processing_fps = roi_processing_fps
                    else:
                        # ETA based on total frames for non-tracking chapters
                        processing_fps = processed_frames / max(1, time_elapsed)
                        eta_seconds = (total_frames - processed_frames) / max(1, processing_fps)
                    
                    progress_callback(
                        current_chapter_idx, total_chapters, chapter_name,
                        current_chunk_idx, total_chunks,
                        processed_frames, total_frames,
                        processing_fps, time_elapsed, eta_seconds
                    )
        
        cap.release()
        
        # Create funscript object
        funscript = DualAxisFunscript()
        for action in primary_actions:
            funscript.add_action(action['at'], action['pos'])
        
        logger.info(f"Mixed Stage 3 completed: {len(primary_actions)} actions generated")
        
        # Log corruption summary if any frames were corrupted
        if hasattr(processor, '_corrupted_frame_count') and processor._corrupted_frame_count > 0:
            logger.warning(f"Data corruption detected: {processor._corrupted_frame_count} frames had invalid ROI coordinates and were skipped")
        
        # Save debug msgpack for visualization and troubleshooting
        if video_path:
            import os
            video_dir = os.path.dirname(video_path)
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            debug_msgpack_path = os.path.join(video_dir, f"{video_basename}_stage3_mixed_debug.msgpack")
            processor.save_debug_msgpack(debug_msgpack_path)
        
        return {
            "success": True,
            "primary_actions": primary_actions,
            "secondary_actions": secondary_actions,
            "funscript": funscript,
            "total_frames_processed": processed_frames,
            "processing_method": "mixed",
            "debug_data_frames": len(processor.debug_data)
        }
        
    except Exception as e:
        error_msg = f"Mixed Stage 3 error: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }
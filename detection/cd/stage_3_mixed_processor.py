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
from detection.cd.data_structures import FrameObject
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
        self.roi_tracker = None  # Lazy import to avoid circular dependency
        
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
        Get the position short name from either VideoSegment or Segment objects.
        Returns the standardized short name (e.g., 'HJ', 'BJ', 'CG/Miss', 'NR').
        """
        if hasattr(segment, 'position_short_name') and isinstance(segment.position_short_name, str):
            # VideoSegment object with valid string data
            return segment.position_short_name
        elif hasattr(segment, 'class_name') and isinstance(segment.class_name, str):
            # VideoSegment object using class_name field
            class_name = segment.class_name
            if class_name == 'Blowjob':
                return 'BJ'
            elif class_name == 'Handjob':
                return 'HJ'
            else:
                return segment.class_name
        elif hasattr(segment, 'major_position'):
            # Segment object - map major_position to short name
            position_mapping = {
                'Handjob': 'HJ',
                'Blowjob': 'BJ',
                'Cowgirl / Missionary': 'CG/Miss',
                'Not Relevant': 'NR'
            }
            return position_mapping.get(segment.major_position, 'Other')
        elif isinstance(segment, dict):
            # Dictionary segment (from SQLite)
            if 'class_name' in segment and isinstance(segment['class_name'], str):
                class_name = segment['class_name']
                if class_name == 'Blowjob':
                    return 'BJ'
                elif class_name == 'Handjob':
                    return 'HJ'
                else:
                    return class_name
            elif 'major_position' in segment and isinstance(segment['major_position'], str):
                position_mapping = {
                    'Handjob': 'HJ',
                    'Blowjob': 'BJ',
                    'Cowgirl / Missionary': 'CG/Miss',
                    'Not Relevant': 'NR'
                }
                return position_mapping.get(segment['major_position'], 'Other')
        else:
            # Debug corrupted data
            import logging
            logger = logging.getLogger(__name__)
            if hasattr(segment, 'position_short_name'):
                logger.warning(f"Corrupted position_short_name: {segment.position_short_name} (type: {type(segment.position_short_name)})")
            if hasattr(segment, 'class_name'):
                logger.warning(f"Corrupted class_name: {segment.class_name} (type: {type(segment.class_name)})")
            return 'Other'
    
    def determine_chapter_type(self, frame_id: int) -> str:
        """
        Determine the chapter type for a given frame based on Stage 2 segments.
        Returns 'BJ', 'HJ', or 'Other'
        """
        # Find the segment containing this frame
        for segment in self.stage2_segments:
            # Handle both object and dict segments
            if isinstance(segment, dict):
                start_frame = segment.get('start_frame_id', 0)
                end_frame = segment.get('end_frame_id', 0)
            else:
                start_frame = segment.start_frame_id
                end_frame = segment.end_frame_id
                
            if start_frame <= frame_id <= end_frame:
                position_short_name = self._get_segment_position_short_name(segment)
                # Debug logging for first few frames
                if frame_id < 5:  # Only log first few frames to avoid spam
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Frame {frame_id}: segment type {type(segment).__name__}, position_short_name='{position_short_name}'")
                if position_short_name in ['BJ', 'HJ']:
                    return position_short_name
                break
        return 'Other'
    
    def extract_roi_from_stage2(self, frame_id: int, frame_shape: Tuple[int, int]) -> Optional[Tuple[int, int, int, int]]:
        """
        Extract ROI from Stage 2 locked penis state for the given frame.
        Applies proper padding and ROI calculation logic matching live YOLO_ROI tracker.
        Returns (x, y, w, h) ROI coordinates or None if no valid ROI.
        """
        frame_obj = self.stage2_frame_objects.get(frame_id)
        if not frame_obj:
            return None
        
        # Check if locked penis is active and has a valid box
        if (frame_obj.locked_penis_state.active and 
            frame_obj.locked_penis_state.box):
            box = frame_obj.locked_penis_state.box
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
                    
                    # Convert from (x1, y1, x2, y2) to (x, y, w, h) format for penis box
                    x1, y1, x2, y2 = int(coords[0]), int(coords[1]), int(coords[2]), int(coords[3])
                    penis_box_xywh = (x1, y1, x2 - x1, y2 - y1)
                    
                    # Get contact boxes from Stage 2 data
                    interacting_objects = []
                    if hasattr(frame_obj, 'detected_contact_boxes') and frame_obj.detected_contact_boxes:
                        for contact_box in frame_obj.detected_contact_boxes:
                            if isinstance(contact_box, dict) and 'bbox' in contact_box:
                                bbox = contact_box['bbox']
                                if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                                    # Convert to (x, y, w, h) format
                                    bx1, by1, bx2, by2 = bbox[:4]
                                    contact_xywh = (int(bx1), int(by1), int(bx2 - bx1), int(by2 - by1))
                                    interacting_objects.append({
                                        'box': contact_xywh,
                                        'class_name': contact_box.get('class_name', 'unknown')
                                    })
                    
                    # Use the same ROI calculation logic as live YOLO_ROI tracker
                    roi_padding = getattr(self.roi_tracker, 'roi_padding', 20) if self.roi_tracker else 20
                    roi_xywh = self._calculate_combined_roi_with_padding(frame_shape, penis_box_xywh, interacting_objects, roi_padding)
                    return roi_xywh
                    
                else:
                    logging.error(f"Frame {frame_id} box is not a valid sequence: {box}")
                    return None
            except (ValueError, TypeError) as e:
                logging.error(f"Frame {frame_id} box conversion error: {e}, box: {box}")
                return None
        
        return None
    
    def _calculate_combined_roi_with_padding(self, frame_shape: Tuple[int, int], penis_box_xywh: Tuple[int, int, int, int], 
                                           interacting_objects: List[Dict], roi_padding: int) -> Tuple[int, int, int, int]:
        """
        Calculate combined ROI with padding, matching the live YOLO_ROI tracker logic.
        Returns (x, y, w, h) ROI coordinates.
        """
        # Combine penis box with all interacting objects
        entities = [penis_box_xywh] + [obj["box"] for obj in interacting_objects]
        
        # Find bounding box of all entities
        min_x = min(e[0] for e in entities)
        min_y = min(e[1] for e in entities)
        max_x_coord = max(e[0] + e[2] for e in entities)
        max_y_coord = max(e[1] + e[3] for e in entities)
        
        # Apply padding
        rx1 = max(0, min_x - roi_padding)
        ry1 = max(0, min_y - roi_padding)
        rx2 = min(frame_shape[1], max_x_coord + roi_padding)  # frame_shape is (height, width)
        ry2 = min(frame_shape[0], max_y_coord + roi_padding)
        
        rw = rx2 - rx1
        rh = ry2 - ry1
        
        # Ensure minimum ROI size (matching live tracker logic)
        min_w, min_h = 128, 128
        if rw < min_w:
            deficit = min_w - rw
            rx1 = max(0, rx1 - deficit // 2)
            rw = min_w
        if rh < min_h:
            deficit = min_h - rh
            ry1 = max(0, ry1 - deficit // 2)
            rh = min_h
        
        # Ensure ROI doesn't exceed frame boundaries
        if rx1 + rw > frame_shape[1]:
            rx1 = frame_shape[1] - rw
        if ry1 + rh > frame_shape[0]:
            ry1 = frame_shape[0] - rh
            
        return (int(rx1), int(ry1), int(rw), int(rh))
    
    def get_stage2_signal(self, frame_id: int) -> float:
        """
        Get the Stage 2 funscript signal for the given frame.
        Returns a value between 0.0 and 1.0.
        """
        # First try the lightweight signal map (preferred)
        if hasattr(self, 'stage2_signal_map') and self.stage2_signal_map:
            return self.stage2_signal_map.get(frame_id, 0.5)
        
        # Fallback: Try frame objects (legacy compatibility)
        frame_obj = self.stage2_frame_objects.get(frame_id)
        if frame_obj and hasattr(frame_obj, 'atr_funscript_distance'):
            try:
                distance = frame_obj.atr_funscript_distance
                if isinstance(distance, (bytes, str)):
                    logging.error(f"Frame {frame_id} atr_funscript_distance is not numeric: {distance} (type: {type(distance)})")
                    return 0.5
                return float(distance) / 100.0
            except (ValueError, TypeError) as e:
                logging.error(f"Frame {frame_id} signal conversion error: {e}, distance: {frame_obj.atr_funscript_distance}")
                return 0.5
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
                # Use TrackerManager API instead of hardcoded ROITracker
                from tracker.tracker_manager import TrackerManager
                self.tracker_manager = TrackerManager(mock_app, self.tracker_model_path)
                
                # Find a suitable YOLO ROI tracker dynamically
                from config.tracker_discovery import get_tracker_discovery
                discovery = get_tracker_discovery()
                yolo_roi_trackers = []
                for tracker_name in discovery.get_all_trackers():
                    info = discovery.get_tracker_info(tracker_name)
                    if info and 'yolo' in info.display_name.lower() and 'roi' in info.display_name.lower():
                        yolo_roi_trackers.append((tracker_name, info))
                
                if not yolo_roi_trackers:
                    raise RuntimeError("No YOLO ROI tracker found in discovery system")
                
                # Use the first available YOLO ROI tracker
                selected_tracker = yolo_roi_trackers[0][0]  # tracker_name
                if not self.tracker_manager.set_tracking_mode(selected_tracker):
                    raise RuntimeError(f"Failed to initialize tracker: {selected_tracker}")
                
                self.roi_tracker = self.tracker_manager._current_tracker
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
            
            if self.initialize_roi_tracker_if_needed(tracker_config, common_app_config):
                # Process frame with ROI tracker
                try:
                    self.live_tracker_active = True
                    debug_info['live_tracker_active'] = True
                    
                    # Increment frame counter (matching live tracker logic)
                    if not hasattr(self.roi_tracker, 'internal_frame_counter'):
                        self.roi_tracker.internal_frame_counter = 0
                    self.roi_tracker.internal_frame_counter += 1
                    
                    # Use the same ROI update logic as live YOLO_ROI tracker
                    frame_shape = video_frame.shape[:2]  # (height, width)
                    run_detection_this_frame = (
                        (self.roi_tracker.internal_frame_counter % self.roi_tracker.roi_update_interval == 0)
                        or (self.roi_tracker.roi is None)
                        or (not hasattr(self.roi_tracker, 'penis_last_known_box') or not self.roi_tracker.penis_last_known_box)
                    )
                    
                    # Only extract and update ROI when needed (matching live tracker)
                    if run_detection_this_frame:
                        roi = self.extract_roi_from_stage2(frame_id, frame_shape)
                        self.current_roi = roi
                        debug_info['current_roi'] = roi
                        debug_info['roi_update_frame'] = True
                        
                        if roi:
                            # Update ROI in tracker (matching live tracker ROI setting logic)
                            should_update_roi = True
                        else:
                            should_update_roi = False
                    else:
                        # Use cached ROI or current ROI from tracker
                        debug_info['roi_update_frame'] = False
                        debug_info['current_roi'] = getattr(self.roi_tracker, 'roi', self.current_roi)
                        should_update_roi = False
                        roi = self.current_roi
                    
                    # Apply ROI update if needed 
                    if should_update_roi and roi:
                        # Set ROI in tracker (matching live tracker logic)
                        self.roi_tracker.roi = roi
                        # Also set oscillation area for oscillation mode fallback
                        if hasattr(self.roi_tracker, 'set_oscillation_area'):
                            self.roi_tracker.set_oscillation_area(roi)
                        elif hasattr(self.roi_tracker, 'oscillation_area_fixed'):
                            self.roi_tracker.oscillation_area_fixed = roi
                        
                        self.last_used_roi = roi
                        debug_info['roi_updated'] = True
                    else:
                        debug_info['roi_updated'] = False
                    
                    # Use the proper YOLO ROI tracking method with Stage 2 data
                    position = self.track_frame_with_stage2_data(frame_id, video_frame)
                    
                    # Update debug info
                    debug_info['penis_box'] = getattr(self.roi_tracker, 'penis_last_known_box', None)
                    debug_info['main_interaction'] = getattr(self.roi_tracker, 'main_interaction_class', None)
                    debug_info['roi_current'] = getattr(self.roi_tracker, 'roi', roi)
                    
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
    
    def _initialize_roi_tracker(self, tracker_config: Dict[str, Any] = None):
        """Initialize ROI tracker for mixed mode processing using TrackerManager."""
        if self.roi_tracker is not None:
            return
        
        # Use default config if none provided
        if tracker_config is None:
            tracker_config = {}
        
        # Mock app object for ROI tracker initialization
        class MockApp:
            def __init__(self):
                self.logger = logging.getLogger('mixed_processor')
                # Mock app_settings for tracker
                class MockSettings:
                    def get(self, key, default=None):
                        return default
                self.app_settings = MockSettings()
        
        mock_app = MockApp()
        
        # Use TrackerManager API instead of hardcoded ROITracker
        from tracker.tracker_manager import TrackerManager
        self.tracker_manager = TrackerManager(mock_app, self.tracker_model_path)
        
        # Find a suitable YOLO ROI tracker dynamically
        from config.tracker_discovery import get_tracker_discovery
        discovery = get_tracker_discovery()
        yolo_roi_trackers = []
        for tracker_name in discovery.get_all_trackers():
            info = discovery.get_tracker_info(tracker_name)
            if info and 'yolo' in info.display_name.lower() and 'roi' in info.display_name.lower():
                yolo_roi_trackers.append((tracker_name, info))
        
        if not yolo_roi_trackers:
            raise RuntimeError("No YOLO ROI tracker found in discovery system")
        
        # Use the first available YOLO ROI tracker
        selected_tracker = yolo_roi_trackers[0][0]  # tracker_name
        if not self.tracker_manager.set_tracking_mode(selected_tracker):
            raise RuntimeError(f"Failed to initialize tracker: {selected_tracker}")
        
        self.roi_tracker = self.tracker_manager._current_tracker

    def track_frame_with_stage2_data(self, frame_id: int, video_frame: np.ndarray) -> float:
        """
        Track a single frame using Stage 2 detection data with exact same ROI logic as live tracker.
        Returns position as 0-1 float.
        """
        try:
            frame_obj = self.stage2_frame_objects.get(frame_id)
            if not frame_obj:
                return 0.5  # Default center position
            
            # Extract locked penis box and contact boxes from Stage 2 data
            penis_box = None
            contact_boxes = []
            
            # Get locked penis box - convert from (x1,y1,x2,y2) to (x,y,w,h)
            if (frame_obj.locked_penis_state.active and 
                frame_obj.locked_penis_state.box):
                box = frame_obj.locked_penis_state.box
                if isinstance(box, (list, tuple)) and len(box) >= 4:
                    try:
                        # Convert from (x1,y1,x2,y2) to (x,y,w,h) format
                        x1, y1, x2, y2 = [float(coord) for coord in box[:4]]
                        penis_box = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                    except (ValueError, TypeError):
                        pass
            
            # Get contact boxes from detected contact objects
            for contact_obj in frame_obj.detected_contact_boxes:
                if isinstance(contact_obj, dict) and 'bbox' in contact_obj:
                    bbox = contact_obj['bbox']
                    class_name = contact_obj.get('class_name', 'unknown')
                    confidence = contact_obj.get('confidence', 0.5)
                    
                    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
                        try:
                            # Convert from (x1,y1,x2,y2) to (x,y,w,h) format
                            x1, y1, x2, y2 = [float(coord) for coord in bbox[:4]]
                            contact_box = (int(x1), int(y1), int(x2-x1), int(y2-y1))
                            contact_boxes.append({
                                'box': contact_box,
                                'class_name': class_name,
                                'confidence': confidence
                            })
                        except (ValueError, TypeError):
                            continue
            
            # Initialize ROI tracker if needed
            if not self.roi_tracker:
                self._initialize_roi_tracker({})
            
            # The tracker is already properly initialized via TrackerManager
            # No need to change tracking modes - use the tracker as-is
            
            # Check if we need to update ROI (exact same logic as live tracker)
            run_detection_this_frame = (
                (self.roi_tracker.internal_frame_counter % self.roi_tracker.roi_update_interval == 0)
                or (self.roi_tracker.roi is None)
                or (not self.roi_tracker.penis_last_known_box
                    and self.roi_tracker.frames_since_target_lost < self.roi_tracker.max_frames_for_roi_persistence
                    and self.roi_tracker.internal_frame_counter % max(1, self.roi_tracker.roi_update_interval // 3) == 0)
            )
            
            # Update ROI if needed using Stage 2 data instead of YOLO detection
            if run_detection_this_frame and penis_box:
                self.roi_tracker.frames_since_target_lost = 0
                self.roi_tracker._update_penis_tracking(penis_box)
                
                # Find interacting objects from contact boxes
                interacting_objs = []
                if contact_boxes and self.roi_tracker.penis_last_known_box:
                    # Create detected objects list in format expected by _find_interacting_objects
                    detected_objects = [{'box': contact['box'], 'class_name': contact['class_name']} for contact in contact_boxes]
                    found_interactions = self.roi_tracker._find_interacting_objects(self.roi_tracker.penis_last_known_box, detected_objects)
                    interacting_objs = found_interactions
                
                # Determine main interaction class (exact same logic as live tracker)
                current_best_interaction_name = None
                if interacting_objs:
                    interacting_objs.sort(key=lambda x: self.roi_tracker.CLASS_PRIORITY.get(x["class_name"].lower(), 99))
                    current_best_interaction_name = interacting_objs[0]["class_name"].lower()
                self.roi_tracker._update_main_interaction_class(current_best_interaction_name)
                
                # Calculate combined ROI using exact same logic as live tracker
                combined_roi_candidate = self.roi_tracker._calculate_combined_roi(
                    video_frame.shape[:2], self.roi_tracker.penis_last_known_box, interacting_objs
                )
                
                # Apply VR-specific ROI width limits (exact same logic as live tracker)
                if self.roi_tracker._is_vr_video() and self.roi_tracker.penis_last_known_box:
                    penis_w = self.roi_tracker.penis_last_known_box[2]
                    rx, ry, rw, rh = combined_roi_candidate
                    new_rw = 0
                    
                    if self.roi_tracker.main_interaction_class in ["face", "hand"]:
                        new_rw = penis_w
                    else:
                        new_rw = min(rw, penis_w * 2)
                    
                    if new_rw > 0:
                        original_center_x = rx + rw / 2
                        new_rx = int(original_center_x - new_rw / 2)
                        
                        frame_width = video_frame.shape[1]
                        final_rw = int(min(new_rw, frame_width))
                        final_rx = max(0, min(new_rx, frame_width - final_rw))
                        
                        combined_roi_candidate = (final_rx, ry, final_rw, rh)
                
                # Apply ROI smoothing (exact same logic as live tracker)
                self.roi_tracker.roi = self.roi_tracker._smooth_roi_transition(combined_roi_candidate)
            else:
                # Handle missing penis detection (exact same logic as live tracker)
                if penis_box is None and self.roi_tracker.penis_last_known_box:
                    self.roi_tracker.penis_last_known_box = None
                    self.roi_tracker._update_main_interaction_class(None)
            
            # Handle ROI persistence timeout (exact same logic as live tracker)
            if not self.roi_tracker.penis_last_known_box and self.roi_tracker.roi is not None:
                self.roi_tracker.frames_since_target_lost += 1
                if self.roi_tracker.frames_since_target_lost > self.roi_tracker.max_frames_for_roi_persistence:
                    self.roi_tracker.roi = None
                    self.roi_tracker.prev_gray_main_roi = None
                    self.roi_tracker.prev_features_main_roi = None
            
            # Process the ROI content for optical flow tracking (exact same logic as live tracker)
            if self.roi_tracker.roi and self.roi_tracker.penis_last_known_box:
                processed_frame = self.roi_tracker._preprocess_frame(video_frame)
                current_frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
                
                rx, ry, rw, rh = self.roi_tracker.roi
                current_roi_patch_gray = current_frame_gray[ry:ry + rh, rx:rx + rw]
                
                primary_pos, secondary_pos, intensity, _, updated_sparse_features = self.roi_tracker.process_main_roi_content(
                    processed_frame, current_roi_patch_gray, self.roi_tracker.prev_gray_main_roi, self.roi_tracker.prev_features_main_roi
                )
                
                self.roi_tracker.prev_gray_main_roi = current_roi_patch_gray
                self.roi_tracker.prev_features_main_roi = updated_sparse_features
                
                # Update tracking state
                self.roi_tracker.internal_frame_counter += 1
                
                # Convert to 0-1 range and return
                return primary_pos / 100.0
            else:
                # No ROI available, return default
                return 0.5
                
        except Exception as e:
            logging.error(f"Error in track_frame_with_stage2_data for frame {frame_id}: {e}")
            return 0.5

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
    s2_frame_objects_map: Dict[int, FrameObject],  # Legacy compatibility
    tracker_config: Dict[str, Any],
    common_app_config: Dict[str, Any],
    progress_callback=None,
    stop_event: Optional[Event] = None,
    parent_logger: Optional[logging.Logger] = None,
    sqlite_db_path: Optional[str] = None,
    stage2_funscript=None  # New: Stage 2 funscript with signal and chapters
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
        
        # Debug: Log segment information
        logger.info(f"DEBUG: atr_segments_list has {len(atr_segments_list)} segments")
        for i, segment in enumerate(atr_segments_list):
            logger.info(f"DEBUG: Segment {i}: type={type(segment).__name__}, attributes={dir(segment) if hasattr(segment, '__dict__') else 'Not an object'}")
            if hasattr(segment, 'start_frame_id'):
                logger.info(f"DEBUG: Segment {i} frames: {segment.start_frame_id}-{segment.end_frame_id}")
            if hasattr(segment, 'position_short_name'):
                logger.info(f"DEBUG: Segment {i} position_short_name: {segment.position_short_name}")
            if hasattr(segment, 'class_name'):
                logger.info(f"DEBUG: Segment {i} class_name: {segment.class_name}")
            if hasattr(segment, 'major_position'):
                logger.info(f"DEBUG: Segment {i} major_position: {segment.major_position}")
            if isinstance(segment, dict):
                logger.info(f"DEBUG: Segment {i} dict keys: {list(segment.keys())}")
                if 'class_name' in segment:
                    logger.info(f"DEBUG: Segment {i} dict class_name: {segment['class_name']}")
        
        # Load frame objects from SQLite if not provided in memory (which is typical after Stage 2 memory optimization)
        import os
        if sqlite_db_path and os.path.exists(sqlite_db_path):
            try:
                from detection.cd.stage_2_sqlite_storage import Stage2SQLiteStorage
                storage = Stage2SQLiteStorage(sqlite_db_path, logger)
                
                # Get frame range and load frame objects
                min_frame, max_frame = storage.get_frame_range()
                if min_frame is not None and max_frame is not None:
                    frame_objects_dict = storage.get_frame_objects_range(min_frame, max_frame)
                    logger.info(f"Loaded {len(frame_objects_dict)} frame objects from SQLite database for mixed mode")
                    s2_frame_objects_map = frame_objects_dict
                else:
                    logger.warning("No frame range found in SQLite database")
                    
                # Also load segments directly from SQLite to avoid corruption issues
                import sqlite3
                try:
                    with sqlite3.connect(sqlite_db_path) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT start_frame_id, end_frame_id, major_position FROM atr_segments")
                        sqlite_segments = cursor.fetchall()
                        
                        # Create proper segment objects for chapter detection
                        corrected_segments = []
                        for start_frame, end_frame, major_position in sqlite_segments:
                            # Create a simple dict with the correct position info
                            segment_dict = {
                                'start_frame_id': start_frame,
                                'end_frame_id': end_frame,
                                'major_position': major_position,
                                'class_name': major_position  # Use major_position as class_name
                            }
                            corrected_segments.append(segment_dict)
                            logger.info(f"SQLite segment: frames {start_frame}-{end_frame}, position='{major_position}'")
                        
                        if corrected_segments:
                            logger.info(f"Loaded {len(corrected_segments)} corrected segments from SQLite")
                            # Replace the corrupted segments with corrected ones
                            atr_segments_list = corrected_segments
                            
                except Exception as e:
                    logger.warning(f"Failed to load segments directly from SQLite: {e}")
                    
                storage.close()
            except Exception as e:
                logger.error(f"Failed to load frame objects from SQLite for mixed mode: {e}")
        
        processor.set_stage2_results(s2_frame_objects_map, atr_segments_list)
        
        # Initialize results
        primary_actions = []
        secondary_actions = []
        
        # Get Stage 2 signal data - prefer funscript over heavy frame objects
        stage2_signal_map = {}
        
        logger.info(f"Mixed processor received stage2_funscript: {stage2_funscript is not None}")
        logger.info(f"Mixed processor received s2_frame_objects_map: {len(s2_frame_objects_map) if s2_frame_objects_map else 0} objects")
        
        if stage2_funscript and hasattr(stage2_funscript, 'primary_actions'):
            # Use Stage 2 funscript actions as signal source (much more efficient)
            logger.info(f"Using Stage 2 funscript with {len(stage2_funscript.primary_actions)} actions as signal source")
            
            # Convert funscript actions to frame-based signal map
            video_fps = common_app_config.get('video_fps', 30.0)
            for action in stage2_funscript.primary_actions:
                frame_id = int((action['at'] / 1000.0) * video_fps)
                stage2_signal_map[frame_id] = action['pos'] / 100.0  # Convert 0-100 to 0.0-1.0
            
            total_frames = max(stage2_signal_map.keys()) + 1 if stage2_signal_map else 0
            logger.info(f"Converted Stage 2 funscript to signal map covering {total_frames} frames")
            
        elif s2_frame_objects_map:
            # Fallback: Use frame objects if available (legacy compatibility)
            logger.info(f"Using {len(s2_frame_objects_map)} Stage 2 frame objects as signal source (legacy mode)")
            for frame_id, frame_obj in s2_frame_objects_map.items():
                if hasattr(frame_obj, 'atr_funscript_distance'):
                    stage2_signal_map[frame_id] = frame_obj.atr_funscript_distance / 100.0
            total_frames = len(s2_frame_objects_map)
            
        else:
            # Try loading from SQLite as last resort
            logger.error("No Stage 2 signal data available - neither funscript nor frame objects provided")
            return {"success": False, "error": "No Stage 2 signal data available"}
        
        if not stage2_signal_map:
            logger.error("Failed to extract Stage 2 signal data")
            return {"success": False, "error": "Failed to extract Stage 2 signal data"}
        
        # Update processor with lightweight signal data instead of heavy frame objects
        processor.stage2_signal_map = stage2_signal_map
        total_roi_tracking_frames = 0  # Frames that need ROI processing (BJ/HJ)
        processed_frames = 0
        processed_roi_frames = 0  # Frames processed with ROI tracking
        
        # Count frames that need ROI tracking (BJ/HJ segments only)
        for segment in atr_segments_list:
            position_short_name = processor._get_segment_position_short_name(segment)
            if position_short_name in ['BJ', 'HJ']:
                # Handle both dict and object segments
                if isinstance(segment, dict):
                    start_frame = segment.get('start_frame_id', 0)
                    end_frame = segment.get('end_frame_id', 0)
                else:
                    start_frame = segment.start_frame_id
                    end_frame = segment.end_frame_id
                total_roi_tracking_frames += end_frame - start_frame + 1
        
        logger.info(f"Mixed Stage 3: Processing ALL {total_frames} frames (preserving Stage 2 signal), with ROI tracking for {total_roi_tracking_frames} BJ/HJ frames")
        
        
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
        
        # Create a mapping of frame_id -> segment for efficient lookup
        frame_to_segment = {}
        bj_hj_frames = set()
        
        for segment in atr_segments_list:
            position_short_name = processor._get_segment_position_short_name(segment)
            # Handle both dict and object segments
            if isinstance(segment, dict):
                start_frame = segment.get('start_frame_id', 0)
                end_frame = segment.get('end_frame_id', 0)
            else:
                start_frame = segment.start_frame_id
                end_frame = segment.end_frame_id
                
            for frame_id in range(start_frame, end_frame + 1):
                frame_to_segment[frame_id] = (segment, position_short_name)
                if position_short_name in ['BJ', 'HJ']:
                    bj_hj_frames.add(frame_id)
        
        logger.info(f"Mapped {len(bj_hj_frames)} frames for ROI tracking (BJ/HJ chapters)")
        
        # Process ALL frames in order using the signal map
        frame_ids = sorted(stage2_signal_map.keys())
        
        for frame_id in frame_ids:
            if stop_event and stop_event.is_set():
                break
            
            # Check if this frame is in a BJ/HJ segment that needs ROI tracking
            is_tracking_frame = frame_id in bj_hj_frames
            
            if is_tracking_frame:
                # Use ROI tracking for BJ/HJ frames
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                ret, frame = cap.read()
                if not ret:
                    logger.warning(f"Could not read frame {frame_id}, falling back to Stage 2 signal")
                    position = processor.get_stage2_signal(frame_id)
                else:
                    try:
                        frame_time_ms = int((frame_id / video_fps) * 1000.0)
                        position, debug_info = processor.process_frame_mixed(
                            frame_id, frame, tracker_config, common_app_config, frame_time_ms
                        )
                        processed_roi_frames += 1
                    except (ValueError, TypeError) as e:
                        logger.error(f"Frame {frame_id} ROI processing error: {e}")
                        position = processor.get_stage2_signal(frame_id)
            else:
                # Use Stage 2 signal directly for all other frames
                position = processor.get_stage2_signal(frame_id)
                processor.signal_source = "stage2"
                processor.live_tracker_active = False
            
            # Convert to funscript action
            try:
                timestamp_ms = int((frame_id / video_fps) * 1000)
                pos_0_100 = int(position * 100)
                primary_actions.append({
                    'at': timestamp_ms,
                    'pos': pos_0_100
                })
            except (ValueError, TypeError) as e:
                logger.error(f"Frame {frame_id} action conversion error: {e}")
                continue
            
            processed_frames += 1
            
            # Update progress
            if progress_callback and processed_frames % 100 == 0:
                # Simple progress calculation for all frames
                time_elapsed = time.time() - start_time
                processing_fps = processed_frames / max(1, time_elapsed)
                eta_seconds = (total_frames - processed_frames) / max(1, processing_fps)
                
                # Get current segment info for display
                current_segment_info = frame_to_segment.get(frame_id)
                if current_segment_info:
                    _, position_short_name = current_segment_info
                    chapter_name = position_short_name
                else:
                    chapter_name = "Mixed Processing"
                
                progress_callback(
                    1, 1, chapter_name,  # Simplified progress reporting
                    processed_frames, total_frames,
                    processed_frames, total_frames,
                    processing_fps, time_elapsed, eta_seconds
                )
        
        cap.release()
        
        # Create funscript object - start with Stage 2 funscript if available
        if stage2_funscript and hasattr(stage2_funscript, 'primary_actions'):
            # Start with the Stage 2 funscript and replace BJ/HJ chapters
            funscript = DualAxisFunscript()
            
            # First, identify time ranges for BJ/HJ chapters to remove
            bj_hj_time_ranges = []
            for segment in atr_segments_list:
                position_short_name = processor._get_segment_position_short_name(segment)
                if position_short_name in ['BJ', 'HJ']:
                    # Handle both dict and object segments
                    if isinstance(segment, dict):
                        start_frame = segment.get('start_frame_id', 0)
                        end_frame = segment.get('end_frame_id', 0)
                    else:
                        start_frame = segment.start_frame_id
                        end_frame = segment.end_frame_id
                    start_time_ms = int((start_frame / video_fps) * 1000)
                    end_time_ms = int((end_frame / video_fps) * 1000)
                    bj_hj_time_ranges.append((start_time_ms, end_time_ms))
            
            logger.info(f"Removing Stage 2 signal from {len(bj_hj_time_ranges)} BJ/HJ chapters")
            
            # Add Stage 2 actions, excluding BJ/HJ time ranges
            actions_kept = 0
            actions_removed = 0
            for action in stage2_funscript.primary_actions:
                action_time = action['at']
                should_keep = True
                for start_time, end_time in bj_hj_time_ranges:
                    if start_time <= action_time <= end_time:
                        should_keep = False
                        actions_removed += 1
                        break
                if should_keep:
                    funscript.add_action(action_time, action['pos'])
                    actions_kept += 1
            
            logger.info(f"Kept {actions_kept} actions from Stage 2, removed {actions_removed} actions from BJ/HJ chapters")
            
            # Add new tracked actions for BJ/HJ chapters
            tracked_actions_added = 0
            for action in primary_actions:
                action_time = action['at']
                # Only add if it's within a BJ/HJ time range
                for start_time, end_time in bj_hj_time_ranges:
                    if start_time <= action_time <= end_time:
                        funscript.add_action(action_time, action['pos'])
                        tracked_actions_added += 1
                        break
            
            logger.info(f"Added {tracked_actions_added} new tracked actions for BJ/HJ chapters")
            
            # Preserve chapters from Stage 2 funscript
            if hasattr(stage2_funscript, 'chapters') and stage2_funscript.chapters:
                funscript.chapters = stage2_funscript.chapters.copy()
                logger.info(f"Preserved {len(funscript.chapters)} chapters from Stage 2")
            
        else:
            # Fallback: Create new funscript from scratch
            funscript = DualAxisFunscript()
            for action in primary_actions:
                funscript.add_action(action['at'], action['pos'])
            
            # Set chapters from ATR segments
            if atr_segments_list:
                funscript.set_chapters_from_segments(atr_segments_list, video_fps)
                
        logger.info(f"Mixed Stage 3 completed: {len(funscript.primary_actions)} total actions with {len(funscript.chapters)} chapters")
        
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
            "debug_data_frames": len(processor.debug_data),
            "video_segments": [seg.to_dict() if hasattr(seg, 'to_dict') else (seg.__dict__ if hasattr(seg, '__dict__') else seg) for seg in atr_segments_list]
        }
        
    except Exception as e:
        error_msg = f"Mixed Stage 3 error: {str(e)}"
        logger.error(error_msg)
        return {
            "success": False,
            "error": error_msg
        }
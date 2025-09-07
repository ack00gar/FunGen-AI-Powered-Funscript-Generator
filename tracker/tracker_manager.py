"""
Tracker manager that directly interfaces with modular trackers.
Replaces ModularTrackerBridge with clean, scalable architecture.
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union

from config.tracker_discovery import get_tracker_discovery
from tracker_modules import tracker_registry
from funscript.dual_axis_funscript import DualAxisFunscript


class TrackerManager:
    """
    Native modular tracker manager with direct instantiation.
    No bridge layers - direct communication between GUI and trackers.
    """
    
    def __init__(self, app_logic_instance: Optional[Any], tracker_model_path: str):
        self.app = app_logic_instance
        self.tracker_model_path = tracker_model_path
        
        # Set up logger
        if app_logic_instance and hasattr(app_logic_instance, 'logger'):
            self.logger = app_logic_instance.logger
        else:
            self.logger = logging.getLogger('NativeTrackerManager')
            
        # Current tracker instance and metadata
        self._current_tracker = None
        self._current_mode = None
        self._tracker_info = None
        self._discovery = get_tracker_discovery()
        
        # Create funscript instance for accumulating tracking data
        self.funscript = DualAxisFunscript(logger=self.logger)
        
        # Tracking state
        self.tracking_active = False
        self.current_fps = 0.0
        
        # Pending configurations (applied when tracker is instantiated)
        self._pending_axis_A = None
        self._pending_axis_B = None
        self._pending_user_roi = None
        self._pending_user_point = None
        
        # UI visualization state (for GUI compatibility)
        self.show_all_boxes = False
        self.show_flow = False
        self.show_stats = False
        self.show_funscript_preview = False
        self.show_masks = False
        self.show_roi = False
        self.show_grid_blocks = False
        
        # Current ROI for visualization overlay
        self.roi = None
        
        # Additional GUI compatibility attributes that were in the old bridge
        self.oscillation_area_fixed = None  # Should be None or (x, y, w, h) tuple
        self.user_roi_fixed = None  # Should be None or (x, y, w, h) tuple
        self.main_interaction_class = None
        self.confidence_threshold = 0.7
        
        # Model paths for GUI compatibility (set by control panel when models change)
        self.det_model_path = self.tracker_model_path  # Detection model path
        self.pose_model_path = None  # Pose model path (if used)
        
        # Live tracker GUI compatibility attributes
        self.enable_inversion_detection = False  # Motion mode feature
        self.motion_mode = "normal"  # Motion mode state
        self.roi_padding = 50
        self.roi_update_interval = 10
        self.roi_smoothing_factor = 0.1
        self.max_frames_for_roi_persistence = 30
        self.use_sparse_flow = False
        self.sensitivity = 1.0
        self.base_amplification_factor = 1.0
        self.class_specific_amplification_multipliers = {}
        self.flow_history_window_smooth = 10
        self.y_offset = 0  # Y-axis offset for positioning
        self.x_offset = 0  # X-axis offset for positioning  
        self.output_delay_frames = 0  # Frame delay compensation
        self.current_video_fps_for_delay = 30.0  # FPS for delay calculations
        self.internal_frame_counter = 0  # Frame counter for processing
        
        # Additional properties that modular trackers might expect
        self.oscillation_history = {}  # Dictionary for oscillation trackers
        self.user_roi_current_flow_vector = (0.0, 0.0)  # For user ROI trackers
        self.user_roi_initial_point_relative = None
        self.user_roi_tracked_point_relative = None
        
        # More oscillation tracker properties
        self.oscillation_cell_persistence = {}  # Dictionary for cell persistence
        self._gray_full_buffer = None  # Gray frame buffer
        self.prev_gray = None  # Previous gray frame
        self.prev_gray_oscillation = None  # Previous gray frame for oscillation detection
        self.grid_size = (8, 8)  # Grid size for oscillation detection
        self.oscillation_grid_size = 8  # Integer for compatibility
        self.oscillation_threshold = 0.5  # Oscillation detection threshold
        self.initialized = False  # Tracker initialization status
        
        self.logger.info("TrackerManager initialized - Direct modular tracker interface")

    def set_tracking_mode(self, mode_name: str) -> bool:
        """Set tracking mode with direct tracker instantiation."""
        try:
            if mode_name == self._current_mode and self._current_tracker:
                self.logger.debug(f"Already using tracker mode: {mode_name}")
                return True
                
            # Clean up previous tracker
            self._cleanup_current_tracker()
            
            # Get tracker info and class
            tracker_info = self._discovery.get_tracker_info(mode_name)
            if not tracker_info:
                self.logger.error(f"Unknown tracker mode: {mode_name}")
                return False
                
            tracker_class = tracker_registry.get_tracker(mode_name)
            if not tracker_class:
                self.logger.error(f"Could not load tracker class for: {mode_name}")
                return False
                
            # Direct instantiation - no bridge layer
            self._current_tracker = tracker_class()
            self._current_mode = mode_name
            self._tracker_info = tracker_info
            
            # Set up tracker with app and model path
            self._setup_tracker_environment()
            
            # Initialize tracker
            if not self._initialize_tracker():
                return False
                
            # Apply any pending configurations
            self._apply_pending_configurations()
            
            self.logger.info(f"Native tracker instantiated: {mode_name} ({tracker_info.display_name})")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set tracking mode {mode_name}: {e}")
            import traceback
            traceback.print_exc()
            return False

    def start_tracking(self) -> bool:
        """Start tracking with direct tracker call."""
        if not self._current_tracker:
            self.logger.error("No tracker set - call set_tracking_mode() first")
            return False
            
        try:
            self.tracking_active = True
            if hasattr(self._current_tracker, 'start_tracking'):
                result = self._current_tracker.start_tracking()
                # Handle different return types
                return result if isinstance(result, bool) else True
            return True
        except Exception as e:
            self.logger.error(f"Failed to start tracking: {e}")
            self.tracking_active = False
            return False

    def stop_tracking(self):
        """Stop tracking with direct tracker call."""
        if not self._current_tracker:
            return
            
        try:
            self.tracking_active = False
            if hasattr(self._current_tracker, 'stop_tracking'):
                self._current_tracker.stop_tracking()
            elif hasattr(self._current_tracker, 'cleanup'):
                self._current_tracker.cleanup()
        except Exception as e:
            self.logger.error(f"Failed to stop tracking: {e}")

    def process_frame(self, frame: np.ndarray, frame_time_ms: int, 
                     frame_index: Optional[int] = None, 
                     min_write_frame_id: Optional[int] = None) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """Process frame with direct tracker call."""
        if not self._current_tracker:
            self.logger.error("No tracker set for process_frame")
            return frame, None
            
        try:
            # Ensure frame is writable for OpenCV operations
            if not frame.flags.writeable:
                frame = frame.copy()
                
            # Direct call to modular tracker
            result = self._current_tracker.process_frame(frame, frame_time_ms, frame_index)
            
            # Handle TrackerResult object or tuple format
            processed_frame, action_log = self._extract_result_data(result, frame)
            
            # Add actions to funscript
            self._add_actions_to_funscript(action_log)
            
            # Update visualization state
            self._update_visualization_state()
            
            return processed_frame, action_log
            
        except Exception as e:
            self.logger.error(f"Error in process_frame with tracker {self._current_mode}: {e}")
            return frame, None

    def process_frame_for_oscillation(self, frame: np.ndarray, frame_time_ms: int, 
                                    frame_index: Optional[int] = None) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """Process frame for oscillation detection - delegates to current tracker."""
        if not self._current_tracker:
            self.logger.error("No tracker set - call set_tracking_mode() first")
            return frame, None
            
        try:
            # Ensure frame is writable for OpenCV operations
            if not frame.flags.writeable:
                frame = frame.copy()
            
            # Try to use the tracker's process_frame method
            result = self._current_tracker.process_frame(frame, frame_time_ms, frame_index)
            
            # Handle TrackerResult object or tuple format
            processed_frame, action_log = self._extract_result_data(result, frame)
            
            # For oscillation trackers, we need to sample positions periodically
            if 'oscillation' in self._current_mode.lower():
                # Oscillation trackers maintain continuous position, sample it
                if hasattr(self._current_tracker, 'oscillation_funscript_pos'):
                    position = self._current_tracker.oscillation_funscript_pos
                    
                    # Only add action if position changed or enough time has passed
                    last_action = self.funscript.primary_actions[-1] if self.funscript.primary_actions else None
                    add_action = False
                    
                    if last_action is None:
                        # First action
                        add_action = True
                    elif position != last_action['pos']:
                        # Position changed
                        add_action = True
                    elif frame_time_ms - last_action['at'] >= 100:
                        # At least 100ms since last action (10 Hz sampling)
                        add_action = True
                    
                    if add_action and self.funscript and position is not None:
                        self.funscript.add_action(frame_time_ms, position)
                        # Create action_log for compatibility
                        action_log = [{'at': frame_time_ms, 'pos': position}]
            else:
                # Regular trackers use action_log
                self._add_actions_to_funscript(action_log)
            
            return processed_frame, action_log
                
        except Exception as e:
            self.logger.error(f"Error in process_frame_for_oscillation: {e}")
            return frame, None

    def reset(self, reason: Optional[str] = None, **kwargs):
        """Reset tracker with direct call."""
        if not self._current_tracker:
            return
            
        try:
            if hasattr(self._current_tracker, 'reset'):
                # Try with parameters first, fallback to no parameters
                try:
                    self._current_tracker.reset(reason=reason, **kwargs)
                except TypeError:
                    self._current_tracker.reset()
        except Exception as e:
            self.logger.error(f"Failed to reset tracker: {e}")

    def cleanup(self):
        """Clean up current tracker and manager state."""
        self._cleanup_current_tracker()
        self.funscript = DualAxisFunscript(logger=self.logger)
        self.tracking_active = False
    
    def update_tracker_settings(self, **kwargs) -> bool:
        """Update current tracker settings dynamically."""
        if not self._current_tracker:
            self.logger.debug("No current tracker to update settings")
            return False
            
        if hasattr(self._current_tracker, 'update_settings'):
            try:
                result = self._current_tracker.update_settings(**kwargs)
                if result:
                    self.logger.debug(f"Tracker settings updated successfully")
                else:
                    self.logger.warning("Tracker settings update failed")
                return result
            except Exception as e:
                self.logger.error(f"Error updating tracker settings: {e}")
                return False
        else:
            self.logger.debug(f"Tracker {type(self._current_tracker).__name__} does not support dynamic settings updates")
            return False

    # Configuration methods with direct tracker interface
    def set_user_defined_roi_and_point(self, roi_abs_coords: Tuple[int, int, int, int], 
                                     point_abs_coords_in_frame: Tuple[int, int], 
                                     current_frame_for_patch: Optional[np.ndarray] = None) -> bool:
        """Set user-defined ROI and point with direct tracker call."""
        if self._current_tracker and hasattr(self._current_tracker, 'set_user_defined_roi_and_point'):
            try:
                result = self._current_tracker.set_user_defined_roi_and_point(
                    roi_abs_coords, point_abs_coords_in_frame, current_frame_for_patch
                )
                if result:
                    self.logger.info(f"✅ User ROI set: ROI={roi_abs_coords}, Point={point_abs_coords_in_frame}")
                    # Sync manager state for GUI compatibility
                    self.user_roi_fixed = roi_abs_coords
                    # Calculate relative point in ROI coordinates
                    x_rel = point_abs_coords_in_frame[0] - roi_abs_coords[0] 
                    y_rel = point_abs_coords_in_frame[1] - roi_abs_coords[1]
                    self.user_roi_initial_point_relative = (x_rel, y_rel)
                    self.user_roi_tracked_point_relative = (x_rel, y_rel)
                else:
                    self.logger.warning("❌ Tracker rejected user ROI setting")
                return result
            except Exception as e:
                self.logger.error(f"Error setting user ROI: {e}")
                return False
        else:
            # Store for later application
            self._pending_user_roi = roi_abs_coords
            self._pending_user_point = point_abs_coords_in_frame
            self.logger.info(f"Stored pending user ROI: {roi_abs_coords}, {point_abs_coords_in_frame}")
            return True

    def set_axis(self, point_a: Tuple[int, int], point_b: Tuple[int, int]) -> bool:
        """Set axis points with direct tracker call."""
        if self._current_tracker and hasattr(self._current_tracker, 'set_axis'):
            try:
                result = self._current_tracker.set_axis(point_a, point_b)
                self.logger.info(f"✅ Axis set: A={point_a}, B={point_b}")
                return result
            except Exception as e:
                self.logger.error(f"Error setting axis: {e}")
                return False
        else:
            # Store for later application
            self._pending_axis_A = point_a
            self._pending_axis_B = point_b
            self.logger.info(f"Stored pending axis: A={point_a}, B={point_b}")
            return True

    def clear_user_defined_roi_and_point(self):
        """Clear user ROI with direct tracker call."""
        self._pending_user_roi = None
        self._pending_user_point = None
        if self._current_tracker and hasattr(self._current_tracker, 'clear_user_defined_roi_and_point'):
            self._current_tracker.clear_user_defined_roi_and_point()

    def clear_oscillation_area_and_point(self):
        """Clear oscillation area with direct tracker call."""
        self.oscillation_area_fixed = None
        if self._current_tracker and hasattr(self._current_tracker, 'clear_oscillation_area_and_point'):
            self._current_tracker.clear_oscillation_area_and_point()

    def set_oscillation_area_and_point(self, area_rect_video_coords, point_video_coords, current_frame):
        """Set oscillation area and point - delegates to current tracker."""
        if self._current_tracker and hasattr(self._current_tracker, 'set_oscillation_area_and_point'):
            self._current_tracker.set_oscillation_area_and_point(area_rect_video_coords, point_video_coords, current_frame)
        elif self._current_tracker and hasattr(self._current_tracker, 'set_user_defined_roi_and_point'):
            # Fallback for trackers that use the user-defined ROI method
            self._current_tracker.set_user_defined_roi_and_point(area_rect_video_coords, point_video_coords, current_frame)
        else:
            self.logger.warning(f"Current tracker {self._current_mode} does not support setting oscillation area")

    def set_oscillation_area(self, area_rect_video_coords):
        """Set oscillation area only (no point needed) - delegates to current tracker."""
        if self._current_tracker and hasattr(self._current_tracker, 'set_oscillation_area'):
            self._current_tracker.set_oscillation_area(area_rect_video_coords)
        elif self._current_tracker and hasattr(self._current_tracker, 'set_roi'):
            # Fallback for trackers that use set_roi method
            self._current_tracker.set_roi(area_rect_video_coords)
        else:
            self.logger.warning(f"Current tracker {self._current_mode} does not support setting oscillation area")

    # Advanced configuration methods
    def update_dis_flow_config(self, preset=None, finest_scale=None):
        """Update optical flow configuration."""
        if self._current_tracker and hasattr(self._current_tracker, 'update_dis_flow_config'):
            self._current_tracker.update_dis_flow_config(preset=preset, finest_scale=finest_scale)

    def update_oscillation_grid_size(self):
        """Update oscillation detection grid size."""
        if self._current_tracker and hasattr(self._current_tracker, 'update_oscillation_grid_size'):
            self._current_tracker.update_oscillation_grid_size()

    def update_oscillation_sensitivity(self):
        """Update oscillation detection sensitivity."""
        if self._current_tracker and hasattr(self._current_tracker, 'update_oscillation_sensitivity'):
            self._current_tracker.update_oscillation_sensitivity()

    def _load_models(self):
        """Reload models in current tracker after model paths change."""
        if not self._current_tracker:
            self.logger.debug("No current tracker to reload models for")
            return
        
        try:
            # Try to reinitialize the tracker if it supports model reloading
            if hasattr(self._current_tracker, '_load_models'):
                self._current_tracker._load_models()
                self.logger.info(f"Models reloaded for tracker {self._current_mode}")
            elif hasattr(self._current_tracker, 'reinitialize'):
                self._current_tracker.reinitialize()
                self.logger.info(f"Tracker {self._current_mode} reinitialized after model path change")
            elif hasattr(self._current_tracker, 'initialize'):
                # Fallback: reinitialize the tracker
                result = self._current_tracker.initialize(self.app)
                if result:
                    self.logger.info(f"Tracker {self._current_mode} reinitialized successfully")
                else:
                    self.logger.warning(f"Tracker {self._current_mode} reinitialization failed")
            else:
                self.logger.info(f"Tracker {self._current_mode} does not support model reloading")
        except Exception as e:
            self.logger.error(f"Error reloading models for tracker {self._current_mode}: {e}")

    def _is_vr_video(self) -> bool:
        """Check if current video is VR format."""
        # First, try to delegate to current tracker if it has the method
        if self._current_tracker and hasattr(self._current_tracker, '_is_vr_video'):
            try:
                return self._current_tracker._is_vr_video()
            except Exception as e:
                self.logger.warning(f"Error calling tracker's _is_vr_video: {e}")
        
        # Fallback implementation using app video dimensions
        try:
            if self.app and hasattr(self.app, 'get_video_dimensions'):
                width, height = self.app.get_video_dimensions()
                if width and height:
                    aspect_ratio = width / height
                    # VR videos typically have aspect ratios >= 1.8
                    is_vr = aspect_ratio >= 1.8
                    self.logger.debug(f"VR detection: {width}x{height} (ratio {aspect_ratio:.2f}) -> {'VR' if is_vr else 'standard'}")
                    return is_vr
            
            # Try alternative method using processor
            if self.app and hasattr(self.app, 'processor') and self.app.processor:
                width = getattr(self.app.processor, 'frame_width', None)
                height = getattr(self.app.processor, 'frame_height', None)
                if width and height:
                    aspect_ratio = width / height
                    is_vr = aspect_ratio >= 1.8
                    self.logger.debug(f"VR detection (processor): {width}x{height} (ratio {aspect_ratio:.2f}) -> {'VR' if is_vr else 'standard'}")
                    return is_vr
        except Exception as e:
            self.logger.warning(f"Error in VR video detection: {e}")
        
        # Default to non-VR if detection fails
        return False

    # Getters for current state
    def get_current_tracker_name(self) -> Optional[str]:
        """Get current tracker mode name."""
        return self._current_mode

    def get_current_tracker(self):
        """Get current tracker instance."""
        return self._current_tracker

    def get_tracker_info(self):
        """Get current tracker metadata."""
        return self._tracker_info

    def is_tracking_active(self) -> bool:
        """Check if tracking is currently active."""
        return self.tracking_active and self._current_tracker is not None

    # Private implementation methods
    def _cleanup_current_tracker(self):
        """Clean up current tracker instance."""
        if self._current_tracker and hasattr(self._current_tracker, 'cleanup'):
            try:
                tracker_name = getattr(self._tracker_info, 'display_name', 'Unknown') if self._tracker_info else 'Unknown'
                self._current_tracker.cleanup()
                self.logger.debug(f"Tracker cleaned up: {tracker_name}")
            except Exception as e:
                self.logger.error(f"Error cleaning up tracker: {e}")
        
        self._current_tracker = None
        self._current_mode = None
        self._tracker_info = None

    def _setup_tracker_environment(self):
        """Set up tracker environment with app context."""
        if not self._current_tracker:
            return
            
        # Set essential attributes
        self._current_tracker.app = self.app
        self._current_tracker.model_path = self.tracker_model_path
        self._current_tracker.logger = self.logger
        
        # Provide compatibility attributes for trackers
        self._provide_tracker_compatibility_attributes()

    def _initialize_tracker(self) -> bool:
        """Initialize tracker with error handling."""
        if not self._current_tracker:
            return False
            
        try:
            if hasattr(self._current_tracker, 'initialize'):
                init_result = self._current_tracker.initialize(self.app)
                if isinstance(init_result, bool) and not init_result:
                    self.logger.error(f"Tracker {self._current_mode} initialization failed")
                    return False
            return True
        except Exception as e:
            self.logger.error(f"Error initializing tracker {self._current_mode}: {e}")
            return False

    def _apply_pending_configurations(self):
        """Apply any pending configurations to the tracker."""
        if not self._current_tracker:
            return
            
        # Apply pending axis settings
        if self._pending_axis_A is not None and self._pending_axis_B is not None:
            if hasattr(self._current_tracker, 'set_axis'):
                try:
                    result = self._current_tracker.set_axis(self._pending_axis_A, self._pending_axis_B)
                    self.logger.info(f"Applied pending axis: A={self._pending_axis_A}, B={self._pending_axis_B}, result={result}")
                except Exception as e:
                    self.logger.error(f"Error applying pending axis: {e}")
            self._pending_axis_A = None
            self._pending_axis_B = None
        
        # Apply pending user ROI settings
        if self._pending_user_roi is not None and self._pending_user_point is not None:
            if hasattr(self._current_tracker, 'set_user_defined_roi_and_point'):
                try:
                    result = self._current_tracker.set_user_defined_roi_and_point(
                        self._pending_user_roi, self._pending_user_point, None
                    )
                    self.logger.info(f"Applied pending user ROI: ROI={self._pending_user_roi}, Point={self._pending_user_point}, result={result}")
                except Exception as e:
                    self.logger.error(f"Error applying pending user ROI: {e}")
            self._pending_user_roi = None
            self._pending_user_point = None

    def _extract_result_data(self, result, original_frame) -> Tuple[np.ndarray, Optional[List[Dict]]]:
        """Extract processed frame and action log from tracker result."""
        # Handle TrackerResult object
        if hasattr(result, 'processed_frame') and hasattr(result, 'action_log'):
            return result.processed_frame, result.action_log
        
        # Handle tuple format
        elif isinstance(result, tuple) and len(result) >= 2:
            processed_frame, action_log = result[0], result[1]
            return processed_frame, action_log
        
        # Handle single frame return
        elif isinstance(result, np.ndarray):
            return result, None
        
        # Fallback
        else:
            self.logger.warning(f"Unexpected tracker result format: {type(result)}")
            return original_frame, None

    def _add_actions_to_funscript(self, action_log: Optional[List[Dict]]):
        """Add action log entries to the funscript."""
        if not action_log or not self.funscript:
            return
            
        try:
            for action in action_log:
                if isinstance(action, dict) and 'at' in action and 'pos' in action:
                    timestamp_ms = action['at']
                    position = action['pos']
                    self.funscript.add_action(timestamp_ms, position)
        except Exception as e:
            self.logger.error(f"Error adding actions to funscript: {e}")

    def _update_visualization_state(self):
        """Update visualization state from current tracker."""
        if not self._current_tracker:
            return
            
        # Update ROI for visualization overlay
        if hasattr(self._current_tracker, 'roi'):
            self.roi = getattr(self._current_tracker, 'roi', None)
        
        # Update FPS if available
        if hasattr(self._current_tracker, 'current_fps'):
            self.current_fps = getattr(self._current_tracker, 'current_fps', 0.0)
        
        # Update live tracker GUI attributes for motion mode overlay
        if hasattr(self._current_tracker, 'enable_inversion_detection'):
            self.enable_inversion_detection = getattr(self._current_tracker, 'enable_inversion_detection', False)
        if hasattr(self._current_tracker, 'motion_mode'):
            self.motion_mode = getattr(self._current_tracker, 'motion_mode', 'normal')
        if hasattr(self._current_tracker, 'main_interaction_class'):
            self.main_interaction_class = getattr(self._current_tracker, 'main_interaction_class', None)

    def _provide_tracker_compatibility_attributes(self):
        """Provide attributes that modular trackers might expect from the old ROITracker."""
        if not self._current_tracker:
            return
            
        # Copy manager properties to the tracker instance so it can access them
        # IMPORTANT: Only set attributes that don't already exist to avoid overwriting tracker's own attributes
        compatibility_attrs = {
            'oscillation_history': self.oscillation_history,
            'oscillation_area_fixed': self.oscillation_area_fixed,
            'oscillation_cell_persistence': self.oscillation_cell_persistence,
            '_gray_full_buffer': self._gray_full_buffer,
            'prev_gray': self.prev_gray,
            'prev_gray_oscillation': self.prev_gray_oscillation,
            'grid_size': self.grid_size,
            'oscillation_grid_size': self.oscillation_grid_size,
            'oscillation_threshold': self.oscillation_threshold,
            'user_roi_fixed': self.user_roi_fixed,
            'user_roi_current_flow_vector': self.user_roi_current_flow_vector,
            'user_roi_initial_point_relative': self.user_roi_initial_point_relative,
            'user_roi_tracked_point_relative': self.user_roi_tracked_point_relative,
            'roi': self.roi,
            'sensitivity': self.sensitivity,
            'confidence_threshold': self.confidence_threshold,
            'show_all_boxes': self.show_all_boxes,
            'show_flow': self.show_flow,
            'show_stats': self.show_stats,
            'funscript': self.funscript,
            'initialized': self.initialized
        }
        
        for attr_name, attr_value in compatibility_attrs.items():
            # Only set if not already present in the tracker
            if not hasattr(self._current_tracker, attr_name):
                setattr(self._current_tracker, attr_name, attr_value)
            # Special case: for dictionary attributes, only set if they're None or not initialized
            elif attr_name in ['oscillation_history', 'oscillation_cell_persistence'] and hasattr(self._current_tracker, attr_name):
                current_val = getattr(self._current_tracker, attr_name)
                # Only override if the tracker's value is None or not a dict
                if current_val is None or not isinstance(current_val, dict):
                    setattr(self._current_tracker, attr_name, attr_value)


# Factory function for creating manager instances
def create_tracker_manager(app_logic_instance: Optional[Any], 
                          tracker_model_path: str) -> TrackerManager:
    """Factory function to create tracker manager instances."""
    return TrackerManager(app_logic_instance, tracker_model_path)
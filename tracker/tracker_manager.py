"""
Tracker manager that directly interfaces with modular trackers.
Replaces ModularTrackerBridge with clean, scalable architecture.
"""
import logging
import numpy as np
from typing import List, Dict, Tuple, Optional, Any, Union

from tracker.tracker_modules import tracker_registry
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
        from config.tracker_discovery import get_tracker_discovery
        self._discovery = get_tracker_discovery()
        
        # Create funscript instance for accumulating tracking data
        self.funscript = DualAxisFunscript(logger=self.logger)

        # Apply point simplification setting from app settings
        if app_logic_instance and hasattr(app_logic_instance, 'app_settings'):
            simplification_enabled = app_logic_instance.app_settings.get('funscript_point_simplification_enabled', True)
            self.funscript.enable_point_simplification = simplification_enabled
        
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
        
        # Device control integration (subscriber feature)
        self.device_bridge = None
        self.live_device_control_enabled = False  # User toggle
        self._init_device_bridge()
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

        # Rolling Ultimate Autotune for live tracking (streamer mode)
        # Load from settings if app instance is available (disabled by default, requires streamer + connected session)
        if app_logic_instance and hasattr(app_logic_instance, 'app_settings'):
            self.rolling_autotune_enabled = app_logic_instance.app_settings.get("live_tracker_rolling_autotune_enabled", False)
            self.rolling_autotune_interval_ms = app_logic_instance.app_settings.get("live_tracker_rolling_autotune_interval_ms", 5000)
            self.rolling_autotune_window_ms = app_logic_instance.app_settings.get("live_tracker_rolling_autotune_window_ms", 5000)
        else:
            self.rolling_autotune_enabled = False  # Disabled by default - requires streamer with connected session
            self.rolling_autotune_interval_ms = 5000  # Apply autotune every 5 seconds
            self.rolling_autotune_window_ms = 5000  # Process last 5 seconds of data
        self.rolling_autotune_last_time = 0  # Last time autotune was applied

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

            # Log final point simplification summary
            if self.funscript and hasattr(self.funscript, 'log_final_simplification_summary'):
                self.funscript.log_final_simplification_summary()
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

            # Apply rolling autotune if enabled (for streamer mode)
            if (self.rolling_autotune_enabled and
                frame_time_ms - self.rolling_autotune_last_time >= self.rolling_autotune_interval_ms):
                self._apply_rolling_autotune(frame_time_ms)
                self.rolling_autotune_last_time = frame_time_ms

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

        # Reapply point simplification setting
        if self.app and hasattr(self.app, 'app_settings'):
            simplification_enabled = self.app.app_settings.get('funscript_point_simplification_enabled', True)
            self.funscript.enable_point_simplification = simplification_enabled

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

    def unload_detection_model(self):
        """Unloads the detection model."""
        self.logger.info("Unloading detection model.")
        self.det_model_path = None
        if self._current_tracker:
            if hasattr(self._current_tracker, 'det_model_path'):
                self._current_tracker.det_model_path = None
            self._load_models()

    def unload_pose_model(self):
        """Unloads the pose model."""
        self.logger.info("Unloading pose model.")
        self.pose_model_path = None
        if self._current_tracker:
            if hasattr(self._current_tracker, 'pose_model_path'):
                self._current_tracker.pose_model_path = None
            self._load_models()

    def unload_models():
        """Unloads models from the current tracker by cleaning up the tracker."""
        self.logger.info("Unloading models by cleaning up the current tracker.")
        self._cleanup_current_tracker()

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
            # Yield before potentially heavy initialization (YOLO model loading)
            import time
            time.sleep(0.001)  # 1ms yield - forces GIL release

            if hasattr(self._current_tracker, 'initialize'):
                init_result = self._current_tracker.initialize(self.app, tracker_model_path=self.tracker_model_path)
                if isinstance(init_result, bool) and not init_result:
                    self.logger.error(f"Tracker {self._current_mode} initialization failed")
                    return False

            # Yield after initialization completes
            time.sleep(0.001)  # 1ms yield - forces GIL release
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
        processed_frame = None
        action_log = None
        
        # Handle TrackerResult object
        if hasattr(result, 'processed_frame') and hasattr(result, 'action_log'):
            processed_frame, action_log = result.processed_frame, result.action_log
        
        # Handle tuple format
        elif isinstance(result, tuple) and len(result) >= 2:
            processed_frame, action_log = result[0], result[1]
        
        # Handle single frame return
        elif isinstance(result, np.ndarray):
            processed_frame, action_log = result, None
        
        # Fallback
        else:
            self.logger.warning(f"Unexpected tracker result format: {type(result)}")
            processed_frame, action_log = original_frame, None
        
        # Send to device control if available and enabled
        if action_log and len(action_log) > 0:
            self.logger.debug(f"Attempting to send action to device control: {action_log[-1]}")
            self._send_to_device_control(action_log[-1])  # Send latest action
        else:
            self.logger.debug(f"No action log to send to device control: action_log={action_log}")
        
        return processed_frame, action_log

    def _add_actions_to_funscript(self, action_log: Optional[List[Dict]]):
        """Add action log entries to the funscript, skipping 'Not Relevant' category chapters."""
        if not action_log or not self.funscript:
            return

        # Check if we're in a "Not Relevant" category chapter - if so, skip scripting
        if hasattr(self, 'app') and self.app:
            try:
                from config.constants import POSITION_INFO_MAPPING
                fs_proc = getattr(self.app, 'funscript_processor', None)
                processor = getattr(self.app, 'processor', None)

                if fs_proc and processor:
                    current_frame = processor.current_frame_index
                    chapter_at_frame = fs_proc.get_chapter_at_frame(current_frame)

                    # Determine category based on position_short_name (reliable for old and new chapters)
                    if chapter_at_frame:
                        position_short_name = chapter_at_frame.position_short_name
                        position_info = POSITION_INFO_MAPPING.get(position_short_name, {})
                        category = position_info.get('category', 'Position')  # Default to Position if not in mapping

                        self.logger.debug(f"Frame {current_frame}: Chapter '{position_short_name}', Category '{category}', Scripting: {category != 'Not Relevant'}")

                        if category == "Not Relevant":
                            return  # Not Relevant category = don't script
                    else:
                        self.logger.debug(f"Frame {current_frame}: No chapter, Scripting: YES")
                    # Otherwise (no chapter or Position category) = continue scripting
            except Exception as e:
                self.logger.warning(f"Could not check chapter type for scripting: {e}")
                # If we can't determine, continue adding actions (fail open)

        try:
            for action in action_log:
                if isinstance(action, dict) and 'at' in action and 'pos' in action:
                    timestamp_ms = action['at']
                    position = action['pos']
                    self.funscript.add_action(timestamp_ms, position)
        except Exception as e:
            self.logger.error(f"Error adding actions to funscript: {e}")

    def _apply_rolling_autotune(self, current_time_ms: int):
        """
        Apply Ultimate Autotune to the last N seconds of funscript data.
        This creates a rolling window of cleaned-up data for streaming scenarios.

        The cleaned data should be ahead of the actual playback position by at least
        the window size, ensuring smooth, optimized output reaches devices/clients.

        Args:
            current_time_ms: Current timestamp in milliseconds
        """
        if not self.funscript:
            return

        # Check which axes have data
        has_primary = self.funscript.primary_actions and len(self.funscript.primary_actions) > 0
        has_secondary = self.funscript.secondary_actions and len(self.funscript.secondary_actions) > 0

        if not has_primary and not has_secondary:
            return

        # Calculate time window
        start_time = current_time_ms - self.rolling_autotune_window_ms

        try:
            # Apply Ultimate Autotune to this window only
            from funscript.plugins.ultimate_autotune_plugin import UltimateAutotunePlugin
            autotune = UltimateAutotunePlugin()

            axes_processed = []

            # Acquire lock to prevent race conditions with timeline rendering
            # This ensures the funscript isn't being modified while the GUI is reading it
            if hasattr(self.funscript, '_lock'):
                lock = self.funscript._lock
            else:
                # Create a lock if it doesn't exist (for backwards compatibility)
                import threading
                lock = threading.RLock()
                self.funscript._lock = lock

            with lock:
                # Process primary axis if it has data
                if has_primary:
                    primary_actions = self.funscript.primary_actions
                    primary_indices = [
                        i for i, action in enumerate(primary_actions)
                        if start_time <= action['at'] <= current_time_ms
                    ]

                    if len(primary_indices) >= 2:
                        result = autotune.transform(
                            self.funscript,
                            axis='primary',
                            selected_indices=primary_indices
                        )
                        if result:
                            axes_processed.append(f"primary({len(primary_indices)} pts)")

                # Process secondary axis if it has data
                if has_secondary:
                    secondary_actions = self.funscript.secondary_actions
                    secondary_indices = [
                        i for i, action in enumerate(secondary_actions)
                        if start_time <= action['at'] <= current_time_ms
                    ]

                    if len(secondary_indices) >= 2:
                        result = autotune.transform(
                            self.funscript,
                            axis='secondary',
                            selected_indices=secondary_indices
                        )
                        if result:
                            axes_processed.append(f"secondary({len(secondary_indices)} pts)")

            if axes_processed:
                self.logger.info(f"🔧 Rolling autotune applied to {', '.join(axes_processed)} "
                               f"({start_time}ms - {current_time_ms}ms)")
            else:
                self.logger.debug(f"Not enough data for rolling autotune in window")

        except Exception as e:
            self.logger.error(f"Error applying rolling autotune: {e}")
            import traceback
            traceback.print_exc()

    def _init_device_bridge(self):
        """Initialize device control bridge if available."""
        try:
            # Check if device_control folder exists (supporter feature)
            if self._is_device_control_available():
                from device_control.bridges.live_tracker_bridge import create_live_tracker_bridge
                
                # Get device manager from app if available
                device_manager = getattr(self.app, 'device_manager', None)
                if device_manager:
                    self.device_bridge = create_live_tracker_bridge(device_manager)
                    self.logger.info("Device control bridge initialized for live tracking")
                else:
                    self.logger.debug("Device manager not available in app")
        except ImportError:
            # Device control not available (non-subscriber)
            self.device_bridge = None
            self.logger.debug("Device control not available - live device control disabled")
        except Exception as e:
            self.logger.warning(f"Failed to initialize device control bridge: {e}")
            self.device_bridge = None
    
    def _is_device_control_available(self) -> bool:
        """Check if device control features are available."""
        from pathlib import Path
        return Path("device_control").exists()
    
    def set_live_device_control_enabled(self, enabled: bool):
        """Enable/disable live device control (user toggle)."""
        self.live_device_control_enabled = enabled
        self.logger.info(f"Live device control {'enabled' if enabled else 'disabled'}")
        
        if enabled and self.device_bridge:
            # Start device bridge (handle no event loop gracefully)
            import asyncio
            try:
                # Check if there's an active event loop
                loop = asyncio.get_running_loop()
                asyncio.create_task(self.device_bridge.start())
            except RuntimeError:
                # No event loop available - manually activate bridge
                self.device_bridge.is_active = True
                self.logger.debug("No event loop - manually activated device bridge")
        elif self.device_bridge:
            # Stop device bridge (handle no event loop gracefully)
            import asyncio
            try:
                # Check if there's an active event loop
                loop = asyncio.get_running_loop()
                asyncio.create_task(self.device_bridge.stop())
            except RuntimeError:
                # No event loop available - manually deactivate bridge
                self.device_bridge.is_active = False
                self.logger.debug("No event loop - manually deactivated device bridge")

    def _send_to_device_control(self, latest_action: Dict):
        """Send latest tracking position to device control."""
        # Check if device bridge needs to be initialized
        if not self.device_bridge and hasattr(self.app, 'device_manager'):
            self.logger.info("Device manager available but no bridge - re-initializing bridge")
            self._init_device_bridge()
            
            # Also check if live tracking should be enabled from settings
            if not self.live_device_control_enabled:
                live_tracking_enabled = self.app.app_settings.get("device_control_live_tracking", False)
                if live_tracking_enabled:
                    self.set_live_device_control_enabled(True)
                    self.logger.info("Auto-enabled live device control from settings")
        
        # Debug logging for device control flow
        device_manager = getattr(self.app, 'device_manager', None)
        device_connected = device_manager.is_connected() if device_manager else False
        connected_devices = list(device_manager.connected_devices.keys()) if device_manager else []
        
        # Use debug level for routine checks to reduce verbosity during normal operation
        self.logger.debug(f"Device control check: bridge={self.device_bridge is not None}, "
                         f"enabled={self.live_device_control_enabled}, active={self.tracking_active}")
        self.logger.debug(f"Device manager: exists={device_manager is not None}, "
                         f"connected={device_connected}, devices={connected_devices}")
        
        if not (self.device_bridge and 
                self.live_device_control_enabled and 
                self.tracking_active):
            # Log what's preventing device control (only warn once per session)
            if not self.device_bridge and not hasattr(self, '_no_bridge_warned'):
                self.logger.info("Live tracking device control: No device bridge available")
                self._no_bridge_warned = True
            if not self.live_device_control_enabled and not hasattr(self, '_not_enabled_warned'):
                self.logger.info("Live tracking device control: Not enabled - check Control Panel → Global Device Settings → 'Enable Live Tracking Control'")
                self._not_enabled_warned = True
            return
            
        try:
            if not isinstance(latest_action, dict):
                return
                
            # Extract positions from action
            primary_pos = latest_action.get('pos')
            secondary_pos = latest_action.get('secondary_pos', 50.0)  # Default center
            
            if primary_pos is not None:
                # Import TrackerResult here to avoid import issues for non-subscribers
                try:
                    from tracker.tracker_modules.core.base_tracker import TrackerResult
                except ImportError:
                    # Create a simple mock if TrackerResult not available
                    class TrackerResult:
                        def __init__(self, processed_frame, action_log, debug_info):
                            self.processed_frame = processed_frame
                            self.action_log = action_log
                            self.debug_info = debug_info
                
                # Create a TrackerResult-like object for the bridge
                mock_result = TrackerResult(
                    processed_frame=None,  # Not needed for device control
                    action_log=None,
                    debug_info={
                        'primary_position': primary_pos,
                        'secondary_position': secondary_pos,
                        'timestamp_ms': latest_action.get('at', 0)
                    }
                )
                
                # Send to device bridge
                self.device_bridge.on_tracker_result(mock_result)
                
        except Exception as e:
            self.logger.error(f"Error sending to device control: {e}")

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
#!/usr/bin/env python3
"""
User Fixed ROI Tracker - Manual ROI-based tracking.

This tracker allows users to manually define a fixed rectangular region of interest
and tracks motion within that region using optical flow. It supports both whole-ROI
tracking and sub-tracking with a smaller tracking box within the ROI.

Author: Migrated from User ROI system
Version: 1.0.0
"""

import logging
import time
import numpy as np
import cv2
from collections import deque
from typing import Dict, Any, Optional, List, Tuple

try:
    from ..core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from ..helpers.signal_amplifier import SignalAmplifier
except ImportError:
    from tracker.tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
    from tracker.tracker_modules.helpers.signal_amplifier import SignalAmplifier


class UserRoiTracker(BaseTracker):
    """
    User-defined fixed ROI tracker.
    
    This tracker excels at:
    - Manual ROI definition by the user
    - Optical flow analysis within fixed regions
    - Sub-tracking with smaller tracking boxes
    - Motion smoothing and history management
    - Adaptive and manual scaling options
    - Tracked point position updates
    """
    
    def __init__(self):
        super().__init__()
        
        # ROI configuration
        self.user_roi_fixed = None  # User-defined ROI (x, y, w, h)
        
        # Sub-tracking configuration (matches original tracker.py performance)
        self.enable_user_roi_sub_tracking = True  # Default enabled like original
        self.user_roi_tracked_point_relative = None  # (x, y) relative to ROI
        self.user_roi_tracking_box_size = (5, 5)  # Match original 5x5 pixel box
        
        # Optical flow
        self.flow_dense = None
        self.prev_gray_user_roi_patch = None
        self.use_sparse_flow = False
        
        # Motion tracking and smoothing (matches original tracker)
        self.flow_history_window_smooth = 3  # Fixed at 3 frames for responsiveness
        self.primary_flow_history_smooth = []  # Use list like original, not deque
        self.secondary_flow_history_smooth = []  # Use list like original, not deque
        self.user_roi_current_flow_vector = (0.0, 0.0)
        
        # Additional smoothing layers for jerk reduction
        self.position_history_smooth = deque(maxlen=3)
        self.last_primary_position = 50
        self.last_secondary_position = 50
        
        # Enhanced signal mastering using helper module
        self.signal_amplifier = SignalAmplifier(
            history_size=120,  # 4 seconds @ 30fps
            enable_live_amp=True,  # Enable dynamic amplification by default
            smoothing_alpha=0.3,  # EMA smoothing factor
            logger=self.logger
        )
        
        # Adaptive flow range attributes (from original tracker.py)
        self.flow_min_primary_adaptive = -0.1
        self.flow_max_primary_adaptive = 0.1
        self.flow_min_secondary_adaptive = -0.1
        self.flow_max_secondary_adaptive = 0.1
        
        # Size factor tracking (from original - used even for User ROI)
        self.penis_max_size_history = deque(maxlen=30)
        self.penis_last_known_box = None
        
        # Position and scaling
        self.sensitivity = 10.0
        self.current_effective_amp_factor = 1.0
        self.adaptive_flow_scale = True  # Enable by default like original
        self.y_offset = 0
        self.x_offset = 0
        
        # Settings
        self.show_roi = True
        
        # Performance tracking
        self.current_fps = 30.0
        self._fps_counter = 0
        self._fps_last_time = time.time()
        self.stats_display = []
        
        # Output delay compensation
        self.output_delay_frames = 0
    
    @property
    def metadata(self) -> TrackerMetadata:
        """Return metadata describing this tracker."""
        return TrackerMetadata(
            name="user_roi",
            display_name="User ROI Tracker",
            description="Manual ROI definition with optical flow tracking and optional sub-tracking",
            category="live",
            version="1.0.0",
            author="User ROI System",
            tags=["manual", "roi", "optical-flow", "fixed", "user-defined"],
            requires_roi=True,  # ROI must be manually set by user
            supports_dual_axis=True
        )
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the user ROI tracker."""
        try:
            self.app = app_instance
            
            # Load settings
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                
                self.enable_user_roi_sub_tracking = settings.get('enable_user_roi_sub_tracking', True)  # Original default: True
                self.user_roi_tracking_box_size = (
                    settings.get('user_roi_tracking_box_width', 5),  # Original default: 5x5 pixels
                    settings.get('user_roi_tracking_box_height', 5)
                )
                # Keep flow history window at 3 for responsiveness (matches original tracker)
                self.flow_history_window_smooth = 3  # Don't override with settings
                self.sensitivity = settings.get('sensitivity', 10.0)
                self.adaptive_flow_scale = settings.get('adaptive_flow_scale', False)
                self.y_offset = settings.get('y_offset', 0)
                self.x_offset = settings.get('x_offset', 0)
                self.show_roi = settings.get('show_roi', True)
                self.use_sparse_flow = settings.get('use_sparse_flow', False)
                self.output_delay_frames = settings.get('output_delay_frames', 0)
                
                self.logger.info(f"User ROI settings: sub_tracking={self.enable_user_roi_sub_tracking}, "
                               f"box_size={self.user_roi_tracking_box_size}, sensitivity={self.sensitivity}")
            
            # Update smoothing deque maxlen
            self.primary_flow_history_smooth = deque(maxlen=self.flow_history_window_smooth)
            self.secondary_flow_history_smooth = deque(maxlen=self.flow_history_window_smooth)
            
            # Initialize optical flow - use DIS with ultrafast preset for better performance
            try:
                self.flow_dense = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_ULTRAFAST)
                self.logger.info("DIS optical flow initialized (ultrafast preset) for User ROI")
            except AttributeError:
                try:
                    # Fallback to medium preset if ultrafast not available
                    self.flow_dense = cv2.DISOpticalFlow.create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
                    self.logger.info("DIS optical flow initialized (medium preset) for User ROI")
                except AttributeError:
                    self.logger.error("No DIS optical flow implementation available")
                    return False
            
            # Reset state
            self.user_roi_fixed = None
            self.user_roi_tracked_point_relative = None
            self.prev_gray_user_roi_patch = None
            self.primary_flow_history_smooth.clear()
            self.secondary_flow_history_smooth.clear()
            self.user_roi_current_flow_vector = (0.0, 0.0)
            
            self._initialized = True
            self.logger.info("User ROI tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, 
                     frame_index: Optional[int] = None) -> TrackerResult:
        """
        Process a frame using user-defined ROI tracking.
        
        This implementation:
        1. Extracts the user-defined ROI region from the frame
        2. Optionally tracks a specific point within the ROI using sub-tracking
        3. Calculates optical flow within the ROI or tracking box
        4. Applies motion smoothing and scaling
        5. Updates tracked point position
        6. Generates funscript actions based on motion
        """
        try:
            self._update_fps()
            processed_frame = self._preprocess_frame(frame)
            current_frame_gray = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2GRAY)
            
            action_log_list = []
            final_primary_pos, final_secondary_pos = 50, 50
            
            self.current_effective_amp_factor = self._get_effective_amplification_factor()
            
            # Initialize stats display
            self.stats_display = [
                f"UserROI FPS:{self.current_fps:.1f} T(ms):{frame_time_ms} Amp:{self.current_effective_amp_factor:.2f}x"
            ]
            if frame_index is not None:
                self.stats_display.append(f"FIdx:{frame_index}")
            
            # Process user ROI if defined and tracking is active
            if self.user_roi_fixed and self.tracking_active:
                final_primary_pos, final_secondary_pos = self._process_user_roi(
                    current_frame_gray, processed_frame.shape[:2]
                )
            else:
                # No ROI set or tracking inactive
                self.prev_gray_user_roi_patch = None
                self.user_roi_current_flow_vector = (0.0, 0.0)
            
            # Generate funscript actions if tracking is active
            if self.tracking_active:
                action_log_list = self._generate_actions(
                    frame_time_ms, final_primary_pos, final_secondary_pos, frame_index
                )
            
            # Apply visualizations
            self._draw_visualizations(processed_frame)
            
            # Prepare debug info
            debug_info = {
                'primary_position': final_primary_pos,
                'secondary_position': final_secondary_pos,
                'roi': self.user_roi_fixed,
                'sub_tracking_enabled': self.enable_user_roi_sub_tracking,
                'tracked_point': self.user_roi_tracked_point_relative,
                'flow_vector': self.user_roi_current_flow_vector,
                'tracking_active': self.tracking_active,
                'smoothing_window': len(self.primary_flow_history_smooth)
            }
            
            status_msg = f"User ROI | Pos: {final_primary_pos},{final_secondary_pos}"
            if self.user_roi_fixed:
                w, h = self.user_roi_fixed[2], self.user_roi_fixed[3]
                status_msg += f" | ROI: {w}x{h}"
            if self.enable_user_roi_sub_tracking:
                status_msg += " | Sub-tracking"
            
            return TrackerResult(
                processed_frame=processed_frame,
                action_log=action_log_list if action_log_list else None,
                debug_info=debug_info,
                status_message=status_msg
            )
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return TrackerResult(
                processed_frame=frame,
                action_log=None,
                debug_info={'error': str(e)},
                status_message=f"Error: {e}"
            )
    
    def start_tracking(self) -> bool:
        """Start user ROI tracking."""
        if not self._initialized:
            self.logger.error("Tracker not initialized")
            return False
        
        if not self.user_roi_fixed:
            self.logger.error("Cannot start tracking: no user ROI defined")
            return False
        
        self.tracking_active = True
        self.prev_gray_user_roi_patch = None
        self.primary_flow_history_smooth.clear()
        self.secondary_flow_history_smooth.clear()
        self.user_roi_current_flow_vector = (0.0, 0.0)
        
        # Reset signal amplifier for new tracking session
        self.signal_amplifier.reset()
        
        self.logger.info("User ROI tracking started")
        return True
    
    def stop_tracking(self) -> bool:
        """Stop user ROI tracking."""
        self.tracking_active = False
        self.logger.info("User ROI tracking stopped")
        return True
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> bool:
        """
        Set the user-defined ROI.
        
        Args:
            roi: Region as (x, y, width, height)
        
        Returns:
            bool: True if ROI was set successfully
        """
        try:
            if len(roi) != 4:
                self.logger.error("ROI must be (x, y, width, height)")
                return False
            
            x, y, w, h = roi
            if w <= 0 or h <= 0:
                self.logger.error("ROI width and height must be positive")
                return False
            
            if x < 0 or y < 0:
                self.logger.error("ROI coordinates must be non-negative")
                return False
            
            self.user_roi_fixed = roi
            self.logger.info(f"User ROI set to: {roi}")
            
            # Reset tracking state when ROI changes
            self.prev_gray_user_roi_patch = None
            self.user_roi_tracked_point_relative = None
            self.primary_flow_history_smooth.clear()
            self.secondary_flow_history_smooth.clear()
            
            # Initialize tracked point to center of ROI if sub-tracking is enabled
            if self.enable_user_roi_sub_tracking:
                self.user_roi_tracked_point_relative = (w / 2.0, h / 2.0)
                self.logger.info(f"Initialized tracked point to ROI center: {self.user_roi_tracked_point_relative}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set ROI: {e}")
            return False
    
    def set_user_defined_roi_and_point(self, roi_abs_coords: Tuple[int, int, int, int], 
                                       point_abs_coords_in_frame: Tuple[int, int], 
                                       current_frame_for_patch: Optional[np.ndarray] = None):
        """
        Set the user-defined ROI and tracking point.
        
        This method is called from app_logic when the user finishes drawing ROI and clicking point.
        
        Args:
            roi_abs_coords: ROI in video coordinates (x, y, width, height)
            point_abs_coords_in_frame: Point coordinates in video space (x, y)
            current_frame_for_patch: Current video frame (unused, kept for compatibility)
        """
        try:
            # Set the ROI
            self.user_roi_fixed = roi_abs_coords
            x, y, w, h = roi_abs_coords
            self.logger.info(f"User ROI set to: {roi_abs_coords}")
            
            # Calculate point relative to ROI
            point_x, point_y = point_abs_coords_in_frame
            rel_x = float(point_x - x)
            rel_y = float(point_y - y)
            
            # Clamp point to be within ROI bounds
            rel_x = max(0, min(rel_x, w - 1))
            rel_y = max(0, min(rel_y, h - 1))
            
            self.user_roi_tracked_point_relative = (rel_x, rel_y)
            self.logger.info(f"Tracked point set to: {self.user_roi_tracked_point_relative} (relative to ROI)")
            
            # Reset tracking state
            self.prev_gray_user_roi_patch = None
            
            # Start tracking if not already active
            if not self.tracking_active:
                self.start_tracking()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set ROI and point: {e}")
            return False
    
    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validate user ROI settings."""
        try:
            history_window = settings.get('flow_history_window_smooth', self.flow_history_window_smooth)
            if not isinstance(history_window, int) or history_window < 1 or history_window > 50:
                self.logger.error("Flow history window must be between 1 and 50")
                return False
            
            sensitivity = settings.get('sensitivity', self.sensitivity)
            if not isinstance(sensitivity, (int, float)) or sensitivity <= 0:
                self.logger.error("Sensitivity must be positive")
                return False
            
            box_width = settings.get('user_roi_tracking_box_width', self.user_roi_tracking_box_size[0])
            box_height = settings.get('user_roi_tracking_box_height', self.user_roi_tracking_box_size[1])
            if (not isinstance(box_width, int) or not isinstance(box_height, int) or 
                box_width < 3 or box_height < 3 or box_width > 200 or box_height > 200):
                self.logger.error("Tracking box dimensions must be between 3 and 200 pixels")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Settings validation error: {e}")
            return False
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get current status information."""
        base_status = super().get_status_info()
        
        custom_status = {
            'roi_set': self.user_roi_fixed is not None,
            'roi_dimensions': f"{self.user_roi_fixed[2]}x{self.user_roi_fixed[3]}" if self.user_roi_fixed else "None",
            'sub_tracking_enabled': self.enable_user_roi_sub_tracking,
            'tracked_point': self.user_roi_tracked_point_relative,
            'tracking_box_size': self.user_roi_tracking_box_size,
            'flow_history_length': len(self.primary_flow_history_smooth),
            'current_flow_vector': self.user_roi_current_flow_vector,
            'sensitivity': self.sensitivity,
            'adaptive_scaling': self.adaptive_flow_scale
        }
        
        base_status.update(custom_status)
        return base_status
    
    def cleanup(self):
        """Clean up resources."""
        self.user_roi_fixed = None
        self.user_roi_tracked_point_relative = None
        self.prev_gray_user_roi_patch = None
        self.flow_dense = None
        self.primary_flow_history_smooth.clear()
        self.secondary_flow_history_smooth.clear()
        # self.logger.info("User ROI tracker cleaned up")
    
    # Private helper methods
    
    def _preprocess_frame(self, frame: np.ndarray) -> np.ndarray:
        """Basic frame preprocessing."""
        return frame.copy()
    
    def _process_user_roi(self, current_frame_gray: np.ndarray, frame_shape: Tuple[int, int]) -> Tuple[int, int]:
        """Process the user-defined ROI region."""
        urx, ury, urw, urh = self.user_roi_fixed
        
        # Clamp ROI to frame bounds
        urx_c, ury_c = max(0, urx), max(0, ury)
        urw_c = min(urw, current_frame_gray.shape[1] - urx_c)
        urh_c = min(urh, current_frame_gray.shape[0] - ury_c)
        
        if urw_c <= 0 or urh_c <= 0:
            self.prev_gray_user_roi_patch = None
            self.user_roi_current_flow_vector = (0.0, 0.0)
            return 50, 50
        
        # Extract ROI patch
        current_user_roi_patch_gray = current_frame_gray[ury_c:ury_c + urh_c, urx_c:urx_c + urw_c]
        
        # Calculate motion using sub-tracking or whole ROI
        if self.enable_user_roi_sub_tracking:
            final_primary_pos, final_secondary_pos = self._process_sub_tracking(
                current_user_roi_patch_gray, urw_c, urh_c
            )
        else:
            final_primary_pos, final_secondary_pos = self._process_whole_roi(
                current_user_roi_patch_gray
            )
        
        # Store current patch for next frame
        self.prev_gray_user_roi_patch = np.ascontiguousarray(current_user_roi_patch_gray)
        
        return final_primary_pos, final_secondary_pos
    
    def _process_sub_tracking(self, current_roi_patch: np.ndarray, roi_w: int, roi_h: int) -> Tuple[int, int]:
        """Process ROI using sub-tracking with a smaller tracking box."""
        dy_raw, dx_raw = 0.0, 0.0
        
        if (self.prev_gray_user_roi_patch is not None and 
            self.user_roi_tracked_point_relative and 
            self.flow_dense and
            self.prev_gray_user_roi_patch.shape == current_roi_patch.shape):
            
            # Calculate optical flow
            flow = self.flow_dense.calc(
                np.ascontiguousarray(self.prev_gray_user_roi_patch), 
                np.ascontiguousarray(current_roi_patch), 
                None
            )
            
            if flow is not None:
                # Extract motion from tracking box
                track_w, track_h = self.user_roi_tracking_box_size
                box_center_x, box_center_y = self.user_roi_tracked_point_relative
                
                # Calculate tracking box bounds
                box_x1 = int(box_center_x - track_w / 2)
                box_y1 = int(box_center_y - track_h / 2)
                box_x2 = box_x1 + track_w
                box_y2 = box_y1 + track_h
                
                patch_h, patch_w = current_roi_patch.shape
                
                # Clamp to patch bounds
                box_x1_c, box_y1_c = max(0, box_x1), max(0, box_y1)
                box_x2_c, box_y2_c = min(patch_w, box_x2), min(patch_h, box_y2)
                
                if box_x2_c > box_x1_c and box_y2_c > box_y1_c:
                    sub_flow = flow[box_y1_c:box_y2_c, box_x1_c:box_x2_c]
                    if sub_flow.size > 0:
                        dx_raw = np.median(sub_flow[..., 0])
                        dy_raw = np.median(sub_flow[..., 1])
            
            # Update tracked point position based on optical flow (for visualization)
            if self.user_roi_tracked_point_relative:
                rel_x, rel_y = self.user_roi_tracked_point_relative
                
                # Update position with flow values, keeping within ROI boundaries  
                new_x = rel_x + dx_raw
                new_y = rel_y + dy_raw
                
                # Constrain to ROI boundaries
                new_x = max(0.0, min(new_x, float(roi_w - 1)))
                new_y = max(0.0, min(new_y, float(roi_h - 1)))
                
                self.user_roi_tracked_point_relative = (new_x, new_y)
        
        return self._apply_motion_processing(dy_raw, dx_raw)
    
    def _process_whole_roi(self, current_roi_patch: np.ndarray) -> Tuple[int, int]:
        """Process motion across the entire ROI."""
        dy_raw, dx_raw = 0.0, 0.0
        
        if self.prev_gray_user_roi_patch is not None:
            dx_raw, dy_raw, _, _ = self._calculate_flow_in_patch(
                current_roi_patch,
                self.prev_gray_user_roi_patch,
                use_sparse=self.use_sparse_flow,
                prev_features_for_sparse=None
            )
        
        return self._apply_motion_processing(dy_raw, dx_raw)
    
    def _calculate_flow_in_patch(self, current_patch: np.ndarray, prev_patch: np.ndarray, 
                               use_sparse: bool = False, prev_features_for_sparse=None) -> Tuple[float, float, None, None]:
        """Calculate optical flow in a patch (simplified version)."""
        if current_patch.shape != prev_patch.shape or not self.flow_dense:
            return 0.0, 0.0, None, None
        
        try:
            flow = self.flow_dense.calc(
                np.ascontiguousarray(prev_patch), 
                np.ascontiguousarray(current_patch), 
                None
            )
            
            if flow is not None:
                dx = np.median(flow[..., 0])
                dy = np.median(flow[..., 1])
                return dx, dy, None, None
            
        except Exception as e:
            self.logger.error(f"Flow calculation error: {e}")
        
        return 0.0, 0.0, None, None
    
    def _apply_motion_processing(self, dy_raw: float, dx_raw: float) -> Tuple[int, int]:
        """Apply motion smoothing and scaling using original tracker.py algorithm."""
        # Add to smoothing history
        self.primary_flow_history_smooth.append(dy_raw)
        self.secondary_flow_history_smooth.append(dx_raw)
        
        # Limit history size
        if len(self.primary_flow_history_smooth) > self.flow_history_window_smooth:
            self.primary_flow_history_smooth.pop(0)
        if len(self.secondary_flow_history_smooth) > self.flow_history_window_smooth:
            self.secondary_flow_history_smooth.pop(0)
        
        # Apply smoothing
        dy_smooth = (np.median(self.primary_flow_history_smooth) 
                    if self.primary_flow_history_smooth else dy_raw)
        dx_smooth = (np.median(self.secondary_flow_history_smooth) 
                    if self.secondary_flow_history_smooth else dx_raw)
        
        # Calculate the base positions using original algorithm
        size_factor = self.get_current_penis_size_factor()
        if self.adaptive_flow_scale:
            base_primary_pos = self._apply_adaptive_scaling_original(dy_smooth, "flow_min_primary_adaptive", "flow_max_primary_adaptive", size_factor, True)
            secondary_pos = self._apply_adaptive_scaling_original(dx_smooth, "flow_min_secondary_adaptive", "flow_max_secondary_adaptive", size_factor, False)
        else:
            effective_amp_factor = self._get_effective_amplification_factor()
            manual_scale_multiplier = (self._get_current_sensitivity() / 10.0) * (1.0 / size_factor) * effective_amp_factor
            base_primary_pos = int(np.clip(50 + dy_smooth * manual_scale_multiplier + self.y_offset, 0, 100))
            secondary_pos = int(np.clip(50 + dx_smooth * manual_scale_multiplier + self.x_offset, 0, 100))

        # Update flow vector
        self.user_roi_current_flow_vector = (dx_smooth, dy_smooth)
        
        # Return directly without extra smoothing layers (matches original tracker)
        # The flow history median smoothing is sufficient
        return base_primary_pos, secondary_pos
    
    def _apply_final_smoothing(self, primary_pos: int, secondary_pos: int) -> Tuple[int, int]:
        """Apply final layer of smoothing to reduce signal jerkiness."""
        # Exponential moving average for smooth transitions
        alpha = 0.3  # Smoothing factor (0 = max smoothing, 1 = no smoothing)
        
        # Apply EMA smoothing
        smoothed_primary = int(alpha * primary_pos + (1 - alpha) * self.last_primary_position)
        smoothed_secondary = int(alpha * secondary_pos + (1 - alpha) * self.last_secondary_position)
        
        # Store position history for additional median smoothing
        self.position_history_smooth.append((smoothed_primary, smoothed_secondary))
        
        # Apply median smoothing over position history for even smoother results
        if len(self.position_history_smooth) >= 3:
            primary_values = [pos[0] for pos in list(self.position_history_smooth)]
            secondary_values = [pos[1] for pos in list(self.position_history_smooth)]
            
            final_primary = int(np.median(primary_values))
            final_secondary = int(np.median(secondary_values))
        else:
            final_primary = smoothed_primary
            final_secondary = smoothed_secondary
        
        # Update last positions
        self.last_primary_position = final_primary
        self.last_secondary_position = final_secondary
        
        return final_primary, final_secondary
    
    def _apply_adaptive_scaling(self, flow_value: float, is_primary: bool = True) -> int:
        """Apply adaptive scaling to flow values (simplified version)."""
        # Simplified adaptive scaling - real implementation would be more complex
        size_factor = 1.0  # No object detection in this mode
        base_scale = 2.0
        scaled_value = 50 + flow_value * base_scale * size_factor
        
        if is_primary:
            scaled_value += self.y_offset
        else:
            scaled_value += self.x_offset
        
        return int(np.clip(scaled_value, 0, 100))
    
    def _generate_actions(self, frame_time_ms: int, final_primary_pos: int, final_secondary_pos: int,
                         frame_index: Optional[int]) -> List[Dict]:
        """Generate funscript actions based on calculated positions."""
        action_log_list = []
        
        # Get current tracking settings
        current_tracking_axis_mode = getattr(self.app, 'tracking_axis_mode', 'both')
        current_single_axis_output = getattr(self.app, 'single_axis_output_target', 'primary')
        
        primary_to_write, secondary_to_write = None, None
        
        if current_tracking_axis_mode == "both":
            primary_to_write, secondary_to_write = final_primary_pos, final_secondary_pos
        elif current_tracking_axis_mode == "vertical":
            if current_single_axis_output == "primary":
                primary_to_write = final_primary_pos
            else:
                secondary_to_write = final_primary_pos
        elif current_tracking_axis_mode == "horizontal":
            if current_single_axis_output == "primary":
                primary_to_write = final_secondary_pos
            else:
                secondary_to_write = final_secondary_pos
        
        # Apply automatic lag compensation
        automatic_smoothing_delay_frames = ((self.flow_history_window_smooth - 1) / 2.0 
                                          if self.flow_history_window_smooth > 1 else 0.0)
        total_delay_frames = self.output_delay_frames + automatic_smoothing_delay_frames
        
        # Convert frame delay to time delay
        effective_delay_ms = total_delay_frames * (1000.0 / max(self.current_fps, 1.0))
        adjusted_frame_time_ms = frame_time_ms - effective_delay_ms
        
        # Add action to funscript
        if hasattr(self.app, 'funscript'):
            self.app.funscript.add_action(
                timestamp_ms=int(round(adjusted_frame_time_ms)), 
                primary_pos=primary_to_write, 
                secondary_pos=secondary_to_write
            )
        
        # Create action log entry
        action_log_entry = {
            "at": int(round(adjusted_frame_time_ms)),
            "pos": primary_to_write,
            "secondary_pos": secondary_to_write,
            "mode": current_tracking_axis_mode,
            "target": current_single_axis_output if current_tracking_axis_mode != "both" else "N/A",
            "raw_at": frame_time_ms,
            "delay_applied_ms": effective_delay_ms,
            "roi_main": self.user_roi_fixed,
            "amp": self.current_effective_amp_factor
        }
        
        if frame_index is not None:
            action_log_entry["frame_index"] = frame_index
        
        action_log_list.append(action_log_entry)
        
        return action_log_list
    
    def _draw_visualizations(self, processed_frame: np.ndarray):
        """Draw visualization overlays on the frame."""
        # Draw User ROI rectangle
        if self.show_roi and self.user_roi_fixed:
            urx, ury, urw, urh = self.user_roi_fixed
            color = (0, 255, 255)  # Yellow for user ROI
            cv2.rectangle(processed_frame, (urx, ury), (urx + urw, ury + urh), color, 2)
            
            # ROI label removed for cleaner display
            
            # Draw the tracked point moving with optical flow (blue, bold)
            if self.user_roi_tracked_point_relative:
                rel_x, rel_y = self.user_roi_tracked_point_relative
                
                # Convert relative coordinates to absolute
                abs_x = urx + rel_x
                abs_y = ury + rel_y
                
                # Draw bold blue point that moves with optical flow
                cv2.circle(processed_frame, (int(abs_x), int(abs_y)), 6, (255, 0, 0), -1)  # Bold blue dot
        
        # Add tracking indicator
        self._draw_tracking_indicator(processed_frame)
    
    def get_current_penis_size_factor(self) -> float:
        """Calculate current penis size factor (EXACT original method adapted for User ROI)."""
        # For User ROI, use ROI size as proxy for object size
        if not self.user_roi_fixed:
            return 1.0
            
        # Use ROI dimensions as size history
        _, _, w, h = self.user_roi_fixed
        roi_size = w * h
        
        # Add to history
        if len(self.penis_max_size_history) == 0:
            self.penis_max_size_history.append(roi_size)
        
        max_hist = max(self.penis_max_size_history) if self.penis_max_size_history else roi_size
        if max_hist < 1:
            return 1.0
            
        return np.clip(roi_size / max_hist, 0.1, 1.5)
    
    def _apply_adaptive_scaling_original(self, value: float, min_val_attr: str, max_val_attr: str, size_factor: float, is_primary: bool) -> int:
        """Security-compliant adaptive scaling method - EXACT original algorithm"""
        # Direct attribute access instead of setattr to comply with security restrictions
        if min_val_attr == "flow_min_primary_adaptive":
            min_h = self.flow_min_primary_adaptive
            self.flow_min_primary_adaptive = min(min_h * 0.995, value * 0.9 if value < -0.1 else value * 1.1)
            min_h = min(self.flow_min_primary_adaptive, -0.2)
        else:  # flow_min_secondary_adaptive
            min_h = self.flow_min_secondary_adaptive
            self.flow_min_secondary_adaptive = min(min_h * 0.995, value * 0.9 if value < -0.1 else value * 1.1)
            min_h = min(self.flow_min_secondary_adaptive, -0.2)
            
        if max_val_attr == "flow_max_primary_adaptive":
            max_h = self.flow_max_primary_adaptive
            self.flow_max_primary_adaptive = max(max_h * 0.995, value * 1.1 if value > 0.1 else value * 0.9)
            max_h = max(self.flow_max_primary_adaptive, 0.2)
        else:  # flow_max_secondary_adaptive
            max_h = self.flow_max_secondary_adaptive
            self.flow_max_secondary_adaptive = max(max_h * 0.995, value * 1.1 if value > 0.1 else value * 0.9)
            max_h = max(self.flow_max_secondary_adaptive, 0.2)
            
        flow_range = max_h - min_h
        if abs(flow_range) < 0.1: 
            flow_range = np.sign(flow_range) * 0.1 if flow_range != 0 else 0.1
        normalized_centered_flow = (2 * (value - min_h) / flow_range) - 1.0 if flow_range != 0 else 0.0
        normalized_centered_flow = np.clip(normalized_centered_flow, -1.0, 1.0)
        effective_amp_factor = self._get_effective_amplification_factor()
        max_deviation = (self._get_current_sensitivity() / 2.5) * effective_amp_factor
        pos_offset = self.y_offset if is_primary else self.x_offset
        return int(np.clip(50 + normalized_centered_flow * max_deviation + pos_offset, 0, 100))

    def _get_effective_amplification_factor(self) -> float:
        """Calculate effective amplification factor with real-time settings updates."""
        # Read from app settings in real-time for control panel responsiveness
        if hasattr(self.app, 'app_settings') and self.app.app_settings:
            base_factor = self.app.app_settings.get('live_tracker_base_amplification', 1.0)
        else:
            base_factor = 1.0
        return base_factor

    def _update_fps(self):
        """Update FPS calculation using high-performance delta time method."""
        current_time_sec = time.time()
        if self._fps_last_time > 0:
            delta_time = current_time_sec - self._fps_last_time
            if delta_time > 0.001:  # Avoid division by zero
                self.current_fps = 1.0 / delta_time
        self._fps_last_time = current_time_sec
    
    def _get_current_sensitivity(self) -> float:
        """Get current sensitivity with real-time settings updates."""
        if hasattr(self.app, 'app_settings') and self.app.app_settings:
            return self.app.app_settings.get('live_tracker_sensitivity', self.sensitivity)
        else:
            return self.sensitivity
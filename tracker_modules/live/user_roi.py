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
import numpy as np
import cv2
from collections import deque
from typing import Dict, Any, Optional, List, Tuple

try:
    from ..core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
except ImportError:
    from tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult


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
        
        # Sub-tracking configuration
        self.enable_user_roi_sub_tracking = False
        self.user_roi_tracked_point_relative = None  # (x, y) relative to ROI
        self.user_roi_tracking_box_size = (40, 40)  # (w, h) of tracking box
        
        # Optical flow
        self.flow_dense = None
        self.prev_gray_user_roi_patch = None
        self.use_sparse_flow = False
        
        # Motion tracking and smoothing
        self.primary_flow_history_smooth = deque(maxlen=10)
        self.secondary_flow_history_smooth = deque(maxlen=10)
        self.flow_history_window_smooth = 10
        self.user_roi_current_flow_vector = (0.0, 0.0)
        
        # Position and scaling
        self.sensitivity = 10.0
        self.current_effective_amp_factor = 1.0
        self.adaptive_flow_scale = False
        self.y_offset = 0
        self.x_offset = 0
        
        # Settings
        self.show_roi = True
        
        # Performance tracking
        self.current_fps = 30.0
        self.stats_display = []
        
        # Output delay compensation
        self.output_delay_frames = 0
    
    @property
    def metadata(self) -> TrackerMetadata:
        """Return metadata describing this tracker."""
        return TrackerMetadata(
            name="user_roi",
            display_name="Live User ROI Tracker",
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
                
                self.enable_user_roi_sub_tracking = settings.get('enable_user_roi_sub_tracking', False)
                self.user_roi_tracking_box_size = (
                    settings.get('user_roi_tracking_box_width', 40),
                    settings.get('user_roi_tracking_box_height', 40)
                )
                self.flow_history_window_smooth = settings.get('flow_history_window_smooth', 10)
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
            
            # Initialize optical flow
            try:
                self.flow_dense = cv2.optflow.createOptFlow_DualTVL1()
                self.logger.info("DualTVL1 optical flow initialized for User ROI")
            except AttributeError:
                try:
                    self.flow_dense = cv2.FarnebackOpticalFlow_create()
                    self.logger.info("Farneback optical flow initialized as fallback")
                except AttributeError:
                    self.logger.error("No optical flow implementation available")
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
                box_width < 10 or box_height < 10 or box_width > 200 or box_height > 200):
                self.logger.error("Tracking box dimensions must be between 10 and 200 pixels")
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
            
            # Update tracked point position
            if self.user_roi_tracked_point_relative:
                prev_x_rel, prev_y_rel = self.user_roi_tracked_point_relative
                new_x_rel = prev_x_rel + dx_raw
                new_y_rel = prev_y_rel + dy_raw
                self.user_roi_tracked_point_relative = (
                    max(0.0, min(new_x_rel, float(roi_w))), 
                    max(0.0, min(new_y_rel, float(roi_h)))
                )
        
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
        """Apply motion smoothing and scaling to raw flow values."""
        # Add to smoothing history
        self.primary_flow_history_smooth.append(dy_raw)
        self.secondary_flow_history_smooth.append(dx_raw)
        
        # Apply smoothing
        dy_smooth = (np.median(self.primary_flow_history_smooth) 
                    if self.primary_flow_history_smooth else dy_raw)
        dx_smooth = (np.median(self.secondary_flow_history_smooth) 
                    if self.secondary_flow_history_smooth else dx_raw)
        
        # Apply scaling
        if self.adaptive_flow_scale:
            final_primary_pos = self._apply_adaptive_scaling(dy_smooth, is_primary=True)
            final_secondary_pos = self._apply_adaptive_scaling(dx_smooth, is_primary=False)
        else:
            effective_amp_factor = self._get_effective_amplification_factor()
            manual_scale_multiplier = (self.sensitivity / 10.0) * effective_amp_factor
            final_primary_pos = int(np.clip(50 + dy_smooth * manual_scale_multiplier + self.y_offset, 0, 100))
            final_secondary_pos = int(np.clip(50 + dx_smooth * manual_scale_multiplier + self.x_offset, 0, 100))
        
        # Update flow vector
        self.user_roi_current_flow_vector = (dx_smooth, dy_smooth)
        
        return final_primary_pos, final_secondary_pos
    
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
            
            # Draw ROI label
            cv2.putText(processed_frame, "User ROI", (urx, ury - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            # Draw tracking box if sub-tracking is enabled
            if (self.enable_user_roi_sub_tracking and 
                self.user_roi_tracked_point_relative and 
                self.user_roi_tracking_box_size):
                
                track_w, track_h = self.user_roi_tracking_box_size
                rel_x, rel_y = self.user_roi_tracked_point_relative
                
                # Convert relative coordinates to absolute
                abs_x = urx + rel_x
                abs_y = ury + rel_y
                
                # Draw tracking box
                box_x1 = int(abs_x - track_w / 2)
                box_y1 = int(abs_y - track_h / 2)
                box_x2 = box_x1 + track_w
                box_y2 = box_y1 + track_h
                
                tracking_color = (255, 0, 255)  # Magenta for tracking box
                cv2.rectangle(processed_frame, (box_x1, box_y1), (box_x2, box_y2), tracking_color, 1)
                
                # Draw center point
                cv2.circle(processed_frame, (int(abs_x), int(abs_y)), 3, tracking_color, -1)
        
        # Draw stats display
        for i, stat in enumerate(self.stats_display):
            cv2.putText(processed_frame, stat, (10, 30 + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw tracking status
        if self.tracking_active:
            cv2.putText(processed_frame, "USER ROI TRACKING", (10, processed_frame.shape[0] - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    def _get_effective_amplification_factor(self) -> float:
        """Calculate effective amplification factor."""
        # Simplified version - real implementation would consider more factors
        return 1.0
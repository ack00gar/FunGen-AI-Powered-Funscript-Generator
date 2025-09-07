"""
Working Axis Projection Tracker - Simplified and robust version.

This is a simplified but working implementation that focuses on reliable motion detection
and axis projection with clear, debuggable logic.
"""

import cv2
import numpy as np
import time
import logging
from typing import Optional, Dict, Any, Tuple
from collections import deque

from tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult


class WorkingAxisProjectionTracker(BaseTracker):
    """
    Simplified but working axis projection tracker.
    
    This version focuses on:
    - Reliable frame differencing motion detection
    - Clear axis projection logic
    - Robust tracking with simple filtering
    - Easy debugging and tuning
    """
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Processing parameters
        self.proc_width = 640
        self.scale = 1.0
        
        # Axis definition (horizontal by default)
        self.axis_A = None  # Start point
        self.axis_B = None  # End point
        
        # Motion detection - optimized for real video content
        self.motion_threshold = 15  # Lower threshold for subtle motion
        self.min_motion_area = 50   # Lower minimum area for detection
        self.adaptive_threshold = True  # Enable adaptive thresholding
        
        # Tracking state
        self.prev_gray = None
        self.current_position = 50
        self.position_history = deque(maxlen=10)
        self.confidence = 0.0
        
        # Smoothing
        self.smoothing_alpha = 0.3
        self.smoothed_position = 50.0
        
        # FPS tracking
        self.current_fps = 30.0
        self._fps_counter = 0
        self._fps_last_time = time.time()
        
        self.tracking_active = False
        
        # Debug info
        self.debug_motion_pixels = 0
        self.debug_motion_center = None
        
        # Adaptive thresholding state
        self.background_level = 0
        self.motion_variance_history = deque(maxlen=30)
        self.frame_count = 0
    
    @property
    def metadata(self) -> TrackerMetadata:
        return TrackerMetadata(
            name="axis_projection_working",
            display_name="Working Axis Projection Tracker",
            description="Simplified but reliable motion tracking with axis projection",
            category="live",
            version="1.0.0",
            author="Working Motion Tracker",
            tags=["frame-diff", "projection", "simple", "reliable"],
            requires_roi=False,
            supports_dual_axis=False
        )
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """Initialize the tracker."""
        try:
            self.app = app_instance
            
            # Set default horizontal axis if video dimensions available
            if hasattr(app_instance, 'get_video_dimensions'):
                width, height = app_instance.get_video_dimensions()
                if width and height:
                    # Horizontal axis across center of frame
                    margin = int(0.1 * width)
                    self.axis_A = (margin, height // 2)
                    self.axis_B = (width - margin, height // 2)
                    self.logger.info(f"Default axis set: A={self.axis_A}, B={self.axis_B}")
            
            # Load settings if available
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                self.proc_width = settings.get('axis_proc_width', 640)
                self.motion_threshold = settings.get('axis_motion_threshold', 15)
                self.smoothing_alpha = settings.get('axis_smoothing', 0.3)
                self.min_motion_area = settings.get('axis_min_motion_area', 50)
                self.adaptive_threshold = settings.get('axis_adaptive_threshold', True)
            
            self._initialized = True
            self.logger.info("Working axis projection tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def set_axis(self, point_a: Tuple[int, int], point_b: Tuple[int, int]) -> bool:
        """Set the projection axis."""
        try:
            self.axis_A = tuple(point_a)
            self.axis_B = tuple(point_b)
            self.logger.info(f"Axis set: A={self.axis_A}, B={self.axis_B}")
            
            # Reset position to center
            self.current_position = 50
            self.smoothed_position = 50.0
            self.position_history.clear()
            
            return True
        except Exception as e:
            self.logger.error(f"Failed to set axis: {e}")
            return False
    
    def _prepare_frame(self, frame: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare frame for processing."""
        h, w = frame.shape[:2]
        
        # Resize if needed
        if self.proc_width and w > self.proc_width:
            self.scale = self.proc_width / w
            new_h = int(h * self.scale)
            frame_small = cv2.resize(frame, (self.proc_width, new_h), interpolation=cv2.INTER_AREA)
        else:
            frame_small = frame
            self.scale = 1.0
        
        # Convert to grayscale with some preprocessing
        gray = cv2.cvtColor(frame_small, cv2.COLOR_BGR2GRAY)
        
        # Apply slight blur to reduce noise
        gray = cv2.GaussianBlur(gray, (3, 3), 0.8)  # Lighter blur to preserve motion details
        
        return gray, frame_small
    
    def _detect_motion(self, gray: np.ndarray) -> Tuple[bool, Optional[Tuple[float, float]], int]:
        """
        Detect motion with adaptive thresholding for real video content.
        
        Returns:
            (motion_detected, motion_center, motion_area)
        """
        if self.prev_gray is None:
            return False, None, 0
        
        self.frame_count += 1
        
        # Frame differencing
        diff = cv2.absdiff(self.prev_gray, gray)
        
        # Adaptive thresholding based on image statistics
        if self.adaptive_threshold:
            # Calculate dynamic threshold based on image variance
            img_mean = np.mean(diff)
            img_std = np.std(diff)
            
            # Store variance history for adaptation
            self.motion_variance_history.append(img_std)
            
            if len(self.motion_variance_history) >= 5:
                avg_std = np.mean(list(self.motion_variance_history))
                # Adaptive threshold: base + factor * std deviation
                adaptive_thresh = max(8, min(40, img_mean + 1.5 * avg_std))
            else:
                adaptive_thresh = self.motion_threshold
            
            threshold = adaptive_thresh
        else:
            threshold = self.motion_threshold
        
        # Apply threshold
        _, motion_mask = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        
        # Store last threshold for debugging
        self._last_threshold = threshold
        
        # Clean up mask with smaller kernel for better detail preservation
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_OPEN, kernel)
        motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_CLOSE, kernel)
        
        # Additional cleanup: remove very small isolated regions
        if self.frame_count > 5:  # After a few frames of adaptation
            kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))
            motion_mask = cv2.morphologyEx(motion_mask, cv2.MORPH_DILATE, kernel_dilate)
        
        # Find largest motion blob
        contours, _ = cv2.findContours(motion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return False, None, 0
        
        # Get largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        if area < self.min_motion_area:
            return False, None, int(area)
        
        # Calculate centroid
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return False, None, int(area)
        
        cx = M["m10"] / M["m00"]
        cy = M["m01"] / M["m00"]
        
        return True, (cx, cy), int(area)
    
    def _project_to_axis(self, point: Tuple[float, float]) -> float:
        """
        Project point onto axis and return position [0, 100].
        
        Args:
            point: Point in processed coordinates
            
        Returns:
            Position on axis [0, 100]
        """
        if self.axis_A is None or self.axis_B is None:
            return 50.0
        
        # Convert axis to processed coordinates
        A_proc = (self.axis_A[0] * self.scale, self.axis_A[1] * self.scale)
        B_proc = (self.axis_B[0] * self.scale, self.axis_B[1] * self.scale)
        
        # Vector from A to B
        AB = np.array([B_proc[0] - A_proc[0], B_proc[1] - A_proc[1]], dtype=np.float32)
        
        # Vector from A to point
        AP = np.array([point[0] - A_proc[0], point[1] - A_proc[1]], dtype=np.float32)
        
        # Project AP onto AB
        AB_length_sq = np.dot(AB, AB)
        if AB_length_sq < 1e-6:
            return 50.0  # Degenerate axis
        
        t = np.dot(AP, AB) / AB_length_sq
        
        # Clamp to [0, 1] and convert to [0, 100]
        t = max(0.0, min(1.0, t))
        return t * 100.0
    
    def _update_position(self, new_position: float, confidence: float):
        """Update position with smoothing."""
        self.position_history.append(new_position)
        
        if confidence > 0.1:  # Only smooth if we have reasonable confidence
            # Simple exponential smoothing
            self.smoothed_position = (self.smoothing_alpha * new_position + 
                                    (1 - self.smoothing_alpha) * self.smoothed_position)
        else:
            # Low confidence - decay toward center
            self.smoothed_position = (0.95 * self.smoothed_position + 0.05 * 50.0)
        
        # Ensure position is properly clamped to funscript range
        self.current_position = int(round(max(0, min(100, self.smoothed_position))))
        self.confidence = confidence
    
    def _update_fps(self):
        """Update FPS counter."""
        self._fps_counter += 1
        if self._fps_counter >= 30:
            current_time = time.time()
            if self._fps_last_time > 0:
                self.current_fps = 30.0 / (current_time - self._fps_last_time)
            self._fps_last_time = current_time
            self._fps_counter = 0
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int,
                     frame_index: Optional[int] = None) -> TrackerResult:
        """Process frame with working motion tracking."""
        try:
            self._update_fps()
            
            # Debug: Log that we're being called
            if frame_index and frame_index == 1:
                self.logger.info("AXIS DEBUG: process_frame() called - tracker is active in GUI")
            
            # Prepare frame
            gray, frame_small = self._prepare_frame(frame)
            
            # Initialize on first frame
            if self.prev_gray is None:
                self.prev_gray = gray.copy()
                return TrackerResult(
                    processed_frame=frame_small,
                    action_log=None,
                    debug_info={'status': 'initializing'},
                    status_message="Initializing working axis tracker..."
                )
            
            # Detect motion
            motion_detected, motion_center, motion_area = self._detect_motion(gray)
            
            # Store debug info
            self.debug_motion_pixels = motion_area
            self.debug_motion_center = motion_center
            
            if motion_detected and motion_center:
                # Project to axis
                axis_position = self._project_to_axis(motion_center)
                
                # Calculate confidence based on motion area, intensity, and consistency
                # Scale area confidence more aggressively for smaller areas
                area_confidence = min(1.0, motion_area / 300.0)  # Lower normalization for real videos
                
                # Motion intensity confidence (how much motion vs just noise)
                if self.adaptive_threshold and len(self.motion_variance_history) >= 3:
                    recent_std = np.mean(list(self.motion_variance_history)[-3:])
                    intensity_confidence = min(1.0, recent_std / 10.0)  # Based on motion variance
                else:
                    intensity_confidence = 0.5
                
                # Consistency with recent history
                consistency_confidence = 1.0
                if len(self.position_history) >= 3:
                    recent_positions = list(self.position_history)[-3:]
                    position_std = np.std(recent_positions)
                    consistency_confidence = max(0.3, np.exp(-position_std / 15.0))  # More forgiving
                
                final_confidence = 0.4 * area_confidence + 0.3 * intensity_confidence + 0.3 * consistency_confidence
                
                # Update position
                self._update_position(axis_position, final_confidence)
            else:
                # No motion detected - maintain current position with reduced confidence
                self._update_position(self.current_position, max(0.0, self.confidence * 0.9))
            
            # Generate action if tracking is active (match format of working trackers)
            action_log = None
            if self.tracking_active:
                # Use the same format as oscillation_legacy and other working trackers
                action_log = [{
                    "at": int(frame_time_ms),  # Ensure integer timestamp
                    "pos": int(self.current_position)  # Ensure integer position
                }]  # Match exact format of working trackers
                
                # Debug logging for GUI troubleshooting  
                if self.frame_count % 30 == 0:  # Log every 30th frame
                    axis_info = f"A={self.axis_A}, B={self.axis_B}" if (self.axis_A and self.axis_B) else "NOT SET"
                    self.logger.info(f"AXIS DEBUG Frame {self.frame_count}: pos={self.current_position}, conf={self.confidence:.3f}, axis={axis_info}")
                    if motion_detected:
                        self.logger.info(f"AXIS DEBUG Frame {self.frame_count}: motion_center={motion_center}, area={motion_area}")
                    else:
                        self.logger.info(f"AXIS DEBUG Frame {self.frame_count}: NO MOTION DETECTED")
            
            # Draw visualization
            vis_frame = self._draw_visualization(frame_small, motion_center)
            
            # Enhanced debug info
            debug_info = {
                'position': self.current_position,
                'raw_position': self.smoothed_position,
                'confidence': self.confidence,
                'motion_detected': motion_detected,
                'motion_area': motion_area,
                'motion_center': motion_center,
                'fps': round(self.current_fps, 1),
                'tracking_active': self.tracking_active,
                'axis_defined': self.axis_A is not None and self.axis_B is not None,
                'adaptive_threshold': getattr(self, 'adaptive_threshold', False),
                'current_threshold': getattr(self, '_last_threshold', self.motion_threshold),
                'frame_count': self.frame_count
            }
            
            status_msg = f"Working Axis | Pos: {self.current_position} | Conf: {self.confidence:.2f}"
            if motion_detected:
                status_msg += f" | Motion: {motion_area}px"
            
            # Update previous frame
            self.prev_gray = gray.copy()
            
            return TrackerResult(
                processed_frame=vis_frame,
                action_log=action_log,
                debug_info=debug_info,
                status_message=status_msg
            )
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            return TrackerResult(
                processed_frame=frame,
                action_log=None,
                debug_info={'error': str(e)},
                status_message=f"Error: {str(e)}"
            )
    
    def _draw_visualization(self, frame: np.ndarray, motion_center: Optional[Tuple[float, float]]) -> np.ndarray:
        """Draw visualization overlays."""
        vis = frame.copy()
        
        try:
            # Draw axis if defined
            if self.axis_A and self.axis_B:
                A_proc = (int(self.axis_A[0] * self.scale), int(self.axis_A[1] * self.scale))
                B_proc = (int(self.axis_B[0] * self.scale), int(self.axis_B[1] * self.scale))
                
                # Draw axis line
                cv2.line(vis, A_proc, B_proc, (0, 255, 0), 2)
                
                # Draw axis endpoints
                cv2.circle(vis, A_proc, 4, (0, 255, 0), -1)
                cv2.circle(vis, B_proc, 4, (0, 255, 0), -1)
                
                # Draw current position on axis
                t = self.current_position / 100.0
                pos_x = int(A_proc[0] + t * (B_proc[0] - A_proc[0]))
                pos_y = int(A_proc[1] + t * (B_proc[1] - A_proc[1]))
                
                # Color based on confidence
                color_intensity = int(255 * self.confidence)
                cv2.circle(vis, (pos_x, pos_y), 8, (255, color_intensity, 0), -1)
                
                # Draw confidence ring
                if self.confidence > 0.1:
                    radius = int(5 + 10 * self.confidence)
                    cv2.circle(vis, (pos_x, pos_y), radius, (255, color_intensity, 0), 1)
            
            # Draw motion center if detected
            if motion_center:
                cv2.circle(vis, (int(motion_center[0]), int(motion_center[1])), 5, (0, 0, 255), -1)
                cv2.circle(vis, (int(motion_center[0]), int(motion_center[1])), 15, (0, 0, 255), 2)
            
            # Status text
            cv2.putText(vis, f"Working Axis Tracker", (10, 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.putText(vis, f"Pos: {self.current_position}/100  Conf: {self.confidence:.2f}", 
                       (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.debug_motion_pixels > 0:
                cv2.putText(vis, f"Motion: {self.debug_motion_pixels}px", 
                           (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            
            if self.tracking_active:
                cv2.putText(vis, "TRACKING", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except:
            pass  # Don't let visualization errors crash the tracker
        
        return vis
    
    def start_tracking(self):
        """Start tracking."""
        self.tracking_active = True
        axis_status = "SET" if (self.axis_A and self.axis_B) else "NOT SET"
        self.logger.info(f"Working axis projection tracking started - Axis: {axis_status}")
        if self.axis_A and self.axis_B:
            self.logger.info(f"Axis coordinates: A={self.axis_A}, B={self.axis_B}")
        else:
            self.logger.warning("WARNING: Axis not set! This will cause flat line output.")
    
    def stop_tracking(self):
        """Stop tracking."""
        self.tracking_active = False
        self.logger.info("Working axis projection tracking stopped")
    
    def cleanup(self):
        """Clean up resources."""
        self.prev_gray = None
        self.logger.debug("Working axis projection tracker cleaned up")
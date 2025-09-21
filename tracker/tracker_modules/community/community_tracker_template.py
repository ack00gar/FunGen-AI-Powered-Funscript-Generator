#!/usr/bin/env python3
"""
Community Tracker Template - Example implementation for custom trackers.

This is a complete working example that demonstrates how to create a custom
tracker that integrates seamlessly with the modular tracker system.

Copy this file, rename it, and modify the implementation to create your own
tracking algorithm. The tracker will be automatically discovered and made
available in the UI.

Author: Community Template
Version: 1.0.0
"""

import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple

try:
    from ..core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult
except ImportError:
    from tracker.tracker_modules.core.base_tracker import BaseTracker, TrackerMetadata, TrackerResult


class CommunityExampleTracker(BaseTracker):
    """
    Example tracker implementation showing all required patterns.
    
    This tracker demonstrates:
    - Basic frame processing with motion detection
    - Settings management and validation
    - ROI support
    - Funscript action generation
    - Visual overlay rendering
    - Error handling and logging
    """
    
    def __init__(self):
        super().__init__()
        
        # Tracker-specific state variables
        self.previous_frame = None
        self.roi_rect = None
        self.motion_history = []
        self.action_buffer = []
        
        # Settings with defaults
        self.motion_threshold = 10.0
        self.sensitivity = 1.0
        self.smoothing_factor = 0.8
        self.min_action_interval_ms = 100
        
        # State tracking
        self.last_action_time = 0
        self.frame_count = 0
    
    @property
    def metadata(self) -> TrackerMetadata:
        """Return metadata describing this tracker."""
        return TrackerMetadata(
            name="community_example",
            display_name="Community Example Tracker",
            description="Template tracker showing basic motion detection and funscript generation",
            category="community",
            version="1.0.0",
            author="Community Template",
            tags=["motion", "template", "example"],
            requires_roi=False,  # ROI is optional but supported
            supports_dual_axis=True
        )
    
    def initialize(self, app_instance, **kwargs) -> bool:
        """
        Initialize the tracker with app instance and settings.
        
        Args:
            app_instance: Main application with access to settings, funscript, etc.
            **kwargs: Additional initialization parameters
        
        Returns:
            bool: True if initialization successful
        """
        try:
            self.app = app_instance
            
            # Load settings from app
            if hasattr(app_instance, 'app_settings'):
                settings = app_instance.app_settings
                
                # Load tracker-specific settings with fallbacks
                self.motion_threshold = settings.get('community_example_motion_threshold', 10.0)
                self.sensitivity = settings.get('community_example_sensitivity', 1.0)
                self.smoothing_factor = settings.get('community_example_smoothing', 0.8)
                self.min_action_interval_ms = settings.get('community_example_min_interval', 100)
                
                self.logger.info(f"Loaded settings: threshold={self.motion_threshold}, sensitivity={self.sensitivity}")
            
            # Reset state
            self.previous_frame = None
            self.motion_history = []
            self.action_buffer = []
            self.last_action_time = 0
            self.frame_count = 0
            
            # Validate settings
            if not self.validate_settings({
                'motion_threshold': self.motion_threshold,
                'sensitivity': self.sensitivity,
                'smoothing_factor': self.smoothing_factor
            }):
                self.logger.error("Settings validation failed")
                return False
            
            self._initialized = True
            self.logger.info("Community example tracker initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Initialization failed: {e}")
            return False
    
    def process_frame(self, frame: np.ndarray, frame_time_ms: int, 
                     frame_index: Optional[int] = None) -> TrackerResult:
        """
        Process a frame and generate tracking results.
        
        This example implementation:
        1. Converts frame to grayscale
        2. Detects motion using frame differencing
        3. Generates funscript actions based on motion intensity
        4. Applies visual overlays
        5. Returns results
        """
        try:
            self.frame_count += 1
            processed_frame = frame.copy()
            action_log = []
            debug_info = {}
            
            # Convert to grayscale for motion detection
            gray_frame = self._convert_to_grayscale(frame)
            
            # Apply ROI if set
            if self.roi_rect:
                x, y, w, h = self.roi_rect
                roi_gray = gray_frame[y:y+h, x:x+w]
                # Draw ROI rectangle
                self._draw_roi_rectangle(processed_frame, self.roi_rect, (0, 255, 0))
            else:
                roi_gray = gray_frame
            
            # Motion detection
            motion_intensity = 0.0
            if self.previous_frame is not None:
                motion_intensity = self._detect_motion(roi_gray, self.previous_frame)
            
            # Update motion history with smoothing
            self.motion_history.append(motion_intensity)
            if len(self.motion_history) > 10:  # Keep last 10 frames
                self.motion_history.pop(0)
            
            smoothed_motion = self._apply_smoothing(self.motion_history)
            
            # Generate funscript actions if tracking is active
            if self.tracking_active and smoothed_motion > self.motion_threshold:
                action = self._generate_funscript_action(smoothed_motion, frame_time_ms)
                if action:
                    action_log.append(action)
            
            # Apply visual overlays
            self._draw_motion_visualization(processed_frame, smoothed_motion)
            self._draw_status_overlay(processed_frame, smoothed_motion, len(action_log) > 0)
            
            # Store current frame for next iteration
            self.previous_frame = roi_gray.copy() if self.roi_rect else gray_frame.copy()
            
            # Prepare debug info
            debug_info = {
                'motion_intensity': float(smoothed_motion),
                'motion_threshold': float(self.motion_threshold),
                'tracking_active': self.tracking_active,
                'frame_count': self.frame_count,
                'roi_active': self.roi_rect is not None,
                'actions_generated': len(action_log)
            }
            
            status_msg = f"Motion: {smoothed_motion:.1f} | Active: {self.tracking_active}"
            
            return TrackerResult(
                processed_frame=processed_frame,
                action_log=action_log,
                debug_info=debug_info,
                status_message=status_msg
            )
            
        except Exception as e:
            self.logger.error(f"Frame processing error: {e}")
            # Return original frame on error to prevent crash
            return TrackerResult(
                processed_frame=frame,
                action_log=None,
                debug_info={'error': str(e)},
                status_message=f"Error: {e}"
            )
    
    def start_tracking(self) -> bool:
        """Start the tracking session."""
        try:
            if not self._initialized:
                self.logger.error("Tracker not initialized")
                return False
            
            self.tracking_active = True
            self.last_action_time = 0
            self.action_buffer = []
            
            self.logger.info("Community example tracker started")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start tracking: {e}")
            return False
    
    def stop_tracking(self) -> bool:
        """Stop the tracking session."""
        try:
            self.tracking_active = False
            self.logger.info("Community example tracker stopped")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to stop tracking: {e}")
            return False
    
    def set_roi(self, roi: Tuple[int, int, int, int]) -> bool:
        """
        Set region of interest for tracking.
        
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
            
            self.roi_rect = roi
            self.logger.info(f"ROI set to: {roi}")
            
            # Reset previous frame when ROI changes
            self.previous_frame = None
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to set ROI: {e}")
            return False
    
    def validate_settings(self, settings: Dict[str, Any]) -> bool:
        """Validate tracker-specific settings."""
        try:
            # Validate motion threshold
            threshold = settings.get('motion_threshold', self.motion_threshold)
            if not isinstance(threshold, (int, float)) or threshold < 0:
                self.logger.error("Motion threshold must be a non-negative number")
                return False
            
            # Validate sensitivity
            sensitivity = settings.get('sensitivity', self.sensitivity)
            if not isinstance(sensitivity, (int, float)) or sensitivity <= 0:
                self.logger.error("Sensitivity must be a positive number")
                return False
            
            # Validate smoothing factor
            smoothing = settings.get('smoothing_factor', self.smoothing_factor)
            if not isinstance(smoothing, (int, float)) or not (0 <= smoothing <= 1):
                self.logger.error("Smoothing factor must be between 0 and 1")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Settings validation error: {e}")
            return False
    
    def get_status_info(self) -> Dict[str, Any]:
        """Get current status information for UI display."""
        base_status = super().get_status_info()
        
        custom_status = {
            'motion_threshold': self.motion_threshold,
            'sensitivity': self.sensitivity,
            'frame_count': self.frame_count,
            'roi_set': self.roi_rect is not None,
            'motion_history_length': len(self.motion_history)
        }
        
        base_status.update(custom_status)
        return base_status
    
    def get_settings_schema(self) -> Dict[str, Any]:
        """Get JSON schema for tracker settings UI."""
        return {
            "type": "object",
            "properties": {
                "community_example_motion_threshold": {
                    "type": "number",
                    "title": "Motion Threshold",
                    "description": "Minimum motion intensity to trigger actions",
                    "minimum": 0,
                    "maximum": 100,
                    "default": 10.0
                },
                "community_example_sensitivity": {
                    "type": "number", 
                    "title": "Sensitivity",
                    "description": "Overall sensitivity multiplier",
                    "minimum": 0.1,
                    "maximum": 5.0,
                    "default": 1.0
                },
                "community_example_smoothing": {
                    "type": "number",
                    "title": "Motion Smoothing",
                    "description": "Smoothing factor for motion detection (0=no smoothing, 1=maximum)",
                    "minimum": 0,
                    "maximum": 1,
                    "default": 0.8
                },
                "community_example_min_interval": {
                    "type": "integer",
                    "title": "Min Action Interval (ms)",
                    "description": "Minimum time between funscript actions",
                    "minimum": 50,
                    "maximum": 1000,
                    "default": 100
                }
            }
        }
    
    def cleanup(self):
        """Clean up resources when tracker is being destroyed."""
        try:
            self.previous_frame = None
            self.motion_history = []
            self.action_buffer = []
            self.roi_rect = None
            
            # self.logger.info("Community example tracker cleaned up")
            
        except Exception as e:
            self.logger.error(f"Cleanup error: {e}")
    
    # Private helper methods
    
    def _convert_to_grayscale(self, frame: np.ndarray) -> np.ndarray:
        """Convert BGR frame to grayscale."""
        if len(frame.shape) == 3:
            return np.mean(frame, axis=2).astype(np.uint8)
        return frame
    
    def _detect_motion(self, current_gray: np.ndarray, previous_gray: np.ndarray) -> float:
        """Detect motion between two grayscale frames."""
        if current_gray.shape != previous_gray.shape:
            return 0.0
        
        # Simple frame difference
        diff = np.abs(current_gray.astype(np.float32) - previous_gray.astype(np.float32))
        motion = np.mean(diff) * self.sensitivity
        
        return float(motion)
    
    def _apply_smoothing(self, motion_values: List[float]) -> float:
        """Apply smoothing to motion values."""
        if not motion_values:
            return 0.0
        
        if len(motion_values) == 1:
            return motion_values[0]
        
        # Exponential moving average
        result = motion_values[0]
        for value in motion_values[1:]:
            result = self.smoothing_factor * result + (1 - self.smoothing_factor) * value
        
        return result
    
    def _generate_funscript_action(self, motion_intensity: float, frame_time_ms: int) -> Optional[Dict]:
        """Generate a funscript action based on motion intensity."""
        # Throttle actions based on minimum interval
        if frame_time_ms - self.last_action_time < self.min_action_interval_ms:
            return None
        
        # Convert motion intensity to funscript position (0-100)
        position = min(100, max(0, int(motion_intensity * 5)))  # Scale motion to position
        
        # Add action to app's funscript
        if hasattr(self.app, 'funscript'):
            self.app.funscript.add_action(frame_time_ms, position)
        
        self.last_action_time = frame_time_ms
        
        return {
            'timestamp': frame_time_ms,
            'position': position,
            'motion_intensity': motion_intensity
        }
    
    def _draw_roi_rectangle(self, frame: np.ndarray, roi: Tuple[int, int, int, int], color: Tuple[int, int, int]):
        """Draw ROI rectangle on frame."""
        try:
            import cv2
            x, y, w, h = roi
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, "ROI", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        except ImportError:
            pass  # OpenCV not available
    
    def _draw_motion_visualization(self, frame: np.ndarray, motion_intensity: float):
        """Draw motion intensity visualization on frame."""
        try:
            import cv2
            
            # Draw motion bar
            bar_width = int((motion_intensity / max(self.motion_threshold * 2, 1)) * 200)
            bar_width = min(bar_width, 200)
            
            # Color based on intensity (green -> yellow -> red)
            if motion_intensity < self.motion_threshold:
                color = (0, 255, 0)  # Green
            elif motion_intensity < self.motion_threshold * 2:
                color = (0, 255, 255)  # Yellow  
            else:
                color = (0, 0, 255)  # Red
            
            cv2.rectangle(frame, (10, 10), (10 + bar_width, 30), color, -1)
            cv2.rectangle(frame, (10, 10), (210, 30), (255, 255, 255), 1)
            
        except ImportError:
            pass  # OpenCV not available
    
    def _draw_status_overlay(self, frame: np.ndarray, motion_intensity: float, action_generated: bool):
        """Draw status information on frame."""
        try:
            import cv2
            
            status_text = f"Motion: {motion_intensity:.1f}"
            if action_generated:
                status_text += " | ACTION"
            
            cv2.putText(frame, status_text, (10, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            if self.tracking_active:
                cv2.putText(frame, "TRACKING", (10, frame.shape[0] - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
        except ImportError:
            pass  # OpenCV not available


# The tracker will be automatically discovered and registered
# No additional registration code needed!